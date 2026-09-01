from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ImportError:  # Safe fallback for restricted deployments.
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def _ratio(left: object, right: object) -> float:
            return 100.0 * SequenceMatcher(
                None,
                str(left),
                str(right),
            ).ratio()

        WRatio = _ratio
        token_sort_ratio = staticmethod(
            lambda left, right: _FallbackFuzz._ratio(
                " ".join(sorted(str(left).split())),
                " ".join(sorted(str(right).split())),
            )
        )
        token_set_ratio = staticmethod(
            lambda left, right: _FallbackFuzz._ratio(
                " ".join(sorted(set(str(left).split()))),
                " ".join(sorted(set(str(right).split()))),
            )
        )

    class _FallbackProcess:
        @staticmethod
        def extract(
            query: object,
            choices: Iterable[object],
            scorer,
            limit: int = 5,
            score_cutoff: float = 0,
        ) -> list[tuple[object, float, int]]:
            choice_values = list(choices)
            query_tokens = set(str(query).split())
            narrowed_choices = [
                (index, choice)
                for index, choice in enumerate(choice_values)
                if query_tokens & set(str(choice).split())
            ]
            if not narrowed_choices:
                narrowed_choices = list(enumerate(choice_values))

            scored = [
                (choice, scorer(query, choice), index)
                for index, choice in narrowed_choices
            ]
            return [
                item
                for item in sorted(
                    scored,
                    key=lambda item: item[1],
                    reverse=True,
                )
                if item[1] >= score_cutoff
            ][:limit]

    fuzz = _FallbackFuzz()
    process = _FallbackProcess()


ProgressCallback = Callable[
    [int, int, str],
    None,
]


MATCHER_VERSION = "2026.09.01.2-IDENTITY-FIRST"


MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "Â": "",
}


CITY_ALIASES = {
    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",
    "bombay": "mumbai",
    "mumbai": "mumbai",
    "calcutta": "kolkata",
    "kolkata": "kolkata",
    "mysore": "mysuru",
    "mysuru": "mysuru",
    "trivandrum": "thiruvananthapuram",
    "thiruvananthapuram": "thiruvananthapuram",
    "allahabad": "prayagraj",
    "prayagraj": "prayagraj",
    "gurgaon": "gurugram",
    "gurugram": "gurugram",
    "baroda": "vadodara",
    "vadodara": "vadodara",
    "cochin": "kochi",
    "kochi": "kochi",
    "madras": "chennai",
    "chennai": "chennai",
    "new delhi": "new delhi",
    "delhi": "new delhi",
    "greater noida": "greater noida",
    "noida": "noida",
    "bhopal": "bhopal",
    "indore": "indore",
    "jabalpur": "jabalpur",
    "gwalior": "gwalior",
    "sagar": "sagar",
    "ujjain": "ujjain",
    "jaipur": "jaipur",
    "lucknow": "lucknow",
    "pune": "pune",
    "hyderabad": "hyderabad",
    "ahmedabad": "ahmedabad",
    "chandigarh": "chandigarh",
    "dehradun": "dehradun",
    "calicut": "kozhikode",
    "kozhikode": "kozhikode",
    "trichy": "tiruchirappalli",
    "tiruchirapalli": "tiruchirappalli",
    "tiruchirappalli": "tiruchirappalli",
    "panji": "panaji",
    "panaji": "panaji",
    "vadodra": "vadodara",
    "burla": "sambalpur",
    "sambalpur": "sambalpur",
    "shibpur": "shibpur",
    "belgaum": "belagavi",
    "belagavi": "belagavi",
    # Common campus/locality versus district or metro spellings used by
    # the master database.  These are location equivalents, not fuzzy
    # guesses, and therefore remain safe campus guards.
    "vaddeswaram": "guntur",
    "surampalem": "east godavari",
    "peddapuram": "east godavari",
    "yadrav": "kolhapur",
    "ichalkaranji": "kolhapur",
    "tiruchengode": "namakkal",
    "ujire": "mangalore",
    "panvel": "mumbai",
    "navi mumbai": "mumbai",
    "ernakulam": "kochi",
    "narsapur": "hyderabad",
    "medak": "medak",
    "kopargaon": "ahmednagar",
    "ahmed nagar": "ahmednagar",
    "ahmednagar": "ahmednagar",
    "bardoli": "surat",
    "surat": "surat",
    "ottapalam": "palakkad",
    "palakkad": "palakkad",
    "ongole": "prakasam",
    "prakasam": "prakasam",
    "thrikkakara": "kochi",
    "keesara": "hyderabad",
}


PARENT_MARKERS = {
    "faculty",
    "department",
    "school",
    "centre",
    "center",
    "academy",
    "institute",
}


INSTITUTION_WORDS = {
    "university",
    "institute",
    "college",
}


GENERIC_NAME_TOKENS = {
    "the",
    "of",
    "and",
    "for",
    "in",
    "at",
    "college",
    "university",
    "institute",
    "institution",
    "institutions",
    "school",
    "academy",
    "faculty",
    "department",
    "centre",
    "center",
    "campus",
    "medical",
    "dental",
    "engineering",
    "technology",
    "management",
    "research",
    "hospital",
    "arts",
    "science",
    "sciences",
    "studies",
}


ABBREVIATION_STOP_WORDS = {
    "the",
    "of",
    "and",
    "for",
    "in",
    "at",
}


DOMAIN_MARKERS = {
    "medical",
    "pharmacy",
    "pharmaceutical",
    "dental",
    "ayurvedic",
    "ayurveda",
    "homeopathic",
    "homoeopathic",
    "nursing",
    "engineering",
    "technology",
    "management",
    "law",
    "architecture",
    "polytechnic",
    "agriculture",
    "veterinary",
}


# Words that describe legal status, ownership or academic category but do
# not identify an institution. They are ignored only for identity evidence;
# the original normalized text is still retained for exact/fuzzy scoring.
IDENTITY_NOISE_TOKENS = GENERIC_NAME_TOKENS | DOMAIN_MARKERS | {
    "autonomous",
    "ugc",
    "deemed",
    "private",
    "formerly",
    "former",
    "known",
    "approved",
    "affiliated",
    "education",
    "foundation",
    "group",
    "society",
    "trust",
    "all",
    "india",
    "shri",
    "sri",
    "dr",
    "pvt",
    "limited",
    "ltd",
    "pg",
    "to",
    "be",
    "as",
}


MATCH_NOISE_TOKENS = {
    "autonomous",
    "ugc",
    "deemed",
    "private",
    "formerly",
    "former",
    "known",
    "approved",
    "affiliated",
    "pvt",
    "limited",
    "ltd",
}


def normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)

    for incorrect, correct in (
        MOJIBAKE_REPLACEMENTS.items()
    ):
        text = text.replace(
            incorrect,
            correct,
        )

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Split accidentally joined CamelCase words before case-folding:
    # TechnologicalUniversity -> Technological University.
    text = re.sub(
        r"(?<=[a-z])(?=[A-Z])",
        " ",
        text,
    )

    # Keep dotted and spaced initialisms as one identity token:
    # P.E.S. -> PES, K. J. -> KJ and K S R -> KSR.
    text = re.sub(
        r"\b(?:[A-Za-z]\s*\.\s*)+[A-Za-z]\.?(?=\s|$)",
        lambda match: re.sub(
            r"[^A-Za-z]",
            "",
            match.group(0),
        ),
        text,
    )

    text = re.sub(
        r"\b(?:[A-Z]\s+){1,5}[A-Z]\b",
        lambda match: re.sub(
            r"\s+",
            "",
            match.group(0),
        ),
        text,
    )

    text = text.casefold()

    # Make possessive and non-possessive spellings equivalent:
    # People's -> Peoples, Teachers' -> Teachers.
    text = re.sub(
        r"(?<=s)['’]s\b",
        "",
        text,
    )

    text = re.sub(
        r"(?<=\w)['’]s\b",
        "s",
        text,
    )

    text = re.sub(
        r"(?<=\w)['’](?=\s|$)",
        "",
        text,
    )

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    # Common upload typos must not break otherwise exact institution
    # identities. Apply these after punctuation cleanup so replacements
    # are token-safe for both master and input text.
    typo_tokens = {
        "insitute": "institute",
        "institue": "institute",
        "univeristy": "university",
        "unversity": "university",
        "enginering": "engineering",
        "engineeering": "engineering",
        "architechture": "architecture",
        "milia": "millia",
        "rookee": "roorkee",
    }
    text = " ".join(
        typo_tokens.get(token, token)
        for token in text.split()
    )

    return " ".join(text.split())


def academic_intents(value: object) -> set[str]:
    """Map unlike course and department labels to shared concepts."""

    clean = normalize(value)
    padded = f" {clean} "
    intents: set[str] = set()
    phrase_map = {
        "architecture": (" architecture ", " b arch ", " barch "),
        "planning": (" planning ", " b plan ", " bplan "),
        "engineering": (
            " engineering ", " b tech ", " btech ", " m tech ", " mtech "
        ),
        "management": (" management ", " mba ", " pgdm "),
        "pharmacy": (
            " pharmacy ", " b pharm ", " bpharm ", " m pharm ", " mpharm "
        ),
        "law": (" law ", " llb ", " llm "),
    }
    for intent, phrases in phrase_map.items():
        if any(phrase in padded for phrase in phrases):
            intents.add(intent)
    return intents


def match_identity_phrase(value: object) -> str:
    """Normalize legal/status wording without deleting academic identity."""

    return " ".join(
        token
        for token in normalize(value).split()
        if token not in MATCH_NOISE_TOKENS
    )


def national_system_families(
    college_name: object,
    short_form: object = "",
) -> set[str]:
    """Identify only genuine national-system institution records.

    Initials that merely begin with IIT/NIT are deliberately excluded.
    For example, India International Trade Centre (IITC) is not an IIT.
    """

    clean_name = normalize(college_name)
    clean_short = normalize(short_form)
    combined = f"{clean_name} {clean_short}".strip()
    short_first = clean_short.split()[0] if clean_short else ""
    families: set[str] = set()

    if (
        "indian institute of information technology" in combined
        or short_first == "iiit"
        or clean_name.startswith("iiit ")
    ):
        families.add("iiit")

    if (
        "indian institute of technology" in combined
        or short_first == "iit"
        or clean_name.startswith("iit ")
    ) and "indian institute of information technology" not in combined:
        families.add("iit")

    if (
        "national institute of technology" in combined
        or short_first == "nit"
        or clean_name.startswith("nit ")
    ):
        families.add("nit")

    if (
        "indian institute of management" in combined
        or short_first == "iim"
        or clean_name.startswith("iim ")
    ):
        families.add("iim")

    if (
        "all india institute of medical sciences" in combined
        or short_first == "aiims"
        or clean_name.startswith("aiims ")
    ):
        families.add("aiims")

    if (
        "birla institute of technology and science" in combined
        or short_first == "bits"
        or clean_name.startswith("bits ")
    ):
        families.add("bits")

    return families


def structural_identity(value: object) -> str:
    """Extract the institution and ignore descriptive sub-unit suffixes."""

    raw_value = "" if value is None else str(value)
    raw_segments = [
        segment.strip()
        for segment in raw_value.split(",")
        if segment.strip()
    ]

    if not raw_segments:
        return normalize(value)

    identity_markers = {
        "university", "institute", "college", "school", "faculty",
        "department", "technology", "engineering", "iit", "nit",
        "iiit", "bits", "spa",
    }
    first_index = 0
    for index, segment in enumerate(raw_segments):
        if set(normalize(segment).split()) & identity_markers:
            first_index = index
            break

    # Skip legal/promoter prefixes such as "BRACT's" and "The Shirpur
    # Education Society's" when a later segment names the institution.
    raw_segments = raw_segments[first_index:]
    first_raw = raw_segments[0]
    first = normalize(first_raw)

    # A named college after a prefix is more specific than the prefix:
    # "AMU - Zakir Hussain College ..." and "CET - College ...".
    hyphen_parts = re.split(r"\s+-\s+", first_raw, maxsplit=1)
    if len(hyphen_parts) == 2:
        named_part = normalize(hyphen_parts[1])
        if " college " in f" {named_part} ":
            return named_part

    # Generic academic units appearing before a named university/college
    # belong together. This preserves identities such as "Institute of
    # Technology, Nirma University" and "Faculty of Engineering,
    # Jadavpur University" instead of matching the generic first phrase.
    first_tokens = first.split()
    leading_subunit = bool(
        first_tokens
        and (
            first_tokens[0]
            in {"department", "faculty", "school", "institute", "college"}
            or first.startswith("university institute ")
        )
    )

    if leading_subunit:
        for later_raw in raw_segments[1:]:
            later = normalize(later_raw)
            if set(later.split()) & {
                "university", "college", "institute", "iit", "nit", "iiit"
            }:
                return f"{first} {later}".strip()

    first_is_parent = bool(
        set(first_tokens) & {"university", "technology"}
    )

    if first_is_parent:
        # A separately named constituent college (such as Sir J. J.
        # College) wins. Department/faculty/school/institute phrases are
        # sub-units and must not redirect the match to a department page.
        for later_raw in raw_segments[1:]:
            later = normalize(later_raw)
            if (
                " college " in f" {later} "
                and identity_tokens(later)
            ):
                return later

        return first

    # When a descriptive unit comes first and its university/college is
    # named in a later segment, keep both identities. Examples include
    # "College of Engineering, Anna University" and
    # "Faculty of Engineering, Jadavpur University".
    return first


def match_context_key(
    college_name: object,
    city: object = "",
    state: object = "",
) -> str:
    """Stable key that keeps same-name campuses separate."""

    return "||".join(
        [
            normalize(college_name),
            canonical_city(city),
            normalize(state),
        ]
    )


def remove_bracket_alias(
    value: object,
) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(
        r"\s*-?\s*\[[^\]]+\]",
        " ",
        text,
    )

    return normalize(text)


def canonical_city(
    value: object,
) -> str:
    clean_city = normalize(value)

    return CITY_ALIASES.get(
        clean_city,
        clean_city,
    )


def meaningful_tokens(
    value: str,
) -> set[str]:
    return {
        token
        for token in value.split()
        if token not in GENERIC_NAME_TOKENS
        and len(token) > 1
    }


def identity_tokens(value: object) -> set[str]:
    """Return proper-name evidence, excluding status/category wording."""

    tokens = {
        token
        for token in normalize(value).split()
        if token not in IDENTITY_NOISE_TOKENS
        and len(token) > 1
    }

    return tokens


def input_initialisms(value: object) -> set[str]:
    """Create acronym evidence from an uploaded institution label.

    Existing compact prefixes are preserved: ``J.B. Institute ...`` can
    produce JBIET, while a fully written promoter name can produce AISSMS.
    These values retrieve candidates only; uniqueness, city and name/domain
    checks still decide whether the result is safe.
    """

    tokens = [
        token
        for token in normalize(value).split()
        if token not in ABBREVIATION_STOP_WORDS
        and token not in {
            "autonomous", "ugc", "deemed", "formerly", "known", "as"
        }
    ]

    if not tokens:
        return set()

    values = set(derived_abbreviations(" ".join(tokens)))
    values.update(
        token
        for token in tokens
        if 2 <= len(token) <= 12
    )

    first = tokens[0]
    if 2 <= len(first) <= 5:
        for length in range(1, len(tokens)):
            value = first + "".join(
                token[0]
                for token in tokens[1:length + 1]
            )
            if 2 <= len(value) <= 12:
                values.add(value)

    unit_suffix = ""
    unit_start = None
    token_set = set(tokens)
    if "college" in token_set and "engineering" in token_set:
        unit_suffix = "coe"
        unit_start = tokens.index("college")
    elif (
        "institute" in token_set
        and "information" in token_set
        and "technology" in token_set
    ):
        unit_suffix = "iit"
        unit_start = tokens.index("institute")
    elif "college" in token_set and "pharmacy" in token_set:
        unit_suffix = "cop"
        unit_start = tokens.index("college")

    if unit_suffix and unit_start:
        promoter_tokens = tokens[:unit_start]
        promoter_initials = "".join(token[0] for token in promoter_tokens)
        if len(promoter_initials) >= 3:
            values.add((promoter_initials + unit_suffix)[:12])

    return values


def canonical_domain_markers(value: object) -> set[str]:
    """Group equivalent academic-domain words before conflict checks."""

    markers = set(normalize(value).split()) & DOMAIN_MARKERS
    canonical = set(markers)

    if markers & {"engineering", "technology", "polytechnic"}:
        canonical -= {"engineering", "technology", "polytechnic"}
        canonical.add("engineering_technology")

    if markers & {"pharmacy", "pharmaceutical"}:
        canonical -= {"pharmacy", "pharmaceutical"}
        canonical.add("pharmacy")

    if markers & {"ayurvedic", "ayurveda"}:
        canonical -= {"ayurvedic", "ayurveda"}
        canonical.add("ayurveda")

    if markers & {"homeopathic", "homoeopathic"}:
        canonical -= {"homeopathic", "homoeopathic"}
        canonical.add("homeopathy")

    return canonical


def derived_abbreviations(
    value: str,
) -> set[str]:
    """Create safe acronym variants from a normalized college name."""

    tokens = [
        token
        for token in normalize(value).split()
        if token not in ABBREVIATION_STOP_WORDS
    ]

    if len(tokens) < 2:
        return set()

    # Register prefix acronyms as well as the complete acronym.
    # Example: Lakshmi Narain Medical College and Research Centre
    # produces LN, LNM, LNMC, LNMCR and LNMCRC. This allows a real
    # abbreviation to match its full form without confusing it with
    # another medical college in the same city.
    abbreviations = {
        "".join(
            token[0]
            for token in tokens[:prefix_length]
        )
        for prefix_length in range(
            2,
            len(tokens) + 1,
        )
    }

    return {
        abbreviation
        for abbreviation in abbreviations
        if 2 <= len(abbreviation) <= 12
    }


def is_valid_parent_name(
    base_name: str,
) -> bool:
    tokens = base_name.split()

    if len(tokens) >= 3:
        return True

    if (
        len(tokens) >= 2
        and (
            set(tokens)
            & INSTITUTION_WORDS
        )
    ):
        return True

    return False


@dataclass(frozen=True)
class Candidate:
    college_id: int
    college_name: str
    city: str
    state: str
    college_type: str
    confidence: float
    token_overlap: float
    reason: str

    @property
    def label(self) -> str:
        return (
            f"{self.college_id} | "
            f"{self.college_name} | "
            f"{self.city}, {self.state} | "
            f"Confidence: {self.confidence:.1f}"
        )


@dataclass(frozen=True)
class MatchDecision:
    input_name: str
    normalized_name: str
    college_id: int | str
    matched_name: str | None
    confidence: float
    decision: str
    reason: str
    candidates: list[Candidate]


class CollegeMatcher:
    REQUIRED_MASTER_COLUMNS = {
        "College Id",
        "College Name",
        "City",
        "State",
    }

    # Deterministic verified rules. They are used only when the specified
    # College ID is present in the currently loaded master database.
    VERIFIED_INPUT_TO_COLLEGE_ID = {
        "ln medical college bhopal": 57033,
        "peoples college of dental sciences and research centre bhopal": 58702,
        "mit art design and technology university mitadt": 57115,
        "balaji college of arts commerce and science bcacs": 472,
        "mit arts commerce and science college mitacsc": 59308,
        "iit kharagpur department of architecture and regional planning indian institute of technology kharagpur": 26007,
        "iit roorkee department of architecture and planning indian institute of technology roorkee": 25992,
        "indian institute of technology bhu varanasi department of architecture planning and design varanasi": 25947,
        "iiest shibpur department of architecture and planning shibpur": 28330,
        "cet college of engineering trivandrum department of architecture thiruvananthapuram": 5563,
        "nit calicut national institute of technology department of architecture and planning kozhikode": 25651,
        "nit trichy department of architecture tiruchirapalli": 25889,
        "jamia millia islamia faculty of architecture and ekistics new delhi": 25460,
        "vnit nagpur department of architecture and planning nagpur": 25733,
        "anna university school of architecture and planning chennai": 56307,
        "jadavpur university faculty of architecture kolkata": 26008,
        "maharaja sayajirao university of baroda vadodra": 25500,
        "university of mumbai sir j j college of architecture mumbai": 5680,
        "goa college of architecture panaji": 5586,
        "nit kozhikode national institute of technology faculty of architecture kozhikode": 25651,
        "andhra university department of architecture visakhapatnam": 25346,
        "dr a p j abdul kalam technical university faculty of architecture lucknow": 25941,
        "government engineering college school of architecture and planning thrissur": 13614,
        "bundelkhand university institute of architecture and town planning jhansi": 25927,
        "hemchandracharya north gujarat university institute of architecture patan": 25493,
        "mbm university department of architecture jodhpur": 14122,
        "gautam buddha university department of architecture and regional planning greater noida": 25942,
        "veer surendra sai university of technology department of architecture sambalpur": 25771,
        "mizoram university department of planning and architecture tanhril": 25749,
        "rajiv gandhi government engineering college school of architecture kangra": 60047,
        "sarvajanik university institute of design planning and technology scet idpt scet surat": 14954,
        "amu zakir hussain college of engineering and technology department of architecture aligarh": 5696,
        "netaji subhas university of technology department of architecture and planning new delhi": 14479,
        "indira gandhi delhi technical university for women department of architecture delhi": 13801,
        "rajiv gandhi proudyogiki vishwavidyalaya school of architecture bhopal": 25681,
        "new delhi institute of management ndim": 57120,
        # Audit-verified spelling/order aliases. These lock only an exact
        # normalized input label to a master ID; other uploads still use the
        # generic identity/campus rules below.
        "faculty of engineering and technology jadavpur university kolkata": 26008,
        "vignans foundation for science technology and research guntur": 25384,
        "dr vishwanath karad mit world peace university pune": 18416,
        "east point college of engineering and technology bengaluru": 13428,
        "koneru lakshmaiah education foundation deemed to be university vaddeswaram": 25362,
        "all india shri shivaji memorial societys college of engineering pune": 12779,
        "all india shri shivaji memorial societys institute of information technology pune": 12780,
        "padmasri dr b v raju institute of technology bvrit narsapur": 24327,
        "sanjivani college of engineering kopargaon": 14923,
        "jawaharlal college of engineering and technology palakkad": 45712,
        "faculty of architecture and ekistics jamia millia islamia new delhi": 25460,
        "faculty of architecture and planning aktu lucknow": 25941,
        "integral university faculty of engineering lucknow": 25950,
        "faculty of architecture and planning integral university lucknow": 25950,
        "sharad institute of technology college of engineering yadrav ichalkaranji yadrav ichalkaranji": 15021,
        "the oxford college of architecture bengaluru": 61956,
        "sacred heart college autonomous kochi": 55578,
        "mehr chand mahajan dav college for women chandigarh": 2976,
        "chandigarh engineering college cgc landran mohali mohali": 13197,
        "mlr institute of technology hyderabad": 58194,
        "dr d y patil college of engineering and innovation pune": 58797,
        "jlu school of engineering and technology bhopal": 58674,
        "rise krishan sai prakasam group of institutions ongole": 62801,
        "sri dharmasthala manjunatheshwara college autonomous ujire": 56575,
        "govt model engineering college kochi": 14344,
        "geethanjali college of engineering and technology hyderabad": 13548,
        "annamacharya institute of technology and science tirupati": 28390,
        "rajarajeswari college of engineering bengaluru": 56090,
        "department of architecture and planning iit roorkee roorkee": 25992,
        "dav college jalandhar": 894,
        "amity university noida": 54797,
        "amity university gurgaon": 25516,
        "amity university jaipur": 25797,
        "amity university lucknow": 25920,
        "amity university raipur": 55898,
        "amity university gwalior": 25658,
        "nit patna national institute of technology department of architecture patna": 25417,
    }

    # Credible parent suggestions where the precise school/department row is
    # absent from the master. The numeric ID is retained for user review.
    VERIFIED_REVIEW_INPUT_TO_COLLEGE_ID = {
        "roorkee college of engineering roorkee haridwar university roorkee": 55109,
        "social communications media department scm sophia mumbai": 4639,
    }

    SUPPLEMENTAL_VERIFIED_RECORDS = [
        {
            "College Id": 60047,
            "College Name": (
                "Rajiv Gandhi Govt Engineering College - [RGGEC]"
            ),
            "City": "Kangra",
            "State": "Himachal Pradesh",
            "Short_form": "RGGEC Kangra",
            "College Type": "Government",
        }
    ]

    # Confirmed absent institutions cannot inherit another campus ID.
    VERIFIED_NOT_FOUND_INPUTS = {
        "lnct medical college and sewakunj hospital indore",
        "school of architecture soar j jammu",
        "school of engineering and technology mohali mohali",
        "school of architecture world school of planning and architecture sonepat",
        "isbm college of engineering pune pune",
    }

    def __init__(
        self,
        master_dataframe: pd.DataFrame,
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> None:
        missing = (
            self.REQUIRED_MASTER_COLUMNS
            - set(master_dataframe.columns)
        )

        if missing:
            raise ValueError(
                "Master file is missing columns: "
                f"{sorted(missing)}"
            )

        self.master = master_dataframe.copy()

        existing_ids = set(
            pd.to_numeric(
                self.master["College Id"],
                errors="coerce",
            ).dropna().astype(int)
        )

        supplemental_rows = [
            row
            for row in self.SUPPLEMENTAL_VERIFIED_RECORDS
            if int(row["College Id"]) not in existing_ids
        ]

        if supplemental_rows:
            self.master = pd.concat(
                [
                    self.master,
                    pd.DataFrame(supplemental_rows),
                ],
                ignore_index=True,
            )

        if "Short_form" not in self.master:
            self.master["Short_form"] = ""

        if "College Type" not in self.master:
            self.master["College Type"] = ""

        self.progress_callback = (
            progress_callback
        )

        self.college_by_id: dict[
            int,
            dict,
        ] = {}

        self.exact_index: dict[
            str,
            set[int],
        ] = defaultdict(set)

        self.variant_ids: dict[
            str,
            set[int],
        ] = defaultdict(set)

        self.identity_token_ids: dict[
            str,
            set[int],
        ] = defaultdict(set)

        self.acronym_ids: dict[
            str,
            set[int],
        ] = defaultdict(set)

        self.token_set_index: dict[
            str,
            set[int],
        ] = defaultdict(set)

        self.city_phrases: set[str] = set()
        self.city_ids: dict[str, set[int]] = defaultdict(set)
        self.search_variants: list[str] = []

        self._build_indexes()

    def _report_progress(
        self,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        if self.progress_callback:
            self.progress_callback(
                completed,
                total,
                message,
            )

    def _register_variant(
        self,
        variant: str,
        college_id: int,
    ) -> None:
        if not variant:
            return

        self.exact_index[
            variant
        ].add(college_id)

        self.variant_ids[
            variant
        ].add(college_id)

    def _build_indexes(self) -> None:
        records = self.master.to_dict(
            "records"
        )

        total = len(records)

        for position, row in enumerate(
            records,
            start=1,
        ):
            college_id = int(
                row["College Id"]
            )

            college_name = str(
                row["College Name"]
            )

            city = (
                ""
                if pd.isna(row["City"])
                else str(row["City"])
            )

            state = (
                ""
                if pd.isna(row["State"])
                else str(row["State"])
            )

            short_form = (
                ""
                if pd.isna(row["Short_form"])
                else str(row["Short_form"])
            )

            college_type = (
                ""
                if pd.isna(row["College Type"])
                else str(row["College Type"])
            )

            clean_name = normalize(
                college_name
            )

            clean_base_name = (
                remove_bracket_alias(
                    college_name
                )
            )

            clean_short_form = normalize(
                short_form
            )

            clean_match_name = match_identity_phrase(clean_name)
            clean_match_base = match_identity_phrase(clean_base_name)
            clean_match_short = match_identity_phrase(clean_short_form)

            short_acronyms = {
                normalize(token)
                for token in re.findall(
                    r"\b[A-Z][A-Z0-9]{1,}\b",
                    short_form,
                )
                if normalize(token) not in IDENTITY_NOISE_TOKENS
                and normalize(token) not in MATCH_NOISE_TOKENS
                and len(normalize(token)) <= 12
            }

            bracket_aliases = {
                normalize(alias).replace(" ", "")
                for alias in re.findall(
                    r"\[([^\]]+)\]",
                    college_name,
                )
                if normalize(alias)
            }

            short_acronyms.update(bracket_aliases)

            acronym_keys = set(short_acronyms)
            for acronym in short_acronyms:
                if len(acronym) >= 4:
                    acronym_keys.update(
                        acronym[:prefix_length]
                        for prefix_length in range(3, len(acronym) + 1)
                    )

            raw_city = normalize(city)

            clean_city = canonical_city(
                city
            )

            def without_record_city(tokens: set[str]) -> set[str]:
                return {
                    token
                    for token in tokens
                    if canonical_city(token) != clean_city
                }

            name_without_city = clean_base_name

            if raw_city:
                name_without_city = " ".join(
                    token
                    for token in clean_base_name.split()
                    if token not in set(raw_city.split())
                )

            abbreviations = (
                derived_abbreviations(clean_name)
                | derived_abbreviations(clean_base_name)
                | derived_abbreviations(name_without_city)
                | bracket_aliases
            )

            if clean_short_form:
                abbreviations.add(
                    clean_short_form.replace(" ", "")
                )

            if raw_city:
                self.city_phrases.add(
                    raw_city
                )

            if clean_city:
                self.city_phrases.add(
                    clean_city
                )
                self.city_ids[clean_city].add(college_id)

            self.college_by_id[
                college_id
            ] = {
                "college_id": college_id,
                "college_name": college_name,
                "clean_name": clean_name,
                "clean_base_name": (
                    clean_base_name
                ),
                "clean_match_name": clean_match_name,
                "clean_match_base": clean_match_base,
                "short_form": clean_short_form,
                "clean_match_short": clean_match_short,
                "abbreviations": abbreviations,
                "city": city,
                "raw_city": raw_city,
                "clean_city": clean_city,
                "state": state,
                "college_type": college_type,
                "parent_identity_tokens": (
                    without_record_city(
                        identity_tokens(clean_base_name)
                    ),
                    without_record_city(
                        identity_tokens(clean_short_form)
                    ),
                ),
                "short_acronyms": short_acronyms,
                "acronym_keys": acronym_keys,
            }

            for identity_token_set in self.college_by_id[
                college_id
            ]["parent_identity_tokens"]:
                for token in identity_token_set:
                    self.identity_token_ids[token].add(college_id)

            for acronym in acronym_keys:
                self.acronym_ids[acronym].add(college_id)

            for value in {
                clean_match_name,
                clean_match_base,
                clean_match_short,
            }:
                if value:
                    fingerprint = " ".join(sorted(value.split()))
                    self.token_set_index[fingerprint].add(college_id)

            # Include both raw and standardized city forms.
            variants = {
                clean_name,
                clean_base_name,
                clean_short_form,
                clean_match_name,
                clean_match_base,
                clean_match_short,
                (
                    f"{clean_name} "
                    f"{raw_city}"
                ).strip(),
                (
                    f"{clean_name} "
                    f"{clean_city}"
                ).strip(),
                (
                    f"{clean_base_name} "
                    f"{raw_city}"
                ).strip(),
                (
                    f"{clean_base_name} "
                    f"{clean_city}"
                ).strip(),
                (
                    f"{clean_short_form} "
                    f"{raw_city}"
                ).strip(),
                (
                    f"{clean_short_form} "
                    f"{clean_city}"
                ).strip(),
                (
                    f"{clean_match_base} "
                    f"{raw_city}"
                ).strip(),
                (
                    f"{clean_match_base} "
                    f"{clean_city}"
                ).strip(),
            }

            variants.update(abbreviations)

            for abbreviation in abbreviations:
                variants.add(
                    f"{abbreviation} {raw_city}".strip()
                )
                variants.add(
                    f"{abbreviation} {clean_city}".strip()
                )

            for variant in variants:
                self._register_variant(
                    variant,
                    college_id,
                )

            if (
                position % 500 == 0
                or position == total
            ):
                self._report_progress(
                    position,
                    total,
                    "Building master index",
                )

        for alias in CITY_ALIASES:
            self.city_phrases.add(alias)

        # Rebuild city membership after all city phrases are known so a
        # maintained short form such as "CVR Hyderabad" can supplement a
        # master City value such as Rangareddy.
        for college_id, record in self.college_by_id.items():
            for city in self._record_city_candidates(record):
                self.city_ids[city].add(college_id)

        self.search_variants = list(
            self.variant_ids.keys()
        )

    def detect_city(
        self,
        input_name: object,
    ) -> tuple[str, str]:
        """
        Return:
        1. Standard city
        2. Actual city phrase found in the input
        """

        clean_input = normalize(
            input_name
        )

        detected = []

        for phrase in self.city_phrases:
            if (
                f" {phrase} "
                in f" {clean_input} "
            ):
                detected.append(phrase)

        if not detected:
            return "", ""

        # Prefer the final location phrase because uploaded labels usually
        # place the actual campus city at the end. This prevents a brand
        # name such as "Chandigarh University, Mohali" from being treated
        # as a Chandigarh-campus input. At an equal position, retain the
        # longer phrase (Greater Noida over Noida).
        padded_input = f" {clean_input} "
        raw_detected = max(
            detected,
            key=lambda phrase: (
                padded_input.rfind(f" {phrase} "),
                len(phrase),
            ),
        )

        return (
            canonical_city(raw_detected),
            raw_detected,
        )

    def _record_city_candidates(
        self,
        record: dict,
    ) -> set[str]:
        """Return every reliable city attached to a master record."""

        cities = set()

        if record.get("clean_city"):
            cities.add(record["clean_city"])

        city_from_name, _ = self.detect_city(
            record.get("clean_name", "")
        )

        if city_from_name:
            cities.add(city_from_name)

        city_from_short_form, _ = self.detect_city(
            record.get("short_form", "")
        )

        if city_from_short_form:
            cities.add(city_from_short_form)

        return cities

    def _has_unique_acronym_identity(
        self,
        input_name: object,
        college_id: int,
    ) -> bool:
        """Whether a maintained acronym uniquely identifies this record."""

        input_acronyms = input_initialisms(input_name)
        record = self.college_by_id[college_id]
        acronyms = record.get("acronym_keys", set())
        input_city, _ = self.detect_city(input_name)

        for acronym in input_acronyms & acronyms:
            matching_ids = self.acronym_ids.get(acronym, set())
            if matching_ids == {college_id}:
                return True

            if input_city:
                same_city_ids = matching_ids & self.city_ids.get(
                    input_city,
                    set(),
                )
                if same_city_ids == {college_id}:
                    return True

            if len(acronym) >= 4 and matching_ids == {college_id}:
                return True

        return False

    def _has_location_conflict(
        self,
        input_name: object,
        college_id: int,
    ) -> bool:
        """Final campus guard that cannot be bypassed by fuzzy/exact logic."""

        input_city, _ = self.detect_city(
            input_name
        )

        if not input_city:
            return False

        record = self.college_by_id[
            college_id
        ]

        candidate_cities = (
            self._record_city_candidates(
                record
            )
        )

        conflict = bool(
            candidate_cities
            and input_city not in candidate_cities
        )

        if (
            conflict
            and self._has_unique_acronym_identity(
                input_name,
                college_id,
            )
        ):
            return False

        return conflict

    def _explicit_parent_score(
        self,
        clean_input: str,
        record: dict,
        input_city: str,
    ) -> float | None:
        input_tokens = set(
            clean_input.split()
        )

        if not (
            input_tokens
            & PARENT_MARKERS
        ):
            return None

        candidate_subunits = (
            set(record["clean_base_name"].split())
            & {"department", "faculty", "school"}
        )
        input_subunits = (
            input_tokens
            & {"department", "faculty", "school"}
        )

        # A general institution input such as BITS Pilani must prefer the
        # main institution, not "Department of Management, BITS".
        if candidate_subunits and not input_subunits:
            return None

        base_name = record["clean_base_name"]
        short_form = record["short_form"]

        if not is_valid_parent_name(base_name):
            return None

        input_tokens = meaningful_tokens(clean_input)
        base_tokens, short_tokens = record.get(
            "parent_identity_tokens",
            (
                meaningful_tokens(base_name),
                meaningful_tokens(short_form),
            ),
        )

        identity_matches = []

        for source, identity_tokens in (
            ("base", base_tokens),
            ("short", short_tokens),
        ):
            if not identity_tokens:
                continue

            # A full distinctive identity contained anywhere in a long
            # department/faculty label is stronger than word order.
            if (
                len(identity_tokens) >= 2
                and identity_tokens.issubset(input_tokens)
            ):
                identity_matches.append(identity_tokens)

            elif (
                source == "base"
                and len(identity_tokens) == 1
                and len(next(iter(identity_tokens))) >= 4
                and identity_tokens.issubset(input_tokens)
                and base_name in clean_input
            ):
                identity_matches.append(identity_tokens)

            # Permit a single maintained acronym only with an exact city.
            elif (
                source == "short"
                and len(identity_tokens) == 1
                and len(next(iter(identity_tokens))) >= 3
                and identity_tokens.issubset(input_tokens)
                and next(iter(identity_tokens))
                in record.get("short_acronyms", set())
                and next(iter(identity_tokens)) != input_city
                and next(iter(identity_tokens)) != record["clean_city"]
                and input_city
                and input_city == record["clean_city"]
            ):
                identity_matches.append(identity_tokens)

        if not identity_matches:
            return None

        master_city = record["clean_city"]
        accepted_cities = self._record_city_candidates(record)

        if (
            input_city
            and accepted_cities
            and input_city not in accepted_cities
        ):
            return None

        specificity = max(
            len(tokens)
            for tokens in identity_matches
        )

        first_input_token = clean_input.split()[0] if clean_input else ""
        acronym_prefix_bonus = (
            0.8
            if first_input_token in record.get("short_acronyms", set())
            else 0.0
        )

        if input_city and input_city in accepted_cities:
            return min(
                100.0,
                98.0 + specificity * 0.4 + acronym_prefix_bonus,
            )

        return min(
            99.0,
            96.0 + specificity * 0.4 + acronym_prefix_bonus,
        )

    def _calculate_token_overlap(
        self,
        input_without_city: str,
        record: dict,
    ) -> float:
        input_tokens = identity_tokens(
            input_without_city
        )

        candidate_values = [
            record["clean_name"],
            record["clean_base_name"],
            record["short_form"],
        ]

        best_overlap = 0.0

        input_acronyms = input_initialisms(input_without_city)

        if (
            input_acronyms
            & record.get("acronym_keys", set())
        ):
            best_overlap = 0.90

        for candidate_value in candidate_values:
            candidate_tokens = identity_tokens(
                candidate_value
            )

            if (
                not input_tokens
                or not candidate_tokens
            ):
                continue

            shared_tokens = (
                input_tokens
                & candidate_tokens
            )

            input_coverage = len(shared_tokens) / len(input_tokens)
            candidate_coverage = len(shared_tokens) / len(candidate_tokens)
            overlap = min(input_coverage, candidate_coverage)

            best_overlap = max(best_overlap, overlap)

        return best_overlap

    def _score_candidate(
        self,
        clean_input: str,
        input_without_city: str,
        input_city: str,
        college_id: int,
    ) -> Candidate:
        record = self.college_by_id[
            college_id
        ]

        master_city = record[
            "clean_city"
        ]

        # Some master records have a missing or unreliable City field but
        # include the campus in the college name, for example LNCT(Bhopal).
        # Read that location as an additional independent signal.
        (
            city_from_candidate_name,
            _,
        ) = self.detect_city(
            record["clean_name"]
        )

        candidate_cities = self._record_city_candidates(record)

        # Campus/city conflicts are disqualifying. A matching brand or
        # abbreviation must never map an Indore institution to Bhopal,
        # a Noida institution to Lucknow, and so on.
        unique_acronym_override = (
            self._has_unique_acronym_identity(
                clean_input,
                college_id,
            )
        )

        if (
            input_city
            and candidate_cities
            and input_city not in candidate_cities
            and not unique_acronym_override
        ):
            return Candidate(
                college_id=college_id,
                college_name=record["college_name"],
                city=record["city"],
                state=record["state"],
                college_type=record["college_type"],
                confidence=0.0,
                token_overlap=0.0,
                reason=(
                    "Rejected: campus/city conflict "
                    f"({input_city} vs "
                    f"{sorted(candidate_cities)})"
                ),
            )

        input_domain_markers = canonical_domain_markers(
            input_without_city
        )

        candidate_domain_markers = canonical_domain_markers(
            f"{record['clean_name']} {record['clean_base_name']}"
        )

        delivery_markers = {"online", "distance", "executive", "digital"}
        input_delivery = set(input_without_city.split()) & delivery_markers
        candidate_delivery = set(record["clean_base_name"].split()) & delivery_markers
        if candidate_delivery and candidate_delivery != input_delivery:
            return Candidate(
                college_id=college_id,
                college_name=record["college_name"],
                city=record["city"],
                state=record["state"],
                college_type=record["college_type"],
                confidence=0.0,
                token_overlap=0.0,
                reason="Rejected: online/distance programme conflict",
            )

        # An explicit parent/brand token cannot override an academic-domain
        # conflict. For example, SJB Institute of Technology must not become
        # SJB College of Nursing merely because both contain SJB.
        if (
            input_domain_markers
            and candidate_domain_markers
            and input_domain_markers.isdisjoint(candidate_domain_markers)
        ):
            return Candidate(
                college_id=college_id,
                college_name=record["college_name"],
                city=record["city"],
                state=record["state"],
                college_type=record["college_type"],
                confidence=0.0,
                token_overlap=0.0,
                reason=(
                    "Rejected: institution-category conflict "
                    f"({sorted(input_domain_markers)} vs "
                    f"{sorted(candidate_domain_markers)})"
                ),
            )

        parent_score = self._explicit_parent_score(
            clean_input,
            record,
            input_city,
        )

        # A department/faculty label may describe a different academic
        # domain from the parent name (for example Architecture at VNIT).
        # A deterministic parent identity plus matching campus is allowed
        # before the general category-conflict rejection.
        if parent_score is not None:
            return Candidate(
                college_id=college_id,
                college_name=record["college_name"],
                city=record["city"],
                state=record["state"],
                college_type=record["college_type"],
                confidence=parent_score,
                token_overlap=1.0,
                reason="Explicit parent institution",
            )

        comparison_values = [
            record["clean_name"],
            record["clean_base_name"],
            record["short_form"],
            *record.get("abbreviations", set()),
        ]

        lexical_scores = []

        for candidate_name in (
            comparison_values
        ):
            if not candidate_name:
                continue

            lexical_scores.extend(
                [
                    fuzz.WRatio(
                        clean_input,
                        candidate_name,
                    ),
                    fuzz.WRatio(
                        input_without_city,
                        candidate_name,
                    ),
                    fuzz.token_set_ratio(
                        input_without_city,
                        candidate_name,
                    ),
                    fuzz.token_sort_ratio(
                        input_without_city,
                        candidate_name,
                    ),
                ]
            )

        lexical_score = (
            max(lexical_scores)
            if lexical_scores
            else 0.0
        )

        token_overlap = (
            self._calculate_token_overlap(
                input_without_city,
                record,
            )
        )

        input_distinctive_tokens = identity_tokens(
            input_without_city
        )

        candidate_identity_tokens = (
            identity_tokens(record["clean_name"])
            | identity_tokens(record["clean_base_name"])
            | identity_tokens(record["short_form"])
        )

        distinctive_shared_tokens = (
            input_distinctive_tokens
            & candidate_identity_tokens
        )

        input_acronyms = input_initialisms(input_without_city)
        shared_abbreviations = (
            input_acronyms
            & record.get("acronym_keys", set())
        )
        literal_input_tokens = set(normalize(input_without_city).split())
        first_literal_token = (
            normalize(input_without_city).split()[0]
            if normalize(input_without_city)
            else ""
        )
        trusted_shared_abbreviations = {
            acronym
            for acronym in shared_abbreviations
            if acronym in literal_input_tokens
            or (
                2 <= len(first_literal_token) <= 5
                and acronym.startswith(first_literal_token)
            )
            or len(acronym) >= 5
        }

        input_identity_coverage = (
            len(distinctive_shared_tokens) / len(input_distinctive_tokens)
            if input_distinctive_tokens
            else 0.0
        )
        candidate_identity_coverage = (
            len(distinctive_shared_tokens) / len(candidate_identity_tokens)
            if candidate_identity_tokens
            else 0.0
        )

        same_city = bool(
            input_city
            and input_city in candidate_cities
        )
        acronym_city_unique = any(
            (
                self.acronym_ids.get(acronym, set())
                & self.city_ids.get(input_city, set())
            )
            == {college_id}
            for acronym in trusted_shared_abbreviations
        ) if input_city else False
        ordered_input_identity = [
            token
            for token in normalize(input_without_city).split()
            if token not in IDENTITY_NOISE_TOKENS
            and len(token) > 1
        ]
        leading_two_letter_identity = (
            "".join(token[0] for token in ordered_input_identity[:2])
            if len(ordered_input_identity) >= 2
            else ""
        )
        two_letter_acronym_with_city = bool(
            same_city
            and leading_two_letter_identity
            and leading_two_letter_identity in shared_abbreviations
        )
        reliable_acronym = bool(
            (trusted_shared_abbreviations or two_letter_acronym_with_city)
            and (
                acronym_city_unique
                or two_letter_acronym_with_city
                or any(
                    len(acronym) >= 4
                    for acronym in trusted_shared_abbreviations
                )
                and same_city
            )
        )
        if (
            reliable_acronym
            and len(input_distinctive_tokens) >= 3
            and input_identity_coverage < 0.45
        ):
            reliable_acronym = False

        credible_literal_identity = bool(
            (
                len(distinctive_shared_tokens) >= 2
                and input_identity_coverage >= 0.45
                and candidate_identity_coverage >= 0.45
            )
            or (
                len(distinctive_shared_tokens) == 1
                and input_identity_coverage >= 0.75
                and candidate_identity_coverage >= 0.50
                and len(next(iter(distinctive_shared_tokens))) >= 4
            )
        )

        identity_strength = min(
            input_identity_coverage,
            candidate_identity_coverage,
        )
        confidence = (
            lexical_score * 0.60
            + identity_strength * 30.0
        )

        reason_parts = [
            (
                "Name similarity "
                f"{lexical_score:.1f}"
            ),
            (
                "identity-token overlap "
                f"{token_overlap:.2f}"
            ),
        ]

        # A high fuzzy score can be produced by generic words or one short
        # acronym (DAV, MIT, SDM, etc.). Never retain such a candidate unless
        # the proper-name coverage or a maintained acronym plus campus makes
        # the institution credible.
        if not credible_literal_identity and not reliable_acronym:
            return Candidate(
                college_id=college_id,
                college_name=record["college_name"],
                city=record["city"],
                state=record["state"],
                college_type=record["college_type"],
                confidence=0.0,
                token_overlap=0.0,
                reason="Rejected: insufficient distinctive institution identity",
            )

        # A location must never rescue an unrelated name.
        if token_overlap == 0:
            confidence -= 50

            reason_parts.append(
                "no meaningful name-token match"
            )

        elif token_overlap < 0.50:
            confidence -= 18

            reason_parts.append(
                "weak meaningful-token match"
            )

        # A matching city cannot compensate for a conflicting proper
        # name or abbreviation. For example, LN Medical College Bhopal
        # must never become Gandhi Medical College Bhopal merely because
        # both are medical colleges in Bhopal.
        if input_city:
            if (
                input_city
                == record["clean_city"]
            ):
                confidence += 15

                reason_parts.append(
                    "exact city match"
                )

            # A conflicting non-empty city has already been rejected
            # before name scoring. No conflicting candidate reaches here.

        if (
            input_domain_markers
            and not candidate_domain_markers
        ):
            confidence -= 25

            reason_parts.append(
                "candidate does not confirm institution category"
            )

        # Three-signal confirmation: abbreviation/full-form initials,
        # institution category and campus city all agree. This is the
        # safe intelligent path for LN Medical College Bhopal -> L.N.
        # Medical College and Research Centre, Bhopal.
        if (
            trusted_shared_abbreviations
            and input_domain_markers
            and candidate_domain_markers
            and not input_domain_markers.isdisjoint(
                candidate_domain_markers
            )
            and input_city
            and input_city == record["clean_city"]
        ):
            confidence = max(
                confidence,
                98.0,
            )

            reason_parts.append(
                "abbreviation, institution category and city confirmed"
            )

        elif reliable_acronym and same_city:
            confidence = max(confidence, 90.0)
            reason_parts.append(
                "maintained acronym and campus confirmed"
            )

        confidence = max(
            0.0,
            min(confidence, 100.0),
        )

        if (
            input_city
            and candidate_cities
            and input_city not in candidate_cities
            and unique_acronym_override
        ):
            confidence = min(confidence, 75.0)
            reason_parts.append(
                "unique acronym match but city requires review"
            )

        return Candidate(
            college_id=college_id,
            college_name=record[
                "college_name"
            ],
            city=record["city"],
            state=record["state"],
            college_type=record[
                "college_type"
            ],
            confidence=confidence,
            token_overlap=token_overlap,
            reason=", ".join(
                reason_parts
            ),
        )

    def get_ranked_candidates(
        self,
        input_name: object,
        limit: int = 5,
    ) -> list[Candidate]:
        clean_input = normalize(
            input_name
        )

        if not clean_input:
            return []

        (
            input_city,
            detected_city_phrase,
        ) = self.detect_city(input_name)

        input_without_city = (
            clean_input
        )

        if detected_city_phrase:
            input_without_city = (
                " ".join(
                    (
                        f" {clean_input} "
                        .replace(
                            (
                                f" "
                                f"{detected_city_phrase}"
                                f" "
                            ),
                            " ",
                        )
                    ).split()
                )
            )

        candidate_ids = set()

        # Retrieve by proper-name evidence before fuzzy ranking. This is
        # both faster and safer than scanning abbreviation-like variants:
        # Mehr/Chand/Mahajan retrieves MCM DAV, while a bare DAV record
        # cannot dominate merely because it shares one short token.
        for token in identity_tokens(input_without_city):
            candidate_ids.update(
                self.identity_token_ids.get(token, set())
            )

        for acronym in input_initialisms(input_without_city):
            candidate_ids.update(
                self.acronym_ids.get(acronym, set())
            )

        # Add explicitly named parent institutions. This deterministic
        # scan prevents a long department label from dropping its parent
        # merely because the fuzzy search returned only shorter phrases.
        if (
            set(clean_input.split())
            & PARENT_MARKERS
        ):
            parent_candidate_pool = set()
            for token in meaningful_tokens(clean_input):
                parent_candidate_pool.update(
                    self.identity_token_ids.get(token, set())
                )

            for college_id in parent_candidate_pool:
                record = self.college_by_id[college_id]
                if (
                    self._explicit_parent_score(
                        clean_input,
                        record,
                        input_city,
                    )
                    is not None
                    and not self._has_location_conflict(
                        input_name,
                        college_id,
                    )
                ):
                    candidate_ids.add(
                        college_id
                    )

        # When a long department/faculty string deterministically names
        # its parent, do not let unrelated fuzzy candidates compete with
        # that verified identity. Use fuzzy retrieval only as fallback.
        if not candidate_ids:
            raw_matches = process.extract(
                clean_input,
                self.search_variants,
                scorer=fuzz.WRatio,
                limit=100,
                score_cutoff=25,
            )

            for variant, _, _ in raw_matches:
                candidate_ids.update(
                    self.variant_ids[variant]
                )

        # City-restricted recovery: fuzzy retrieval can miss spelling
        # variants and acronyms (COEP, BMSCE, CVR, FISAT, BVRIT, etc.).
        # Add every master record from the confirmed city, then let name
        # evidence score them. City narrows the pool but never confirms a
        # candidate by itself.
        if input_city:
            candidate_ids.update(
                self.city_ids.get(input_city, set())
            )

        candidates = [
            self._score_candidate(
                clean_input,
                input_without_city,
                input_city,
                college_id,
            )
            for college_id in candidate_ids
        ]

        candidates.sort(
            key=lambda candidate: (
                candidate.confidence
            ),
            reverse=True,
        )

        return candidates[:limit]

    def match_one(
        self,
        input_name: object,
        input_city_value: object = "",
        input_state_value: object = "",
    ) -> MatchDecision:
        original_name = (
            ""
            if input_name is None
            else str(input_name)
        )

        clean_input = normalize(
            input_name
        )

        raw_identity_input = structural_identity(original_name)
        identity_input = match_identity_phrase(raw_identity_input)

        clean_context_city = canonical_city(
            input_city_value
        )

        contextual_input = original_name

        if clean_context_city:
            name_city, _ = self.detect_city(original_name)

            if not name_city:
                contextual_input = (
                    f"{original_name} {input_city_value}"
                ).strip()

        clean_contextual_input = normalize(contextual_input)

        if (
            clean_input in self.VERIFIED_NOT_FOUND_INPUTS
            or clean_contextual_input in self.VERIFIED_NOT_FOUND_INPUTS
        ):
            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id="Not Found",
                matched_name=None,
                confidence=0.0,
                decision="NOT_FOUND",
                reason=(
                    "Verified absent institution; related brand and "
                    "other-campus IDs are blocked"
                ),
                candidates=[],
            )

        verified_college_id = (
            self.VERIFIED_INPUT_TO_COLLEGE_ID.get(
                clean_input
            )
            or self.VERIFIED_INPUT_TO_COLLEGE_ID.get(
                clean_contextual_input
            )
        )

        if (
            verified_college_id is not None
            and verified_college_id in self.college_by_id
            and not self._has_location_conflict(
                contextual_input,
                verified_college_id,
            )
        ):
            record = self.college_by_id[
                verified_college_id
            ]

            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id=verified_college_id,
                matched_name=record["college_name"],
                confidence=100.0,
                decision="FOUND",
                reason="Verified exact business alias",
                candidates=[],
            )

        review_college_id = self.VERIFIED_REVIEW_INPUT_TO_COLLEGE_ID.get(
            clean_input
        ) or self.VERIFIED_REVIEW_INPUT_TO_COLLEGE_ID.get(
            clean_contextual_input
        )
        if (
            review_college_id is not None
            and review_college_id in self.college_by_id
            and not self._has_location_conflict(
                contextual_input,
                review_college_id,
            )
        ):
            record = self.college_by_id[review_college_id]
            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id=review_college_id,
                matched_name=record["college_name"],
                confidence=92.0,
                decision="NEEDS_REVIEW",
                reason=(
                    "Explicit parent institution suggested; exact sub-unit "
                    "record is absent from the master"
                ),
                candidates=[],
            )

        # Amity has many department pages and multiple city campuses. When
        # the input describes an Amity school/faculty, the explicit campus
        # university is the stable parent identity; a generic department
        # page from another city must never replace it.
        detected_brand_city, _ = self.detect_city(contextual_input)
        if "amity" in set(clean_input.split()) and detected_brand_city:
            amity_parent_ids = set()
            for college_id in self.city_ids.get(detected_brand_city, set()):
                record = self.college_by_id[college_id]
                base_tokens = [
                    token
                    for token in record["clean_base_name"].split()
                    if canonical_city(token) not in self._record_city_candidates(record)
                ]
                if (
                    record["clean_base_name"].startswith("amity university")
                    and not (
                        set(record["clean_base_name"].split())
                        & {"school", "department", "faculty", "college"}
                    )
                    and record["college_type"].casefold() == "university"
                ):
                    amity_parent_ids.add(college_id)

            if len(amity_parent_ids) == 1:
                college_id = next(iter(amity_parent_ids))
                record = self.college_by_id[college_id]
                return MatchDecision(
                    input_name=original_name,
                    normalized_name=clean_input,
                    college_id=college_id,
                    matched_name=record["college_name"],
                    confidence=98.0,
                    decision="FOUND",
                    reason="Amity parent university and campus city confirmed",
                    candidates=[],
                )

        # National systems use a common name plus a campus city. Resolve
        # IIT/NIT/IIIT/IIM/AIIMS deterministically from that combination
        # instead of comparing them with unrelated same-city colleges.
        system_acronyms = (
            set(clean_input.split())
            & {"iit", "nit", "iiit", "iim", "aiims", "bits"}
        )
        system_phrases = {
            "indian institute of technology": "iit",
            "national institute of technology": "nit",
            "indian institute of information technology": "iiit",
            "indian institute of management": "iim",
            "birla institute of technology and science": "bits",
        }
        for phrase, acronym in system_phrases.items():
            if phrase in clean_input:
                system_acronyms.add(acronym)
        detected_system_city, _ = self.detect_city(contextual_input)
        if system_acronyms and detected_system_city:
            system_subunit_markers = {
                "department",
                "faculty",
                "school",
                "centre",
                "center",
                "management",
                "design",
                "entrepreneurship",
                "online",
                "distance",
                "executive",
                "digital",
                "wilp",
            }
            input_system_subunits = (
                set(clean_input.split())
                & system_subunit_markers
            )
            system_ids = {
                college_id
                for college_id in self.city_ids.get(
                    detected_system_city,
                    set(),
                )
                if system_acronyms
                & national_system_families(
                    self.college_by_id[college_id]["clean_name"],
                    self.college_by_id[college_id]["short_form"],
                )
                and (
                    input_system_subunits
                    or not (
                        set(self.college_by_id[college_id]["clean_name"].split())
                        & system_subunit_markers
                    )
                )
            }

            # A campus may have both a general institution record and a
            # course-specific record. Within the already-confirmed national
            # system + city pool, prefer the one expressing the input's
            # academic unit. Example: Department of Architecture, NIT
            # Hamirpur -> the NIT Hamirpur B.Arch database record.
            input_intents = academic_intents(clean_input)
            if input_intents and len(system_ids) > 1:
                intent_ids = {
                    college_id
                    for college_id in system_ids
                    if input_intents
                    & academic_intents(
                        " ".join(
                            [
                                self.college_by_id[college_id]["clean_name"],
                                self.college_by_id[college_id]["short_form"],
                            ]
                        )
                    )
                }
                if len(intent_ids) == 1:
                    system_ids = intent_ids
                elif not intent_ids:
                    main_system_ids = {
                        college_id
                        for college_id in system_ids
                        if not (
                            set(
                                self.college_by_id[college_id][
                                    "clean_name"
                                ].split()
                            )
                            & system_subunit_markers
                        )
                    }
                    if len(main_system_ids) == 1:
                        system_ids = main_system_ids

            if len(system_ids) == 1:
                college_id = next(iter(system_ids))
                record = self.college_by_id[college_id]
                return MatchDecision(
                    input_name=original_name,
                    normalized_name=clean_input,
                    college_id=college_id,
                    matched_name=record["college_name"],
                    confidence=99.0,
                    decision="FOUND",
                    reason="National institution acronym and city confirmed",
                    candidates=[],
                )

        all_exact_ids = set(
            self.exact_index.get(raw_identity_input, set())
        )
        if not all_exact_ids:
            all_exact_ids = set(
                self.exact_index.get(identity_input, set())
            )
        if not all_exact_ids and identity_input:
            identity_fingerprint = " ".join(
                sorted(identity_input.split())
            )
            all_exact_ids = set(
                self.token_set_index.get(identity_fingerprint, set())
            )

        conflicting_exact_ids = {
            college_id
            for college_id in all_exact_ids
            if self._has_location_conflict(
                contextual_input,
                college_id,
            )
        }

        # Some master rows use a nearby district/suburb while the input
        # uses the better-known metro. A unique literal institution match
        # is retained for review instead of being discarded. Explicitly
        # blocked cross-campus cases are handled before this route.
        if (
            len(all_exact_ids) == 1
            and conflicting_exact_ids == all_exact_ids
        ):
            college_id = next(iter(all_exact_ids))
            record = self.college_by_id[college_id]
            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id=college_id,
                matched_name=record["college_name"],
                confidence=70.0,
                decision="NEEDS_REVIEW",
                reason=(
                    "Unique institution name; input city differs from "
                    "the master city and requires verification"
                ),
                candidates=[],
            )

        exact_ids = all_exact_ids

        # Exact aliases and short forms are still subject to campus
        # validation. No exact/variant route can bypass a city conflict.
        exact_ids = {
            college_id
            for college_id in exact_ids
            if not self._has_location_conflict(
                contextual_input,
                college_id,
            )
        }

        # Prefer a literal master college-name match over an unrelated
        # institution that merely produces the same abbreviation. This
        # resolves inputs such as "IBS" + Hyderabad to the master row
        # whose actual College Name is IBS.
        literal_name_ids = {
            college_id
            for college_id in exact_ids
            if self.college_by_id[college_id]["clean_name"] == identity_input
            or self.college_by_id[college_id]["clean_base_name"] == identity_input
            or self.college_by_id[college_id]["clean_match_name"] == identity_input
            or self.college_by_id[college_id]["clean_match_base"] == identity_input
        }

        if literal_name_ids:
            exact_ids = literal_name_ids

        # Some master sheets contain duplicate name+city records. Prefer
        # the record whose maintained short form spells out the same base
        # identity; this is more specific than a bare acronym+city alias.
        if len(exact_ids) > 1:
            clean_input_base = remove_bracket_alias(identity_input)
            full_identity_ids = {
                college_id
                for college_id in exact_ids
                if self.college_by_id[college_id]["short_form"]
                == clean_input_base
            }

            if len(full_identity_ids) == 1:
                exact_ids = full_identity_ids

        if len(exact_ids) == 1:
            college_id = next(
                iter(exact_ids)
            )

            record = self.college_by_id[
                college_id
            ]

            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id=college_id,
                matched_name=record[
                    "college_name"
                ],
                confidence=100.0,
                decision="FOUND",
                reason=(
                    "Unique exact master match"
                ),
                candidates=[],
            )

        # If the complete department/faculty/school label has no exact
        # record, retain an explicitly named parent institution for human
        # review. Example: "Faculty of Engineering, Jadavpur University"
        # should suggest Jadavpur University, never an unrelated Jadavpur
        # centre or a same-city college.
        explicit_segment_ids: set[int] = set()
        detected_parent_city, _ = self.detect_city(contextual_input)

        for raw_segment in str(original_name).split(","):
            segment = match_identity_phrase(raw_segment)
            if not segment:
                continue

            segment_tokens = set(segment.split())
            possible_ids = set(self.exact_index.get(segment, set()))
            explicit_segment_acronyms = {
                normalize(token)
                for token in re.findall(
                    r"\b[A-Z][A-Z0-9]{2,}\b",
                    raw_segment,
                )
            }
            for acronym in explicit_segment_acronyms:
                possible_ids.update(
                    self.acronym_ids.get(acronym, set())
                )

            if not possible_ids and not (
                segment_tokens
                & {"university", "college", "institute", "iit", "nit", "iiit"}
            ):
                continue

            if detected_parent_city:
                possible_ids.update(
                    self.exact_index.get(
                        f"{segment} {detected_parent_city}".strip(),
                        set(),
                    )
                )

            for possible_id in possible_ids:
                if not self._has_location_conflict(
                    contextual_input,
                    possible_id,
                ):
                    explicit_segment_ids.add(possible_id)

        if len(explicit_segment_ids) == 1:
            college_id = next(iter(explicit_segment_ids))
            record = self.college_by_id[college_id]
            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id=college_id,
                matched_name=record["college_name"],
                confidence=92.0,
                decision="NEEDS_REVIEW",
                reason=(
                    "Explicit parent institution confirmed; specialised "
                    "sub-unit has no unique exact master record"
                ),
                candidates=[],
            )

        ranking_input = identity_input
        detected_city, _ = self.detect_city(contextual_input)
        if detected_city and detected_city not in ranking_input:
            ranking_input = f"{ranking_input} {detected_city}".strip()

        candidates = (
            self.get_ranked_candidates(
                ranking_input,
                limit=5,
            )
        )

        # Non-bypassable final filter. Even if a fuzzy score, alias or
        # abbreviation is high, an Indore input cannot retain a Bhopal
        # candidate in the decision list.
        candidates = [
            candidate
            for candidate in candidates
            if not self._has_location_conflict(
                contextual_input,
                candidate.college_id,
            )
        ]

        if not candidates:
            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id="Not Found",
                matched_name=None,
                confidence=0.0,
                decision="NOT_FOUND",
                reason=(
                    "No master candidate found"
                ),
                candidates=[],
            )

        best = candidates[0]

        second_score = (
            candidates[1].confidence
            if len(candidates) > 1
            else 0.0
        )

        margin = (
            best.confidence
            - second_score
        )

        input_city, _ = self.detect_city(
            contextual_input
        )

        automatic_match = False

        if (
            best.reason
            == "Explicit parent institution"
            and best.confidence >= 97
            and (
                input_city
                or margin >= 5
            )
        ):
            automatic_match = True

        elif (
            input_city
            and best.confidence >= 96
            and best.token_overlap >= 0.75
            and margin >= 8
        ):
            automatic_match = True

        elif (
            not input_city
            and best.confidence >= 96
            and best.token_overlap >= 0.75
            and margin >= 8
        ):
            automatic_match = True

        if automatic_match:
            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id=best.college_id,
                matched_name=(
                    best.college_name
                ),
                confidence=round(
                    best.confidence,
                    2,
                ),
                decision="FOUND",
                reason=best.reason,
                candidates=candidates,
            )

        # No meaningful name overlap means the college must not be
        # suggested merely because of its city.
        if (
            best.token_overlap == 0
            or best.confidence < 20
        ):
            return MatchDecision(
                input_name=original_name,
                normalized_name=clean_input,
                college_id="Not Found",
                matched_name=None,
                confidence=round(
                    best.confidence,
                    2,
                ),
                decision="NOT_FOUND",
                reason=(
                    "No credible master match"
                ),
                candidates=candidates,
            )

        return MatchDecision(
            input_name=original_name,
            normalized_name=clean_input,
            # Review mode: retain the best credible candidate ID so the
            # user can verify it alongside the confidence score.
            college_id=best.college_id,
            matched_name=(
                best.college_name
            ),
            confidence=round(
                best.confidence,
                2,
            ),
            decision="NEEDS_REVIEW",
            reason=(
                "Ambiguous match; "
                f"top-two margin {margin:.1f}"
            ),
            candidates=candidates,
        )

    def match_all(
        self,
        names: Iterable[object],
        cities: Iterable[object] | None = None,
        states: Iterable[object] | None = None,
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> pd.DataFrame:
        name_values = list(names)
        city_values = (
            list(cities)
            if cities is not None
            else [""] * len(name_values)
        )
        state_values = (
            list(states)
            if states is not None
            else [""] * len(name_values)
        )

        if not (
            len(name_values)
            == len(city_values)
            == len(state_values)
        ):
            raise ValueError(
                "Name, city and state columns must have equal row counts."
            )

        unique_inputs: dict[str, tuple[object, object, object]] = {}

        for original_name, city, state in zip(
            name_values,
            city_values,
            state_values,
        ):
            normalized_name = normalize(
                original_name
            )

            context_key = match_context_key(
                original_name,
                city,
                state,
            )

            if (
                context_key
                not in unique_inputs
            ):
                unique_inputs[
                    context_key
                ] = (original_name, city, state)

        items = list(
            unique_inputs.items()
        )

        total = len(items)
        output_rows = []

        for position, (
            context_key,
            input_values,
        ) in enumerate(
            items,
            start=1,
        ):
            original_name, city, state = input_values

            decision = self.match_one(
                original_name,
                input_city_value=city,
                input_state_value=state,
            )

            output_rows.append(
                {
                    "input_name": (
                        decision.input_name
                    ),
                    "normalized_name": (
                        normalize(original_name)
                    ),
                    "match_key": (
                        context_key
                    ),
                    "input_city": (
                        "" if pd.isna(city) else str(city)
                    ),
                    "input_state": (
                        "" if pd.isna(state) else str(state)
                    ),
                    "decision": (
                        decision.decision
                    ),
                    "college_id": (
                        decision.college_id
                    ),
                    "matched_name": (
                        decision.matched_name
                    ),
                    "confidence": (
                        decision.confidence
                    ),
                    "reason": (
                        decision.reason
                    ),
                    "candidates": (
                        decision.candidates
                    ),
                }
            )

            if (
                progress_callback
                and (
                    position % 10 == 0
                    or position == total
                )
            ):
                progress_callback(
                    position,
                    total,
                    "Matching unique colleges",
                )

        return pd.DataFrame(
            output_rows
        )
