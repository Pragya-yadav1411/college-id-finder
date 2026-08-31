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


MATCHER_VERSION = "2026.08.31.6-CITY-RECOVERY"


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

    text = text.casefold()

    # Keep dotted initialisms as one identity token: B.M.S. -> bms,
    # R.M.K. -> rmk and K.S.R. -> ksr. Treating every letter as a
    # separate word previously prevented obvious master matches.
    text = re.sub(
        r"\b(?:[a-z]\s*\.\s*){2,}",
        lambda match: (
            re.sub(r"[^a-z]", "", match.group(0)) + " "
        ),
        text,
    )

    # Make possessive and non-possessive spellings equivalent:
    # People's -> Peoples, Teachers' -> Teachers.
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

    return " ".join(text.split())


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

    first_is_parent = bool(
        set(first.split()) & {"university", "technology"}
    )

    if first_is_parent:
        # A separately named constituent college (such as Sir J. J.
        # College) wins. Department/faculty/school/institute phrases are
        # sub-units and must not redirect the match to a department page.
        for later_raw in raw_segments[1:]:
            later = normalize(later_raw)
            if " college " in f" {later} ":
                return later

        return first

    # When a descriptive unit comes first and its university/college is
    # named in a later segment, keep both identities. Examples include
    # "College of Engineering, Anna University" and
    # "Faculty of Engineering, Jadavpur University".
    leading_subunit = bool(
        set(first.split())
        & {"department", "faculty", "school", "college", "institute"}
    )
    if leading_subunit:
        for later_raw in raw_segments[1:]:
            later = normalize(later_raw)
            if set(later.split()) & {
                "university", "college", "institute", "iit", "nit"
            }:
                return f"{first} {later}".strip()

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

            short_acronyms = {
                normalize(token)
                for token in re.findall(
                    r"\b[A-Z][A-Z0-9]{2,}\b",
                    short_form,
                )
            }

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
                "short_form": clean_short_form,
                "abbreviations": abbreviations,
                "city": city,
                "raw_city": raw_city,
                "clean_city": clean_city,
                "state": state,
                "college_type": college_type,
                "parent_identity_tokens": (
                    without_record_city(
                        meaningful_tokens(clean_base_name)
                    ),
                    without_record_city(
                        meaningful_tokens(clean_short_form)
                    ),
                ),
                "short_acronyms": short_acronyms,
            }

            for identity_tokens in self.college_by_id[
                college_id
            ]["parent_identity_tokens"]:
                for token in identity_tokens:
                    self.identity_token_ids[token].add(college_id)

            # Include both raw and standardized city forms.
            variants = {
                clean_name,
                clean_base_name,
                clean_short_form,
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

        raw_detected = max(
            detected,
            key=len,
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

        input_tokens = set(normalize(input_name).split())
        record = self.college_by_id[college_id]
        acronyms = {
            value
            for value in (
                record.get("short_acronyms", set())
                | record.get("abbreviations", set())
            )
            if len(value) >= 3
        }

        for acronym in input_tokens & acronyms:
            matching_ids = self.exact_index.get(acronym, set())
            if matching_ids == {college_id}:
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
        input_tokens = meaningful_tokens(
            input_without_city
        )

        candidate_values = [
            record["clean_name"],
            record["clean_base_name"],
            record["short_form"],
        ]

        best_overlap = 0.0

        compact_input_tokens = {
            token.replace(".", "")
            for token in input_without_city.split()
            if 2 <= len(token.replace(".", "")) <= 12
        }

        if (
            compact_input_tokens
            & record.get("abbreviations", set())
        ):
            best_overlap = 1.0

        for candidate_value in candidate_values:
            candidate_tokens = meaningful_tokens(
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

            denominator = min(
                len(input_tokens),
                len(candidate_tokens),
            )

            if denominator:
                overlap = (
                    len(shared_tokens)
                    / denominator
                )

                best_overlap = max(
                    best_overlap,
                    overlap,
                )

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

        input_domain_markers = (
            set(input_without_city.split())
            & DOMAIN_MARKERS
        )

        candidate_domain_markers = (
            (
                set(record["clean_name"].split())
                | set(record["clean_base_name"].split())
            )
            & DOMAIN_MARKERS
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

        # A shared brand/abbreviation is not enough when the institution
        # category conflicts. LN Medical College must never map to an LN
        # Pharmacy, Dental, Engineering or Management institution.
        if (
            input_domain_markers
            and candidate_domain_markers
            and input_domain_markers.isdisjoint(
                candidate_domain_markers
            )
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

        input_distinctive_tokens = meaningful_tokens(
            input_without_city
        )

        candidate_identity_tokens = (
            meaningful_tokens(record["clean_name"])
            | meaningful_tokens(record["clean_base_name"])
            | meaningful_tokens(record["short_form"])
            | record.get("abbreviations", set())
        )

        distinctive_shared_tokens = (
            input_distinctive_tokens
            & candidate_identity_tokens
        )

        shared_abbreviations = (
            input_distinctive_tokens
            & record.get("abbreviations", set())
        )

        confidence = (
            lexical_score * 0.85
        )

        reason_parts = [
            (
                "Name similarity "
                f"{lexical_score:.1f}"
            ),
            (
                "meaningful-token overlap "
                f"{token_overlap:.2f}"
            ),
        ]

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
        if (
            input_distinctive_tokens
            and not distinctive_shared_tokens
        ):
            confidence -= 55
            token_overlap = 0.0

            reason_parts.append(
                "distinctive name or abbreviation conflict"
            )

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
            shared_abbreviations
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

        identity_input = structural_identity(
            original_name
        )

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

        if clean_input in self.VERIFIED_NOT_FOUND_INPUTS:
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
            input_system_subunits = (
                set(clean_input.split())
                & {"department", "faculty", "school"}
            )
            system_ids = {
                college_id
                for college_id in self.city_ids.get(
                    detected_system_city,
                    set(),
                )
                if system_acronyms
                & (
                    set(self.college_by_id[college_id]["clean_name"].split())
                    | set(self.college_by_id[college_id]["short_form"].split())
                )
                and (
                    input_system_subunits
                    or not (
                        set(self.college_by_id[college_id]["clean_name"].split())
                        & {"department", "faculty", "school", "digital", "wilp"}
                    )
                )
            }

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

        all_exact_ids = self.exact_index.get(
            identity_input,
            set(),
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
