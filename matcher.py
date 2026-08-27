from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd
from rapidfuzz import fuzz, process


ProgressCallback = Callable[
    [int, int, str],
    None,
]


MATCHER_VERSION = "2026.08.25.5-STRICT"


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
}


PARENT_MARKERS = {
    "faculty",
    "department",
    "school",
    "centre",
    "center",
    "academy",
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
        "peoples college of dental sciences and research centre bhopal": 58702,
    }

    # Confirmed absent institutions cannot inherit another campus ID.
    VERIFIED_NOT_FOUND_INPUTS = {
        "lnct medical college and sewakunj hospital indore",
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

        self.city_phrases: set[str] = set()
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

            raw_city = normalize(city)

            clean_city = canonical_city(
                city
            )

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
            }

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

        return cities

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

        return bool(
            candidate_cities
            and input_city not in candidate_cities
        )

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

        base_name = record[
            "clean_base_name"
        ]

        if not is_valid_parent_name(
            base_name
        ):
            return None

        if (
            f" {base_name} "
            not in f" {clean_input} "
        ):
            return None

        master_city = record[
            "clean_city"
        ]

        if (
            input_city
            and master_city
            and input_city != master_city
        ):
            return None

        if (
            input_city
            and input_city == master_city
        ):
            return 99.5

        return 97.0

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

        candidate_cities = {
            city
            for city in (
                master_city,
                city_from_candidate_name,
            )
            if city
        }

        # Campus/city conflicts are disqualifying. A matching brand or
        # abbreviation must never map an Indore institution to Bhopal,
        # a Noida institution to Lucknow, and so on.
        if (
            input_city
            and candidate_cities
            and input_city not in candidate_cities
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

        parent_score = (
            self._explicit_parent_score(
                clean_input,
                record,
                input_city,
            )
        )

        if parent_score is not None:
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
                confidence=parent_score,
                token_overlap=1.0,
                reason=(
                    "Explicit parent institution"
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

        raw_matches = process.extract(
            clean_input,
            self.search_variants,
            scorer=fuzz.WRatio,
            limit=100,
            score_cutoff=25,
        )

        candidate_ids = set()

        for variant, _, _ in raw_matches:
            candidate_ids.update(
                self.variant_ids[variant]
            )

        # Add explicitly named parent institutions.
        if (
            set(clean_input.split())
            & PARENT_MARKERS
        ):
            padded_input = (
                f" {clean_input} "
            )

            for college_id, record in (
                self.college_by_id.items()
            ):
                base_name = record[
                    "clean_base_name"
                ]

                if (
                    is_valid_parent_name(
                        base_name
                    )
                    and f" {base_name} "
                    in padded_input
                ):
                    candidate_ids.add(
                        college_id
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
    ) -> MatchDecision:
        original_name = (
            ""
            if input_name is None
            else str(input_name)
        )

        clean_input = normalize(
            input_name
        )

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
                input_name,
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

        exact_ids = self.exact_index.get(
            clean_input,
            set(),
        )

        # Exact aliases and short forms are still subject to campus
        # validation. No exact/variant route can bypass a city conflict.
        exact_ids = {
            college_id
            for college_id in exact_ids
            if not self._has_location_conflict(
                input_name,
                college_id,
            )
        }

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

        candidates = (
            self.get_ranked_candidates(
                input_name,
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
                input_name,
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
            input_name
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

        # No meaningful name overlap means the college
        # must not be assigned merely because of its city.
        if (
            best.token_overlap == 0
            or best.confidence < 60
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
            # Strict mode: an uncertain candidate never becomes a
            # numeric College ID in the output.
            college_id="Needs Review",
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
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> pd.DataFrame:
        unique_inputs: dict[
            str,
            object,
        ] = {}

        for original_name in names:
            normalized_name = normalize(
                original_name
            )

            if (
                normalized_name
                not in unique_inputs
            ):
                unique_inputs[
                    normalized_name
                ] = original_name

        items = list(
            unique_inputs.items()
        )

        total = len(items)
        output_rows = []

        for position, (
            normalized_name,
            original_name,
        ) in enumerate(
            items,
            start=1,
        ):
            decision = self.match_one(
                original_name
            )

            output_rows.append(
                {
                    "input_name": (
                        decision.input_name
                    ),
                    "normalized_name": (
                        normalized_name
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