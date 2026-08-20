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

        if input_city:
            if (
                input_city
                == record["clean_city"]
            ):
                confidence += 15

                reason_parts.append(
                    "exact city match"
                )

            else:
                confidence -= 30

                reason_parts.append(
                    "city conflict"
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

        exact_ids = self.exact_index.get(
            clean_input,
            set(),
        )

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
            and best.confidence >= 88
            and best.token_overlap >= 0.50
            and margin >= 3
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