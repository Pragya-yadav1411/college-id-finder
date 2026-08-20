from __future__ import annotations

import pandas as pd
import streamlit as st

from excel_io import (
    read_first_sheet,
    write_match_results,
)
from matcher import CollegeMatcher


st.set_page_config(
    page_title="College ID Finder",
    page_icon="🎓",
    layout="wide",
)

st.title("College ID Finder")

st.write(
    """
    Upload the Collegedunia master database and target
    workbook. The system will automatically find College
    IDs, identify unavailable colleges and assign a
    confidence score.
    """
)

st.caption(
    """
    Location and campus names are used to separate
    institutions such as Amity University Noida,
    Lucknow, Jaipur, Mumbai and other campuses.
    """
)


def safe_percentage(
    completed: int,
    total: int,
) -> float:
    if total <= 0:
        return 0.0

    return max(
        0.0,
        min(completed / total, 1.0),
    )


# ---------------------------------------------------------
# Upload screen
# ---------------------------------------------------------

if "matching_results" not in st.session_state:
    master_file = st.file_uploader(
        "Upload the Collegedunia master database",
        type=["xlsx"],
        key="master_upload",
    )

    target_file = st.file_uploader(
        "Upload the target college workbook",
        type=["xlsx"],
        key="target_upload",
    )

    if (
        master_file is not None
        and target_file is not None
    ):
        try:
            master_dataframe, _ = (
                read_first_sheet(
                    master_file
                )
            )

            target_dataframe, target_sheet_name = (
                read_first_sheet(
                    target_file
                )
            )

        except Exception as error:
            st.error(
                "Unable to read the Excel files: "
                f"{error}"
            )
            st.stop()

        master_count = len(
            master_dataframe
        )

        target_count = len(
            target_dataframe
        )

        load_column_1, load_column_2 = (
            st.columns(2)
        )

        load_column_1.metric(
            "Master records",
            f"{master_count:,}",
        )

        load_column_2.metric(
            "Target rows",
            f"{target_count:,}",
        )

        default_name_column = (
            "Name of Institute/ University"
        )

        if (
            default_name_column
            in target_dataframe.columns
        ):
            default_name_index = list(
                target_dataframe.columns
            ).index(default_name_column)
        else:
            default_name_index = 0

        default_id_column = "College ID"

        if (
            default_id_column
            in target_dataframe.columns
        ):
            default_id_index = list(
                target_dataframe.columns
            ).index(default_id_column)
        else:
            default_id_index = 0

        college_name_column = st.selectbox(
            "College-name column",
            options=list(
                target_dataframe.columns
            ),
            index=default_name_index,
        )

        college_id_column = st.selectbox(
            "College-ID output column",
            options=list(
                target_dataframe.columns
            ),
            index=default_id_index,
        )

        if st.button(
            "Start Automatic Matching",
            type="primary",
            use_container_width=True,
        ):
            try:
                st.subheader(
                    "Processing colleges"
                )

                progress_bar = st.progress(
                    0.0
                )

                progress_text = st.empty()
                progress_detail = st.empty()

                def update_master_progress(
                    completed: int,
                    total: int,
                    message: str,
                ) -> None:
                    stage_progress = (
                        safe_percentage(
                            completed,
                            total,
                        )
                    )

                    # Master indexing represents
                    # the first 30% of processing.
                    overall_progress = (
                        stage_progress * 0.30
                    )

                    progress_bar.progress(
                        overall_progress
                    )

                    progress_text.info(
                        "Stage 1 of 2: "
                        "Building master index"
                    )

                    progress_detail.write(
                        f"{completed:,} / "
                        f"{total:,} master records "
                        f"({stage_progress * 100:.1f}%)"
                    )

                matcher = CollegeMatcher(
                    master_dataframe,
                    progress_callback=(
                        update_master_progress
                    ),
                )

                def update_matching_progress(
                    completed: int,
                    total: int,
                    message: str,
                ) -> None:
                    stage_progress = (
                        safe_percentage(
                            completed,
                            total,
                        )
                    )

                    # Automatic matching represents
                    # the remaining 70%.
                    overall_progress = (
                        0.30
                        + stage_progress * 0.70
                    )

                    progress_bar.progress(
                        overall_progress
                    )

                    progress_text.info(
                        "Stage 2 of 2: "
                        "Matching colleges"
                    )

                    progress_detail.write(
                        f"{completed:,} / "
                        f"{total:,} unique colleges "
                        f"({stage_progress * 100:.1f}%)"
                    )

                matching_results = (
                    matcher.match_all(
                        target_dataframe[
                            college_name_column
                        ],
                        progress_callback=(
                            update_matching_progress
                        ),
                    )
                )

                progress_bar.progress(
                    1.0
                )

                progress_text.success(
                    "Automatic matching completed."
                )

                progress_detail.write(
                    "100% completed"
                )

                st.session_state[
                    "matching_results"
                ] = matching_results

                st.session_state[
                    "target_file_bytes"
                ] = target_file.getvalue()

                st.session_state[
                    "target_sheet_name"
                ] = target_sheet_name

                st.session_state[
                    "college_name_column"
                ] = college_name_column

                st.session_state[
                    "college_id_column"
                ] = college_id_column

                st.session_state[
                    "master_count"
                ] = master_count

                st.session_state[
                    "target_count"
                ] = target_count

                st.rerun()

            except Exception as error:
                st.error(
                    "Matching failed: "
                    f"{error}"
                )


# ---------------------------------------------------------
# Results screen
# ---------------------------------------------------------

if "matching_results" in st.session_state:
    results_dataframe = (
        st.session_state[
            "matching_results"
        ]
    )

    total_unique = len(
        results_dataframe
    )

    found_count = int(
        (
            results_dataframe["decision"]
            == "FOUND"
        ).sum()
    )

    not_found_count = int(
        (
            results_dataframe["decision"]
            == "NOT_FOUND"
        ).sum()
    )

    review_count = int(
        (
            results_dataframe["decision"]
            == "NEEDS_REVIEW"
        ).sum()
    )

    st.success(
        "Automatic matching completed."
    )

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Unique colleges",
        total_unique,
    )

    metric_2.metric(
        "College IDs found",
        found_count,
    )

    metric_3.metric(
        "Not Found",
        not_found_count,
    )

    metric_4.metric(
        "Needs Review",
        review_count,
    )

    # -----------------------------------------------------
    # Build downloadable workbook
    # -----------------------------------------------------

    result_mapping = {
        row["normalized_name"]: {
            "college_id": row[
                "college_id"
            ],
            "confidence": row[
                "confidence"
            ],
        }
        for _, row in (
            results_dataframe.iterrows()
        )
    }

    try:
        completed_workbook = (
            write_match_results(
                original_file_bytes=(
                    st.session_state[
                        "target_file_bytes"
                    ]
                ),
                sheet_name=(
                    st.session_state[
                        "target_sheet_name"
                    ]
                ),
                input_name_column=(
                    st.session_state[
                        "college_name_column"
                    ]
                ),
                output_id_column=(
                    st.session_state[
                        "college_id_column"
                    ]
                ),
                results=result_mapping,
                confidence_column_name=(
                    "Confidence Score"
                ),
            )
        )

        st.download_button(
            "Download College ID Result",
            data=completed_workbook,
            file_name=(
                "college_id_finder_result.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            type="primary",
            use_container_width=True,
        )

    except Exception as error:
        st.error(
            "Unable to generate the output workbook: "
            f"{error}"
        )

    # -----------------------------------------------------
    # Display review cases only
    # -----------------------------------------------------

    if review_count > 0:
        st.subheader(
            "Ambiguous colleges requiring review"
        )

        st.warning(
            f"{review_count:,} unique colleges have "
            "credible candidates but cannot be assigned "
            "safely without verification."
        )

        review_dataframe = (
            results_dataframe[
                results_dataframe["decision"]
                == "NEEDS_REVIEW"
            ][
                [
                    "input_name",
                    "college_id",
                    "matched_name",
                    "confidence",
                    "reason",
                ]
            ].copy()
        )

        review_dataframe.columns = [
            "Input College Name",
            "College ID Result",
            "Best Suggested College",
            "Confidence Score",
            "Reason",
        ]

        st.dataframe(
            review_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # Confidence distribution
    # -----------------------------------------------------

    st.subheader(
        "Confidence summary"
    )

    high_confidence = int(
        (
            results_dataframe[
                "confidence"
            ] >= 95
        ).sum()
    )

    medium_confidence = int(
        (
            (
                results_dataframe[
                    "confidence"
                ] >= 80
            )
            & (
                results_dataframe[
                    "confidence"
                ] < 95
            )
        ).sum()
    )

    low_confidence = int(
        (
            results_dataframe[
                "confidence"
            ] < 80
        ).sum()
    )

    confidence_1, confidence_2, confidence_3 = (
        st.columns(3)
    )

    confidence_1.metric(
        "High confidence (95–100)",
        high_confidence,
    )

    confidence_2.metric(
        "Medium confidence (80–94.99)",
        medium_confidence,
    )

    confidence_3.metric(
        "Low confidence (below 80)",
        low_confidence,
    )

    # -----------------------------------------------------
    # Audit-log download
    # -----------------------------------------------------

    st.subheader(
        "Audit log"
    )

    audit_columns = [
        "input_name",
        "decision",
        "college_id",
        "matched_name",
        "confidence",
        "reason",
    ]

    audit_dataframe = (
        results_dataframe[
            audit_columns
        ].copy()
    )

    audit_dataframe.columns = [
        "Input College Name",
        "Decision",
        "College ID",
        "Matched College Name",
        "Confidence Score",
        "Reason",
    ]

    st.dataframe(
        audit_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    audit_csv = audit_dataframe.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Complete Audit Log",
        data=audit_csv,
        file_name=(
            "college_id_finder_audit.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    if st.button(
        "Reset and Upload New Files",
        use_container_width=True,
    ):
        keys_to_remove = [
            "matching_results",
            "target_file_bytes",
            "target_sheet_name",
            "college_name_column",
            "college_id_column",
            "master_count",
            "target_count",
        ]

        for key in keys_to_remove:
            st.session_state.pop(
                key,
                None,
            )

        st.rerun()