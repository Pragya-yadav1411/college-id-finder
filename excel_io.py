from __future__ import annotations

from copy import copy
from io import BytesIO

import openpyxl
import pandas as pd

from matcher import normalize


def read_first_sheet(
    uploaded_file,
) -> tuple[pd.DataFrame, str]:
    """
    Read the first worksheet from an uploaded Excel file.
    """

    uploaded_file.seek(0)

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    sheet_name = (
        excel_file.sheet_names[0]
    )

    uploaded_file.seek(0)

    dataframe = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
    )

    return dataframe, sheet_name


def get_header_columns(
    worksheet,
) -> dict[str, int]:
    """
    Return worksheet headers and column numbers.
    """

    return {
        str(cell.value): cell.column
        for cell in worksheet[1]
        if cell.value is not None
    }


def copy_cell_format(
    source_cell,
    target_cell,
) -> None:
    """
    Copy formatting from one Excel cell to another.
    """

    if source_cell.has_style:
        target_cell._style = copy(
            source_cell._style
        )

    if source_cell.number_format:
        target_cell.number_format = (
            source_cell.number_format
        )

    if source_cell.font:
        target_cell.font = copy(
            source_cell.font
        )

    if source_cell.fill:
        target_cell.fill = copy(
            source_cell.fill
        )

    if source_cell.border:
        target_cell.border = copy(
            source_cell.border
        )

    if source_cell.alignment:
        target_cell.alignment = copy(
            source_cell.alignment
        )

    if source_cell.protection:
        target_cell.protection = copy(
            source_cell.protection
        )


def ensure_confidence_column(
    worksheet,
    college_id_column_number: int,
    confidence_column_name: str,
) -> int:
    """
    Find or create the Confidence Score column.

    If it does not exist, create it immediately after
    the selected College ID column.
    """

    header_columns = get_header_columns(
        worksheet
    )

    if (
        confidence_column_name
        in header_columns
    ):
        return header_columns[
            confidence_column_name
        ]

    confidence_column_number = (
        college_id_column_number + 1
    )

    worksheet.insert_cols(
        confidence_column_number,
        amount=1,
    )

    id_header_cell = worksheet.cell(
        row=1,
        column=college_id_column_number,
    )

    confidence_header_cell = (
        worksheet.cell(
            row=1,
            column=confidence_column_number,
        )
    )

    confidence_header_cell.value = (
        confidence_column_name
    )

    copy_cell_format(
        id_header_cell,
        confidence_header_cell,
    )

    # Copy the College ID column width.
    id_column_letter = (
        openpyxl.utils.get_column_letter(
            college_id_column_number
        )
    )

    confidence_column_letter = (
        openpyxl.utils.get_column_letter(
            confidence_column_number
        )
    )

    id_width = (
        worksheet.column_dimensions[
            id_column_letter
        ].width
    )

    worksheet.column_dimensions[
        confidence_column_letter
    ].width = max(
        id_width or 12,
        18,
    )

    return confidence_column_number


def write_match_results(
    original_file_bytes: bytes,
    sheet_name: str,
    input_name_column: str,
    output_id_column: str,
    results: dict[str, dict],
    confidence_column_name: str = (
        "Confidence Score"
    ),
) -> bytes:
    """
    Write College IDs and confidence scores into the
    original target workbook.

    Safety rules:
    - Original row order is preserved.
    - No source row is removed.
    - Duplicate names receive the same result.
    - The existing College ID column is used.
    - Confidence Score is added beside College ID.
    - Other source values remain unchanged.
    """

    workbook = openpyxl.load_workbook(
        BytesIO(original_file_bytes)
    )

    if sheet_name not in workbook.sheetnames:
        raise ValueError(
            f"Worksheet not found: {sheet_name}"
        )

    worksheet = workbook[
        sheet_name
    ]

    header_columns = get_header_columns(
        worksheet
    )

    if (
        input_name_column
        not in header_columns
    ):
        raise ValueError(
            "College-name column not found: "
            f"{input_name_column}"
        )

    if (
        output_id_column
        not in header_columns
    ):
        raise ValueError(
            "College-ID column not found: "
            f"{output_id_column}"
        )

    name_column_number = (
        header_columns[
            input_name_column
        ]
    )

    id_column_number = (
        header_columns[
            output_id_column
        ]
    )

    confidence_column_number = (
        ensure_confidence_column(
            worksheet,
            id_column_number,
            confidence_column_name,
        )
    )

    for row_number in range(
        2,
        worksheet.max_row + 1,
    ):
        original_college_name = (
            worksheet.cell(
                row=row_number,
                column=name_column_number,
            ).value
        )

        normalized_name = normalize(
            original_college_name
        )

        if normalized_name not in results:
            continue

        result = results[
            normalized_name
        ]

        college_id_value = result[
            "college_id"
        ]

        confidence_value = result[
            "confidence"
        ]

        id_cell = worksheet.cell(
            row=row_number,
            column=id_column_number,
        )

        confidence_cell = worksheet.cell(
            row=row_number,
            column=confidence_column_number,
        )

        id_cell.value = college_id_value
        confidence_cell.value = (
            confidence_value
        )

        confidence_cell.number_format = (
            '0.00'
        )

        # Copy the nearby row formatting so the new
        # confidence cell matches the original workbook.
        copy_cell_format(
            id_cell,
            confidence_cell,
        )

        confidence_cell.number_format = (
            '0.00'
        )

    output_file = BytesIO()

    workbook.save(
        output_file
    )

    return output_file.getvalue()