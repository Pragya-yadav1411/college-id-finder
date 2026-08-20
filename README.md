# College ID Finder

College ID Finder is an internal automation tool for matching college names from an uploaded Excel workbook with the Collegedunia master database.

## Main objective

The system must return the correct Collegedunia College ID.

It must never assign an ID by guessing.

## Matching rules

1. College IDs always come from the uploaded Collegedunia master database.
2. A unique exact college-name or short-form match can be accepted automatically.
3. Fuzzy matching generates suggestions only.
4. Every uncertain match requires human verification.
5. A reviewer must select the correct master record or mark the institution as `Not Found`.
6. Duplicate college rows receive the same verified College ID.
7. The target workbook structure remains unchanged.
8. Only the selected `College ID` column is updated.
9. The final workbook cannot be downloaded while unresolved institutions remain.
10. An audit log records how every institution was resolved.

## Required master-database columns

The master Excel file must contain:

- `College Id`
- `College Name`
- `City`
- `State`

Optional but recommended:

- `Short_form`
- `College Type`

## Required target-workbook columns

The target workbook must contain:

- A college-name column
- An existing College ID output column

For the current project, the expected columns are:

- `Name of Institute/ University`
- `College ID`

The application also allows the user to select different columns.

## Application process

1. Upload the Collegedunia master workbook.
2. Upload the target college workbook.
3. Select the college-name and College ID columns.
4. Click **Start College ID Matching**.
5. Review uncertain institutions one at a time.
6. Select a verified master candidate or choose `Not Found`.
7. Complete all pending reviews.
8. Download the completed workbook.
9. Download the audit log.

## Project files

```text
college-id-finder
├── data
├── app.py
├── matcher.py
├── excel_io.py
├── requirements.txt
└── README.md