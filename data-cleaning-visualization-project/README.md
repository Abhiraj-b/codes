# Data Cleaning & Visualization Project

This project turns a messy retail sales dataset into cleaned data, summary metrics, and a visual HTML report.

## What It Covers

- Missing value handling for dates, categories, regions, emails, prices, and units sold
- Duplicate detection and removal
- Outlier treatment with IQR capping
- Feature engineering for revenue and order month
- Visual reporting with a browser-ready dashboard

## Project Structure

```text
.
|-- data/
|   |-- raw_sales_data.csv
|   `-- processed/
|-- reports/
|-- src/
|   `-- clean_visualize.py
|-- requirements.txt
`-- README.md
```

## Run

Use the bundled Codex Python path if `python` is not available on your machine:

```powershell
& 'C:\Users\abhir\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\clean_visualize.py
```

If Python is on your PATH:

```powershell
python src\clean_visualize.py
```

## Outputs

- `data/processed/cleaned_sales_data.csv`
- `reports/cleaning_summary.json`
- `reports/dashboard.html`

Open `reports/dashboard.html` in a browser to view the final visual report.
