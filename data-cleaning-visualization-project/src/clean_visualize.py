from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw_sales_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def normalize_category(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"
    cleaned = str(value).strip().title()
    replacements = {
        "Furnture": "Furniture",
        "Electronics ": "Electronics",
    }
    return replacements.get(cleaned, cleaned)


def cap_iqr(series: pd.Series) -> tuple[pd.Series, int, dict[str, float]]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return series.clip(lower, upper), int(mask.sum()), {"lower": float(lower), "upper": float(upper)}


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    summary: dict[str, object] = {
        "raw_rows": int(len(raw)),
        "raw_columns": list(raw.columns),
        "missing_values_before": raw.isna().sum().to_dict(),
    }

    df = raw.copy()
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    invalid_dates = int(df["order_date"].isna().sum())
    df = df.dropna(subset=["order_date"])

    df["customer_email"] = (
        df["customer_email"].fillna("unknown@example.com").astype(str).str.strip().str.lower()
    )
    df["region"] = df["region"].fillna("Unknown").astype(str).str.strip().str.title()
    df["category"] = df["category"].apply(normalize_category)
    df["product"] = df["product"].fillna("Unknown Product").astype(str).str.strip()
    df["payment_method"] = df["payment_method"].fillna("Unknown").astype(str).str.strip().str.title()

    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    negative_units = int((df["units_sold"] < 0).sum())
    df.loc[df["units_sold"] < 0, "units_sold"] = pd.NA

    df["units_sold"] = df["units_sold"].fillna(df["units_sold"].median()).round().astype(int)
    df["unit_price"] = df["unit_price"].fillna(df.groupby("category")["unit_price"].transform("median"))
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

    df["units_sold"], unit_outliers, unit_bounds = cap_iqr(df["units_sold"].astype(float))
    df["unit_price"], price_outliers, price_bounds = cap_iqr(df["unit_price"].astype(float))

    df["units_sold"] = df["units_sold"].round().astype(int)
    df["unit_price"] = df["unit_price"].round(2)
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)

    summary.update(
        {
            "duplicates_removed": duplicate_count,
            "invalid_dates_removed": invalid_dates,
            "negative_units_replaced": negative_units,
            "units_sold_outliers_capped": unit_outliers,
            "unit_price_outliers_capped": price_outliers,
            "outlier_bounds": {
                "units_sold": unit_bounds,
                "unit_price": price_bounds,
            },
            "clean_rows": int(len(df)),
            "missing_values_after": df.isna().sum().to_dict(),
            "total_revenue": float(df["revenue"].sum()),
            "average_order_value": float(df["revenue"].mean()),
        }
    )
    return df, summary


def svg_bar(data: pd.Series, title: str, color: str) -> str:
    items = data.sort_values(ascending=False)
    width, height = 700, 310
    left, right, top, bottom = 140, 30, 35, 36
    plot_width = width - left - right
    bar_h = 28
    gap = 15
    max_value = float(items.max()) if len(items) else 1.0
    rows = []
    for index, (label, value) in enumerate(items.items()):
        y = top + index * (bar_h + gap)
        bar_w = 0 if max_value == 0 else (float(value) / max_value) * plot_width
        rows.append(
            f'<text x="{left - 12}" y="{y + 19}" text-anchor="end">{html.escape(str(label))}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{color}"/>'
            f'<text x="{left + bar_w + 8}" y="{y + 19}">{float(value):,.0f}</text>'
        )
    svg_height = max(height, top + len(items) * (bar_h + gap) + bottom)
    return (
        f'<svg viewBox="0 0 {width} {svg_height}" role="img" aria-label="{html.escape(title)}">'
        f'<text x="0" y="20" class="chart-title">{html.escape(title)}</text>'
        + "".join(rows)
        + "</svg>"
    )


def svg_line(data: pd.Series, title: str) -> str:
    items = data.sort_index()
    width, height = 700, 310
    left, right, top, bottom = 60, 30, 40, 48
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = float(items.max()) if len(items) else 1.0
    points = []
    labels = []
    for index, (label, value) in enumerate(items.items()):
        x = left + (index / max(len(items) - 1, 1)) * plot_width
        y = top + plot_height - ((float(value) / max_value) * plot_height if max_value else 0)
        points.append(f"{x:.1f},{y:.1f}")
        labels.append(f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle">{html.escape(str(label))}</text>')
    circles = "".join(
        f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" r="5" fill="#1f7a8c"/>'
        for point in points
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<text x="0" y="20" class="chart-title">{html.escape(title)}</text>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#c9d3dc"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#1f7a8c" stroke-width="4"/>'
        f"{circles}{''.join(labels)}</svg>"
    )


def make_dashboard(df: pd.DataFrame, summary: dict[str, object]) -> str:
    revenue_by_category = df.groupby("category")["revenue"].sum()
    revenue_by_region = df.groupby("region")["revenue"].sum()
    monthly_revenue = df.groupby("order_month")["revenue"].sum()
    payment_mix = df.groupby("payment_method")["order_id"].count()

    cards = [
        ("Clean Rows", f"{summary['clean_rows']:,}"),
        ("Revenue", f"${summary['total_revenue']:,.0f}"),
        ("Avg Order", f"${summary['average_order_value']:,.0f}"),
        ("Duplicates Removed", f"{summary['duplicates_removed']:,}"),
    ]
    card_html = "".join(
        f'<article class="metric"><span>{label}</span><strong>{value}</strong></article>'
        for label, value in cards
    )

    top_products = (
        df.groupby("product", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(6)
    )
    table_rows = "".join(
        f"<tr><td>{html.escape(row.product)}</td><td>${row.revenue:,.2f}</td></tr>"
        for row in top_products.itertuples(index=False)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sales Data Cleaning Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #5c6b75;
      --line: #d8e0e7;
      --panel: #ffffff;
      --page: #f5f7f9;
      --teal: #1f7a8c;
      --gold: #c98b2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--page);
      color: var(--ink);
    }}
    header {{
      padding: 32px 28px 20px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 40px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.5; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .metric, .chart, .table-panel, .note {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 16px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 8px; font-size: 28px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .chart {{ padding: 16px; overflow-x: auto; }}
    svg {{ width: 100%; min-width: 520px; height: auto; }}
    text {{ fill: var(--ink); font-size: 14px; }}
    .chart-title {{ font-size: 18px; font-weight: 700; }}
    .table-panel {{ margin-top: 18px; padding: 18px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 12px; border-top: 1px solid var(--line); }}
    th {{ color: var(--muted); font-size: 13px; }}
    .note {{ margin-top: 18px; padding: 16px; color: var(--muted); }}
    @media (max-width: 800px) {{
      .metrics, .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Sales Data Cleaning Dashboard</h1>
    <p>Raw retail orders were cleaned for duplicates, invalid dates, missing values, negative units, and outliers before analysis.</p>
  </header>
  <main>
    <section class="metrics">{card_html}</section>
    <section class="grid">
      <article class="chart">{svg_bar(revenue_by_category, "Revenue by Category", "#1f7a8c")}</article>
      <article class="chart">{svg_bar(revenue_by_region, "Revenue by Region", "#c98b2c")}</article>
      <article class="chart">{svg_line(monthly_revenue, "Monthly Revenue Trend")}</article>
      <article class="chart">{svg_bar(payment_mix, "Orders by Payment Method", "#4f6d7a")}</article>
    </section>
    <section class="table-panel">
      <h2>Top Products</h2>
      <table>
        <thead><tr><th>Product</th><th>Revenue</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>
    <section class="note">
      Outliers were capped using the IQR method, which preserves rows while reducing the influence of extreme values.
    </section>
  </main>
</body>
</html>"""


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_DATA)
    cleaned, summary = clean_data(raw)

    cleaned.to_csv(PROCESSED_DIR / "cleaned_sales_data.csv", index=False)
    (REPORTS_DIR / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "dashboard.html").write_text(make_dashboard(cleaned, summary), encoding="utf-8")

    print("Cleaning and visualization complete.")
    print(f"Rows: {summary['raw_rows']} raw -> {summary['clean_rows']} clean")
    print(f"Revenue: ${summary['total_revenue']:,.2f}")
    print(f"Dashboard: {REPORTS_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    main()
