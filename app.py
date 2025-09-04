
import os
import json
import sqlite3
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Tuple
import io
from fastapi.responses import StreamingResponse
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from charset_normalizer import detect

# ========= Logging =========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zalando-analytics")

# ========= ENV & Paths =========
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "Zalando"))
EAN_PATH = Path(os.getenv("EAN_CSV", BASE_DIR / "EAN.csv"))
if not EAN_PATH.exists():
    alt = BASE_DIR / "EAN.scv"
    if alt.exists():
        EAN_PATH = alt
VAT_PATH = Path(os.getenv("VAT_XLSX", BASE_DIR / "VAT.xlsx"))

DB_PATH = BASE_DIR / "zalando_data.db"
os.makedirs(BASE_DIR, exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
logger.info("DB PATH: %s", DB_PATH)

# ========= App (define BEFORE using @app.*) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up dashboard…")
    # DB tables should already exist from your ETL app.
    yield
    logger.info("Shutting down dashboard…")
    conn.close()

app = FastAPI(
    title="Zalando Pro Analytics",
    description="Comfortable, high-signal analytics UI over your SQLite dataset",
    lifespan=lifespan,
)


# ========= Helpers =========
def _norm_cols(cols: List[str]) -> List[str]:
    return [
        c.lower().replace(" ", "").replace("/", "").replace(".", "").replace("-", "").replace("\ufeff", "")
        for c in cols
    ]

def normalize_ean_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s

def to_float_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\xa0", "", regex=False).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _month_sort_key(label: str) -> Tuple[int]:
    try:
        n = int(str(label).split(".")[0].strip())
        return (n,)
    except Exception:
        return (9999,)

def _eu(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)

# ========= Data access / KPIs =========
def _load_sales(conn, start_date: str, end_date: str) -> pd.DataFrame:
    q = (
        "SELECT ean, type, gross_revenue, currency, country, month, document_date, order_date, return_date, status "
        "FROM sales WHERE document_date BETWEEN ? AND ?"
    )
    df = pd.read_sql(q, conn, params=(start_date, end_date))
    df["gross_revenue"] = pd.to_numeric(df["gross_revenue"], errors="coerce").fillna(0.0)
    # Your pipeline signs sales negative, returns positive; flip to "human" sign:
    df["signed_rev"] = -1.0 * df["gross_revenue"]
    return df

def _load_ean_base(conn) -> pd.DataFrame:
    q = "SELECT ean, article, color, size, url, season, category, heel_height, cost, zalando_sku FROM ean_base"
    df = pd.read_sql(q, conn)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
    return df

def _load_zfs(conn, start_date: str, end_date: str) -> pd.DataFrame:
    # ZFS lacks dates; we’ll join via month to sales for context, but also surface raw grouping by month.
    q = "SELECT ean, month, event_type, amount FROM zfs"
    df = pd.read_sql(q, conn)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df

def compute_summary(conn, start_date: str, end_date: str,
                    country: Optional[str] = None,
                    category: Optional[str] = None) -> Dict[str, Any]:
    sales = _load_sales(conn, start_date, end_date)
    eans = _load_ean_base(conn)
    zfs = _load_zfs(conn, start_date, end_date)

    # Join ean info
    sales = sales.merge(eans, on="ean", how="left")

    # Optional filters
    if country:
        sales = sales[sales["country"] == country]
    if category:
        sales = sales[sales["category"] == category]

    # Basic KPIs
    s_sales = sales[sales["type"] == "Sale"]["signed_rev"].sum()
    s_returns = sales[sales["type"] == "Return"]["signed_rev"].sum()
    net_rev = s_sales - s_returns

    sales_cnt = sales[sales["type"] == "Sale"].groupby(["ean", "month"]).size()
    returns_cnt = sales[sales["type"] == "Return"].groupby(["ean", "month"]).size()
    tot_sales_units = int(sales_cnt.sum())
    tot_return_units = int(returns_cnt.sum())
    overall_return_pct = (tot_return_units / tot_sales_units * 100.0) if tot_sales_units else 0.0

    # Net revenue by month
    sales_m = sales[sales["type"] == "Sale"].groupby("month")["signed_rev"].sum()
    ret_m = sales[sales["type"] == "Return"].groupby("month")["signed_rev"].sum()
    net_by_m = (sales_m - ret_m).fillna(0.0)
    net_by_m = net_by_m.sort_index(key=lambda idx: [_month_sort_key(i) for i in idx])

    # Per EAN×Month: return %
    return_pct = (returns_cnt / sales_cnt * 100.0).dropna()

    # Per EAN×Month: net revenue
    s_rev = sales[sales["type"] == "Sale"].groupby(["ean", "month"])["signed_rev"].sum()
    r_rev = sales[sales["type"] == "Return"].groupby(["ean", "month"])["signed_rev"].sum()
    net_rev_em = (s_rev - r_rev).fillna(0.0)

    # ZFS costs (flip sign so costs are positive numbers)
    zfs_cost = pd.Series(dtype=float)
    if not zfs.empty:
        zfs["norm_event"] = zfs["event_type"].astype(str)
        storage = zfs[zfs["norm_event"].str.contains("Storage", na=False)]
        storage_cost_em = (-1.0 * storage.groupby(["ean", "month"])["amount"].sum()).fillna(0.0)

        # Map main logistics buckets
        outbound = zfs[zfs["norm_event"].str.contains("Outbound Shipping|Outbound Cross-Border Fee|Warehouse Processing", na=False)]
        outbound_cost = (-1.0 * outbound.groupby(["ean", "month"])["amount"].sum()).fillna(0.0)

        returns_log = zfs[zfs["norm_event"].str.contains("Returns Processing|Returns Shipping|Returns Refurbishment", na=False)]
        returns_log_cost = (-1.0 * returns_log.groupby(["ean", "month"])["amount"].sum()).fillna(0.0)

        r2p = zfs[zfs["norm_event"].str.contains("Return to Partner Processing", na=False)]
        r2p_cost = (-1.0 * r2p.groupby(["ean", "month"])["amount"].sum()).fillna(0.0)

        # Unclassified = everything else cost-positive
        known = storage.index.tolist()
        # we’ll compute total monthly and subtract known buckets
        zfs_total = (-1.0 * zfs.groupby(["ean", "month"])["amount"].sum()).fillna(0.0)
        unclassified = (zfs_total
                        - storage_cost_em.reindex(zfs_total.index, fill_value=0.0)
                        - outbound_cost.reindex(zfs_total.index, fill_value=0.0)
                        - returns_log_cost.reindex(zfs_total.index, fill_value=0.0)
                        - r2p_cost.reindex(zfs_total.index, fill_value=0.0))

        zfs_cost = {
            "storage": storage_cost_em,
            "outbound": outbound_cost,
            "returns": returns_log_cost,
            "return_to_partner": r2p_cost,
            "other": unclassified
        }
    else:
        zfs_cost = {"storage": pd.Series(dtype=float),
                    "outbound": pd.Series(dtype=float),
                    "returns": pd.Series(dtype=float),
                    "return_to_partner": pd.Series(dtype=float),
                    "other": pd.Series(dtype=float)}

    # Commission approximation (marketing + payment svc fee) — use Sales if you have columns, otherwise 0
    # Placeholder: compute % of positive sales revenue as a proxy if needed
    # (Easy to swap to real columns once present)
    commission_proxy = 0.0

    # VAT (if table present)
    try:
        vat = pd.read_sql("SELECT country, vat_percent FROM vat", conn)
        sales_v = sales.merge(vat, on="country", how="left")
        # VAT formula described in spec (per month or per EAN, we sum global here)
        sold_eur = sales_v[sales_v["type"] == "Sale"]["signed_rev"].sum()
        returned_eur = sales_v[sales_v["type"] == "Return"]["signed_rev"].sum()
        vatrate = (sales_v["vat_percent"].dropna().median() or 0.0) / 100.0 if not sales_v["vat_percent"].dropna().empty else 0.0
        # The spec’s line looked off; a common VAT estimate on net sales:
        # VAT ≈ (Sold - Returned) * vatrate
        vat_amount = (sold_eur - returned_eur) * vatrate
    except Exception:
        vat_amount = 0.0

    # Cost of goods (COGS) using EAN cost × net units
    # net units = sales units - return units
    units_sales = sales[sales["type"] == "Sale"].groupby(["ean", "month"]).size().rename("u_sale")
    units_ret = sales[sales["type"] == "Return"].groupby(["ean", "month"]).size().rename("u_ret")
    units_em = pd.concat([units_sales, units_ret], axis=1).fillna(0.0)
    units_em["u_net"] = units_em["u_sale"] - units_em["u_ret"]

    cost_map = eans.set_index("ean")["cost"]
    def _cogs_for(pair):
        ean = pair[0]
        u = units_em.loc[pair]["u_net"] if pair in units_em.index else 0.0
        return max(u, 0.0) * float(cost_map.get(ean, 0.0))

    net_keys = list(set(list(net_rev_em.index)))
    cogs_sum = 0.0
    for k in net_keys:
        try:
            cogs_sum += _cogs_for(k)
        except Exception:
            pass

    # Build top lists
    def _top(series: pd.Series, n: int, desc: bool = True, as_rows=True):
        if series is None or series.empty:
            return []
        s = series.sort_values(ascending=not desc).head(n)
        if not as_rows:
            return s
        out = []
        for (ean, month), val in s.items():
            out.append({"ean": str(ean), "month": str(month), "value": float(val)})
        return out

    top_return_pct = _top(return_pct, 15, desc=True, as_rows=True)
    worst_net = _top(net_rev_em, 15, desc=False, as_rows=True)
    top_storage = _top(zfs_cost["storage"], 15, desc=True, as_rows=True)

    # Net revenue by month (chart)
    chart_months = list(net_by_m.index)
    chart_values = [float(net_by_m[m]) for m in chart_months]

    # Summarize ZFS buckets (total €)
    zfs_totals = {k: float(v.sum()) if isinstance(v, pd.Series) else 0.0 for k, v in zfs_cost.items()}

    return {
        "filters": {"start": start_date, "end": end_date, "country": country, "category": category},
        "kpis": {
            "total_sales_eur": float(s_sales),
            "total_returns_eur": float(s_returns),
            "net_revenue_eur": float(net_rev),
            "overall_return_pct": float(overall_return_pct),
            "units_sold": int(tot_sales_units),
            "units_returned": int(tot_return_units),
            "vat_eur": float(vat_amount),
            "commission_eur": float(commission_proxy),
            "cogs_eur": float(cogs_sum),
        },
        "zfs": {"buckets_eur": zfs_totals},
        "charts": {
            "net_revenue_by_month": {"months": chart_months, "values": chart_values},
            "top_return_pct": top_return_pct,
            "top_storage_costs": top_storage
        },
        "tables": {
            "worst_net_revenue": worst_net,
            "top_return_pct": top_return_pct,
            "top_storage_costs": top_storage
        }
    }

# ========= API for frontend =========
@app.get("/healthz")
def healthz():
    try:
        cursor.execute("SELECT COUNT(*) FROM sales")
        sales_cnt = cursor.fetchone()[0]
        return {"ok": True, "db": str(DB_PATH), "sales_rows": sales_cnt}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/rowcounts")
def rowcounts():
    tables = [
        "sales","zfs","invoices","stock","goods_received",
        "tbone_desadv","tbone_returns","ean_base","vat","returns_policy"
    ]
    out = {}
    for t in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            out[t] = cursor.fetchone()[0]
        except Exception as e:
            out[t] = f"ERR: {e}"
    return out

@app.get("/api/summary")
def api_summary(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    country: Optional[str] = None,
    category: Optional[str] = None,
):
    data = compute_summary(conn, start, end, country, category)
    return JSONResponse(data)

@app.get("/export/payments")
def export_payments(month: str = Query(..., description="e.g. '7. 2025 July'")):
    try:
        df = payments_recon(conn, month)
        if isinstance(df, dict):  # if your function returns dict
            df = pd.DataFrame(df.get("unpaid_orders", []))
        if df.empty:
            df = pd.DataFrame(columns=["order_number", "gross_revenue", "status"])

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=payments_{month.replace(' ', '_')}.csv"}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/export/vat")
def export_vat(countries: Optional[str] = Query(None)):
    lst = [c.strip() for c in countries.split(",")] if countries else None
    df = vat_avask(conn, lst)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vat_report.csv"}
    )

@app.get("/export/vat")
def export_vat(countries: Optional[str] = Query(None)):
    lst = [c.strip() for c in countries.split(",")] if countries else None
    df = vat_avask(conn, lst)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vat_report.csv"}
    )
    


# try to import report helpers; otherwise define inline fallbacks
try:
    from reports import payments_recon as _payments_recon_mod
except Exception:
    _payments_recon_mod = None

try:
    from reports import vat_avask as _vat_avask_mod
except Exception:
    _vat_avask_mod = None


def _prev_month_label(all_months: list[str], month: str) -> str:
    if month not in all_months:
        return all_months[0]
    i = all_months.index(month)
    return all_months[i-1] if i > 0 else all_months[0]


def payments_recon(conn, month: str):
    """Return a CSV StreamingResponse with unpaid orders and planned amount, even if reports.py is missing."""
    if _payments_recon_mod is not None:
        # delegate to reports.py if available
        return _payments_recon_mod(conn, month)

    # --- inline fallback (matches logic you used in /query) ---
    this_m = month
    prev_m = _prev_month_label(months, this_m)

    sales_curr = pd.read_sql(
        "SELECT order_number, gross_revenue, status FROM sales WHERE month=?",
        conn, params=(this_m,)
    )
    sales_prev = pd.read_sql(
        "SELECT order_number, gross_revenue FROM sales WHERE month=?",
        conn, params=(prev_m,)
    )
    inv = pd.read_sql(
        "SELECT order_number FROM invoices WHERE month=?",
        conn, params=(this_m,)
    )

    all_sales = pd.concat([sales_curr, sales_prev], ignore_index=True)
    inv_set = set(inv["order_number"].astype(str))
    all_sales["order_number"] = all_sales["order_number"].astype(str)

    unpaid = all_sales[~all_sales["order_number"].isin(inv_set)].copy()
    # flip sign convention: store positive values for readability
    unpaid["gross_revenue_eur"] = pd.to_numeric(unpaid["gross_revenue"], errors="coerce").fillna(0.0) * -1.0
    unpaid = unpaid[["order_number", "gross_revenue_eur", "status"]].sort_values("order_number")

    planned_next = sales_curr[
        (sales_curr.get("status", pd.Series(dtype=str)) == "settled already") &
        (~sales_curr["order_number"].astype(str).isin(inv_set))
    ]
    planned_total = (pd.to_numeric(planned_next["gross_revenue"], errors="coerce").fillna(0.0) * -1.0).sum()

    # build a CSV in-memory
    buf = io.StringIO()
    # header row with month & planned summary
    buf.write(f"# Payments Reconciliation for {this_m}\n")
    buf.write(f"# Planned next month total (EUR): {planned_total:.2f}\n")
    buf.write("order_number,gross_revenue_eur,status\n")
    unpaid.to_csv(buf, index=False, header=False)
    buf.seek(0)

    filename = f"payments_{this_m.replace(' ', '_').replace('.', '')}.csv"
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


def vat_avask(conn, countries: list[str] | None):
    """Return a CSV StreamingResponse ready for AVASK, even if reports.py is missing."""
    if _vat_avask_mod is not None:
        return _vat_avask_mod(conn, countries)

    sales = pd.read_sql(
        "SELECT ean, order_number, document_date, type, gross_revenue, currency, country FROM sales",
        conn
    )
    tbone = pd.read_sql(
        "SELECT order_number, product_name, street, city FROM tbone_desadv",
        conn
    )
    vat = pd.read_sql(
        "SELECT country, vat_percent FROM vat",
        conn
    )

    if countries:
        countries = [c.strip() for c in countries if c and c.strip()]
        if countries:
            sales = sales[sales["country"].isin(countries)]

    merged = sales.merge(tbone, on="order_number", how="left").merge(vat, on="country", how="left")

    # transform rows
    out = []
    for _, row in merged.iterrows():
        gross = pd.to_numeric(row.get("gross_revenue"), errors="coerce")
        gross = float(gross) * -1.0 if pd.notna(gross) else 0.0
        vat_rate = row.get("vat_percent")
        vat_rate = float(vat_rate) / 100.0 if pd.notna(vat_rate) else 0.0
        out.append({
            "DATE": row.get("document_date"),
            "ORDER NUMBER/REF/ID": row.get("order_number"),
            "TRANSACTION TYPE": row.get("type"),
            "SKU": row.get("ean"),
            "DESCRIPTION": row.get("product_name"),
            "QUANTITY": 1,
            "ORDER TOTAL": gross,
            "CURRENCY": row.get("currency"),
            "SALE DEPARTURE COUNTRY/FULFILMENT COUNTRY": row.get("country"),
            "SALE ARRIVAL COUNTRY/DESTINATION COUNTRY": row.get("country"),
            "TAX RATE": vat_rate,
            "BUYER VAT NUMBER": "PRIVATE INDIVIDUAL",
            "SHIPPING ADDRESS 1": row.get("street"),
            "SHIPPING CITY": row.get("city"),
            "SHIPPING COUNTRY": row.get("country"),
            "TAX COLLECTION RESPONSIBILITY": "SELLER",
            "MARKETPLACE": "ZALANDO",
        })

    df = pd.DataFrame(out)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    suffix = "_".join(countries) if countries else "ALL"
    filename = f"vat_avask_{suffix}.csv"
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
@app.get("/export/payments")
def export_payments(month: str = Query(..., description="e.g. '7. 2025 July'")):
    return payments_recon(conn, month)

@app.get("/export/vat")
def export_vat(countries: Optional[str] = Query(None, description="Comma-separated, e.g. 'DE,PL,SE'")):
    lst = [c.strip() for c in countries.split(",")] if countries else None
    return vat_avask(conn, lst)
    


# ========= Clean, comfortable HTML dashboard =========
@app.get("/", response_class=HTMLResponse)
def dashboard_home(
    start: str = Query("2025-03-01"),
    end: str = Query("2025-07-31"),
    country: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    # Precompute once for initial render
    data = compute_summary(conn, start, end, country, category)
    payload = json.dumps(data)

    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Zalando Pro Analytics</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Bootstrap + Plotly + Tabler Icons -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.22.0/tabler-icons.min.css" rel="stylesheet">
<style>
  :root {{
    --bg: #0f1221; --panel:#151834; --muted:#99a1c2; --text:#e6e8ef; --line:#2b2f55; --accent:#9ad0ff;
  }}
  body {{ background:var(--bg); color:var(--text); }}
  .panel {{ background:var(--panel); border-radius:16px; padding:18px; }}
  .kpi {{ background:#171a2f; border:none; border-radius:16px; padding:16px; }}
  .kpi small {{ color:var(--muted); display:block; }}
  .kpi strong {{ font-size:1.5rem; }}
  .subtle {{ color:var(--muted); }}
  .table thead th {{ color:var(--muted); border-bottom-color:var(--line); }}
  .table tbody td {{ border-bottom-color:var(--line); }}
  a, a:hover {{ color:var(--accent); }}
  .chip {{ background:#0e1737; border:1px solid #2b2f55; border-radius:999px; padding:4px 10px; font-size:.85rem; color:var(--muted); }}
  .header {{ display:flex; gap:12px; align-items:center; }}
  .header h3 {{ margin:0; }}
</style>
</head>
<body class="p-3 p-md-4">
  <div class="container-fluid">
    <div class="header mb-3">
      <h3 class="me-2"><i class="ti ti-chart-line"></i> Zalando Pro Analytics</h3>
      <span class="chip">Period: <strong>{_safe_str(start)}</strong> → <strong>{_safe_str(end)}</strong></span>
      {"<span class='chip'>Country: <strong>"+_safe_str(country)+"</strong></span>" if country else ""}
      {"<span class='chip'>Category: <strong>"+_safe_str(category)+"</strong></span>" if category else ""}
    </div>

    <!-- Filters -->
    <form class="row g-2 mb-3" method="get" action="/">
      <div class="col-6 col-md-3">
        <label class="form-label subtle">Start</label>
        <input class="form-control" type="date" name="start" value="{_safe_str(start)}">
      </div>
      <div class="col-6 col-md-3">
        <label class="form-label subtle">End</label>
        <input class="form-control" type="date" name="end" value="{_safe_str(end)}">
      </div>
      <div class="col-6 col-md-3">
        <label class="form-label subtle">Country (optional)</label>
        <input class="form-control" type="text" name="country" value="{_safe_str(country)}" placeholder="DE / PL / ...">
      </div>
      <div class="col-6 col-md-3">
        <label class="form-label subtle">Category (optional)</label>
        <input class="form-control" type="text" name="category" value="{_safe_str(category)}" placeholder="e.g. Boots">
      </div>
      <div class="col-12 col-md-3">
        <label class="form-label subtle">&nbsp;</label>
        <div><button class="btn btn-primary">Apply</button>
        <a class="btn btn-outline-light ms-2" href="/">Reset</a></div>
      </div>
    </form>

    <!-- KPIs -->
    <div class="row g-3 mb-3">
      <div class="col-6 col-xl-2"><div class="kpi"><small>Total Sales (€)</small><strong id="kpiSales">{data["kpis"]["total_sales_eur"]:.2f}</strong></div></div>
      <div class="col-6 col-xl-2"><div class="kpi"><small>Total Returns (€)</small><strong id="kpiRet">{data["kpis"]["total_returns_eur"]:.2f}</strong></div></div>
      <div class="col-6 col-xl-2"><div class="kpi"><small>Net Revenue (€)</small><strong id="kpiNet">{data["kpis"]["net_revenue_eur"]:.2f}</strong></div></div>
      <div class="col-6 col-xl-2"><div class="kpi"><small>Return Rate</small><strong id="kpiRR">{data["kpis"]["overall_return_pct"]:.2f}%</strong></div></div>
      <div class="col-6 col-xl-2"><div class="kpi"><small>VAT (€)</small><strong id="kpiVAT">{data["kpis"]["vat_eur"]:.2f}</strong></div></div>
      <div class="col-6 col-xl-2"><div class="kpi"><small>COGS (€)</small><strong id="kpiCOGS">{data["kpis"]["cogs_eur"]:.2f}</strong></div></div>
    </div>

    <div class="row g-3 mb-3">
      <div class="col-12 col-lg-6"><div class="panel">
        <h6>Net Revenue by Month</h6>
        <div id="netByMonth"></div>
      </div></div>
      <div class="col-12 col-lg-6"><div class="panel">
        <h6>Top Return % (EAN × Month)</h6>
        <div id="topReturnPct"></div>
      </div></div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-12 col-lg-6"><div class="panel">
        <h6>Top Storage Costs (EAN × Month)</h6>
        <div id="topStorageCosts"></div>
      </div></div>
      <div class="col-12 col-lg-6"><div class="panel">
        <h6>Worst Net Revenue (Top 15)</h6>
        <div class="table-responsive">
          <table class="table table-sm align-middle">
            <thead><tr><th>#</th><th>EAN</th><th>Month</th><th class="text-end">€</th></tr></thead>
            <tbody id="tblWorst"></tbody>
          </table>
        </div>
      </div></div>
    </div>

    <div class="row g-3">
      <div class="col-12"><div class="panel">
        <h6>ZFS Cost Buckets (Total €)</h6>
        <div class="row">
          <div class="col-6 col-md-3"><div class="kpi"><small>Storage</small><strong id="zfsStorage">0</strong></div></div>
          <div class="col-6 col-md-3"><div class="kpi"><small>Outbound</small><strong id="zfsOutbound">0</strong></div></div>
          <div class="col-6 col-md-3"><div class="kpi"><small>Returns Logistics</small><strong id="zfsReturns">0</strong></div></div>
          <div class="col-6 col-md-3"><div class="kpi"><small>Return→Partner</small><strong id="zfsR2P">0</strong></div></div>
        </div>
      </div></div>
    </div>

    <p class="subtle mt-3">API: <code>/api/summary?start=YYYY-MM-DD&end=YYYY-MM-DD&country=DE&category=Boots</code></p>
  </div>

<script>
const DATA = {payload};

function euros(n) {{
  return (Math.round(n*100)/100).toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
}}

// Net Revenue by Month
Plotly.newPlot('netByMonth', [{{
  x: DATA.charts.net_revenue_by_month.months,
  y: DATA.charts.net_revenue_by_month.values,
  type: 'bar'
}}], {{
  margin: {{t:10,l:40,r:10,b:40}},
  paper_bgcolor:'#151834',
  plot_bgcolor:'#151834',
  font:{{color:'#e6e8ef'}},
  xaxis:{{tickangle:-30}}
}}, {{displayModeBar:false}});

// Top Return %
const trp = DATA.charts.top_return_pct || [];
Plotly.newPlot('topReturnPct', [{{
  x: trp.map(r => `${{r.ean}} • ${{r.month}}`),
  y: trp.map(r => r.value),
  type: 'bar'
}}], {{
  margin: {{t:10,l:40,r:10,b:120}},
  paper_bgcolor:'#151834',
  plot_bgcolor:'#151834',
  font:{{color:'#e6e8ef'}},
  yaxis:{{title:'%'}}
}}, {{displayModeBar:false}});

// Top Storage Costs
const tsc = DATA.charts.top_storage_costs || [];
Plotly.newPlot('topStorageCosts', [{{
  x: tsc.map(r => `${{r.ean}} • ${{r.month}}`),
  y: tsc.map(r => r.value),
  type: 'bar'
}}], {{
  margin: {{t:10,l:40,r:10,b:120}},
  paper_bgcolor:'#151834',
  plot_bgcolor:'#151834',
  font:{{color:'#e6e8ef'}},
  yaxis:{{title:'€'}}
}}, {{displayModeBar:false}});

// Worst Net Revenue Table
const worst = DATA.tables.worst_net_revenue || [];
document.getElementById('tblWorst').innerHTML = worst.map((r, i) => `
  <tr>
    <td>${{i+1}}</td>
    <td><code>${{r.ean}}</code></td>
    <td>${{r.month}}</td>
    <td class="text-end">${{euros(r.value)}}</td>
  </tr>
`).join('');

// ZFS KPIs
document.getElementById('zfsStorage').innerText = euros((DATA.zfs && DATA.zfs.buckets_eur && DATA.zfs.buckets_eur.storage) || 0);
document.getElementById('zfsOutbound').innerText = euros((DATA.zfs && DATA.zfs.buckets_eur && DATA.zfs.buckets_eur.outbound) || 0);
document.getElementById('zfsReturns').innerText = euros((DATA.zfs && DATA.zfs.buckets_eur && DATA.zfs.buckets_eur.returns) || 0);
document.getElementById('zfsR2P').innerText = euros((DATA.zfs && DATA.zfs.buckets_eur && DATA.zfs.buckets_eur.return_to_partner) || 0);
</script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)

# ========= CLI (optional) =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8006")), reload=False)
