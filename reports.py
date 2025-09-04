import os
import io
import json
import math
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from fastapi.responses import StreamingResponse
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


from reports import (
    non_our_eans, late_returns, payments_recon, vat_avask,
    analytics as analytics_report, forecast_month, brandbook
)
export = APIRouter(prefix="/export", tags=["export"])
@export.get("/non-our-eans")
def export_non_our_eans_route():
    return non_our_eans(conn)

@export.get("/late-returns")
def export_late_returns_route():
    return late_returns(conn)

@export.get("/payments")
def export_payments_route(month: str = Query(..., description="e.g. '7. 2025 July'")):
    try: return payments_recon(conn, month)
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

@export.get("/vat")
def export_vat_route(countries: Optional[str] = Query(None, description="e.g. 'DE,PL,SE'")):
    lst = [c.strip() for c in countries.split(",")] if countries else None
    return vat_avask(conn, lst)

@export.get("/analytics")
def export_analytics_route(start_date: str = Query(...), end_date: str = Query(...)):
    return analytics_report(conn, start_date, end_date)

@export.get("/forecast")
def export_forecast_route(target_month: str = Query(...), weeks: int = 4):
    return forecast_month(conn, target_month, weeks)

@export.get("/brandbook")
def export_brandbook_route(category: Optional[str] = None, kind: str = "good", top_n: int = 60):
    if kind not in {"good","bad"}:
        raise HTTPException(status_code=400, detail="kind must be 'good' or 'bad'")
    return brandbook(conn, category, kind, top_n)
def _to_csv(df: pd.DataFrame, name: str) -> str:
    p = EXPORT_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    return str(p.resolve())

def _to_xlsx(df: pd.DataFrame, name: str, sheet: str = "Sheet1") -> str:
    p = EXPORT_DIR / f"{name}.xlsx"
    with pd.ExcelWriter(p, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet)
    return str(p.resolve())

def _empty_df(msg: str = "No data") -> pd.DataFrame:
    return pd.DataFrame([{"message": msg}])

# ---------- 1) Non-our EANs ----------
def non_our_eans(conn):
    df = pd.read_sql("""
        SELECT s.ean, s.order_number, s.month, s.country
        FROM sales s
        LEFT JOIN ean_base e ON s.ean = e.ean
        WHERE e.ean IS NULL
    """, conn)

    if df.empty:
        df = pd.DataFrame(columns=["ean","order_number","month","country"])

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=non_our_eans.csv"}
    )
# ---------- 2) Late returns ----------
def late_returns(conn) -> Dict[str, str]:
    sales = pd.read_sql("SELECT ean, order_number, order_date, return_date, country FROM sales WHERE type='Return'", conn)
    pol = pd.read_sql("SELECT country, start_date, max_days FROM returns_policy", conn)

    rows = []
    for _, r in sales.iterrows():
        if pd.isna(r["order_date"]) or pd.isna(r["return_date"]):
            continue
        od = pd.to_datetime(r["order_date"], errors="coerce")
        rd = pd.to_datetime(r["return_date"], errors="coerce")
        if pd.isna(od) or pd.isna(rd):
            continue
        p = pol[(pol["country"] == str(r["country"])) & (pd.to_datetime(pol["start_date"], errors="coerce") <= od)]
        maxd = int(p["max_days"].iloc[-1]) if not p.empty else 100
        days = (rd - od).days
        if days > maxd:
            rec = dict(r)
            rec["days_over"] = days - maxd
            rows.append(rec)

    df = pd.DataFrame(rows) if rows else _empty_df("No late returns detected.")
    csvp = _to_csv(df, "late_returns")
    return {"csv": csvp}

# ---------- 3) Payments reconciliation ----------
def payments_recon(conn, month: str) -> Dict[str, str]:
    months = pd.read_sql("SELECT DISTINCT month FROM sales", conn)["month"].tolist()
    if month not in months:
        raise ValueError(f"Unknown month '{month}'. Known: {months}")

    # figure previous month if present
    try:
        idx = months.index(month)
    except ValueError:
        idx = 0
    prev = months[idx - 1] if idx > 0 else month

    sales_curr = pd.read_sql("SELECT order_number, gross_revenue, status FROM sales WHERE month=?", conn, params=(month,))
    sales_prev = pd.read_sql("SELECT order_number, gross_revenue FROM sales WHERE month=?", conn, params=(prev,))
    inv = pd.read_sql("SELECT order_number FROM invoices WHERE month=?", conn, params=(month,))

    all_sales = pd.concat([sales_curr, sales_prev], ignore_index=True)
    unpaid = all_sales[~all_sales["order_number"].isin(inv["order_number"])].copy()
    planned_next = sales_curr[
        (sales_curr.get("status", pd.Series(dtype=str)) == "settled already") &
        (~sales_curr["order_number"].isin(inv["order_number"]))
    ].copy()

    out = unpaid.copy()
    out["planned_next_flag"] = out["order_number"].isin(planned_next["order_number"])
    if out.empty:
        out = _empty_df("Everything invoiced / no unpaid orders.")

    csvp = _to_csv(out, f"payments_unpaid_{month.replace(' ', '_')}")
    return {"csv": csvp}

# ---------- 4) VAT dataset for AVASK ----------
def vat_avask(conn, countries: Optional[List[str]] = None) -> Dict[str, str]:
    sales = pd.read_sql("SELECT ean, order_number, document_date, type, gross_revenue, currency, country FROM sales", conn)
    tbone = pd.read_sql("SELECT order_number, product_name, street, city FROM tbone_desadv", conn)
    vat = pd.read_sql("SELECT country, vat_percent FROM vat", conn)

    if countries:
        sales = sales[sales["country"].isin(countries)]

    merged = sales.merge(tbone, on="order_number", how="left").merge(vat, on="country", how="left")

    rows = []
    for _, r in merged.iterrows():
        try:
            gross_rev = float(r.get("gross_revenue", 0.0)) * -1
        except Exception:
            gross_rev = 0.0
        rows.append({
            "DATE": r.get("document_date"),
            "ORDER NUMBER/REF/ID": r.get("order_number"),
            "TRANSACTION TYPE": r.get("type"),
            "SKU": r.get("ean"),
            "DESCRIPTION": r.get("product_name"),
            "QUANTITY": 1,
            "ORDER TOTAL": gross_rev,
            "CURRENCY": r.get("currency"),
            "SALE DEPARTURE COUNTRY/FULFILMENT COUNTRY": r.get("country"),
            "SALE ARRIVAL COUNTRY/DESTINATION COUNTRY": r.get("country"),
            "TAX RATE": (float(r.get("vat_percent", 0.0)) / 100.0) if pd.notnull(r.get("vat_percent")) else 0.0,
            "BUYER VAT NUMBER": "PRIVATE INDIVIDUAL",
            "SHIPPING ADDRESS 1": r.get("street"),
            "SHIPPING CITY": r.get("city"),
            "SHIPPING COUNTRY": r.get("country"),
            "TAX COLLECTION RESPONSIBILITY": "SELLER",
            "MARKETPLACE": "ZALANDO",
        })
    df = pd.DataFrame(rows) if rows else _empty_df("No sales for the filter.")
    xlsx = _to_xlsx(df, "vat_avask", sheet="VAT")
    return {"xlsx": xlsx}

# ---------- 5) Analytics (export, not LLM) ----------
def analytics(conn, start_date: str, end_date: str) -> Dict[str, str]:
    sales = pd.read_sql(
        "SELECT ean, type, gross_revenue, country, month, document_date FROM sales "
        "WHERE document_date BETWEEN ? AND ?",
        conn, params=(start_date, end_date)
    )
    ean_base = pd.read_sql("SELECT ean, category, cost FROM ean_base", conn)
    zfs = pd.read_sql("SELECT ean, month, event_type, amount FROM zfs", conn)

    merged = sales.merge(ean_base, on="ean", how="left")

    sale_mask = merged["type"] == "Sale"
    ret_mask = merged["type"] == "Return"

    sales_count = merged[sale_mask].groupby(["ean", "month"]).size()
    return_count = merged[ret_mask].groupby(["ean", "month"]).size()
    return_pct = (return_count / sales_count * 100).fillna(0)

    sales_rev = (merged[sale_mask].groupby(["ean", "month"])["gross_revenue"].sum() * -1).fillna(0)
    return_rev = (merged[ret_mask].groupby(["ean", "month"])["gross_revenue"].sum() * -1).fillna(0)
    net_rev = (sales_rev - return_rev).fillna(0)

    storage = (
        zfs[zfs["event_type"].astype(str).str.contains("Storage", na=False)]
        .groupby(["ean", "month"])["amount"].sum() * -1
    ).fillna(0)

    df_ret = return_pct.reset_index()
    df_ret.columns = ["ean", "month", "return_pct"]

    df_rev = net_rev.reset_index()
    df_rev.columns = ["ean", "month", "net_revenue"]

    df_store = storage.reset_index()
    df_store.columns = ["ean", "month", "storage_cost"]

    # merge to one sheet
    out = df_ret.merge(df_rev, on=["ean","month"], how="outer").merge(df_store, on=["ean","month"], how="outer")
    if out.empty:
        out = _empty_df("No analytics rows in period.")

    xlsx = _to_xlsx(out, f"analytics_{start_date}_to_{end_date}", sheet="Analytics")
    return {"xlsx": xlsx}

# ---------- 6) Simple forecast (toy) ----------
def forecast_month(conn, target_month: str, weeks: int = 4) -> Dict[str, str]:
    # naive approach: average weekly net revenue from prior months and project
    sales = pd.read_sql("SELECT month, type, gross_revenue FROM sales", conn)
    if sales.empty:
        df = _empty_df("No sales to forecast.")
        return {"csv": _to_csv(df, f"forecast_{target_month.replace(' ', '_')}")}

    # compute monthly net revenue
    sales["gr"] = pd.to_numeric(sales["gross_revenue"], errors="coerce").fillna(0.0)
    month_net = sales.groupby(["month","type"])["gr"].sum().unstack(fill_value=0)
    # sales are negative in source, flip: Sale * -1, Return * -1
    month_net["net"] = month_net.get("Sale", 0) * -1 - (month_net.get("Return", 0) * -1)

    hist = month_net["net"].reset_index().rename(columns={"net":"net_revenue"})
    hist = hist[hist["month"] != target_month]

    avg_weekly = (hist["net_revenue"].mean() / 4.0) if not hist.empty else 0.0
    rows = [{"target_month": target_month, "week": i+1, "forecast_net_revenue": round(avg_weekly, 2)} for i in range(weeks)]
    df = pd.DataFrame(rows)
    return {"csv": _to_csv(df, f"forecast_{target_month.replace(' ', '_')}")}

# ---------- 7) Brandbook (good/bad SKUs) ----------
def brandbook(conn, category: Optional[str], kind: str = "good", top_n: int = 60) -> Dict[str, str]:
    # Build basic KPIs per EAN: sales count, return %, net revenue
    sales = pd.read_sql("SELECT ean, type, gross_revenue, country, month FROM sales", conn)
    base = pd.read_sql("SELECT ean, article, color, size, category, url FROM ean_base", conn)

    merged = sales.merge(base, on="ean", how="left")
    if category:
        merged = merged[merged["category"] == category]

    sale_mask = merged["type"] == "Sale"
    ret_mask = merged["type"] == "Return"

    sc = merged[sale_mask].groupby("ean").size().rename("sales_cnt")
    rc = merged[ret_mask].groupby("ean").size().rename("returns_cnt")
    rpct = ((rc / sc) * 100).fillna(0).rename("return_pct")

    s_rev = (merged[sale_mask].groupby("ean")["gross_revenue"].sum() * -1).rename("sales_rev").fillna(0)
    r_rev = (merged[ret_mask].groupby("ean")["gross_revenue"].sum() * -1).rename("return_rev").fillna(0)
    net_rev = (s_rev - r_rev).rename("net_revenue").fillna(0)

    out = pd.concat([sc, rpct, net_rev], axis=1).fillna(0).reset_index()
    out = out.merge(base, on="ean", how="left")

    if out.empty:
        return {"csv": _to_csv(_empty_df("No data for brandbook."), "brandbook_empty")}

    if kind == "good":
        # prioritize high net revenue, low return %
        out = out.sort_values(by=["net_revenue", "return_pct", "sales_cnt"], ascending=[False, True, False]).head(top_n)
    else:
        # bad: high return %, low net revenue
        out = out.sort_values(by=["return_pct", "net_revenue"], ascending=[False, True]).head(top_n)

    csvp = _to_csv(out, f"brandbook_{kind}{'_'+category if category else ''}")
    # also a quick HTML for easy review
    htmlp = EXPORT_DIR / f"brandbook_{kind}{'_'+category if category else ''}.html"
    out.to_html(htmlp, index=False)
    return {"csv": csvp, "html": str(htmlp.resolve())}
