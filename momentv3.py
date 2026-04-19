"""
ValuationB3 — Motor de DCF Fundamentalista para o Mercado Brasileiro (B3)
=========================================================================
Desenvolvido para Streamlit Cloud | v1.0
Arquitetura: Modular, PEP-8, @st.cache_data, st.session_state

Pilares:
  1. Diagnóstico do Cenário Atual   (Aba 1)
  2. Premissas de Projeção          (Aba 2)
  3. Motor DCF (FCFF / FCFE)        (Aba 3)
  4. Output Financeiro e Decisão    (Aba 4)
  5. Análise de Sensibilidade       (Aba 5)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.optimize import brentq

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

SGS_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados/ultimos/{}?formato=json"
SGS_IPCA = 433        # IPCA mensal
SGS_CDI = 4391        # CDI acumulado 12 meses
SGS_SELIC = 432       # Meta SELIC anualizada

FALLBACK_IPCA = 0.045
FALLBACK_CDI = 0.107
FALLBACK_SELIC = 0.105
FALLBACK_PIB_NOMINAL = 0.085   # IPCA + crescimento real estimado
PIB_REAL_ESTIMATE = 0.020      # crescimento real estimado do PIB

CHART_BG = "#0F172A"
CHART_PAPER = "#0F172A"
CHART_FONT = "#E2E8F0"

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ValuationB3 | DCF Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
  [data-testid="stSidebar"]        { background:#0F172A; }
  .stMetricValue                   { font-size:1.35rem !important; font-weight:700; }
  .stMetricLabel                   { font-size:.78rem !important; color:#94A3B8; }
  .stTabs [data-baseweb="tab"]     { background:#1E293B; border-radius:8px;
                                     padding:8px 18px; color:#94A3B8; }
  .stTabs [aria-selected="true"]   { background:#3B82F6 !important; color:#fff !important; }
  div[data-testid="stExpander"]    { border:1px solid #1E293B; border-radius:8px; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

_STATE_DEFAULTS: Dict[str, Any] = {
    "ticker_loaded": False,
    "ticker_yf": "",
    "quarterly_df": pd.DataFrame(),
    "macro": {},
    "current_price": np.nan,
    "shares_outstanding": np.nan,
    "net_debt_ss": np.nan,
    "last_ebitda": np.nan,
    "last_ebit": np.nan,
    "last_revenue": np.nan,
    "last_da": np.nan,
    "last_capex": np.nan,
    "last_nwc": 0.0,
    "dcf_results": {},
    "premissas_inputs": {},
}


def _init_state() -> None:
    """Inicializa session_state com valores padrão se ausentes."""
    for key, val in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_state() -> None:
    """Reinicia estado ao trocar de ticker."""
    for key, val in _STATE_DEFAULTS.items():
        st.session_state[key] = val


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1 — DATA FETCHING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _sgs(series_id: int, n: int = 12) -> pd.DataFrame:
    """Consome a API do SGS do Banco Central do Brasil."""
    try:
        resp = requests.get(SGS_BASE.format(series_id, n), timeout=12)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df
    except Exception as exc:  # noqa: BLE001
        st.toast(f"⚠️ SGS série {series_id}: {exc}", icon="⚠️")
        return pd.DataFrame(columns=["data", "valor"])


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def fetch_macro() -> Dict[str, float]:
    """
    Retorna dict com indicadores macroeconômicos atuais via SGS/BCB.
    Fallbacks automáticos se a API falhar.
    """
    macro: Dict[str, float] = {}

    # IPCA — acumulado 12 meses
    df_ipca = _sgs(SGS_IPCA, 12)
    if not df_ipca.empty and df_ipca["valor"].notna().any():
        macro["ipca_12m"] = float((1 + df_ipca["valor"].dropna() / 100).prod() - 1)
    else:
        macro["ipca_12m"] = FALLBACK_IPCA

    # CDI anual
    df_cdi = _sgs(SGS_CDI, 1)
    if not df_cdi.empty and df_cdi["valor"].notna().any():
        macro["cdi_anual"] = float(df_cdi["valor"].dropna().iloc[-1] / 100)
    else:
        macro["cdi_anual"] = FALLBACK_CDI

    # SELIC meta
    df_selic = _sgs(SGS_SELIC, 1)
    if not df_selic.empty and df_selic["valor"].notna().any():
        macro["selic"] = float(df_selic["valor"].dropna().iloc[-1] / 100)
    else:
        macro["selic"] = FALLBACK_SELIC

    # PIB nominal estimado = IPCA + crescimento real
    macro["pib_nominal"] = macro["ipca_12m"] + PIB_REAL_ESTIMATE

    return macro


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def fetch_info(ticker_yf: str) -> Dict[str, Any]:
    """Busca metadados do ativo via yfinance."""
    try:
        return yf.Ticker(ticker_yf).info or {}
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao buscar info de {ticker_yf}: {exc}")
        return {}


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def fetch_quarterly(ticker_yf: str) -> pd.DataFrame:
    """
    Extrai até 12 trimestres de dados financeiros consolidados:
    Revenue, EBIT, EBITDA, D&A, Capex, NetDebt, NWC.

    Fontes: quarterly_financials, quarterly_balance_sheet, quarterly_cashflow
    via yfinance.  Retorna DataFrame ordenado do mais antigo ao mais recente.
    """
    def _get(df: pd.DataFrame, keys: List[str], col: Any) -> float:
        """Busca segura em DataFrame transposto do yfinance."""
        for k in keys:
            try:
                if k in df.index and col in df.columns:
                    v = df.loc[k, col]
                    if pd.notna(v):
                        return float(v)
            except Exception:
                continue
        return np.nan

    try:
        t = yf.Ticker(ticker_yf)
        fin = t.quarterly_financials
        bs = t.quarterly_balance_sheet
        cf = t.quarterly_cashflow

        if fin is None or fin.empty:
            return pd.DataFrame()

        dates = fin.columns.tolist()[:12]
        rows: List[Dict[str, Any]] = []

        for d in dates:
            row: Dict[str, Any] = {"Date": d}

            # ── DRE ──────────────────────────────────────────────────────────
            row["Revenue"] = _get(fin, [
                "Total Revenue", "Revenue", "Net Revenue",
            ], d)
            row["EBIT"] = _get(fin, [
                "EBIT", "Operating Income",
            ], d)
            row["Net_Income"] = _get(fin, [
                "Net Income", "Net Income Common Stockholders",
            ], d)

            # ── FLUXO DE CAIXA ────────────────────────────────────────────
            row["DA"] = _get(cf, [
                "Depreciation And Amortization",
                "Depreciation Amortization Depletion",
                "Depreciation",
            ], d)
            if np.isnan(row["DA"]):
                row["DA"] = _get(fin, [
                    "Reconciled Depreciation", "Depreciation",
                ], d)

            raw_capex = _get(cf, [
                "Capital Expenditure",
                "Purchase Of PPE",
                "Capital Expenditures",
            ], d)
            row["Capex"] = abs(raw_capex) if not np.isnan(raw_capex) else np.nan

            # ── EBITDA ───────────────────────────────────────────────────────
            if not np.isnan(row["EBIT"]) and not np.isnan(row["DA"]):
                row["EBITDA"] = row["EBIT"] + abs(row["DA"])
            else:
                row["EBITDA"] = _get(fin, ["EBITDA", "Normalized EBITDA"], d)

            # ── BALANÇO ───────────────────────────────────────────────────
            total_debt = _get(bs, [
                "Total Debt",
                "Long Term Debt And Capital Lease Obligation",
            ], d)
            st_debt = _get(bs, [
                "Current Debt",
                "Short Long Term Debt",
                "Current Portion Of Long Term Debt",
            ], d)
            cash = _get(bs, [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
            ], d)
            if np.isnan(cash):
                cash = 0.0
            if np.isnan(total_debt):
                total_debt = (st_debt if not np.isnan(st_debt) else 0.0)

            row["Cash"] = cash
            row["GrossDebt"] = total_debt
            row["NetDebt"] = total_debt - cash

            # ── NWC (Capital de Giro Líquido) ─────────────────────────────
            curr_a = _get(bs, ["Current Assets", "Total Current Assets"], d)
            curr_l = _get(bs, ["Current Liabilities", "Total Current Liabilities"], d)
            st_d_safe = st_debt if not np.isnan(st_debt) else 0.0
            if not np.isnan(curr_a) and not np.isnan(curr_l):
                row["NWC"] = (curr_a - cash) - (curr_l - st_d_safe)
            else:
                row["NWC"] = np.nan

            rows.append(row)

        df = (
            pd.DataFrame(rows)
            .sort_values("Date")
            .reset_index(drop=True)
        )
        df["Delta_NWC"] = df["NWC"].diff().fillna(0.0)
        df["Leverage"] = df["NetDebt"] / df["EBITDA"].replace(0, np.nan)
        return df

    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao extrair trimestres: {exc}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2 — MOTOR DCF
# ──────────────────────────────────────────────────────────────────────────────

def calc_fcff(
    ebit: float,
    tax_rate: float,
    da: float,
    capex: float,
    delta_nwc: float,
) -> float:
    """FCFF = EBIT × (1 – t) + D&A – CapEx – ΔNWC"""
    return ebit * (1 - tax_rate) + da - capex - delta_nwc


def calc_fcfe(
    net_income: float,
    da: float,
    capex: float,
    delta_nwc: float,
    net_borrowing: float = 0.0,
) -> float:
    """FCFE = Lucro Líquido + D&A – CapEx – ΔNWC + Net Borrowing"""
    return net_income + da - capex - delta_nwc + net_borrowing


def gordon_tv(terminal_fcf: float, wacc: float, g: float) -> float:
    """TV = FCF_{t+1} / (WACC – g)  —  Fórmula de Gordon."""
    if wacc <= g:
        raise ValueError(
            f"WACC ({wacc:.2%}) deve ser maior que g ({g:.2%}) para calcular o Valor Terminal."
        )
    return terminal_fcf / (wacc - g)


def project_fcfs(
    base_revenue: float,
    base_da: float,
    base_capex: float,
    base_nwc: float,
    tax_rate: float,
    revenue_growth_rates: List[float],
    ebitda_margin_path: List[float],
    capex_pct_revenue: float,
    da_growth_pa: float,
    nwc_pct_revenue: float,
    is_fcfe: bool = False,
    net_income_margin_path: Optional[List[float]] = None,
    net_borrowing: float = 0.0,
) -> List[float]:
    """
    Projeta FCFs (FCFF ou FCFE) para os anos explícitos.

    Parâmetros
    ----------
    base_revenue          : Receita base anualizada (R$)
    base_da               : D&A base anualizado (R$)
    base_capex            : Capex base anualizado (R$) — não utilizado diretamente
    base_nwc              : NWC atual (R$) — ponto de partida do capital de giro
    tax_rate              : Alíquota efetiva de IR+CSLL
    revenue_growth_rates  : Taxa de crescimento da receita por ano [list]
    ebitda_margin_path    : Margem EBITDA por ano [list]
    capex_pct_revenue     : Capex como % da receita (constante)
    da_growth_pa          : Crescimento anual do D&A (constante)
    nwc_pct_revenue       : NWC como % da receita (constante)
    is_fcfe               : True → calcula FCFE; False → FCFF
    net_income_margin_path: Margem líquida por ano (apenas FCFE)
    net_borrowing         : Captação líquida anual (apenas FCFE)
    """
    fcf_list: List[float] = []
    revenue = base_revenue
    nwc_prev = base_nwc

    for i, g_rev in enumerate(revenue_growth_rates):
        revenue *= (1 + g_rev)
        ebitda = revenue * ebitda_margin_path[i]
        da = base_da * ((1 + da_growth_pa) ** (i + 1))
        ebit = ebitda - da
        capex = revenue * capex_pct_revenue
        nwc = revenue * nwc_pct_revenue
        delta_nwc = nwc - nwc_prev
        nwc_prev = nwc

        if is_fcfe and net_income_margin_path:
            net_income = revenue * net_income_margin_path[i]
            fcf = calc_fcfe(net_income, da, capex, delta_nwc, net_borrowing)
        else:
            fcf = calc_fcff(ebit, tax_rate, da, capex, delta_nwc)

        fcf_list.append(fcf)

    return fcf_list


def dcf_engine(
    fcf_projections: List[float],
    wacc: float,
    g: float,
    net_debt: float,
    shares: float,
    is_fcfe: bool = False,
) -> Dict[str, float]:
    """
    Motor DCF completo.

    Retorna
    -------
    dict com: PV_FCF, PV_TV, TV, Enterprise_Value, Equity_Value, Fair_Price
    """
    n = len(fcf_projections)

    # Valor presente dos fluxos explícitos
    pv_fcf = sum(
        cf / (1 + wacc) ** (t + 1)
        for t, cf in enumerate(fcf_projections)
    )

    # Valor Terminal (Gordon Growth)
    terminal_fcf = fcf_projections[-1] * (1 + g) if fcf_projections else 0.0
    tv = gordon_tv(terminal_fcf, wacc, g)
    pv_tv = tv / (1 + wacc) ** n

    if is_fcfe:
        equity_value = pv_fcf + pv_tv
        enterprise_value = equity_value + net_debt
    else:
        enterprise_value = pv_fcf + pv_tv
        equity_value = enterprise_value - net_debt

    fair_price = equity_value / shares if (shares and shares > 0) else np.nan

    return {
        "PV_FCF": pv_fcf,
        "PV_TV": pv_tv,
        "TV": tv,
        "Enterprise_Value": enterprise_value,
        "Equity_Value": equity_value,
        "Fair_Price": fair_price,
    }


def calc_irr(
    current_price: float,
    fcf_projections: List[float],
    g: float,
    net_debt: float,
    shares: float,
    is_fcfe: bool = False,
) -> float:
    """
    TIR implícita: taxa que iguala o preço atual ao Equity Value
    calculado pelo DCF via método de Brent (scipy).
    """
    n = len(fcf_projections)

    def _equity_at_rate(r: float) -> float:
        pv = sum(cf / (1 + r) ** (t + 1) for t, cf in enumerate(fcf_projections))
        tv = fcf_projections[-1] * (1 + g) / (r - g) if r > g else 0.0
        pv_tv = tv / (1 + r) ** n
        if is_fcfe:
            eq_val = pv + pv_tv
        else:
            eq_val = pv + pv_tv - net_debt
        return eq_val / shares if shares > 0 else np.nan

    def _obj(r: float) -> float:
        return _equity_at_rate(r) - current_price

    try:
        return brentq(_obj, g + 1e-4, 0.80, xtol=1e-6, maxiter=300)
    except Exception:
        return np.nan


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3 — ANÁLISE DE SENSIBILIDADE
# ──────────────────────────────────────────────────────────────────────────────

def build_sensitivity(
    base_revenue: float,
    base_da: float,
    base_nwc: float,
    tax_rate: float,
    revenue_growth_rates: List[float],
    capex_pct: float,
    da_growth: float,
    nwc_pct: float,
    net_debt: float,
    shares: float,
    wacc_range: np.ndarray,
    y_range: np.ndarray,
    y_axis: str,               # "ebitda_margin" | "g"
    fixed_g: float,
    fixed_margin: float,
    current_price: float,
    is_fcfe: bool = False,
    base_nim: float = 0.10,
) -> pd.DataFrame:
    """
    Gera matriz de sensibilidade: WACC (eixo X) × Margem EBITDA|g (eixo Y).
    Célula = Margem de Segurança se current_price disponível, senão Preço Justo.
    """
    rows: List[List[float]] = []

    for y_val in y_range:
        row_vals: List[float] = []
        for w in wacc_range:
            try:
                if y_axis == "ebitda_margin":
                    margin_path = [float(y_val)] * len(revenue_growth_rates)
                    g = fixed_g
                else:
                    margin_path = [fixed_margin] * len(revenue_growth_rates)
                    g = float(y_val)

                if w <= g + 1e-4:
                    row_vals.append(np.nan)
                    continue

                nim_path = [base_nim] * len(revenue_growth_rates) if is_fcfe else None

                fcf_proj = project_fcfs(
                    base_revenue=base_revenue,
                    base_da=base_da,
                    base_capex=0.0,
                    base_nwc=base_nwc,
                    tax_rate=tax_rate,
                    revenue_growth_rates=revenue_growth_rates,
                    ebitda_margin_path=margin_path,
                    capex_pct_revenue=capex_pct,
                    da_growth_pa=da_growth,
                    nwc_pct_revenue=nwc_pct,
                    is_fcfe=is_fcfe,
                    net_income_margin_path=nim_path,
                )

                res = dcf_engine(fcf_proj, w, g, net_debt, shares, is_fcfe)
                fp = res["Fair_Price"]

                if (
                    not np.isnan(current_price)
                    and current_price > 0
                    and not np.isnan(fp)
                ):
                    cell = (fp - current_price) / current_price
                else:
                    cell = fp

                row_vals.append(round(cell, 4) if not np.isnan(cell) else np.nan)

            except Exception:
                row_vals.append(np.nan)

        rows.append(row_vals)

    return pd.DataFrame(
        rows,
        index=np.round(y_range, 4),
        columns=np.round(wacc_range, 4),
    )


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4 — GRÁFICOS PLOTLY
# ──────────────────────────────────────────────────────────────────────────────

def _dark_layout(**kwargs) -> Dict[str, Any]:
    """Layout base dark theme para todos os gráficos."""
    base = dict(
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_PAPER,
        font=dict(color=CHART_FONT, size=12),
        margin=dict(t=60, b=40, l=60, r=40),
    )
    base.update(kwargs)
    return base


def chart_leverage(df: pd.DataFrame) -> go.Figure:
    """Barras de Alavancagem (ND/EBITDA) com limiares de atenção."""
    fig = go.Figure()
    if df.empty or "Leverage" not in df.columns:
        return fig

    df_p = df.dropna(subset=["Leverage"]).copy()
    df_p["DateStr"] = df_p["Date"].dt.strftime("%Y-%m")
    colors = df_p["Leverage"].apply(
        lambda x: "#EF4444" if x > 3.5 else ("#F59E0B" if x > 2.5 else "#22C55E")
    )

    fig.add_trace(go.Bar(
        x=df_p["DateStr"],
        y=df_p["Leverage"],
        marker_color=colors,
        text=df_p["Leverage"].round(2),
        textposition="outside",
        name="ND/EBITDA",
    ))
    fig.add_hline(y=2.5, line_dash="dash", line_color="#F59E0B",
                  annotation_text="Atenção (2,5x)", annotation_position="top left")
    fig.add_hline(y=3.5, line_dash="dash", line_color="#EF4444",
                  annotation_text="Crítico (3,5x)", annotation_position="top left")

    fig.update_layout(
        title="📊 Evolução da Alavancagem — Dívida Líquida / EBITDA",
        xaxis_title="Trimestre",
        yaxis_title="ND/EBITDA (x)",
        showlegend=False,
        **_dark_layout(),
    )
    return fig


def chart_capex_da(df: pd.DataFrame) -> go.Figure:
    """Barras agrupadas Capex vs D&A + linha Capex/D&A."""
    fig = go.Figure()
    if df.empty or "Capex" not in df.columns:
        return fig

    df_p = df.dropna(subset=["Capex", "DA"], how="all").copy()
    df_p["DateStr"] = df_p["Date"].dt.strftime("%Y-%m")
    cap_bi = df_p["Capex"] / 1e9
    da_bi = df_p["DA"].abs() / 1e9

    fig.add_trace(go.Bar(x=df_p["DateStr"], y=cap_bi,
                         name="Capex", marker_color="#3B82F6", opacity=0.85))
    fig.add_trace(go.Bar(x=df_p["DateStr"], y=da_bi,
                         name="D&A", marker_color="#8B5CF6", opacity=0.85))

    ratio = (cap_bi / da_bi.replace(0, np.nan)).round(2)
    fig.add_trace(go.Scatter(
        x=df_p["DateStr"], y=ratio,
        name="Capex/D&A (x)", yaxis="y2",
        line=dict(color="#F59E0B", width=2, dash="dot"),
        mode="lines+markers",
    ))

    fig.update_layout(
        title="🔧 Capex Executado vs D&A (R$ Bilhões)",
        barmode="group",
        xaxis_title="Trimestre",
        yaxis_title="R$ Bilhões",
        yaxis2=dict(title="Capex/D&A (x)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", y=1.12),
        **_dark_layout(),
    )
    return fig


def chart_waterfall(result: Dict[str, float], net_debt: float, label: str) -> go.Figure:
    """Waterfall: PV FCFs → PV TV → EV → (−) ND → Equity."""
    pv_fcf = result.get("PV_FCF", 0) / 1e9
    pv_tv = result.get("PV_TV", 0) / 1e9
    ev = result.get("Enterprise_Value", 0) / 1e9
    nd = net_debt / 1e9
    eq = result.get("Equity_Value", 0) / 1e9

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total", "relative", "total"],
        x=["PV FCFs Explícitos", "PV Valor Terminal",
           "Enterprise Value", "(−) Dívida Líquida", "Equity Value"],
        y=[pv_fcf, pv_tv, 0, -nd, 0],
        connector={"line": {"color": "#334155"}},
        increasing={"marker": {"color": "#22C55E"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#3B82F6"}},
        text=[f"R${v:.1f}Bi" for v in [pv_fcf, pv_tv, ev, -nd, eq]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"🌊 Waterfall DCF — {label}",
        yaxis_title="R$ Bilhões",
        **_dark_layout(),
    )
    return fig


def chart_fcf_projection(
    fcf_cons: List[float],
    fcf_mod: List[float],
    years: int,
) -> go.Figure:
    """Barras agrupadas da projeção de FCF por cenário."""
    xlabels = [f"Ano {i + 1}" for i in range(years)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[v / 1e9 for v in fcf_cons],
        name="🐢 Conservador",
        marker_color="#EF4444",
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=xlabels,
        y=[v / 1e9 for v in fcf_mod],
        name="🚀 Moderado",
        marker_color="#22C55E",
        opacity=0.85,
    ))
    fig.update_layout(
        title="📈 FCF Projetado por Cenário (R$ Bilhões)",
        barmode="group",
        xaxis_title="Ano de Projeção",
        yaxis_title="FCF (R$ Bi)",
        legend=dict(orientation="h", y=1.12),
        **_dark_layout(),
    )
    return fig


def chart_heatmap(
    df_sens: pd.DataFrame,
    y_label: str,
    current_price: float,
) -> go.Figure:
    """Heatmap de Margem de Segurança: WACC (X) × Variável Y."""
    is_mos = not np.isnan(current_price) and current_price > 0
    z = df_sens.values.astype(float)

    colorscale = [
        [0.00, "#7F1D1D"],
        [0.35, "#DC2626"],
        [0.48, "#F59E0B"],
        [0.52, "#FAFAFA"],
        [0.65, "#22C55E"],
        [1.00, "#14532D"],
    ] if is_mos else "RdYlGn"

    if is_mos:
        text_m = [[f"{v:.1%}" if not np.isnan(v) else "N/D" for v in row] for row in z]
        cb_title = "Margem de Seg."
        zmid = 0.0
    else:
        text_m = [[f"R${v:.2f}" if not np.isnan(v) else "N/D" for v in row] for row in z]
        cb_title = "Preço Justo (R$)"
        zmid = None

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{v:.1%}" for v in df_sens.columns],
        y=[f"{v:.1%}" for v in df_sens.index],
        colorscale=colorscale,
        zmid=zmid,
        text=text_m,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title=cb_title),
        hoverongaps=False,
    ))

    if is_mos:
        fig.add_annotation(
            text="🟢 Verde = Margem de Segurança Positiva  |  🔴 Vermelho = Ativo Sobreavaliado",
            xref="paper", yref="paper",
            x=0.5, y=-0.14,
            showarrow=False,
            font=dict(color="#94A3B8", size=11),
        )

    fig.update_layout(
        title=f"🔥 Sensibilidade: WACC × {y_label}",
        xaxis_title="WACC",
        yaxis_title=y_label,
        height=500,
        **_dark_layout(),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5 — SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict[str, Any]:
    """Sidebar: seleção de ticker e tipo de empresa."""
    with st.sidebar:
        st.markdown("## 📊 ValuationB3")
        st.markdown("*DCF Engine para o Mercado Brasileiro*")
        st.divider()

        st.markdown("### 🎯 Ativo")
        ticker_raw = st.text_input(
            "Ticker (sem .SA)",
            value="WEGE3",
            help="Ex: PETR4, VALE3, ITUB4, RADL3, RENT3",
        ).upper().strip()

        is_financial = st.toggle(
            "🏦 Instituição Financeira",
            value=False,
            help="Bancos e seguradoras usam FCFE. Empresas reais usam FCFF.",
        )

        if is_financial:
            st.info("**Modo FCFE** ativado — adequado para bancos, seguradoras e fintechs.")
        else:
            st.info("**Modo FCFF** ativado — adequado para empresas do setor real.")

        st.divider()
        load_btn = st.button(
            "🔄 Carregar Dados do Ativo",
            type="primary",
            use_container_width=True,
        )

        st.divider()
        st.caption(
            "Fontes: yfinance, Banco Central do Brasil (SGS).\n"
            "Os dados trimestrais cobrem até 12 trimestres históricos."
        )

    return {
        "ticker_raw": ticker_raw,
        "ticker_yf": f"{ticker_raw}.SA",
        "is_financial": is_financial,
        "load_btn": load_btn,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6 — ABA 1: DIAGNÓSTICO
# ──────────────────────────────────────────────────────────────────────────────

def render_diagnostico() -> None:
    """Aba 1 — Diagnóstico do Cenário Atual."""
    st.header("🔬 Diagnóstico do Cenário Atual")
    ticker_yf: str = st.session_state["ticker_yf"]

    # ── Macroeconômico (BCB/SGS) ──────────────────────────────────────────
    with st.spinner("Consultando Banco Central (SGS/BCB)..."):
        macro = fetch_macro()
        st.session_state["macro"] = macro

    st.subheader("🏛️ Indicadores Macroeconômicos (BCB)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📈 IPCA 12m", f"{macro['ipca_12m']:.2%}")
    m2.metric("💰 CDI Anual", f"{macro['cdi_anual']:.2%}")
    m3.metric("🏛️ Meta SELIC", f"{macro['selic']:.2%}")
    m4.metric("🌱 PIB Nominal Est.", f"{macro['pib_nominal']:.2%}")

    st.divider()

    # ── Dados Trimestrais ─────────────────────────────────────────────────
    st.subheader(f"📋 Histórico Trimestral — {ticker_yf}")

    df_q: pd.DataFrame = st.session_state["quarterly_df"]

    # Fallback: upload CSV do RI
    st.markdown("##### 📂 Fallback — Upload de Dados do RI (opcional)")
    uploaded = st.file_uploader(
        "CSV com colunas: Date, Revenue, EBIT, EBITDA, DA, Capex, NetDebt, NWC",
        type=["csv"],
        key="ri_upload",
    )
    if uploaded is not None:
        try:
            df_ri = pd.read_csv(uploaded)
            df_ri["Date"] = pd.to_datetime(df_ri["Date"])
            df_ri = df_ri.sort_values("Date").reset_index(drop=True)
            df_ri["Delta_NWC"] = df_ri["NWC"].diff().fillna(0.0)
            df_ri["Leverage"] = df_ri["NetDebt"] / df_ri["EBITDA"].replace(0, np.nan)
            df_q = df_ri
            st.session_state["quarterly_df"] = df_q
            st.success("✅ Dados do RI carregados com sucesso!")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erro ao processar arquivo: {exc}")

    if df_q.empty:
        st.warning(
            "⚠️ Dados trimestrais não disponíveis. "
            "Faça upload via RI ou verifique se o ticker possui histórico no Yahoo Finance."
        )
        return

    # Persiste últimos valores para as abas seguintes
    last = df_q.iloc[-1]

    def _safe(v: Any) -> float:
        return float(v) if pd.notna(v) else np.nan

    st.session_state["last_ebitda"] = _safe(last.get("EBITDA"))
    st.session_state["last_ebit"] = _safe(last.get("EBIT"))
    st.session_state["last_revenue"] = _safe(last.get("Revenue"))
    st.session_state["last_da"] = abs(_safe(last.get("DA")) or 0)
    st.session_state["last_capex"] = abs(_safe(last.get("Capex")) or 0)
    st.session_state["net_debt_ss"] = _safe(last.get("NetDebt", 0))
    st.session_state["last_nwc"] = _safe(last.get("NWC")) or 0.0

    # ── LTM (Last Twelve Months) ──────────────────────────────────────────
    def _ltm_sum(col: str) -> float:
        if col in df_q.columns:
            return df_q[col].tail(4).sum()
        return np.nan

    st.markdown("##### 📊 LTM — Últimos 12 Meses (soma dos 4 últimos trimestres)")
    l1, l2, l3, l4, l5 = st.columns(5)
    ltm_rev = _ltm_sum("Revenue")
    ltm_ebitda = _ltm_sum("EBITDA")
    ltm_capex = _ltm_sum("Capex")
    ltm_da = abs(_ltm_sum("DA"))
    nd_cur = st.session_state["net_debt_ss"]

    def _bi(v: float) -> str:
        return f"R$ {v / 1e9:.2f} Bi" if not np.isnan(v) else "N/D"

    l1.metric("Receita LTM", _bi(ltm_rev))
    l2.metric("EBITDA LTM", _bi(ltm_ebitda))
    l3.metric("Capex LTM", _bi(ltm_capex))
    l4.metric("D&A LTM", _bi(ltm_da))
    l5.metric(
        "Dívida Líquida",
        _bi(nd_cur),
        delta="Positiva (Endividada)" if not np.isnan(nd_cur) and nd_cur > 0 else "Caixa Líquido",
        delta_color="inverse" if not np.isnan(nd_cur) and nd_cur > 0 else "normal",
    )

    # ── Gráficos ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_leverage(df_q), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_capex_da(df_q), use_container_width=True)

    # ── Tabela detalhada ──────────────────────────────────────────────────
    with st.expander("📋 Ver tabela trimestral completa"):
        cols_show = [c for c in
                     ["Date", "Revenue", "EBITDA", "EBIT", "DA",
                      "Capex", "NetDebt", "Leverage", "NWC"]
                     if c in df_q.columns]
        df_disp = df_q[cols_show].copy()
        for col in ["Revenue", "EBITDA", "EBIT", "DA", "Capex", "NetDebt", "NWC"]:
            if col in df_disp.columns:
                df_disp[col] = df_disp[col] / 1e9
        fmt = {
            "Revenue": "R$ {:.2f}Bi", "EBITDA": "R$ {:.2f}Bi",
            "EBIT": "R$ {:.2f}Bi", "DA": "R$ {:.2f}Bi",
            "Capex": "R$ {:.2f}Bi", "NetDebt": "R$ {:.2f}Bi",
            "NWC": "R$ {:.2f}Bi", "Leverage": "{:.2f}x",
        }
        st.dataframe(
            df_disp.style.format({k: v for k, v in fmt.items() if k in df_disp.columns}),
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 7 — ABA 2: PREMISSAS
# ──────────────────────────────────────────────────────────────────────────────

def render_premissas(is_financial: bool) -> Dict[str, Any]:
    """Aba 2 — Formulário de Premissas para os dois cenários."""
    st.header("⚙️ Premissas de Projeção")

    macro: Dict[str, float] = st.session_state.get("macro", {})
    ipca = macro.get("ipca_12m", FALLBACK_IPCA)
    cdi = macro.get("cdi_anual", FALLBACK_CDI)
    pib_nom = macro.get("pib_nominal", FALLBACK_PIB_NOMINAL)

    # Valores base do session_state (preenchidos pela Aba 1)
    last_rev = st.session_state.get("last_revenue", np.nan)
    last_ebitda = st.session_state.get("last_ebitda", np.nan)
    last_da = st.session_state.get("last_da", np.nan)
    last_capex = st.session_state.get("last_capex", np.nan)
    last_nwc = st.session_state.get("last_nwc", 0.0)

    # Defaults calculados a partir do último trimestre
    safe_rev = last_rev if (not np.isnan(last_rev) and last_rev > 0) else 1e10
    safe_ebitda = last_ebitda if (not np.isnan(last_ebitda) and last_ebitda > 0) else 2.5e9
    safe_da = last_da if (not np.isnan(last_da) and last_da > 0) else 5e8
    safe_capex = last_capex if (not np.isnan(last_capex) and last_capex > 0) else 6e8

    # Anualizados (× 4 para fluxos de resultado; NWC é saldo, não anualizar)
    ann_rev = safe_rev * 4
    ann_ebitda = safe_ebitda * 4
    ann_da = safe_da * 4
    ann_capex = safe_capex * 4

    base_margin = ann_ebitda / ann_rev
    base_capex_pct = ann_capex / ann_rev
    base_nim = (ann_ebitda * 0.50) / ann_rev   # proxy lucro líquido

    st.info(
        f"📌 **Base (último trimestre anualizado)** | "
        f"Receita: R$ {ann_rev/1e9:.2f}Bi | "
        f"Margem EBITDA: {base_margin:.1%} | "
        f"Capex/Receita: {base_capex_pct:.1%}"
    )

    # ── Parâmetros Gerais ─────────────────────────────────────────────────
    st.markdown("### 🔧 Parâmetros Gerais")
    cg1, cg2, cg3 = st.columns(3)
    with cg1:
        tax_rate = st.number_input(
            "Alíquota IR + CSLL (%)",
            min_value=0.0, max_value=50.0,
            value=34.0, step=0.5,
            key="tax_rate",
        ) / 100
    with cg2:
        years = st.selectbox(
            "Período Explícito (anos)",
            [3, 5, 7],
            index=0,
            key="proj_years",
        )
    with cg3:
        da_growth = st.number_input(
            "Crescimento D&A (% a.a.)",
            min_value=0.0, max_value=20.0,
            value=round(ipca * 100, 1),
            step=0.5,
            key="da_growth",
        ) / 100

    if is_financial:
        net_borrowing = st.number_input(
            "Net Borrowing Anual (R$ Milhões) — somente FCFE",
            value=0.0,
            step=100.0,
            help="Captação líquida de dívida: nova dívida − amortizações",
        ) * 1e6
    else:
        net_borrowing = 0.0

    st.divider()

    # ── Dois cenários lado a lado ─────────────────────────────────────────
    col_c, col_m = st.columns(2, gap="large")

    # ─── Cenário Conservador ───────────────────────────────────────────────
    with col_c:
        st.markdown(
            "### 🐢 Cenário Conservador",
            help="Crescimento limitado ao IPCA, margens pressionadas, WACC estressado",
        )
        st.caption("Macro adverso | Sem ganhos de eficiência | Custo de capital elevado")

        cons_rev_g: List[float] = []
        for y in range(years):
            v = st.number_input(
                f"Crescimento Receita Ano {y + 1} (%)",
                min_value=-20.0, max_value=50.0,
                value=round(ipca * 100, 1),
                step=0.5,
                key=f"c_rev_{y}",
            ) / 100
            cons_rev_g.append(v)

        cons_margin = st.slider(
            "Margem EBITDA (%)",
            min_value=3.0, max_value=60.0,
            value=round(max(base_margin * 0.88, 0.08) * 100, 1),
            step=0.5,
            key="c_margin",
        ) / 100

        cons_capex_pct = st.slider(
            "Capex / Receita (%)",
            min_value=0.0, max_value=30.0,
            value=round(min(base_capex_pct * 1.10, 20.0), 1),
            step=0.5,
            key="c_capex",
        ) / 100

        cons_nwc_pct = st.slider(
            "NWC / Receita (%)",
            min_value=-10.0, max_value=30.0,
            value=5.0,
            step=0.5,
            key="c_nwc",
        ) / 100

        cons_wacc = st.slider(
            "WACC Estressado (%)",
            min_value=5.0, max_value=30.0,
            value=round(min(cdi * 100 + 5.0, 20.0), 1),
            step=0.25,
            key="c_wacc",
        ) / 100

        if is_financial:
            cons_nim = st.slider(
                "Margem Líquida (%)",
                min_value=1.0, max_value=40.0,
                value=round(max(base_nim * 0.85 * 100, 5.0), 1),
                step=0.5,
                key="c_nim",
            ) / 100
        else:
            cons_nim = base_nim

    # ─── Cenário Moderado ──────────────────────────────────────────────────
    with col_m:
        st.markdown(
            "### 🚀 Cenário Moderado",
            help="Crescimento real baseado em pipeline, margens sustentadas, WACC histórico",
        )
        st.caption("Crescimento real | Eficiência mantida | Custo de capital histórico")

        mod_rev_g: List[float] = []
        for y in range(years):
            v = st.number_input(
                f"Crescimento Receita Ano {y + 1} (%)",
                min_value=-20.0, max_value=80.0,
                value=round(min((ipca + 0.04) * 100, 25.0), 1),
                step=0.5,
                key=f"m_rev_{y}",
            ) / 100
            mod_rev_g.append(v)

        mod_margin = st.slider(
            "Margem EBITDA (%)",
            min_value=3.0, max_value=70.0,
            value=round(base_margin * 100, 1),
            step=0.5,
            key="m_margin",
        ) / 100

        mod_capex_pct = st.slider(
            "Capex / Receita (%)",
            min_value=0.0, max_value=30.0,
            value=round(base_capex_pct * 100, 1),
            step=0.5,
            key="m_capex",
        ) / 100

        mod_nwc_pct = st.slider(
            "NWC / Receita (%)",
            min_value=-10.0, max_value=30.0,
            value=5.0,
            step=0.5,
            key="m_nwc",
        ) / 100

        mod_wacc = st.slider(
            "WACC Médio Histórico (%)",
            min_value=5.0, max_value=25.0,
            value=round(min(cdi * 100 + 3.0, 16.0), 1),
            step=0.25,
            key="m_wacc",
        ) / 100

        if is_financial:
            mod_nim = st.slider(
                "Margem Líquida (%)",
                min_value=1.0, max_value=40.0,
                value=round(base_nim * 100, 1),
                step=0.5,
                key="m_nim",
            ) / 100
        else:
            mod_nim = base_nim

    # ── Taxa g na Perpetuidade ─────────────────────────────────────────────
    st.divider()
    st.markdown("### 🌱 Crescimento na Perpetuidade (g)")
    g_max = round(pib_nom * 100, 2)
    g_default = round(min(ipca * 100 + 0.5, g_max), 2)

    g_val = st.slider(
        f"Taxa g — máximo travado no PIB Nominal estimado ({g_max:.1f}%)",
        min_value=0.5,
        max_value=float(g_max),
        value=float(g_default),
        step=0.25,
        help=(
            f"A taxa de crescimento na perpetuidade não pode exceder o PIB Nominal "
            f"estimado ({g_max:.1f}%), conforme boas práticas de valuation."
        ),
    ) / 100

    return {
        "years": int(years),
        "tax_rate": tax_rate,
        "da_growth": da_growth,
        "g": g_val,
        "ann_rev": ann_rev,
        "ann_da": ann_da,
        "base_margin": base_margin,
        "base_nim": base_nim,
        "last_nwc": last_nwc,
        "net_borrowing": net_borrowing,
        # Conservador
        "cons_rev_g": cons_rev_g,
        "cons_margin": cons_margin,
        "cons_capex_pct": cons_capex_pct,
        "cons_nwc_pct": cons_nwc_pct,
        "cons_wacc": cons_wacc,
        "cons_nim": cons_nim,
        # Moderado
        "mod_rev_g": mod_rev_g,
        "mod_margin": mod_margin,
        "mod_capex_pct": mod_capex_pct,
        "mod_nwc_pct": mod_nwc_pct,
        "mod_wacc": mod_wacc,
        "mod_nim": mod_nim,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 8 — ABA 3: MOTOR DCF
# ──────────────────────────────────────────────────────────────────────────────

def render_motor_dcf(is_financial: bool) -> None:
    """Aba 3 — Motor de Fluxo de Caixa Descontado."""
    st.header("⚡ Motor de Fluxo de Caixa Descontado")

    p: Dict[str, Any] = st.session_state.get("premissas_inputs", {})
    if not p:
        st.info("👈 Defina as premissas na Aba 2 primeiro.")
        return

    net_debt = st.session_state.get("net_debt_ss", 0.0) or 0.0
    shares = st.session_state.get("shares_outstanding", np.nan)

    if np.isnan(shares) or shares <= 0:
        st.error(
            "❌ Número de ações não disponível. "
            "Verifique se o ticker foi carregado corretamente."
        )
        return

    mode = "🏦 FCFE (Instituição Financeira)" if is_financial else "🏭 FCFF (Empresa Real)"
    st.info(f"**Modo de cálculo:** {mode}")

    # Fórmula exibida
    if is_financial:
        st.latex(
            r"\text{FCFE} = \text{Lucro Líquido} + D\&A - CapEx - \Delta NWC + \text{Net Borrowing}"
        )
    else:
        st.latex(
            r"\text{FCFF} = EBIT \times (1 - t) + D\&A - CapEx - \Delta NWC"
        )
    st.latex(
        r"TV = \frac{FCF_{t+1}}{WACC - g} \quad \text{(Gordon Growth)}"
    )

    st.divider()

    # ── Projeção dos FCFs ─────────────────────────────────────────────────
    nim_cons = [p["cons_nim"]] * p["years"] if is_financial else None
    nim_mod = [p["mod_nim"]] * p["years"] if is_financial else None

    fcf_cons = project_fcfs(
        base_revenue=p["ann_rev"],
        base_da=p["ann_da"],
        base_capex=0.0,
        base_nwc=p["last_nwc"],
        tax_rate=p["tax_rate"],
        revenue_growth_rates=p["cons_rev_g"],
        ebitda_margin_path=[p["cons_margin"]] * p["years"],
        capex_pct_revenue=p["cons_capex_pct"],
        da_growth_pa=p["da_growth"],
        nwc_pct_revenue=p["cons_nwc_pct"],
        is_fcfe=is_financial,
        net_income_margin_path=nim_cons,
        net_borrowing=p["net_borrowing"],
    )

    fcf_mod = project_fcfs(
        base_revenue=p["ann_rev"],
        base_da=p["ann_da"],
        base_capex=0.0,
        base_nwc=p["last_nwc"],
        tax_rate=p["tax_rate"],
        revenue_growth_rates=p["mod_rev_g"],
        ebitda_margin_path=[p["mod_margin"]] * p["years"],
        capex_pct_revenue=p["mod_capex_pct"],
        da_growth_pa=p["da_growth"],
        nwc_pct_revenue=p["mod_nwc_pct"],
        is_fcfe=is_financial,
        net_income_margin_path=nim_mod,
        net_borrowing=p["net_borrowing"],
    )

    # ── DCF Engine ────────────────────────────────────────────────────────
    errors: List[str] = []
    result_cons: Dict[str, float] = {}
    result_mod: Dict[str, float] = {}

    try:
        result_cons = dcf_engine(fcf_cons, p["cons_wacc"], p["g"],
                                 net_debt, shares, is_financial)
    except ValueError as exc:
        errors.append(f"❌ DCF Conservador: {exc}")

    try:
        result_mod = dcf_engine(fcf_mod, p["mod_wacc"], p["g"],
                                net_debt, shares, is_financial)
    except ValueError as exc:
        errors.append(f"❌ DCF Moderado: {exc}")

    for err in errors:
        st.error(err)
    if errors:
        return

    # Persiste resultados
    st.session_state["dcf_results"] = {
        "conservador": result_cons,
        "moderado": result_mod,
        "premissas": p,
        "is_financial": is_financial,
        "net_debt": net_debt,
        "shares": shares,
        "fcf_cons": fcf_cons,
        "fcf_mod": fcf_mod,
    }

    # ── Gráfico de projeção ───────────────────────────────────────────────
    st.plotly_chart(chart_fcf_projection(fcf_cons, fcf_mod, p["years"]),
                    use_container_width=True)

    # ── Waterfall ─────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            chart_waterfall(result_cons, net_debt, "🐢 Conservador"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            chart_waterfall(result_mod, net_debt, "🚀 Moderado"),
            use_container_width=True,
        )

    # ── Tabela detalhada ──────────────────────────────────────────────────
    with st.expander("📊 Detalhamento dos FCFs Projetados (R$ Bilhões)"):
        year_cols = {f"Ano {i + 1}": [
            fcf_cons[i] / 1e9, fcf_mod[i] / 1e9
        ] for i in range(p["years"])}
        df_det = pd.DataFrame({
            "Cenário": ["🐢 Conservador", "🚀 Moderado"],
            **year_cols,
            "Σ PV FCFs": [result_cons["PV_FCF"] / 1e9, result_mod["PV_FCF"] / 1e9],
            "PV TV": [result_cons["PV_TV"] / 1e9, result_mod["PV_TV"] / 1e9],
            "Equity Value": [
                result_cons["Equity_Value"] / 1e9,
                result_mod["Equity_Value"] / 1e9,
            ],
            "Preço Justo (R$)": [
                result_cons["Fair_Price"],
                result_mod["Fair_Price"],
            ],
        }).set_index("Cenário")
        st.dataframe(
            df_det.style.format({
                **{c: "{:.2f} Bi" for c in df_det.columns if c != "Preço Justo (R$)"},
                "Preço Justo (R$)": "R$ {:.2f}",
            }),
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 9 — ABA 4: OUTPUT & DECISÃO
# ──────────────────────────────────────────────────────────────────────────────

def render_output() -> None:
    """Aba 4 — Output Financeiro e Decisão de Investimento."""
    st.header("💡 Output Financeiro e Decisão de Investimento")

    dcf = st.session_state.get("dcf_results", {})
    if not dcf:
        st.info("👈 Execute o DCF na Aba 3 primeiro.")
        return

    result_cons: Dict[str, float] = dcf["conservador"]
    result_mod: Dict[str, float] = dcf["moderado"]
    p: Dict[str, Any] = dcf["premissas"]
    is_financial: bool = dcf["is_financial"]
    net_debt: float = dcf["net_debt"]
    shares: float = dcf["shares"]
    fcf_cons: List[float] = dcf["fcf_cons"]
    fcf_mod: List[float] = dcf["fcf_mod"]
    cdi: float = st.session_state.get("macro", {}).get("cdi_anual", FALLBACK_CDI)

    fp_cons = result_cons.get("Fair_Price", np.nan)
    fp_mod = result_mod.get("Fair_Price", np.nan)
    px_atual = st.session_state.get("current_price", np.nan)

    # ── Métricas principais ───────────────────────────────────────────────
    st.subheader("📌 Preço Justo vs Preço de Tela")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🐢 Preço Justo Conservador",
        f"R$ {fp_cons:.2f}" if not np.isnan(fp_cons) else "N/D",
    )
    col2.metric(
        "🚀 Preço Justo Moderado",
        f"R$ {fp_mod:.2f}" if not np.isnan(fp_mod) else "N/D",
    )
    col3.metric(
        "📌 Preço de Tela Atual",
        f"R$ {px_atual:.2f}" if not np.isnan(px_atual) else "N/D",
    )

    ms_pct: float = np.nan
    if not np.isnan(fp_mod) and not np.isnan(px_atual) and px_atual > 0:
        ms_pct = (fp_mod - px_atual) / px_atual
        col4.metric(
            "🛡️ Margem de Segurança",
            f"{ms_pct:.1%}",
            delta="Subavaliado ✅" if ms_pct > 0 else "Sobreavaliado ❌",
            delta_color="normal" if ms_pct > 0 else "inverse",
        )
    else:
        col4.metric("🛡️ Margem de Segurança", "N/D")

    st.divider()

    # ── TIR Implícita ─────────────────────────────────────────────────────
    st.subheader("📐 TIR Implícita da Compra ao Preço Atual")
    st.caption(
        "Taxa Interna de Retorno que o investidor obteria comprando hoje "
        "e realizando ao Preço Justo do modelo."
    )

    if not np.isnan(px_atual) and px_atual > 0 and shares > 0:
        ti1, ti2 = st.columns(2)
        for col_ui, fcf_list, wacc_used, label in [
            (ti1, fcf_cons, p["cons_wacc"], "🐢 Conservador"),
            (ti2, fcf_mod, p["mod_wacc"], "🚀 Moderado"),
        ]:
            try:
                irr = calc_irr(
                    current_price=px_atual,
                    fcf_projections=fcf_list,
                    g=p["g"],
                    net_debt=net_debt,
                    shares=shares,
                    is_fcfe=is_financial,
                )
                spread = irr - cdi if not np.isnan(irr) else np.nan
                col_ui.metric(
                    f"TIR — {label}",
                    f"{irr:.2%}" if not np.isnan(irr) else "N/D",
                    delta=f"Spread vs CDI: {spread:.2%}" if not np.isnan(spread) else None,
                    delta_color="normal" if (not np.isnan(spread) and spread > 0) else "inverse",
                )
            except Exception as exc:
                col_ui.metric(f"TIR — {label}", "Erro")
                col_ui.caption(str(exc))
    else:
        st.warning("Preço atual não disponível para cálculo da TIR.")

    st.divider()

    # ── Painel de Decisão ─────────────────────────────────────────────────
    st.subheader("🎯 Painel de Decisão")

    if not np.isnan(ms_pct) and not np.isnan(px_atual):
        fp_medio = (
            (fp_cons + fp_mod) / 2
            if (not np.isnan(fp_cons) and not np.isnan(fp_mod))
            else fp_mod
        )

        if ms_pct > 0.25:
            verdict = "🟢 COMPRA FORTE"
            subtext = f"Margem de Segurança confortável de {ms_pct:.1%} (>25%)"
            bg = "#14532D"
        elif 0.10 < ms_pct <= 0.25:
            verdict = "🟡 COMPRA MODERADA"
            subtext = f"Ativo levemente subavaliado — Margem de {ms_pct:.1%}"
            bg = "#713F12"
        elif 0.0 < ms_pct <= 0.10:
            verdict = "⚪ NEUTRO / ACUMULAR"
            subtext = f"Preço próximo do valor justo (+{ms_pct:.1%})"
            bg = "#1E3A5F"
        elif -0.10 <= ms_pct <= 0.0:
            verdict = "🟠 NEUTRO / AGUARDAR"
            subtext = f"Ativo levemente sobreavaliado ({ms_pct:.1%})"
            bg = "#7C2D12"
        else:
            verdict = "🔴 EVITAR / REDUZIR"
            subtext = f"Ativo significativamente sobreavaliado ({ms_pct:.1%})"
            bg = "#7F1D1D"

        st.markdown(
            f"""
            <div style="
                background:{bg};
                padding:24px 32px;
                border-radius:14px;
                text-align:center;
                margin:8px 0;
            ">
                <h2 style="color:#FFFFFF;margin:0;letter-spacing:1px">{verdict}</h2>
                <p style="color:#D1D5DB;margin:10px 0 0;font-size:1rem">{subtext}</p>
                <p style="color:#94A3B8;margin:6px 0 0;font-size:.9rem">
                    Preço Atual: <b>R$ {px_atual:.2f}</b> &nbsp;|&nbsp;
                    Preço Justo (Moderado): <b>R$ {fp_mod:.2f}</b> &nbsp;|&nbsp;
                    Preço Justo Médio: <b>R$ {fp_medio:.2f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Defina o preço atual e os resultados do DCF para ver o painel de decisão.")

    # ── Resumo completo ───────────────────────────────────────────────────
    with st.expander("📋 Resumo Completo do Modelo de Valuation"):
        def _fbi(v: float) -> str:
            return f"R$ {v/1e9:.2f} Bi" if not np.isnan(v) else "N/D"

        rows_summary = {
            "Parâmetro": [
                "WACC", "Taxa g",
                "PV FCFs Explícitos", "PV Valor Terminal",
                "Enterprise Value", "(−) Dívida Líquida",
                "Equity Value", "Preço Justo",
            ],
            "🐢 Conservador": [
                f"{p['cons_wacc']:.2%}", f"{p['g']:.2%}",
                _fbi(result_cons["PV_FCF"]), _fbi(result_cons["PV_TV"]),
                _fbi(result_cons["Enterprise_Value"]), _fbi(net_debt),
                _fbi(result_cons["Equity_Value"]),
                f"R$ {fp_cons:.2f}" if not np.isnan(fp_cons) else "N/D",
            ],
            "🚀 Moderado": [
                f"{p['mod_wacc']:.2%}", f"{p['g']:.2%}",
                _fbi(result_mod["PV_FCF"]), _fbi(result_mod["PV_TV"]),
                _fbi(result_mod["Enterprise_Value"]), _fbi(net_debt),
                _fbi(result_mod["Equity_Value"]),
                f"R$ {fp_mod:.2f}" if not np.isnan(fp_mod) else "N/D",
            ],
        }
        st.dataframe(
            pd.DataFrame(rows_summary).set_index("Parâmetro"),
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 10 — ABA 5: SENSIBILIDADE
# ──────────────────────────────────────────────────────────────────────────────

def render_sensibilidade() -> None:
    """Aba 5 — Estresse de Modelo / Análise de Sensibilidade."""
    st.header("🎲 Estresse de Modelo — Análise de Sensibilidade")

    dcf = st.session_state.get("dcf_results", {})
    if not dcf:
        st.info("👈 Execute o DCF nas Abas 2–3 primeiro.")
        return

    p: Dict[str, Any] = dcf["premissas"]
    is_financial: bool = dcf["is_financial"]
    net_debt: float = dcf["net_debt"]
    shares: float = dcf["shares"]
    px_atual: float = st.session_state.get("current_price", np.nan)
    macro: Dict[str, float] = st.session_state.get("macro", {})
    pib_nom: float = macro.get("pib_nominal", FALLBACK_PIB_NOMINAL)

    st.markdown(
        "Varie o **WACC** (eixo X) e escolha a variável do **eixo Y** "
        "(Margem EBITDA ou Taxa g) para mapear onde o ativo perde "
        "a margem de segurança."
    )

    # ── Configurações da Matriz ───────────────────────────────────────────
    cf1, cf2 = st.columns(2)
    with cf1:
        wacc_min = st.number_input("WACC Mínimo (%)", 5.0, 20.0, 8.0, 0.5) / 100
        wacc_max = st.number_input("WACC Máximo (%)", 8.0, 30.0, 20.0, 0.5) / 100
        wacc_steps = st.slider("Pontos no WACC", 3, 10, 7)
    with cf2:
        y_mode = st.radio(
            "Variável no Eixo Y",
            ["Margem EBITDA", "Taxa g (Perpetuidade)"],
            horizontal=True,
            key="y_mode",
        )
        if y_mode == "Margem EBITDA":
            y_min = st.slider("Margem EBITDA Mín (%)", 3.0, 25.0, 8.0, 1.0) / 100
            y_max = st.slider("Margem EBITDA Máx (%)", 10.0, 60.0, 45.0, 1.0) / 100
            y_steps = st.slider("Pontos na Margem", 3, 10, 7)
            y_range = np.linspace(y_min, y_max, y_steps)
            y_label = "Margem EBITDA"
            y_axis_key = "ebitda_margin"
            fixed_g = p["g"]
            fixed_margin = p["mod_margin"]
        else:
            y_min_g = st.slider("g Mínimo (%)", 0.5, 4.0, 1.0, 0.25) / 100
            y_max_g = st.slider(
                f"g Máximo (% — PIB Nominal: {pib_nom:.1%})",
                2.0,
                round(pib_nom * 100, 1),
                round(pib_nom * 100, 1),
                0.25,
            ) / 100
            y_steps = st.slider("Pontos em g", 3, 9, 6)
            y_range = np.linspace(y_min_g, y_max_g, y_steps)
            y_label = "Taxa g"
            y_axis_key = "g"
            fixed_g = p["g"]
            fixed_margin = p["mod_margin"]

    wacc_range = np.linspace(wacc_min, wacc_max, wacc_steps)

    st.divider()
    run_btn = st.button("🔥 Calcular Matriz de Sensibilidade", type="primary")

    if run_btn:
        with st.spinner("Calculando heatmap de sensibilidade..."):
            df_sens = build_sensitivity(
                base_revenue=p["ann_rev"],
                base_da=p["ann_da"],
                base_nwc=p["last_nwc"],
                tax_rate=p["tax_rate"],
                revenue_growth_rates=p["mod_rev_g"],
                capex_pct=p["mod_capex_pct"],
                da_growth=p["da_growth"],
                nwc_pct=p["mod_nwc_pct"],
                net_debt=net_debt,
                shares=shares,
                wacc_range=wacc_range,
                y_range=y_range,
                y_axis=y_axis_key,
                fixed_g=fixed_g,
                fixed_margin=fixed_margin,
                current_price=px_atual,
                is_fcfe=is_financial,
                base_nim=p["base_nim"],
            )

        st.plotly_chart(
            chart_heatmap(df_sens, y_label, px_atual),
            use_container_width=True,
        )

        # Contagem de quadrantes
        total_cells = df_sens.size
        not_nan = df_sens.notna().sum().sum()
        is_mos = not np.isnan(px_atual) and px_atual > 0
        if is_mos:
            positivos = (df_sens > 0).sum().sum()
            pct_safe = positivos / not_nan if not_nan > 0 else 0
            st.info(
                f"✅ **{positivos}/{not_nan}** combinações ({pct_safe:.0%}) "
                f"oferecem margem de segurança positiva ao preço atual."
            )

        with st.expander("📊 Ver dados da matriz"):
            is_mos2 = not np.isnan(px_atual) and px_atual > 0
            fmt_dict = {
                c: "{:.1%}" if is_mos2 else "R$ {:.2f}"
                for c in df_sens.columns
            }
            styled = df_sens.style.format(fmt_dict).background_gradient(
                cmap="RdYlGn", axis=None, vmin=-0.3, vmax=0.3 if is_mos2 else None
            )
            st.dataframe(styled, use_container_width=True)

        st.markdown("""
        **📖 Como interpretar o heatmap:**
        - **🟢 Verde** — O preço justo supera o preço atual: há margem de segurança.
        - **🔴 Vermelho** — O ativo perde a margem de segurança nessas premissas.
        - Use o heatmap para entender a **robustez** da tese de investimento a variações macro e operacionais.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Ponto de entrada principal do aplicativo ValuationB3."""
    _init_state()

    sidebar = render_sidebar()
    ticker_yf: str = sidebar["ticker_yf"]
    is_financial: bool = sidebar["is_financial"]

    # ── Header ────────────────────────────────────────────────────────────
    col_h, col_p = st.columns([5, 1])
    with col_h:
        st.title(f"📊 ValuationB3 — DCF Engine")
        st.caption(
            f"Motor de Valuation Fundamentalista para o Mercado Brasileiro (B3)  |  "
            f"Ativo selecionado: **{sidebar['ticker_raw']}**"
        )

    # ── Carregamento do ativo ─────────────────────────────────────────────
    if sidebar["load_btn"]:
        _reset_state()
        st.session_state["ticker_yf"] = ticker_yf

        with st.status(f"Carregando {ticker_yf}...", expanded=True) as status:
            status.write("📡 Consultando yfinance...")
            info = fetch_info(ticker_yf)

            if not info:
                st.error(f"❌ Ticker '{ticker_yf}' não encontrado.")
                status.update(label="Erro", state="error")
                st.stop()

            px = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            sh = (
                info.get("sharesOutstanding")
                or info.get("impliedSharesOutstanding")
            )

            st.session_state["current_price"] = float(px) if px else np.nan
            st.session_state["shares_outstanding"] = float(sh) if sh else np.nan
            st.session_state["ticker_loaded"] = True

            status.write("📊 Extraindo dados financeiros trimestrais...")
            df_q = fetch_quarterly(ticker_yf)
            st.session_state["quarterly_df"] = df_q

            if not df_q.empty:
                last = df_q.iloc[-1]

                def _s(v: Any) -> float:
                    return float(v) if pd.notna(v) else np.nan

                st.session_state["last_ebitda"] = _s(last.get("EBITDA"))
                st.session_state["last_ebit"] = _s(last.get("EBIT"))
                st.session_state["last_revenue"] = _s(last.get("Revenue"))
                st.session_state["last_da"] = abs(_s(last.get("DA")) or 0)
                st.session_state["last_capex"] = abs(_s(last.get("Capex")) or 0)
                st.session_state["net_debt_ss"] = _s(last.get("NetDebt", 0))
                st.session_state["last_nwc"] = _s(last.get("NWC")) or 0.0
                status.write(f"✅ {len(df_q)} trimestres carregados.")
            else:
                status.write("⚠️ Dados trimestrais não disponíveis — use upload de RI na Aba 1.")

            nome = info.get("longName", ticker_yf)
            setor = info.get("sector", "N/D")
            px_fmt = (
                f"R$ {st.session_state['current_price']:.2f}"
                if not np.isnan(st.session_state["current_price"])
                else "N/D"
            )

            status.update(
                label=f"✅ {nome} ({setor}) — Preço Atual: {px_fmt}",
                state="complete",
                expanded=False,
            )

    # ── Abas principais ───────────────────────────────────────────────────
    if not st.session_state.get("ticker_loaded"):
        st.info(
            "👈 **Insira o ticker e clique em 'Carregar Dados do Ativo'** na barra lateral "
            "para iniciar a análise fundamentalista."
        )
        st.markdown("""
        ### Como usar o ValuationB3:
        1. **Aba 1 — Diagnóstico**: Revise os últimos 12 trimestres, alavancagem e Capex.
        2. **Aba 2 — Premissas**: Defina crescimento, margens e WACC para dois cenários.
        3. **Aba 3 — Motor DCF**: Execute o DCF (FCFF ou FCFE) com Gordon Growth.
        4. **Aba 4 — Output**: Veja o Preço Justo, TIR implícita e Margem de Segurança.
        5. **Aba 5 — Sensibilidade**: Explore o heatmap WACC × Margem / g.
        """)
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Diagnóstico",
        "⚙️ Premissas",
        "⚡ Motor DCF",
        "💡 Output & Decisão",
        "🎲 Sensibilidade",
    ])

    with tab1:
        render_diagnostico()

    with tab2:
        premissas = render_premissas(is_financial)
        st.session_state["premissas_inputs"] = premissas

    with tab3:
        render_motor_dcf(is_financial)

    with tab4:
        render_output()

    with tab5:
        render_sensibilidade()


if __name__ == "__main__":
    main()
