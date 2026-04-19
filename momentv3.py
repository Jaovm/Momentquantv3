"""
==============================================================================
Valuation B3 — DCF Engine Pro
Motor de Valuation Fundamentalista para Ações da B3
Desenvolvido para deploy no Streamlit Cloud

Pilares:
  1. Diagnóstico do Cenário Atual (Aba 1)
  2. Premissas de Projeção (Aba 2)
  3. Motor FCFF / FCFE (Aba 3)
  4. Output Financeiro & Decisão (Aba 3 - métricas)
  5. Estresse de Modelo / Sensibilidade (Aba 4)
==============================================================================
"""

import warnings
import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Valuation B3 | DCF Engine Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CONSTANTES
# ==============================================================================
BCB_SGS_URL = (
    "https://api.bcb.gov.br/dados/serie/"
    "bcdata.sgs.{code}/dados/ultimos/{n}?formato=json"
)
# Séries SGS Banco Central do Brasil
SERIES_IPCA_ACUM_12M = 13522   # IPCA acumulado 12 meses
SERIES_DI_OVER = 11            # DI over diário (base 252)
SERIES_SELIC_META = 432        # Meta Selic
SERIES_PIB_REAL = 7326         # Expectativa PIB real (Focus)

DEFAULT_TICKERS = [
    "WEGE3", "ITUB3", "BBAS3", "PETR4", "VALE3",
    "RENT3", "EGIE3", "BBSE3", "PRIO3", "TOTS3",
    "MDIA3", "TAEE3", "B3SA3", "VIVT3", "AGRO3",
]

CSV_TEMPLATE_HELP = """
**Formato do CSV esperado** (valores em R$ bilhões):

| Date       | Revenue | EBITDA | EBIT | DA   | Capex | Net_Debt | NWC  | Net_Income |
|------------|---------|--------|------|------|-------|----------|------|------------|
| 2024-09-30 | 3.5     | 1.05   | 0.9  | 0.15 | 0.35  | 2.1      | 0.8  | 0.65       |
| 2024-06-30 | 3.2     | 0.96   | 0.82 | 0.14 | 0.32  | 2.0      | 0.75 | 0.60       |

Inclua ao menos 4 linhas (1 ano = 4 trimestres).
"""


# ==============================================================================
# MÓDULO 1: MACRO — API SGS BANCO CENTRAL DO BRASIL
# ==============================================================================

@st.cache_data(ttl=3600 * 6, show_spinner=False)
def fetch_bcb_series(serie_code: int, n_periodos: int = 24) -> pd.DataFrame:
    """
    Busca série temporal no SGS do Banco Central do Brasil.

    Args:
        serie_code: Código da série no SGS.
        n_periodos: Número de períodos a retornar.

    Returns:
        DataFrame com colunas ['data', 'valor'] ou vazio em caso de falha.
    """
    url = BCB_SGS_URL.format(code=serie_code, n=n_periodos)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        if df.empty:
            return pd.DataFrame(columns=["data", "valor"])
        df["data"] = pd.to_datetime(df["data"], dayfirst=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df.sort_values("data").reset_index(drop=True)
    except Exception as exc:
        st.warning(f"⚠️ SGS BCB indisponível (série {serie_code}): {exc}")
        return pd.DataFrame(columns=["data", "valor"])


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def get_macro_indicators() -> dict:
    """
    Retorna IPCA 12m, DI anual, Selic meta e estimativa de PIB nominal.

    Returns:
        dict com chaves: ipca_12m, di_anual, selic, pib_nominal (valores decimais).
    """
    indicators: dict = {}

    # IPCA acumulado 12 meses
    df_ipca = fetch_bcb_series(SERIES_IPCA_ACUM_12M, n_periodos=2)
    indicators["ipca_12m"] = (
        float(df_ipca["valor"].iloc[-1]) / 100
        if not df_ipca.empty
        else 0.045
    )

    # DI over diário → anualizado (base 252)
    df_di = fetch_bcb_series(SERIES_DI_OVER, n_periodos=5)
    if not df_di.empty:
        di_daily_pct = float(df_di["valor"].iloc[-1]) / 100
        indicators["di_anual"] = (1 + di_daily_pct / 252) ** 252 - 1
    else:
        indicators["di_anual"] = 0.135

    # Meta Selic (% a.a.)
    df_selic = fetch_bcb_series(SERIES_SELIC_META, n_periodos=2)
    indicators["selic"] = (
        float(df_selic["valor"].iloc[-1]) / 100
        if not df_selic.empty
        else 0.135
    )

    # PIB Nominal estimado = IPCA + crescimento real assumido de 2%
    indicators["pib_nominal"] = indicators["ipca_12m"] + 0.02

    return indicators


# ==============================================================================
# MÓDULO 2: DADOS FINANCEIROS — YFINANCE
# ==============================================================================

@st.cache_data(ttl=3600 * 8, show_spinner=False)
def fetch_quarterly_financials(ticker_sa: str) -> dict:
    """
    Extrai demonstrativos financeiros trimestrais via yfinance.

    Args:
        ticker_sa: Ticker no formato yfinance, ex: 'WEGE3.SA'.

    Returns:
        dict com DataFrames: income, balance, cashflow, info.
    """
    empty = {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "info": {},
        "ticker": ticker_sa,
    }
    try:
        t = yf.Ticker(ticker_sa)
        income = t.quarterly_financials.T
        balance = t.quarterly_balance_sheet.T
        cashflow = t.quarterly_cashflow.T
        info = t.info or {}
        return {
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
            "info": info,
            "ticker": ticker_sa,
        }
    except Exception as exc:
        st.warning(f"⚠️ Erro ao buscar demonstrativos de {ticker_sa}: {exc}")
        return empty


def _get_col(df: pd.DataFrame, candidates: list) -> pd.Series:
    """Busca a primeira coluna que contenha um dos nomes candidatos."""
    for c in candidates:
        matches = [col for col in df.columns if c.lower() in str(col).lower()]
        if matches:
            return df[matches[0]]
    return pd.Series(dtype=float)


def parse_financial_quarterly(data: dict) -> pd.DataFrame:
    """
    Processa demonstrativos financeiros e extrai métricas por trimestre.

    Returns:
        DataFrame com colunas: Revenue, EBITDA, EBIT, DA, Capex,
        Net_Debt, NWC, Net_Income, Net_Borrowing (em R$ bilhões).
        Índice: DatetimeIndex, ordenado do mais recente ao mais antigo.
    """
    income = data.get("income", pd.DataFrame())
    balance = data.get("balance", pd.DataFrame())
    cashflow = data.get("cashflow", pd.DataFrame())

    if income.empty and balance.empty and cashflow.empty:
        return pd.DataFrame()

    results = pd.DataFrame()

    # ── Income Statement ──────────────────────────────────────────────────────
    if not income.empty:
        results["Revenue"] = _get_col(
            income, ["Total Revenue", "Revenue", "Gross Profit"]
        )
        results["EBIT"] = _get_col(
            income, ["EBIT", "Operating Income", "Ebit"]
        )
        results["Net_Income"] = _get_col(
            income, ["Net Income", "Net Income Common Stockholders"]
        )

    # ── Cash Flow Statement ───────────────────────────────────────────────────
    if not cashflow.empty:
        da_raw = _get_col(
            cashflow,
            ["Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion"],
        )
        capex_raw = _get_col(
            cashflow,
            ["Capital Expenditure", "Capex", "Purchase Of Ppe", "Capital Expenditures"],
        )
        net_borrow = _get_col(
            cashflow,
            ["Net Issuance Payments Of Debt", "Changes In Debt", "Net Long Term Debt Issuance"],
        )

        results["DA"] = da_raw.abs() if not da_raw.empty else pd.Series(dtype=float)
        results["Capex"] = capex_raw.abs() if not capex_raw.empty else pd.Series(dtype=float)
        results["Net_Borrowing"] = (
            net_borrow if not net_borrow.empty else pd.Series(dtype=float)
        )

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    if not balance.empty:
        total_debt = _get_col(
            balance, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]
        )
        cash = _get_col(
            balance,
            ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"],
        )
        curr_assets = _get_col(balance, ["Current Assets", "Total Current Assets"])
        curr_liab = _get_col(balance, ["Current Liabilities", "Total Current Liabilities"])

        if not total_debt.empty and not cash.empty:
            results["Net_Debt"] = total_debt.fillna(0) - cash.fillna(0)
        elif not total_debt.empty:
            results["Net_Debt"] = total_debt.fillna(0)

        if not curr_assets.empty and not curr_liab.empty:
            results["NWC"] = curr_assets.fillna(0) - curr_liab.fillna(0)

    # ── EBITDA = EBIT + D&A ───────────────────────────────────────────────────
    if "EBIT" in results.columns and "DA" in results.columns:
        results["EBITDA"] = results["EBIT"].fillna(0) + results["DA"].fillna(0)

    if results.empty:
        return pd.DataFrame()

    # ── Formatação final ──────────────────────────────────────────────────────
    results.index = pd.to_datetime(results.index)
    results = results.sort_index(ascending=False)

    # Converte para bilhões R$
    num_cols = results.select_dtypes(include=[np.number]).columns
    results[num_cols] = results[num_cols] / 1e9

    return results.head(12)


@st.cache_data(ttl=900, show_spinner=False)
def get_current_price(ticker_sa: str) -> dict:
    """
    Busca preço atual, ações em circulação e market cap.

    Returns:
        dict com chaves: price, shares, market_cap, currency.
    """
    try:
        t = yf.Ticker(ticker_sa)
        info = t.info or {}
        price = float(
            info.get("currentPrice")
            or info.get("previousClose")
            or info.get("regularMarketPrice")
            or 0.0
        )
        shares = float(info.get("sharesOutstanding") or 0)
        mkt_cap = float(info.get("marketCap") or (price * shares))
        return {
            "price": price,
            "shares": shares,
            "market_cap": mkt_cap,
            "currency": info.get("currency", "BRL"),
        }
    except Exception:
        return {"price": 0.0, "shares": 0.0, "market_cap": 0.0, "currency": "BRL"}


# ==============================================================================
# MÓDULO 3: MOTOR DCF
# ==============================================================================

def calculate_fcff(
    ebit: float,
    tax_rate: float,
    da: float,
    capex: float,
    delta_nwc: float,
) -> float:
    """
    FCFF = EBIT × (1 − t) + D&A − CapEx − ΔNWC

    Args:
        ebit: Lucro Operacional (R$ bi).
        tax_rate: Alíquota efetiva de IR/CSLL (decimal).
        da: Depreciação e Amortização (R$ bi).
        capex: Capital Expenditure (R$ bi, valor positivo).
        delta_nwc: Variação do Capital de Giro (R$ bi).

    Returns:
        FCFF em R$ bilhões.
    """
    return ebit * (1.0 - tax_rate) + da - capex - delta_nwc


def calculate_fcfe(
    net_income: float,
    da: float,
    capex: float,
    delta_nwc: float,
    net_borrowing: float,
) -> float:
    """
    FCFE = Net Income + D&A − CapEx − ΔNWC + Net Borrowing

    Args:
        net_income: Lucro Líquido (R$ bi).
        da: D&A (R$ bi).
        capex: CapEx (R$ bi, positivo).
        delta_nwc: Variação do Capital de Giro (R$ bi).
        net_borrowing: Captação líquida de dívida (R$ bi).

    Returns:
        FCFE em R$ bilhões.
    """
    return net_income + da - capex - delta_nwc + net_borrowing


def gordon_terminal_value(fcf_next_year: float, wacc: float, g: float) -> float:
    """
    Perpetuidade de Gordon: TV = FCFF_{t+1} / (WACC − g)

    Args:
        fcf_next_year: FCF do primeiro ano da perpetuidade (R$ bi).
        wacc: Custo médio ponderado de capital (decimal).
        g: Taxa de crescimento perpétuo (decimal, deve ser < WACC).

    Returns:
        Valor Terminal em R$ bilhões, ou 0 se WACC ≤ g.
    """
    if wacc <= g:
        return 0.0
    return fcf_next_year / (wacc - g)


def run_dcf_projection(
    base_revenue: float,
    base_ebitda_margin: float,
    base_da: float,
    base_capex_pct_revenue: float,
    base_nwc: float,
    tax_rate: float,
    scenario: dict,
    is_financial: bool = False,
    base_net_income: float = 0.0,
    base_net_borrowing: float = 0.0,
) -> dict:
    """
    Projeta FCFs para os anos 1–3 e calcula Valor Terminal e Enterprise Value.

    Args:
        base_revenue: Receita LTM (R$ bi).
        base_ebitda_margin: Margem EBITDA LTM (decimal).
        base_da: D&A LTM (R$ bi).
        base_capex_pct_revenue: CapEx / Receita LTM (decimal).
        base_nwc: NWC LTM (R$ bi).
        tax_rate: Alíquota efetiva (decimal).
        scenario: dict com chaves: revenue_growth (list[3]), ebitda_margin (list[3]),
                  wacc, g, da_growth, capex_pct_revenue, nwc_pct_revenue.
        is_financial: Se True, usa FCFE em vez de FCFF.
        base_net_income: Lucro Líquido LTM (R$ bi), usado no FCFE.
        base_net_borrowing: Captação líquida LTM (R$ bi), usado no FCFE.

    Returns:
        dict com chaves: projections (DataFrame), fcfs (list), tv, pv_fcfs,
        pv_tv, enterprise_value.
    """
    n_years = 3
    revenue = base_revenue
    da = base_da
    nwc_prev = base_nwc
    net_income = base_net_income
    net_borrowing = base_net_borrowing
    wacc = scenario["wacc"]
    g = scenario["g"]
    capex_pct = scenario.get("capex_pct_revenue", base_capex_pct_revenue)
    nwc_pct = scenario.get("nwc_pct_revenue", base_nwc / max(base_revenue, 1e-6))
    da_growth_rate = scenario.get("da_growth", 0.03)

    projections: list[dict] = []
    fcfs: list[float] = []

    for year in range(1, n_years + 1):
        g_rev = scenario["revenue_growth"][year - 1]
        margin = scenario["ebitda_margin"][year - 1]

        revenue_proj = revenue * (1.0 + g_rev)
        ebitda_proj = revenue_proj * margin
        da_proj = da * (1.0 + da_growth_rate)
        ebit_proj = ebitda_proj - da_proj
        capex_proj = revenue_proj * capex_pct
        nwc_proj = revenue_proj * nwc_pct
        delta_nwc = nwc_proj - nwc_prev

        if not is_financial:
            fcf = calculate_fcff(ebit_proj, tax_rate, da_proj, capex_proj, delta_nwc)
        else:
            # Crescimento proporcional do lucro líquido
            ni_proj = net_income * (1.0 + g_rev) * (margin / max(base_ebitda_margin, 1e-6))
            fcf = calculate_fcfe(ni_proj, da_proj, capex_proj, delta_nwc, net_borrowing)

        projections.append(
            {
                "Ano": f"Ano {year}",
                "Receita (R$bi)": round(revenue_proj, 3),
                "EBITDA (R$bi)": round(ebitda_proj, 3),
                "Margem EBITDA": margin,
                "EBIT (R$bi)": round(ebit_proj, 3),
                "D&A (R$bi)": round(da_proj, 3),
                "CapEx (R$bi)": round(capex_proj, 3),
                "ΔNWC (R$bi)": round(delta_nwc, 3),
                "FCF (R$bi)": round(fcf, 3),
            }
        )
        fcfs.append(fcf)

        # Atualiza base para próximo ano
        revenue = revenue_proj
        da = da_proj
        nwc_prev = nwc_proj
        if is_financial:
            net_income = ni_proj

    # Valor Terminal (FCF do ano 4 = FCF_{n} × (1+g))
    fcf_terminal = fcfs[-1] * (1.0 + g)
    tv = gordon_terminal_value(fcf_terminal, wacc, g)

    # PV dos FCFs explícitos e do TV
    pv_fcfs = sum(fcf / (1.0 + wacc) ** (i + 1) for i, fcf in enumerate(fcfs))
    pv_tv = tv / (1.0 + wacc) ** n_years
    enterprise_value = pv_fcfs + pv_tv

    return {
        "projections": pd.DataFrame(projections).set_index("Ano"),
        "fcfs": fcfs,
        "tv": tv,
        "pv_fcfs": pv_fcfs,
        "pv_tv": pv_tv,
        "enterprise_value": enterprise_value,
    }


def equity_value_per_share(
    enterprise_value: float,
    net_debt: float,
    shares_outstanding: float,
    is_financial: bool = False,
) -> float:
    """
    Calcula o preço justo por ação.

    Para setor real: Equity Value = EV − Dívida Líquida.
    Para financeiras: EV já representa o equity (FCFE).

    Args:
        enterprise_value: EV em R$ bilhões.
        net_debt: Dívida Líquida em R$ bilhões.
        shares_outstanding: Ações em circulação (unidades absolutas).
        is_financial: Se True, FCFE já é direto ao acionista.

    Returns:
        Preço justo por ação em R$.
    """
    if shares_outstanding <= 0:
        return 0.0

    equity_bi = enterprise_value if is_financial else (enterprise_value - net_debt)
    equity_bi = max(equity_bi, 0.0)

    # Converte bilhões para unidades absolutas / (bilhões de ações)
    shares_bi = shares_outstanding / 1e9
    return equity_bi / max(shares_bi, 1e-9)


def calculate_implicit_irr(
    current_price: float,
    shares_outstanding: float,
    fcfs: list,
    tv: float,
    n_years: int = 3,
) -> float:
    """
    TIR implícita da compra no preço de tela atual.

    Resolve: Market_Cap = sum(FCFi/(1+IRR)^i) + TV/(1+IRR)^n → IRR

    Args:
        current_price: Preço atual em R$.
        shares_outstanding: Ações em circulação (unidades absolutas).
        fcfs: Lista de FCFs projetados (R$ bi).
        tv: Valor Terminal (R$ bi).
        n_years: Horizonte explícito de projeção.

    Returns:
        TIR implícita (decimal) ou nan se não convergir.
    """
    market_cap_bi = current_price * shares_outstanding / 1e9  # em R$ bilhões

    def npv_func(r: float) -> float:
        pv = sum(fcf / (1.0 + r) ** (i + 1) for i, fcf in enumerate(fcfs))
        pv += tv / (1.0 + r) ** n_years
        return pv - market_cap_bi

    try:
        # Tenta encontrar sinal trocado para o brentq
        irr = brentq(npv_func, -0.5, 5.0, xtol=1e-8, maxiter=1000)
        return irr
    except Exception:
        return np.nan


# ==============================================================================
# MÓDULO 4: ANÁLISE DE SENSIBILIDADE
# ==============================================================================

def build_sensitivity_matrix(
    base_revenue: float,
    base_ebitda_margin: float,
    base_da: float,
    base_capex_pct: float,
    base_nwc: float,
    tax_rate: float,
    revenue_growth: list,
    g_base: float,
    wacc_base: float,
    net_debt: float,
    shares_outstanding: float,
    current_price: float,
    is_financial: bool = False,
    sensitivity_y: str = "g",  # "g" ou "ebitda_margin"
) -> pd.DataFrame:
    """
    Gera matriz de preços justos iterando WACC (X) e g ou Margem EBITDA (Y).

    Args:
        sensitivity_y: "g" varia a taxa de perpetuidade; "ebitda_margin" varia a margem.

    Returns:
        DataFrame cujo índice é Y e cujas colunas são WACC (formatados em %).
    """
    step = 0.01
    wacc_min = max(0.06, wacc_base - 0.04)
    wacc_max = wacc_base + 0.045
    wacc_range = np.arange(wacc_min, wacc_max, step)

    if sensitivity_y == "g":
        y_min = max(0.01, g_base - 0.03)
        y_max = min(g_base + 0.035, wacc_base - 0.01)
        y_range = np.arange(y_min, y_max, step)
        y_title = "g (Perp.)"
    else:
        y_min = max(0.03, base_ebitda_margin - 0.08)
        y_max = base_ebitda_margin + 0.085
        y_range = np.arange(y_min, y_max, 0.02)
        y_title = "Margem EBITDA"

    matrix = pd.DataFrame(
        index=np.round(y_range, 4), columns=np.round(wacc_range, 4), dtype=float
    )

    nwc_pct = base_nwc / max(base_revenue, 1e-6)

    for y_val in y_range:
        for wacc_val in wacc_range:
            g_use = y_val if sensitivity_y == "g" else g_base
            if wacc_val <= g_use + 0.001:
                matrix.loc[round(y_val, 4), round(wacc_val, 4)] = np.nan
                continue
            margin_use = (
                [base_ebitda_margin] * 3 if sensitivity_y == "g" else [y_val] * 3
            )
            try:
                scenario_s = {
                    "revenue_growth": revenue_growth,
                    "ebitda_margin": margin_use,
                    "wacc": wacc_val,
                    "g": g_use,
                    "da_growth": 0.03,
                    "capex_pct_revenue": base_capex_pct,
                    "nwc_pct_revenue": nwc_pct,
                }
                result = run_dcf_projection(
                    base_revenue, base_ebitda_margin, base_da,
                    base_capex_pct, base_nwc, tax_rate, scenario_s, is_financial
                )
                fair = equity_value_per_share(
                    result["enterprise_value"], net_debt, shares_outstanding, is_financial
                )
                matrix.loc[round(y_val, 4), round(wacc_val, 4)] = round(fair, 2)
            except Exception:
                matrix.loc[round(y_val, 4), round(wacc_val, 4)] = np.nan

    matrix.index = [f"{v:.1%}" for v in matrix.index]
    matrix.columns = [f"{v:.1%}" for v in matrix.columns]
    matrix.index.name = y_title
    matrix.columns.name = "WACC"

    return matrix


# ==============================================================================
# MÓDULO 5: VISUALIZAÇÕES
# ==============================================================================

def plot_leverage_chart(df_q: pd.DataFrame) -> go.Figure:
    """
    Plota evolução da Dívida Líquida e do índice DL/EBITDA.
    """
    fig = go.Figure()
    if df_q.empty or "Net_Debt" not in df_q.columns:
        fig.update_layout(title="Dados insuficientes para Alavancagem")
        return fig

    df = df_q.copy().reset_index()
    df.columns = [str(c) for c in df.columns]
    date_col = df.columns[0]
    labels = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m")

    leverage = []
    for _, row in df.iterrows():
        ebitda = row.get("EBITDA", np.nan)
        nd = row.get("Net_Debt", np.nan)
        if pd.notna(ebitda) and ebitda != 0 and pd.notna(nd):
            leverage.append(round(nd / ebitda, 2))
        else:
            leverage.append(np.nan)

    fig.add_trace(
        go.Bar(
            x=labels, y=df.get("Net_Debt", pd.Series()),
            name="Dívida Líquida (R$ bi)",
            marker_color="#EF553B", opacity=0.75,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=leverage,
            name="DL / EBITDA (x)",
            mode="lines+markers", yaxis="y2",
            line=dict(color="#00CC96", width=2.5),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title="Alavancagem — Dívida Líquida & DL/EBITDA",
        yaxis=dict(title="R$ bi"),
        yaxis2=dict(
            title="DL/EBITDA (x)", overlaying="y", side="right", showgrid=False
        ),
        legend=dict(orientation="h", y=1.08),
        template="plotly_dark", height=370,
    )
    return fig


def plot_capex_da_chart(df_q: pd.DataFrame) -> go.Figure:
    """Plota CapEx executado vs. D&A."""
    fig = go.Figure()
    if df_q.empty:
        return fig

    df = df_q.copy().reset_index()
    df.columns = [str(c) for c in df.columns]
    date_col = df.columns[0]
    labels = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m")

    if "Capex" in df.columns:
        fig.add_trace(
            go.Bar(
                x=labels, y=df["Capex"],
                name="CapEx (R$ bi)", marker_color="#636EFA",
            )
        )
    if "DA" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=labels, y=df["DA"],
                name="D&A (R$ bi)", mode="lines+markers",
                line=dict(color="#FFA15A", width=2.5, dash="dot"),
            )
        )
    fig.update_layout(
        title="CapEx Executado vs. D&A (R$ bi)",
        barmode="group",
        template="plotly_dark", height=370,
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def plot_ebitda_evolution(df_q: pd.DataFrame) -> go.Figure:
    """Plota Receita, EBITDA e Margem EBITDA."""
    fig = go.Figure()
    if df_q.empty or "EBITDA" not in df_q.columns or "Revenue" not in df_q.columns:
        return fig

    df = df_q[["Revenue", "EBITDA"]].dropna().reset_index()
    df.columns = ["Data", "Receita", "EBITDA"]
    df["Margem (%)"] = (df["EBITDA"] / df["Receita"].replace(0, np.nan) * 100).round(1)
    labels = pd.to_datetime(df["Data"]).dt.strftime("%Y-%m")

    fig.add_trace(
        go.Bar(x=labels, y=df["Receita"], name="Receita (R$ bi)",
               marker_color="#636EFA", opacity=0.55)
    )
    fig.add_trace(
        go.Bar(x=labels, y=df["EBITDA"], name="EBITDA (R$ bi)",
               marker_color="#00CC96")
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=df["Margem (%)"],
            name="Margem EBITDA (%)", mode="lines+markers",
            yaxis="y2", line=dict(color="#FFA15A", width=2.5),
        )
    )
    fig.update_layout(
        title="Receita, EBITDA & Margem",
        barmode="overlay",
        yaxis=dict(title="R$ bi"),
        yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False),
        template="plotly_dark", height=370,
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def plot_sensitivity_heatmap(
    matrix: pd.DataFrame,
    current_price: float,
    title: str = "Sensibilidade — Preço Justo (R$)",
) -> go.Figure:
    """
    Heatmap de sensibilidade com preço justo em cada quadrante.
    Células verdes = upside vs. preço atual; vermelhas = downside.
    """
    if matrix.empty:
        return go.Figure()

    z = matrix.values.astype(float)

    # Anotações com MS em relação ao preço atual
    cell_text: list[list[str]] = []
    for row in z:
        row_text: list[str] = []
        for val in row:
            if np.isnan(val):
                row_text.append("N/D")
            else:
                ms = (val / max(current_price, 0.01) - 1) * 100
                sign = "+" if ms >= 0 else ""
                row_text.append(f"R${val:.1f}<br>{sign}{ms:.0f}%")
        cell_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            text=cell_text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale="RdYlGn",
            zmid=current_price,
            colorbar=dict(title="Preço Justo<br>(R$)", thickness=14),
        )
    )
    fig.add_annotation(
        text=f"Preço Atual: R$ {current_price:.2f}",
        xref="paper", yref="paper",
        x=0.01, y=-0.08, showarrow=False,
        font=dict(size=12, color="#FFA15A"),
    )
    fig.update_layout(
        title=title,
        xaxis_title="WACC",
        yaxis_title=matrix.index.name or "Y",
        template="plotly_dark",
        height=500,
    )
    return fig


def plot_ev_waterfall(result: dict, scenario_label: str, wacc: float) -> go.Figure:
    """Waterfall do bridge de EV: PV FCFs + PV TV = EV."""
    fcfs = result["fcfs"]
    pv_fcfs_parts = [fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(fcfs)]

    measures = ["relative"] * len(pv_fcfs_parts) + ["relative", "total"]
    x_labels = [f"PV FCF A{i+1}" for i in range(len(pv_fcfs_parts))] + ["PV TV", "EV Total"]
    y_values = pv_fcfs_parts + [result["pv_tv"], 0]

    fig = go.Figure(
        go.Waterfall(
            orientation="v", measure=measures,
            x=x_labels, y=y_values,
            connector={"line": {"color": "#555"}},
            increasing={"marker": {"color": "#00CC96"}},
            decreasing={"marker": {"color": "#EF553B"}},
            totals={"marker": {"color": "#636EFA"}},
        )
    )
    fig.update_layout(
        title=f"Bridge do Enterprise Value — {scenario_label}",
        yaxis_title="R$ Bilhões",
        template="plotly_dark", height=380,
    )
    return fig


# ==============================================================================
# HELPER: LTM SUM / AVG
# ==============================================================================

def ltm_sum(df: pd.DataFrame, col: str, n: int = 4) -> float:
    """Soma os últimos n trimestres de uma coluna."""
    if df.empty or col not in df.columns:
        return 0.0
    return float(df[col].dropna().head(n).sum())


def ltm_avg(df: pd.DataFrame, col: str, n: int = 4) -> float:
    """Média dos últimos n trimestres de uma coluna."""
    if df.empty or col not in df.columns:
        return 0.0
    series = df[col].dropna().head(n)
    return float(series.mean()) if not series.empty else 0.0


# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("📊 Valuation B3 — DCF Engine Pro")
    st.markdown(
        "**Motor de Valuation Fundamentalista | B3** — "
        "FCFF · FCFE · Gordon Growth · TIR Implícita · Análise de Sensibilidade"
    )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configurações")

        st.subheader("📌 Universo de Ações")
        multi_select = st.multiselect(
            "Tickers da Carteira",
            options=sorted(DEFAULT_TICKERS),
            default=["WEGE3"],
            help="Adicione ou remova ações. Análise individual via seletor abaixo.",
        )
        custom_ticker = st.text_input(
            "Adicionar Ticker (sem .SA)",
            value="",
            help="Digite qualquer ticker da B3, ex: CYRE3",
        ).strip().upper()
        if custom_ticker and custom_ticker not in multi_select:
            multi_select.append(custom_ticker)

        all_tickers = multi_select if multi_select else ["WEGE3"]
        selected_ticker = st.selectbox("🎯 Ativo em Análise", options=all_tickers)
        ticker_sa = f"{selected_ticker}.SA"

        st.divider()
        st.subheader("🏦 Perfil da Empresa")
        is_financial = st.toggle(
            "Instituição Financeira",
            value=False,
            help="ON → FCFE (bancos, seguradoras) | OFF → FCFF (setor real)",
        )
        tax_rate = (
            st.slider("Alíquota Efetiva IR/CSLL (%)", 15, 40, 34, step=1) / 100.0
        )

        st.divider()
        run_btn = st.button(
            "🚀 Carregar / Atualizar Dados",
            type="primary",
            use_container_width=True,
        )
        st.caption("Fonte: yFinance · SGS Banco Central do Brasil")

    # ── Session State Init ───────────────────────────────────────────────────
    for key in (
        "quarterly_df", "price_info", "macro",
        "conservative_scenario", "moderate_scenario", "base_inputs",
        "result_cons", "result_mod", "fair_price_cons", "fair_price_mod",
        "loaded_ticker",
    ):
        if key not in st.session_state:
            st.session_state[key] = {} if "scenario" in key or key == "base_inputs" \
                else (pd.DataFrame() if "df" in key else (None if "result" in key else {}))

    # ── Data Loading ─────────────────────────────────────────────────────────
    ticker_changed = st.session_state.get("loaded_ticker") != selected_ticker
    if run_btn or ticker_changed:
        with st.spinner(f"⏳ Carregando dados de {selected_ticker}…"):
            st.session_state["macro"] = get_macro_indicators()
            fin_raw = fetch_quarterly_financials(ticker_sa)
            st.session_state["quarterly_df"] = parse_financial_quarterly(fin_raw)
            st.session_state["price_info"] = get_current_price(ticker_sa)
            st.session_state["loaded_ticker"] = selected_ticker

    macro: dict = st.session_state["macro"]
    quarterly_df: pd.DataFrame = st.session_state.get("quarterly_df", pd.DataFrame())
    price_info: dict = st.session_state.get("price_info", {})

    ipca = macro.get("ipca_12m", 0.045)
    selic = macro.get("selic", 0.135)
    pib_nominal = macro.get("pib_nominal", 0.065)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Diagnóstico do Cenário",
        "🎛️ Premissas de Projeção",
        "💰 Valuation & Decisão",
        "🌡️ Sensibilidade",
    ])

    # =========================================================================
    # ABA 1 — DIAGNÓSTICO DO CENÁRIO ATUAL
    # =========================================================================
    with tab1:
        st.subheader(f"Diagnóstico Fundamentalista — {selected_ticker}")

        # ── Indicadores Macro ─────────────────────────────────────────────────
        st.markdown("### 🌐 Indicadores Macroeconômicos (SGS Banco Central)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IPCA 12m", f"{ipca:.2%}", help="SGS série 13522")
        c2.metric("DI Anual", f"{macro.get('di_anual', 0):.2%}", help="SGS série 11")
        c3.metric("Selic Meta", f"{selic:.2%}", help="SGS série 432")
        c4.metric(
            "PIB Nominal Est.",
            f"{pib_nominal:.2%}",
            help="IPCA + 2% crescimento real → teto do g",
        )

        # ── Preço & Market Cap ────────────────────────────────────────────────
        if price_info.get("price", 0) > 0:
            st.markdown("### 💹 Mercado")
            p1, p2, p3 = st.columns(3)
            p1.metric("Preço Atual", f"R$ {price_info['price']:.2f}")
            p2.metric(
                "Market Cap",
                f"R$ {price_info['market_cap'] / 1e9:.2f}B",
            )
            p3.metric(
                "Ações em Circulação",
                f"{price_info['shares'] / 1e6:.0f}M",
            )

        st.divider()

        # ── Upload Fallback ───────────────────────────────────────────────────
        if quarterly_df.empty:
            st.warning(
                "⚠️ Dados financeiros não encontrados via yFinance. "
                "Faça upload do CSV exportado do RI da empresa."
            )
            with st.expander("ℹ️ Formato esperado do CSV", expanded=False):
                st.markdown(CSV_TEMPLATE_HELP)

            uploaded_file = st.file_uploader(
                "📂 Upload CSV de Dados Financeiros",
                type=["csv"],
                help="Valores em R$ bilhões. Colunas: Date, Revenue, EBITDA, EBIT, DA, Capex, Net_Debt, NWC, Net_Income",
            )
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file, parse_dates=["Date"])
                    df_upload = df_upload.set_index("Date").sort_index(ascending=False)
                    st.session_state["quarterly_df"] = df_upload
                    quarterly_df = df_upload
                    st.success(f"✅ {len(quarterly_df)} períodos carregados com sucesso.")
                except Exception as exc:
                    st.error(f"Erro ao processar CSV: {exc}")

        # ── Tabela Financeira ─────────────────────────────────────────────────
        if not quarterly_df.empty:
            st.markdown("### 📊 Demonstrativos Trimestrais (R$ bi)")
            show_cols = [
                c for c in
                ["Revenue", "EBITDA", "EBIT", "DA", "Capex", "Net_Debt", "NWC", "Net_Income"]
                if c in quarterly_df.columns
            ]
            st.dataframe(
                quarterly_df[show_cols].style.format("{:.3f}"),
                use_container_width=True, height=300,
            )

            st.markdown("### 📈 Gráficos Interativos")
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(plot_leverage_chart(quarterly_df), use_container_width=True)
            with g2:
                st.plotly_chart(plot_capex_da_chart(quarterly_df), use_container_width=True)

            st.plotly_chart(plot_ebitda_evolution(quarterly_df), use_container_width=True)

            # Intensidade de Reinvestimento
            if "Capex" in quarterly_df.columns and "DA" in quarterly_df.columns:
                capex_ltm = ltm_sum(quarterly_df, "Capex")
                da_ltm = ltm_sum(quarterly_df, "DA")
                reinvest_ratio = capex_ltm / max(da_ltm, 1e-6)
                rev_ltm = ltm_sum(quarterly_df, "Revenue")
                ebitda_ltm = ltm_sum(quarterly_df, "EBITDA")

                st.markdown("### 🔑 KPIs LTM")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Receita LTM", f"R$ {rev_ltm:.2f}B")
                k2.metric("EBITDA LTM", f"R$ {ebitda_ltm:.2f}B", delta=f"{ebitda_ltm/max(rev_ltm,1e-6):.1%} Mg")
                k3.metric("CapEx LTM", f"R$ {capex_ltm:.2f}B")
                k4.metric(
                    "CapEx / D&A",
                    f"{reinvest_ratio:.1f}x",
                    delta="Alto reinvest." if reinvest_ratio > 1.5 else "Baixo reinvest.",
                    delta_color="off",
                )
        else:
            st.info(
                "📭 Sem dados financeiros. Clique em **Carregar / Atualizar Dados** "
                "ou faça upload do CSV."
            )

    # =========================================================================
    # ABA 2 — PREMISSAS DE PROJEÇÃO
    # =========================================================================
    with tab2:
        st.subheader("🎛️ Premissas de Projeção — Cenário Conservador & Moderado")

        # ── Dados-Base LTM ────────────────────────────────────────────────────
        st.markdown("### 📌 Dados-Base LTM (inputs editáveis)")

        rev_ltm_default = max(0.1, round(ltm_sum(quarterly_df, "Revenue"), 2))
        ebitda_ltm_default = ltm_sum(quarterly_df, "EBITDA")
        da_ltm_default = max(0.0, round(ltm_sum(quarterly_df, "DA"), 3))
        capex_ltm_default = max(0.0, round(ltm_sum(quarterly_df, "Capex"), 3))
        nwc_ltm_default = max(0.0, round(ltm_avg(quarterly_df, "NWC"), 3))
        nd_ltm_default = round(ltm_avg(quarterly_df, "Net_Debt"), 3)
        ni_ltm_default = round(ltm_sum(quarterly_df, "Net_Income"), 3)

        margin_default = round(
            ebitda_ltm_default / max(rev_ltm_default, 1e-6) * 100, 1
        )
        capex_pct_default = round(
            capex_ltm_default / max(rev_ltm_default, 1e-6) * 100, 1
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            base_revenue = st.number_input(
                "Receita Líquida LTM (R$ bi)", value=rev_ltm_default,
                min_value=0.01, step=0.1, format="%.3f", key="base_revenue",
            )
            base_ebitda_margin = st.number_input(
                "Margem EBITDA LTM (%)", value=margin_default,
                min_value=0.0, max_value=100.0, step=0.5, format="%.1f", key="base_ebitda_margin",
            ) / 100.0
            base_da = st.number_input(
                "D&A LTM (R$ bi)", value=da_ltm_default,
                min_value=0.0, step=0.05, format="%.3f", key="base_da",
            )
        with bc2:
            base_capex_pct = st.number_input(
                "CapEx / Receita LTM (%)", value=capex_pct_default,
                min_value=0.0, max_value=100.0, step=0.5, format="%.1f", key="base_capex_pct",
            ) / 100.0
            base_nwc = st.number_input(
                "Capital de Giro Líquido LTM (R$ bi)", value=nwc_ltm_default,
                min_value=0.0, step=0.1, format="%.3f", key="base_nwc",
            )
            net_debt_input = st.number_input(
                "Dívida Líquida Atual (R$ bi)", value=nd_ltm_default,
                step=0.1, format="%.3f", key="net_debt_input",
            )

        st.divider()

        # ── Cenário Conservador ───────────────────────────────────────────────
        with st.expander("🔴 Cenário Conservador", expanded=True):
            st.caption(
                "Crescimento limitado ao IPCA · Opex pressionando margens · WACC estressado"
            )
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown("**Ano 1**")
                c_g1 = st.number_input("Crescimento Receita (%)", value=round(ipca * 100, 1), key="c_g1", min_value=-20.0, max_value=50.0, step=0.5) / 100
                c_m1 = st.number_input("Margem EBITDA (%)", value=max(0.0, margin_default - 1.0), key="c_m1", min_value=0.0, max_value=100.0, step=0.5) / 100
            with cc2:
                st.markdown("**Ano 2**")
                c_g2 = st.number_input("Crescimento Receita (%)", value=round(ipca * 100, 1), key="c_g2", min_value=-20.0, max_value=50.0, step=0.5) / 100
                c_m2 = st.number_input("Margem EBITDA (%)", value=max(0.0, margin_default - 1.5), key="c_m2", min_value=0.0, max_value=100.0, step=0.5) / 100
            with cc3:
                st.markdown("**Ano 3**")
                c_g3 = st.number_input("Crescimento Receita (%)", value=round(ipca * 100, 1), key="c_g3", min_value=-20.0, max_value=50.0, step=0.5) / 100
                c_m3 = st.number_input("Margem EBITDA (%)", value=max(0.0, margin_default - 2.0), key="c_m3", min_value=0.0, max_value=100.0, step=0.5) / 100

            cw1, cw2 = st.columns(2)
            with cw1:
                wacc_cons = st.number_input(
                    "WACC Conservador (%)",
                    value=round((selic + 0.04) * 100, 1),
                    key="wacc_cons", min_value=5.0, max_value=40.0, step=0.25,
                ) / 100
            with cw2:
                g_cons = st.slider(
                    "g Perpetuidade Conservador (%)",
                    min_value=1.0,
                    max_value=round(min(pib_nominal * 100, (wacc_cons - 0.01) * 100), 1),
                    value=min(round(ipca * 100, 1), round(pib_nominal * 100, 1)),
                    step=0.25, key="g_cons",
                ) / 100

            conservative_scenario = {
                "revenue_growth": [c_g1, c_g2, c_g3],
                "ebitda_margin": [c_m1, c_m2, c_m3],
                "wacc": wacc_cons,
                "g": g_cons,
                "da_growth": 0.03,
                "capex_pct_revenue": base_capex_pct * 1.10,  # capex 10% acima do base
                "nwc_pct_revenue": base_nwc / max(base_revenue, 1e-6),
            }

        # ── Cenário Moderado ──────────────────────────────────────────────────
        with st.expander("🟢 Cenário Moderado", expanded=True):
            st.caption(
                "Crescimento real baseado em pipeline · Margens constantes · WACC histórico médio"
            )
            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                st.markdown("**Ano 1**")
                m_g1 = st.number_input("Crescimento Receita (%)", value=round(ipca * 100 + 4, 1), key="m_g1", min_value=-20.0, max_value=80.0, step=0.5) / 100
                m_m1 = st.number_input("Margem EBITDA (%)", value=margin_default, key="m_m1", min_value=0.0, max_value=100.0, step=0.5) / 100
            with cm2:
                st.markdown("**Ano 2**")
                m_g2 = st.number_input("Crescimento Receita (%)", value=round(ipca * 100 + 5, 1), key="m_g2", min_value=-20.0, max_value=80.0, step=0.5) / 100
                m_m2 = st.number_input("Margem EBITDA (%)", value=margin_default, key="m_m2", min_value=0.0, max_value=100.0, step=0.5) / 100
            with cm3:
                st.markdown("**Ano 3**")
                m_g3 = st.number_input("Crescimento Receita (%)", value=round(ipca * 100 + 5, 1), key="m_g3", min_value=-20.0, max_value=80.0, step=0.5) / 100
                m_m3 = st.number_input("Margem EBITDA (%)", value=margin_default, key="m_m3", min_value=0.0, max_value=100.0, step=0.5) / 100

            mw1, mw2 = st.columns(2)
            with mw1:
                wacc_mod = st.number_input(
                    "WACC Moderado (%)",
                    value=round((selic + 0.025) * 100, 1),
                    key="wacc_mod", min_value=5.0, max_value=40.0, step=0.25,
                ) / 100
            with mw2:
                g_max_mod = round(min(pib_nominal * 100, (wacc_mod - 0.01) * 100), 1)
                g_mod = st.slider(
                    "g Perpetuidade Moderado (%)",
                    min_value=1.0,
                    max_value=g_max_mod,
                    value=min(round(ipca * 100 + 1.0, 1), g_max_mod),
                    step=0.25, key="g_mod",
                ) / 100

            moderate_scenario = {
                "revenue_growth": [m_g1, m_g2, m_g3],
                "ebitda_margin": [m_m1, m_m2, m_m3],
                "wacc": wacc_mod,
                "g": g_mod,
                "da_growth": 0.03,
                "capex_pct_revenue": base_capex_pct,
                "nwc_pct_revenue": base_nwc / max(base_revenue, 1e-6),
            }

        # Persiste no session_state
        st.session_state["conservative_scenario"] = conservative_scenario
        st.session_state["moderate_scenario"] = moderate_scenario
        st.session_state["base_inputs"] = {
            "base_revenue": base_revenue,
            "base_ebitda_margin": base_ebitda_margin,
            "base_da": base_da,
            "base_capex_pct": base_capex_pct,
            "base_nwc": base_nwc,
            "net_debt": net_debt_input,
            "tax_rate": tax_rate,
            "ni_ltm": ni_ltm_default,
            "net_borrowing": 0.0,
        }

        st.info(
            f"💡 **g máximo travado ao PIB Nominal estimado ({pib_nominal:.1%})** — "
            "disciplina do Gordon Growth Model impede crescimento perpétuo acima da economia."
        )

    # =========================================================================
    # ABA 3 — VALUATION & DECISÃO
    # =========================================================================
    with tab3:
        st.subheader("💰 Motor DCF & Output de Decisão de Investimento")

        base_inputs = st.session_state.get("base_inputs", {})
        cons_scenario = st.session_state.get("conservative_scenario", {})
        mod_scenario = st.session_state.get("moderate_scenario", {})

        if not base_inputs or not cons_scenario or not mod_scenario:
            st.warning("⚠️ Configure as premissas na **Aba 2** antes de executar o valuation.")
            st.stop()

        current_price = price_info.get("price", 0.0)
        shares_abs = price_info.get("shares", 1.0)

        # ── Executa DCF para os dois cenários ────────────────────────────────
        result_cons = result_mod = None
        fair_price_cons = fair_price_mod = 0.0

        with st.spinner("Calculando modelos DCF…"):
            for label, scenario, key_fp, key_res in [
                ("Conservador", cons_scenario, "fair_price_cons", "result_cons"),
                ("Moderado", mod_scenario, "fair_price_mod", "result_mod"),
            ]:
                try:
                    res = run_dcf_projection(
                        base_revenue=base_inputs["base_revenue"],
                        base_ebitda_margin=base_inputs["base_ebitda_margin"],
                        base_da=base_inputs["base_da"],
                        base_capex_pct_revenue=base_inputs["base_capex_pct"],
                        base_nwc=base_inputs["base_nwc"],
                        tax_rate=base_inputs["tax_rate"],
                        scenario=scenario,
                        is_financial=is_financial,
                        base_net_income=base_inputs.get("ni_ltm", 0.0),
                        base_net_borrowing=base_inputs.get("net_borrowing", 0.0),
                    )
                    fp = equity_value_per_share(
                        res["enterprise_value"],
                        base_inputs["net_debt"],
                        shares_abs,
                        is_financial,
                    )
                    st.session_state[key_res] = res
                    st.session_state[key_fp] = fp
                except Exception as exc:
                    st.error(f"Erro no DCF {label}: {exc}")

        result_cons = st.session_state.get("result_cons")
        result_mod = st.session_state.get("result_mod")
        fair_price_cons = st.session_state.get("fair_price_cons", 0.0)
        fair_price_mod = st.session_state.get("fair_price_mod", 0.0)

        # ── KPIs de Decisão ───────────────────────────────────────────────────
        st.markdown("### 🎯 Preço Justo, TIR & Margem de Segurança")

        safety_margin = (
            (fair_price_mod / max(current_price, 1e-6) - 1.0) * 100
            if current_price > 0 else 0.0
        )

        irr_value = np.nan
        if result_mod and current_price > 0 and shares_abs > 0:
            try:
                irr_value = calculate_implicit_irr(
                    current_price, shares_abs,
                    result_mod["fcfs"], result_mod["tv"],
                )
            except Exception:
                pass

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("💹 Preço Atual", f"R$ {current_price:.2f}")
        m2.metric(
            "🔴 Preço Justo (Cons.)",
            f"R$ {fair_price_cons:.2f}",
            delta=f"{(fair_price_cons / max(current_price, 1e-6) - 1) * 100:+.1f}%" if current_price > 0 else None,
        )
        m3.metric(
            "🟢 Preço Justo (Mod.)",
            f"R$ {fair_price_mod:.2f}",
            delta=f"{(fair_price_mod / max(current_price, 1e-6) - 1) * 100:+.1f}%" if current_price > 0 else None,
        )
        m4.metric(
            "📐 Margem de Segurança",
            f"{safety_margin:+.1f}%",
            delta="Upside" if safety_margin > 0 else "Downside",
            delta_color="normal" if safety_margin > 0 else "inverse",
        )
        m5.metric(
            "📈 TIR Implícita",
            f"{irr_value:.1%}" if not np.isnan(irr_value) else "N/D",
            delta=f"WACC Mod: {mod_scenario.get('wacc', 0):.1%}" if not np.isnan(irr_value) else None,
        )

        # ── Veredicto de Investimento ─────────────────────────────────────────
        if current_price > 0 and fair_price_mod > 0:
            if safety_margin > 30:
                vc, vt = "#00CC96", "🚀 COMPRA FORTE — Margem de Segurança expressiva (>30%)"
            elif safety_margin > 15:
                vc, vt = "#ADFF2F", "✅ COMPRA — Ativo abaixo do valor justo moderado"
            elif safety_margin > -10:
                vc, vt = "#FFA500", "⚖️ NEUTRO — Ativo precificado próximo ao valor justo"
            else:
                vc, vt = "#EF553B", "❌ AGUARDAR / VENDA — Ativo acima do valor justo"

            st.markdown(
                f"""
                <div style='background:{vc}1a; border-left:4px solid {vc};
                     padding:12px 20px; border-radius:6px; margin:12px 0;'>
                  <strong style='color:{vc}; font-size:1.05rem;'>{vt}</strong><br>
                  <span style='color:#ccc; font-size:0.9rem;'>
                    Preço Atual: R${current_price:.2f} &nbsp;|&nbsp;
                    Justo Mod.: R${fair_price_mod:.2f} &nbsp;|&nbsp;
                    MS: {safety_margin:+.1f}% &nbsp;|&nbsp;
                    TIR: {"N/D" if np.isnan(irr_value) else f"{irr_value:.1%}"}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Projeções Lado a Lado ─────────────────────────────────────────────
        st.markdown("### 📋 Projeções Detalhadas por Cenário")
        col_l, col_r = st.columns(2)

        for col, label, result, wacc, fp in [
            (col_l, "🔴 Conservador", result_cons, cons_scenario.get("wacc", 0), fair_price_cons),
            (col_r, "🟢 Moderado", result_mod, mod_scenario.get("wacc", 0), fair_price_mod),
        ]:
            with col:
                st.markdown(f"#### {label}")
                if result:
                    df_proj = result["projections"].copy()
                    # Formata porcentagem
                    df_display = df_proj.copy()
                    if "Margem EBITDA" in df_display.columns:
                        df_display["Margem EBITDA"] = df_display["Margem EBITDA"].map("{:.1%}".format)

                    st.dataframe(
                        df_display.style.format(
                            {c: "R${:.3f}B" for c in df_proj.columns if "R$bi" in c or "bi)" in c}
                        ),
                        use_container_width=True, height=180,
                    )

                    ev = result["enterprise_value"]
                    nd = base_inputs.get("net_debt", 0)
                    equity_bi = max(ev - nd, 0) if not is_financial else ev

                    st.markdown(
                        f"""
                        | | R$ bi |
                        |---|---|
                        | PV FCFs Explícitos | {result['pv_fcfs']:.3f} |
                        | Valor Terminal (TV) | {result['tv']:.3f} |
                        | PV do TV | {result['pv_tv']:.3f} |
                        | **Enterprise Value** | **{ev:.3f}** |
                        | (−) Dívida Líquida | {nd:.3f} |
                        | **Equity Value** | **{equity_bi:.3f}** |
                        | **Preço Justo/Ação** | **R$ {fp:.2f}** |
                        """
                    )

        # ── Gráficos: FCF Projetado + Waterfall ──────────────────────────────
        if result_cons and result_mod:
            df_fcf = pd.DataFrame(
                {
                    "Ano": ["Ano 1", "Ano 2", "Ano 3"],
                    "FCF Conservador": result_cons["fcfs"],
                    "FCF Moderado": result_mod["fcfs"],
                }
            )
            fig_bar = px.bar(
                df_fcf, x="Ano",
                y=["FCF Conservador", "FCF Moderado"],
                barmode="group",
                title="FCF Projetado — Conservador vs. Moderado (R$ bi)",
                color_discrete_map={
                    "FCF Conservador": "#EF553B",
                    "FCF Moderado": "#00CC96",
                },
                labels={"value": "R$ bi", "variable": "Cenário"},
                template="plotly_dark",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            wf1, wf2 = st.columns(2)
            with wf1:
                st.plotly_chart(
                    plot_ev_waterfall(result_cons, "Conservador", cons_scenario.get("wacc", 0.15)),
                    use_container_width=True,
                )
            with wf2:
                st.plotly_chart(
                    plot_ev_waterfall(result_mod, "Moderado", mod_scenario.get("wacc", 0.13)),
                    use_container_width=True,
                )

            # Fórmula exibida
            fcf_type = "FCFE" if is_financial else "FCFF"
            formula = (
                "FCFE = Net Income + D&A − CapEx − ΔNWC + Net Borrowing"
                if is_financial
                else "FCFF = EBIT × (1 − t) + D&A − CapEx − ΔNWC"
            )
            st.info(
                f"**Metodologia:** {fcf_type} · Perpetuidade de Gordon: "
                f"TV = FCF_{{t+1}} / (WACC − g) · {formula}"
            )

    # =========================================================================
    # ABA 4 — SENSIBILIDADE (HEATMAP)
    # =========================================================================
    with tab4:
        st.subheader("🌡️ Análise de Sensibilidade — Matriz de Preços Justos")

        base_inputs = st.session_state.get("base_inputs", {})
        mod_scenario = st.session_state.get("moderate_scenario", {})

        if not base_inputs or not mod_scenario:
            st.warning("⚠️ Configure as premissas na **Aba 2** antes de executar a sensibilidade.")
            st.stop()

        s1, s2 = st.columns([1, 2])
        with s1:
            sens_y = st.radio(
                "Eixo Y da Matriz",
                options=["g — Taxa de Perpetuidade", "Margem EBITDA"],
                index=0,
            )
            y_key = "g" if "g" in sens_y else "ebitda_margin"

            st.markdown(
                f"""
                **Configuração da Iteração:**
                - Eixo X (WACC): ±4pp em torno de {mod_scenario.get('wacc', 0):.1%}
                - Eixo Y ({'g' if y_key=='g' else 'Margem'}): ±3pp em torno do valor base
                - Passo: 1pp
                """
            )
            run_sens = st.button(
                "▶️ Calcular Sensibilidade", type="primary", use_container_width=True
            )

        with s2:
            st.info(
                "**Como ler o heatmap:**  \n"
                "Cada célula mostra o **Preço Justo (R$)** e a **Margem de Segurança (%)** "
                "em relação ao preço atual para aquela combinação de WACC e parâmetro Y.  \n"
                "🟢 Verde = Upside · 🔴 Vermelho = Downside"
            )

        if run_sens:
            with st.spinner("⏳ Calculando matriz (~15s)…"):
                try:
                    matrix = build_sensitivity_matrix(
                        base_revenue=base_inputs["base_revenue"],
                        base_ebitda_margin=base_inputs["base_ebitda_margin"],
                        base_da=base_inputs["base_da"],
                        base_capex_pct=base_inputs["base_capex_pct"],
                        base_nwc=base_inputs["base_nwc"],
                        tax_rate=base_inputs["tax_rate"],
                        revenue_growth=mod_scenario["revenue_growth"],
                        g_base=mod_scenario["g"],
                        wacc_base=mod_scenario["wacc"],
                        net_debt=base_inputs["net_debt"],
                        shares_outstanding=price_info.get("shares", 1.0),
                        current_price=price_info.get("price", 1.0),
                        is_financial=is_financial,
                        sensitivity_y=y_key,
                    )

                    cp = price_info.get("price", 1.0)
                    y_title = "g (Perpetuidade)" if y_key == "g" else "Margem EBITDA"
                    matrix.index.name = y_title

                    fig_heat = plot_sensitivity_heatmap(
                        matrix,
                        cp,
                        f"Sensibilidade: WACC × {y_title} → Preço Justo (R$) | {selected_ticker}",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # Tabela com gradiente
                    st.markdown("#### 📋 Tabela de Preços Justos (R$)")
                    st.dataframe(
                        matrix.style.background_gradient(
                            cmap="RdYlGn", axis=None
                        ).format("{:.2f}", na_rep="N/D"),
                        use_container_width=True,
                    )

                    # Análise dos quadrantes
                    n_total = int(matrix.notna().sum().sum())
                    n_above_15 = int((matrix > cp * 1.15).sum().sum())
                    n_above_0 = int((matrix > cp).sum().sum())
                    pct_15 = n_above_15 / max(n_total, 1) * 100
                    pct_0 = n_above_0 / max(n_total, 1) * 100

                    qa, qb, qc = st.columns(3)
                    qa.metric(
                        "✅ Quadrantes c/ MS > 15%",
                        f"{n_above_15}/{n_total}",
                        delta=f"{pct_15:.0f}% do total",
                    )
                    qb.metric(
                        "🟡 Quadrantes c/ Upside > 0",
                        f"{n_above_0}/{n_total}",
                        delta=f"{pct_0:.0f}% do total",
                    )
                    qc.metric(
                        "🔴 Quadrantes em Downside",
                        f"{n_total - n_above_0}/{n_total}",
                        delta=f"{100 - pct_0:.0f}% do total",
                        delta_color="inverse",
                    )

                except Exception as exc:
                    st.error(f"Erro na análise de sensibilidade: {exc}")
                    st.exception(exc)


if __name__ == "__main__":
    main()
