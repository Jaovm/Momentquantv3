import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v3.7 (Full Hybrid)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
BRAPI_TOKEN = "5gVedSQ928pxhFuTvBFPfr"

# ==============================================================================
# MÓDULO 1: DATA FETCHING (HÍBRIDO: BRAPI + YFINANCE FALLBACK)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados via YFinance."""
    t_list = list(tickers)
    # Garante benchmarks
    for bench in ['BOVA11.SA', 'DIVO11.SA']:
        if bench not in t_list:
            t_list.append(bench)
    
    try:
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False,
            threads=True
        )['Adj Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = data.dropna(axis=1, how='all')
        return data
    except Exception as e:
        st.error(f"Erro crítico ao baixar preços YF: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def fetch_fundamentals_hybrid(tickers: list, token: str) -> pd.DataFrame:
    """
    Busca fundamentos. Tenta Brapi primeiro. 
    Se faltar dados (P/VP, ROE, etc.), preenche com YFinance (.info).
    """
    # Limpa tickers para o formato Brapi (sem .SA)
    clean_tickers = [t.replace('.SA', '') for t in tickers if 'BOVA11' not in t and 'DIVO11' not in t]
    
    if not clean_tickers:
        return pd.DataFrame()

    fundamental_data = []
    
    # UI Feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_tickers = len(clean_tickers)
    
    def safe_float(val):
        if val is None or val == '' or str(val).lower() == 'nan': return np.nan
        try:
            return float(val)
        except:
            return np.nan

    def get_nested_val(item, keys_list):
        # Busca recursiva simples em JSON
        for key in keys_list:
            if key in item and item[key] is not None:
                return item[key]
        # Tenta subníveis comuns
        for module in ['defaultKeyStatistics', 'financialData', 'summaryProfile', 'price']:
            if module in item and isinstance(item[module], dict):
                for key in keys_list:
                    if key in item[module]:
                        return item[module][key]
        return None

    for i, ticker in enumerate(clean_tickers):
        # 1. TENTATIVA VIA BRAPI
        status_text.text(f"Analisando: {ticker} ({i+1}/{total_tickers}) - Fonte: Brapi...")
        
        # Inicializa variáveis com NaN
        price = market_cap = pe_ratio = p_vp = ev_ebitda = roe = net_margin = np.nan
        sector = 'Outros'
        
        url = f"https://brapi.dev/api/quote/{ticker}"
        params = {'token': token, 'fundamental': 'true'}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data_json = response.json()
                results = data_json.get('results', [])
                if results:
                    item = results[0]
                    
                    price = safe_float(item.get('regularMarketPrice'))
                    market_cap = safe_float(item.get('marketCap'))
                    sector = item.get('sector') or item.get('summaryProfile', {}).get('sector', 'Outros')
                    
                    pe_ratio = safe_float(get_nested_val(item, ['priceEarnings', 'trailingPE', 'peRatio']))
                    p_vp = safe_float(get_nested_val(item, ['priceToBook', 'priceToBookRatio', 'p_vp']))
                    ev_ebitda = safe_float(get_nested_val(item, ['enterpriseToEbitda', 'enterpriseValueToEBITDA', 'ev_ebitda']))
                    roe = safe_float(get_nested_val(item, ['returnOnEquity', 'roe']))
                    net_margin = safe_float(get_nested_val(item, ['profitMargin', 'netMargin', 'netProfitMargin']))

        except Exception:
            pass # Falha na Brapi, segue para Fallback silenciosamente

        # 2. VERIFICAÇÃO E FALLBACK VIA YFINANCE
        # Se P/VP ou ROE estiverem vazios, acionamos o Yahoo
        if np.isnan(p_vp) or np.isnan(roe) or np.isnan(ev_ebitda):
            status_text.text(f"Complementando dados: {ticker} via Yahoo Finance...")
            try:
                yf_t = yf.Ticker(f"{ticker}.SA")
                info = yf_t.info
                
                if np.isnan(price): price = info.get('currentPrice') or info.get('previousClose')
                if np.isnan(market_cap): market_cap = info.get('marketCap')
                if sector == 'Outros': sector = info.get('sector', 'Outros')
                
                if np.isnan(pe_ratio): pe_ratio = info.get('trailingPE')
                if np.isnan(p_vp): p_vp = info.get('priceToBook')
                if np.isnan(ev_ebitda): ev_ebitda = info.get('enterpriseToEbitda')
                if np.isnan(roe): roe = info.get('returnOnEquity')
                if np.isnan(net_margin): net_margin = info.get('profitMargins')
                
            except Exception:
                pass
        
        fundamental_data.append({
            'ticker': f"{ticker}.SA", # Normaliza saída para .SA
            'sector': sector,
            'currentPrice': price,
            'marketCap': market_cap,
            'PE': pe_ratio,
            'P_VP': p_vp,
            'EV_EBITDA': ev_ebitda,
            'ROE': roe,
            'Net_Margin': net_margin,
        })
        
        progress_bar.progress((i + 1) / total_tickers)
        time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(fundamental_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['ticker'])
        df = df.set_index('ticker')
        
        # Limpeza final de zeros
        cols_check = ['PE', 'P_VP', 'EV_EBITDA', 'ROE', 'Net_Margin']
        for col in cols_check:
            if col in df.columns:
                df[col] = df[col].replace([0, 0.0], np.nan)
            
    return df

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES
# ==============================================================================

def compute_residual_momentum_enhanced(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Residual Momentum (Blitz) com Volatility Scaling."""
    df = price_df.copy()
    monthly = df.resample('ME').last() 
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    
    regression_window = 36 
    
    for ticker in rets.columns:
        if ticker in ['BOVA11.SA', 'DIVO11.SA']: continue
        
        y_full = rets[ticker].tail(regression_window + skip)
        x_full = market.tail(regression_window + skip)
        
        if len(y_full) < 12: continue
            
        try:
            common_idx = y_full.index.intersection(x_full.index)
            y_full = y_full.loc[common_idx]
            x_full = x_full.loc[common_idx]

            X = sm.add_constant(x_full.values)
            model = sm.OLS(y_full.values, X).fit()
            residuals = pd.Series(model.resid, index=y_full.index)
            
            resid_12m = residuals.iloc[-(12 + skip) : -skip]
            
            if len(resid_12m) == 0:
                scores[ticker] = 0
                continue

            raw_momentum = resid_12m.sum()
            resid_vol = residuals.std()
            
            if resid_vol == 0:
                scores[ticker] = 0
            else:
                scores[ticker] = raw_momentum / resid_vol 
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_value_robust(fund_df: pd.DataFrame) -> pd.Series:
    """Composite Value Score."""
    scores = pd.DataFrame(index=fund_df.index)
    
    def invert_metric(series):
        return 1.0 / series.replace(0, np.nan)

    if 'PE' in fund_df: scores['Earnings_Yield'] = invert_metric(fund_df['PE'])
    if 'P_VP' in fund_df: scores['Book_Yield'] = invert_metric(fund_df['P_VP'])
    if 'EV_EBITDA' in fund_df: scores['EBITDA_Yield'] = invert_metric(fund_df['EV_EBITDA'])

    if scores.empty or scores.dropna(how='all').empty:
        return pd.Series(0, index=fund_df.index, name="Value_Score")

    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        if filled.std() > 0:
            scores[col] = (filled - filled.mean()) / filled.std()
        else:
            scores[col] = 0

    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Composite Quality Score."""
    scores = pd.DataFrame(index=fund_df.index)
    
    if 'ROE' in fund_df: scores['ROE'] = fund_df['ROE']
    if 'Net_Margin' in fund_df: scores['Margin'] = fund_df['Net_Margin']

    if scores.empty or scores.dropna(how='all').empty:
        return pd.Series(0, index=fund_df.index, name="Quality_Score")
    
    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        if filled.std() > 0:
            scores[col] = (filled - filled.mean()) / filled.std()
        else:
            scores[col] = 0
            
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: MATEMÁTICA E MÉTRICAS
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: return series - median 
    z = (series - median) / (mad * 1.4826) 
    return z.clip(-3, 3) 

def calculate_advanced_metrics(prices_series: pd.Series, risk_free_rate_annual: float = 0.10):
    if prices_series.empty or len(prices_series) < 2:
        return {}
    
    daily_rets = prices_series.pct_change().dropna()
    if daily_rets.empty: return {}
    
    total_ret = (prices_series.iloc[-1] / prices_series.iloc[0]) - 1
    days = (prices_series.index[-1] - prices_series.index[0]).days
    cagr = (1 + total_ret)**(365/days) - 1 if days > 0 else 0
    vol_ann = daily_rets.std() * np.sqrt(252)
    
    rf_daily = (1 + risk_free_rate_annual)**(1/252) - 1
    excess_rets = daily_rets - rf_daily
    sharpe = (excess_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    
    downside_rets = excess_rets[excess_rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (excess_rets.mean() * 252) / downside_std if (downside_std > 0 and not np.isnan(downside_std)) else 0
    
    cum_rets = (1 + daily_rets).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    ulcer_index = np.sqrt((drawdown**2).mean())
    
    return {
        'Retorno Total': total_ret,
        'CAGR': cagr,
        'Volatilidade': vol_ann,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_dd,
        'Ulcer Index': ulcer_index
    }

# ==============================================================================
# MÓDULO 4: SIMULAÇÃO MONTE CARLO
# ==============================================================================

def run_monte_carlo(initial_balance, monthly_contrib, mu_annual, sigma_annual, years, simulations=1000):
    if np.isnan(mu_annual) or np.isnan(sigma_annual):
        return pd.DataFrame()
        
    months = int(years * 12)
    dt = 1/12
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    
    if sigma_annual == 0: sigma_annual = 0.01
        
    shock = sigma_annual * np.sqrt(dt) * np.random.normal(0, 1, (months, simulations))
    monthly_returns = np.exp(drift + shock) - 1
    
    portfolio_paths = np.zeros((months + 1, simulations))
    portfolio_paths[0] = initial_balance
    
    for t in range(1, months + 1):
        portfolio_paths[t] = portfolio_paths[t-1] * (1 + monthly_returns[t-1]) + monthly_contrib
        
    percentiles = np.percentile(portfolio_paths, [5, 50, 95], axis=1)
    dates = [datetime.now() + timedelta(days=30*i) for i in range(months + 1)]
    
    return pd.DataFrame({
        'Pessimista (5%)': percentiles[0],
        'Base (50%)': percentiles[1],
        'Otimista (95%)': percentiles[2]
    }, index=dates)

# ==============================================================================
# MÓDULO 5: BACKTEST & ENGINE
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: bool = False):
    """Constrói pesos."""
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    if vol_target:
        valid_sel = [s for s in selected if s in prices.columns]
        if not valid_sel: return pd.Series()
        
        recent_rets = prices[valid_sel].pct_change().tail(63) 
        vols = recent_rets.std() * (252**0.5)
        vols = vols.replace(0, 1e-6)
        raw_weights_inv = 1 / vols
        
        if raw_weights_inv.sum() == 0:
            weights = pd.Series(1.0/len(valid_sel), index=valid_sel)
        else:
            weights = raw_weights_inv / raw_weights_inv.sum() 
    else:
        weights = pd.Series(1.0/len(selected), index=selected)
    return weights.sort_values(ascending=False)

def run_dca_backtest_robust(all_prices: pd.DataFrame, top_n: int, dca_amount: float, use_vol_target: bool, start_date: datetime, end_date: datetime):
    """Backtest Robusto."""
    all_prices = all_prices.ffill()
    dca_start = start_date + timedelta(days=30)
    
    market_calendar = pd.Series(all_prices.index, index=all_prices.index)
    dates_series = market_calendar.loc[dca_start:end_date].resample('MS').first()
    dates = dates_series.dropna().tolist()

    if not dates or len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    portfolio_value = pd.Series(0.0, index=all_prices.index)
    portfolio_holdings = {} 
    monthly_transactions = []
    cash = 0.0 

    for i, month_start in enumerate(dates):
        # 1. Definição das janelas
        eval_date = month_start - timedelta(days=1)
        mom_start = month_start - timedelta(days=365*3) 
        
        prices_historical = all_prices.loc[:eval_date]
        prices_window = prices_historical.loc[mom_start:]
        
        if prices_window.empty: continue

        # 2. Screening
        res_mom = compute_residual_momentum_enhanced(prices_window, lookback=12, skip=1)
        
        if res_mom.empty:
            continue
            
        df_rank = pd.DataFrame(index=res_mom.index)
        df_rank['Score'] = robust_zscore(res_mom)
        df_rank = df_rank.sort_values('Score', ascending=False)
        
        # 3. Pesos
        risk_window = prices_historical.tail(90)
        target_weights = construct_portfolio(df_rank, risk_window, top_n, use_vol_target)
        
        # 4. Execução
        try:
            if month_start not in all_prices.index:
                next_days = all_prices.index[all_prices.index > month_start]
                if next_days.empty: break
                exec_date = next_days[0]
            else:
                exec_date = month_start
                
            current_date_prices = all_prices.loc[exec_date]
        except KeyError:
            continue

        current_portfolio_val_mtm = cash
        for t, qtd in portfolio_holdings.items():
            if t in current_date_prices and not np.isnan(current_date_prices[t]):
                current_portfolio_val_mtm += qtd * current_date_prices[t]
        
        total_portfolio_val = current_portfolio_val_mtm + dca_amount
        
        new_holdings = {}
        
        for ticker, weight in target_weights.items():
            if ticker in current_date_prices and not np.isnan(current_date_prices[ticker]):
                price = current_date_prices[ticker]
                if price > 0:
                    alloc_val = total_portfolio_val * weight
                    qty = alloc_val / price
                    new_holdings[ticker] = qty
                    
                    monthly_transactions.append({
                        'Date': exec_date,
                        'Ticker': ticker,
                        'Action': 'Rebalance/Buy',
                        'Price': price,
                        'Weight': weight
                    })
        
        portfolio_holdings = new_holdings
        
        # 5. MTM
        next_rebalance = dates[i+1] if i < len(dates)-1 else end_date
        valid_end = min(next_rebalance, all_prices.index[-1])
        
        if exec_date > valid_end: continue
            
        valuation_dates = all_prices.loc[exec_date:valid_end].index
        
        for d in valuation_dates:
            val = 0
            for t, q in portfolio_holdings.items():
                p = all_prices.at[d, t]
                if not np.isnan(p):
                    val += q * p
            portfolio_value[d] = val

    portfolio_value = portfolio_value[portfolio_value > 0].sort_index()
    equity_curve = pd.DataFrame({'Strategy_DCA': portfolio_value})
    transactions_df = pd.DataFrame(monthly_transactions)
    final_holdings = portfolio_holdings 

    return equity_curve, transactions_df, final_holdings

def run_benchmark_dca(price_series: pd.Series, dates: list, dca_amount: float):
    """Simula DCA Benchmark."""
    if price_series.empty:
        return pd.Series()
    
    price_series = price_series.dropna()
    
    df_flow = pd.DataFrame(index=price_series.index)
    df_flow['Price'] = price_series
    df_flow['Units'] = 0.0
    
    sorted_dates = sorted(dates)
    
    for d in sorted_dates:
        idx_loc = price_series.index.asof(d)
        if idx_loc is not None:
            price = price_series.loc[idx_loc]
            if price > 0:
                buy_units = dca_amount / price
                if idx_loc in df_flow.index:
                    df_flow.at[idx_loc, 'Add_Units'] = buy_units

    df_flow['Add_Units'] = df_flow.get('Add_Units', pd.Series(0, index=df_flow.index)).fillna(0)
    df_flow['Cumulative_Units'] = df_flow['Add_Units'].cumsum()
    
    equity_curve = df_flow['Cumulative_Units'] * df_flow['Price']
    
    return equity_curve[equity_curve > 0]

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Pro v3.7 (Full Hybrid)")
    st.markdown("""
    **Otimização Multifator Institucional**
    * **Motor Híbrido:** Prioriza API Brapi.dev, mas usa Yahoo Finance para preencher dados faltantes (P/VP, ROE).
    * **Aviso:** O carregamento inicial pode demorar devido ao sistema de redundância.
    """)

    st.sidebar.header("1. Universo e Dados")
    default_univ = "ITUB3, TOTS3, MDIA3, TAEE3, BBSE3, WEGE3, PSSA3, EGIE3, B3SA3, VIVT3, AGRO3, PRIO3, BBAS3, BPAC11, SBSP3, SAPR4, CMIG3, UNIP6, FRAS3, CPFE3"
    ticker_input = st.sidebar.text_area("Tickers (Brapi Format - Sem .SA)", default_univ, height=100)
    raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    yf_tickers = [f"{t}.SA" for t in raw_tickers]
    
    st.sidebar.header("2. Pesos (Ranking Atual)")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_val = st.sidebar.slider("Value (P/L, P/VP, EBITDA)", 0.0, 1.0, 0.40)
    w_qual = st.sidebar.slider("Quality (ROE, Margem)", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Parâmetros de Gestão")
    top_n = st.sidebar.number_input("Número de Ativos", 4, 30, 10)
    use_vol_target = st.sidebar.checkbox("Risk Parity (Inv Vol)", True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("4. Backtest & Monte Carlo")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 100000, 2000)
    dca_years = st.sidebar.slider("Anos de Histórico", 2, 10, 5)
    mc_years = st.sidebar.slider("Projeção Futura (Anos)", 1, 20, 5)
    
    run_btn = st.sidebar.button("🚀 Executar Análise Institucional", type="primary")

    if run_btn:
        if not raw_tickers:
            st.error("Insira pelo menos um ticker.")
            return

        with st.status("Processando Pipeline Quantitativo...", expanded=True) as status:
            end_date = datetime.now()
            start_date_total = end_date - timedelta(days=365 * (dca_years + 3)) 
            start_date_backtest = end_date - timedelta(days=365 * dca_years)

            # 1. Dados de Preço
            status.write("📥 Baixando Histórico de Preços (YFinance)...")
            prices = fetch_price_data(yf_tickers, start_date_total, end_date)
            
            if prices.empty:
                st.error("Falha ao baixar preços.")
                status.update(label="Erro", state="error")
                return

            # 2. Dados Fundamentais (HÍBRIDO)
            status.write("🔍 Consultando Fundamentos (Híbrido: Brapi + Yahoo Fallback)...")
            fundamentals = fetch_fundamentals_hybrid(raw_tickers, BRAPI_TOKEN)
            
            if not fundamentals.empty:
                status.write(f"✅ Fundamentos carregados para {len(fundamentals)} ativos.")
            else:
                status.write("⚠️ Atenção: Falha crítica nos fundamentos. Usando apenas Momentum.")

            # 3. Cálculo do RANKING ATUAL
            status.write("🧮 Calculando Scores Atuais...")
            curr_mom = compute_residual_momentum_enhanced(prices)
            
            if not fundamentals.empty:
                curr_val = compute_value_robust(fundamentals)
                curr_qual = compute_quality_score(fundamentals)
            else:
                curr_val = pd.Series(0, index=prices.columns)
                curr_qual = pd.Series(0, index=prices.columns)

            df_master = pd.DataFrame(index=prices.columns)
            df_master['Res_Mom'] = curr_mom
            df_master['Value'] = curr_val
            df_master['Quality'] = curr_qual
            
            if not fundamentals.empty and 'sector' in fundamentals.columns:
                df_master['Sector'] = fundamentals['sector']
                
            df_master.dropna(thresh=1, inplace=True)

            cols_map = {'Res_Mom': w_rm, 'Value': w_val, 'Quality': w_qual}
            df_master['Composite_Score'] = 0.0
            
            for col, weight in cols_map.items():
                if col in df_master.columns:
                    z = robust_zscore(df_master[col])
                    df_master[f'{col}_Z'] = z
                    df_master['Composite_Score'] += z * weight
            
            df_master = df_master.sort_values('Composite_Score', ascending=False)

            # 4. Execução do BACKTEST
            status.write("⚙️ Rodando Backtest Robusto...")
            dca_curve, dca_transactions, dca_holdings = run_dca_backtest_robust(
                prices, top_n, dca_amount, use_vol_target, start_date_backtest, end_date
            )

            status.update(label="Análise Concluída!", state="complete", expanded=False)

        # ==============================================================================
        # DASHBOARD & BENCHMARKS (RESTAURADO)
        # ==============================================================================
        
        bench_curves = {}
        if not dca_transactions.empty:
            dca_dates = sorted(list(set(pd.to_datetime(dca_transactions['Date']).tolist())))
        else:
            dca_dates = []

        if dca_dates:
            for bench_ticker in ['BOVA11.SA', 'DIVO11.SA']:
                if bench_ticker in prices.columns:
                    bench_curve = run_benchmark_dca(prices[bench_ticker], dca_dates, dca_amount)
                    common_idx = dca_curve.index.intersection(bench_curve.index)
                    if not common_idx.empty:
                        bench_curves[bench_ticker] = bench_curve.loc[common_idx]

        tab1, tab2, tab6, tab3, tab4, tab5 = st.tabs([
            "🏆 Ranking Atual", 
            "📈 Performance DCA", 
            "🆚 Comparativo Benchmarks",
            "💰 Histórico & Custódia",
            "🔮 Monte Carlo", 
            "📋 Dados Brutos"
        ])

        with tab1:
            st.subheader("🎯 Carteira Recomendada (Hoje)")
            
            top_picks = df_master.head(top_n).copy()
            latest_prices = prices.iloc[-1]
            top_picks['Preço Atual'] = latest_prices.reindex(top_picks.index)
            
            risk_window = prices.tail(90)
            sug_weights = construct_portfolio(top_picks, risk_window, top_n, use_vol_target)
            
            top_picks['Peso (%)'] = (sug_weights * 100)
            top_picks['Alocação (R$)'] = (sug_weights * dca_amount)
            top_picks['Qtd Sugerida'] = (top_picks['Alocação (R$)'] / top_picks['Preço Atual'])
            
            cols_show = ['Sector', 'Preço Atual', 'Composite_Score', 'Peso (%)', 'Alocação (R$)', 'Qtd Sugerida', 'Value', 'Quality']
            cols_final = [c for c in cols_show if c in top_picks.columns]
            
            display_df = top_picks[cols_final].style.format({
                'Preço Atual': 'R$ {:.2f}',
                'Composite_Score': '{:.2f}',
                'Value': '{:.2f}',
                'Quality': '{:.2f}',
                'Peso (%)': '{:.1f}%',
                'Alocação (R$)': 'R$ {:.0f}',
                'Qtd Sugerida': '{:.0f}'
            }).background_gradient(subset=['Composite_Score'], cmap='Greens')
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.plotly_chart(px.pie(values=sug_weights, names=sug_weights.index, title="Alocação Sugerida"), use_container_width=True)
            with col_chart2:
                if 'Sector' in top_picks.columns:
                    st.plotly_chart(px.pie(top_picks, names='Sector', values='Peso (%)', title="Exposição Setorial"), use_container_width=True)

        with tab2:
            st.subheader("Simulação de Acumulação (DCA)")
            if not dca_curve.empty:
                end_val = dca_curve.iloc[-1,0]
                unique_months = pd.to_datetime(dca_transactions['Date']).dt.to_period('M').nunique()
                total_invested_real = unique_months * dca_amount
                
                profit = end_val - total_invested_real
                roi = (profit / total_invested_real) if total_invested_real > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Patrimônio Final", f"R$ {end_val:,.2f}")
                m2.metric("Total Investido", f"R$ {total_invested_real:,.2f}")
                m3.metric("Lucro Líquido", f"R$ {profit:,.2f}", delta=f"{roi:.1%}")
                
                fig = px.line(dca_curve, title="Curva de Patrimônio (Estratégia)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Análise de Risco")
                metrics = calculate_advanced_metrics(dca_curve['Strategy_DCA'])
                st.json(metrics)

        with tab6:
            st.subheader("🆚 Estratégia vs Benchmarks")
            if not dca_curve.empty and bench_curves:
                df_compare = dca_curve.copy()
                for b_name, b_series in bench_curves.items():
                    df_compare[b_name] = b_series
                
                df_compare = df_compare.ffill().dropna()
                fig_comp = px.line(df_compare, title="Evolução Patrimonial Comparativa")
                st.plotly_chart(fig_comp, use_container_width=True)
                
                comp_metrics = []
                m_strat = calculate_advanced_metrics(df_compare['Strategy_DCA'])
                m_strat['Asset'] = '🚀 Estratégia'
                m_strat['Saldo Final'] = df_compare['Strategy_DCA'].iloc[-1]
                comp_metrics.append(m_strat)
                
                for b_name in bench_curves.keys():
                    if b_name in df_compare.columns:
                        m_bench = calculate_advanced_metrics(df_compare[b_name])
                        m_bench['Asset'] = b_name
                        m_bench['Saldo Final'] = df_compare[b_name].iloc[-1]
                        comp_metrics.append(m_bench)
                
                df_comp_metrics = pd.DataFrame(comp_metrics).set_index('Asset')
                cols_order = ['Saldo Final', 'Retorno Total', 'CAGR', 'Volatilidade', 'Sharpe', 'Max Drawdown']
                
                st.dataframe(
                    df_comp_metrics[cols_order].style.format({
                        'Saldo Final': 'R$ {:,.2f}',
                        'Retorno Total': '{:.1%}',
                        'CAGR': '{:.1%}',
                        'Volatilidade': '{:.1%}',
                        'Sharpe': '{:.2f}',
                        'Max Drawdown': '{:.1%}'
                    }).highlight_max(subset=['Saldo Final'], color='#d4edda'),
                    use_container_width=True
                )
            else:
                st.warning("Dados insuficientes para comparação ou período muito curto.")

        with tab3:
            col_h1, col_h2 = st.columns([1, 1])
            with col_h1:
                st.subheader("💰 Posição Final (Backtest)")
                if dca_holdings:
                    final_df = pd.DataFrame.from_dict(dca_holdings, orient='index', columns=['Qtd'])
                    last_date_idx = dca_curve.index[-1]
                    if last_date_idx in prices.index:
                        last_prices = prices.loc[last_date_idx]
                        final_df['Preço Fechamento'] = last_prices.reindex(final_df.index)
                        final_df['Valor Total (R$)'] = final_df['Qtd'] * final_df['Preço Fechamento']
                        total_nav = final_df['Valor Total (R$)'].sum()
                        final_df['Peso (%)'] = (final_df['Valor Total (R$)'] / total_nav) * 100
                        final_df = final_df.sort_values('Peso (%)', ascending=False)
                        
                        st.dataframe(final_df.style.format({'Qtd': '{:.0f}', 'Preço Fechamento': 'R$ {:.2f}', 'Valor Total (R$)': 'R$ {:,.2f}', 'Peso (%)': '{:.1f}%'}), use_container_width=True)
                        st.metric("Patrimônio em Custódia", f"R$ {total_nav:,.2f}")
                else:
                    st.info("Nenhuma posição mantida.")

            with col_h2:
                st.subheader("📊 Alocação Final")
                if dca_holdings:
                     st.plotly_chart(px.pie(final_df, values='Valor Total (R$)', names=final_df.index, hole=0.4), use_container_width=True)

            st.divider()
            if not dca_transactions.empty:
                st.subheader("Histórico de Transações")
                st.dataframe(pd.DataFrame(dca_transactions).sort_values('Date', ascending=False), use_container_width=True)

        with tab4:
            st.subheader("Projeção Probabilística")
            if not dca_curve.empty:
                daily_rets = dca_curve['Strategy_DCA'].pct_change().dropna()
                if not daily_rets.empty:
                    mu = daily_rets.mean() * 252
                    sigma = daily_rets.std() * np.sqrt(252)
                    sim_df = run_monte_carlo(dca_curve.iloc[-1,0], dca_amount, mu, sigma, mc_years)
                    if not sim_df.empty:
                        st.plotly_chart(px.line(sim_df, title=f"Cone de Probabilidade - {mc_years} Anos"), use_container_width=True)
                    else:
                        st.warning("Erro ao calcular Monte Carlo (dados insuficientes).")

        with tab5:
            st.subheader("Dados Fundamentais (Brapi + YF)")
            if not fundamentals.empty:
                st.dataframe(fundamentals)
                st.caption("Nota: Dados obtidos via Brapi.dev com fallback automático para Yahoo Finance em caso de lacunas (P/VP, ROE).")
            else:
                st.error("Falha na recuperação de fundamentos. Verifique o token ou a conexão.")

if __name__ == "__main__":
    main()
