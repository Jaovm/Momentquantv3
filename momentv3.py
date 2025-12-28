import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import requests
from datetime import datetime, timedelta
import time

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v3.6 (Hybrid Fix)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Token Brapi (Mantenha o seu token)
BRAPI_TOKEN = "5gVedSQ928pxhFuTvBFPfr"

# ==============================================================================
# MÓDULO 1: DATA FETCHING (HÍBRIDO: BRAPI + YFINANCE FALLBACK)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados via YFinance."""
    t_list = list(tickers)
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
        
        # Flag para saber se precisaremos do Yahoo
        needs_fallback = False
        
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
                
                # Preenche o que estiver faltando (prioriza Brapi se existir, senão usa YF)
                if np.isnan(price): price = info.get('currentPrice') or info.get('previousClose')
                if np.isnan(market_cap): market_cap = info.get('marketCap')
                if sector == 'Outros': sector = info.get('sector', 'Outros')
                
                if np.isnan(pe_ratio): pe_ratio = info.get('trailingPE')
                if np.isnan(p_vp): p_vp = info.get('priceToBook')
                if np.isnan(ev_ebitda): ev_ebitda = info.get('enterpriseToEbitda')
                if np.isnan(roe): roe = info.get('returnOnEquity')
                if np.isnan(net_margin): net_margin = info.get('profitMargins')
                
            except Exception as e:
                # Se falhar no Yahoo também, paciência
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
        # Delay pequeno para não travar APIs
        time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(fundamental_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['ticker'])
        df = df.set_index('ticker')
        
        # Limpeza final de zeros que deveriam ser NaN
        cols_check = ['PE', 'P_VP', 'EV_EBITDA', 'ROE', 'Net_Margin']
        for col in cols_check:
            if col in df.columns:
                df[col] = df[col].replace([0, 0.0], np.nan)
            
    return df

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Mantido igual, apenas garantindo robustez)
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
    scores = pd.DataFrame(index=fund_df.index)
    
    def invert_metric(series):
        # Trata NaN e zeros antes de inverter
        s = series.replace(0, np.nan)
        return 1.0 / s

    if 'PE' in fund_df: scores['Earnings_Yield'] = invert_metric(fund_df['PE'])
    if 'P_VP' in fund_df: scores['Book_Yield'] = invert_metric(fund_df['P_VP'])
    if 'EV_EBITDA' in fund_df: scores['EBITDA_Yield'] = invert_metric(fund_df['EV_EBITDA'])

    if scores.empty or scores.dropna(how='all').empty:
        return pd.Series(0, index=fund_df.index, name="Value_Score")

    # Z-Score robusto por coluna
    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        if filled.std() > 0:
            scores[col] = (filled - filled.mean()) / filled.std()
        else:
            scores[col] = 0

    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
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

def robust_zscore(series: pd.Series) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: return series - median 
    z = (series - median) / (mad * 1.4826) 
    return z.clip(-3, 3) 

# ==============================================================================
# MÓDULO 3 E 4: BACKTEST E METRICAS (Mantidos)
# ==============================================================================

def calculate_advanced_metrics(prices_series: pd.Series):
    if prices_series.empty or len(prices_series) < 2: return {}
    daily_rets = prices_series.pct_change().dropna()
    total_ret = (prices_series.iloc[-1] / prices_series.iloc[0]) - 1
    days = (prices_series.index[-1] - prices_series.index[0]).days
    cagr = (1 + total_ret)**(365/days) - 1 if days > 0 else 0
    vol_ann = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    max_dd = ((1 + daily_rets).cumprod() / (1 + daily_rets).cumprod().cummax() - 1).min()
    
    return {'CAGR': cagr, 'Volatilidade': vol_ann, 'Sharpe': sharpe, 'Max Drawdown': max_dd}

def run_monte_carlo(initial_balance, monthly_contrib, mu_annual, sigma_annual, years, simulations=1000):
    if np.isnan(mu_annual) or np.isnan(sigma_annual): return pd.DataFrame()
    months = int(years * 12)
    dt = 1/12
    monthly_returns = np.exp((mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * np.random.normal(0, 1, (months, simulations))) - 1
    portfolio_paths = np.zeros((months + 1, simulations))
    portfolio_paths[0] = initial_balance
    for t in range(1, months + 1):
        portfolio_paths[t] = portfolio_paths[t-1] * (1 + monthly_returns[t-1]) + monthly_contrib
    percentiles = np.percentile(portfolio_paths, [5, 50, 95], axis=1)
    dates = [datetime.now() + timedelta(days=30*i) for i in range(months + 1)]
    return pd.DataFrame({'Pessimista': percentiles[0], 'Base': percentiles[1], 'Otimista': percentiles[2]}, index=dates)

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: bool = False):
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()
    if vol_target:
        valid_sel = [s for s in selected if s in prices.columns]
        if not valid_sel: return pd.Series()
        vols = prices[valid_sel].pct_change().tail(63).std() * (252**0.5)
        raw_weights_inv = 1 / vols.replace(0, 1e-6)
        weights = raw_weights_inv / raw_weights_inv.sum()
    else:
        weights = pd.Series(1.0/len(selected), index=selected)
    return weights.sort_values(ascending=False)

def run_dca_backtest_robust(all_prices, top_n, dca_amount, use_vol_target, start_date, end_date):
    all_prices = all_prices.ffill()
    dates = pd.date_range(start=start_date + timedelta(days=30), end=end_date, freq='MS')
    portfolio_value = pd.Series(0.0, index=all_prices.index)
    holdings = {}
    transactions = []
    
    for i, date in enumerate(dates):
        hist_prices = all_prices.loc[:date]
        if len(hist_prices) < 252: continue
        
        # Momento
        mom = compute_residual_momentum_enhanced(hist_prices.loc[date-timedelta(days=365*3):])
        if mom.empty: continue
        
        # Ranking simples baseado em Momento (No backtest histórico não temos Brapi do passado)
        # Assumimos que no backtest histórico usamos apenas Momentum Técnico, 
        # pois não temos base de dados fundamentalista point-in-time histórica gratuita.
        rank = mom.sort_values(ascending=False)
        target_weights = construct_portfolio(rank.to_frame(), hist_prices, top_n, use_vol_target)
        
        # Rebalanceamento
        try:
            curr_prices = all_prices.loc[date]
        except KeyError: continue # Feriado/Fim de semana
            
        current_val = sum(holdings.get(t,0) * curr_prices.get(t,0) for t in holdings) + dca_amount
        new_holdings = {}
        for ticker, w in target_weights.items():
            if ticker in curr_prices and not np.isnan(curr_prices[ticker]):
                new_holdings[ticker] = (current_val * w) / curr_prices[ticker]
                transactions.append({'Date': date, 'Ticker': ticker, 'Action': 'Buy', 'Price': curr_prices[ticker]})
        holdings = new_holdings
        
        # Mark to Market até próximo rebalanceamento
        next_date = dates[i+1] if i < len(dates)-1 else end_date
        period_idx = all_prices.loc[date:next_date].index
        for d in period_idx:
            portfolio_value[d] = sum(holdings.get(t,0) * all_prices.at[d, t] for t in holdings if t in all_prices.columns)
            
    return portfolio_value[portfolio_value > 0], pd.DataFrame(transactions), holdings

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Pro v3.6 (Brapi + YF Fallback)")
    st.markdown("""
    **Correção Aplicada:**
    * O sistema agora tenta pegar dados da Brapi.
    * Se faltar P/VP, ROE ou EV/EBITDA, ele busca automaticamente no **Yahoo Finance**.
    """)

    st.sidebar.header("1. Universo e Dados")
    default_univ = "ITUB4, VALE3, WEGE3, PRIO3, BBAS3, PETR4, RENT3, B3SA3, EQTL3, LREN3, RADL3, RAIL3, SUZB3, JBSS3, VIVT3, CMIG4, ELET3, BBSE3, GOAU4, TOTS3, MDIA3, TAEE11"
    ticker_input = st.sidebar.text_area("Tickers (Sem .SA)", default_univ, height=100)
    raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    yf_tickers = [f"{t}.SA" for t in raw_tickers]
    
    st.sidebar.header("2. Pesos do Ranking")
    w_rm = st.sidebar.slider("Momentum", 0.0, 1.0, 0.40)
    w_val = st.sidebar.slider("Value", 0.0, 1.0, 0.40)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Parâmetros")
    top_n = st.sidebar.number_input("Top N", 4, 30, 10)
    dca_amount = st.sidebar.number_input("Aporte Mensal", 100, 100000, 2000)
    
    if st.sidebar.button("🚀 Executar", type="primary"):
        with st.status("Executando Pipeline...", expanded=True) as status:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*5)

            # 1. Preços
            status.write("📥 Baixando Preços...")
            prices = fetch_price_data(yf_tickers, start_date, end_date)
            
            # 2. Fundamentos Híbridos
            status.write("🔍 Buscando Fundamentos (Brapi + Yahoo Backup)...")
            fundamentals = fetch_fundamentals_hybrid(raw_tickers, BRAPI_TOKEN)
            
            if fundamentals.empty:
                st.error("Não foi possível obter fundamentos.")
                return

            # 3. Ranking
            status.write("🧮 Calculando Ranking...")
            # Momentum
            curr_mom = compute_residual_momentum_enhanced(prices)
            # Value & Quality
            curr_val = compute_value_robust(fundamentals)
            curr_qual = compute_quality_score(fundamentals)

            # Consolidar
            df_master = pd.DataFrame(index=prices.columns)
            df_master['Res_Mom'] = curr_mom
            
            # Merge com fundamentos (garantindo index igual)
            df_master = df_master.join(curr_val).join(curr_qual).join(fundamentals[['sector', 'currentPrice']], how='left')
            
            # Cálculo Final Score
            for col, w in [('Res_Mom', w_rm), ('Value_Score', w_val), ('Quality_Score', w_qual)]:
                if col in df_master.columns:
                    df_master[f'{col}_Z'] = robust_zscore(df_master[col].fillna(0))
            
            df_master['Final_Score'] = (
                df_master.get('Res_Mom_Z', 0) * w_rm +
                df_master.get('Value_Score_Z', 0) * w_val +
                df_master.get('Quality_Score_Z', 0) * w_qual
            )
            
            df_master = df_master.sort_values('Final_Score', ascending=False).dropna(subset=['Final_Score'])
            
            # 4. Backtest (Simplificado para Momentum Histórico)
            status.write("⚙️ Rodando Backtest...")
            dca_curve, dca_trans, dca_holdings = run_dca_backtest_robust(prices, top_n, dca_amount, True, start_date, end_date)
            
            status.update(label="Concluído!", state="complete", expanded=False)

        # DASHBOARD
        tab1, tab2, tab3 = st.tabs(["🏆 Ranking Hoje", "📈 Backtest DCA", "📋 Dados Brutos"])
        
        with tab1:
            st.subheader("Sugestão de Carteira")
            top_picks = df_master.head(top_n).copy()
            
            # Pesos sugeridos (Inverse Volatility se possível)
            risk_window = prices.tail(60)
            weights = construct_portfolio(top_picks, risk_window, top_n, True)
            
            top_picks['Peso %'] = (weights * 100).map('{:.1f}%'.format)
            top_picks['Aporte R$'] = (weights * dca_amount).map('R$ {:,.2f}'.format)
            
            st.dataframe(
                top_picks[['sector', 'currentPrice', 'Final_Score', 'Peso %', 'Aporte R$', 'Value_Score', 'Quality_Score']]
                .style.background_gradient(subset=['Final_Score'], cmap='Greens'),
                use_container_width=True
            )
            
        with tab2:
            st.subheader("Performance Histórica (Simulação)")
            if not dca_curve.empty:
                st.plotly_chart(px.line(dca_curve, title="Patrimônio Acumulado"), use_container_width=True)
                metrics = calculate_advanced_metrics(dca_curve)
                c1, c2, c3 = st.columns(3)
                c1.metric("Saldo Final", f"R$ {dca_curve.iloc[-1]:,.2f}")
                c2.metric("CAGR", f"{metrics.get('CAGR',0):.1%}")
                c3.metric("Max Drawdown", f"{metrics.get('Max Drawdown',0):.1%}")

        with tab3:
            st.subheader("Dados Fundamentais Baixados")
            st.dataframe(fundamentals)
            st.caption("Nota: Células preenchidas via Yahoo Finance onde a Brapi falhou.")

if __name__ == "__main__":
    main()
