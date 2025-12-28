import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab: Enhanced Edition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING & MATH UTILS
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados via YFinance."""
    t_list = list(tickers)
    # Garante benchmarks
    if 'BOVA11.SA' not in t_list: t_list.append('BOVA11.SA')
    
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
        st.error(f"Erro crítico ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*12)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """Busca snapshots fundamentais (YFinance)."""
    data = []
    clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
    
    # Placeholder visual será gerenciado fora desta função para performance no cache
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            t_obj = yf.Ticker(t)
            info = t_obj.info
            
            # Tratamento de Setor
            sector = info.get('sector', 'Outros')
            if sector in ['Unknown', 'N/A', 'Outros'] and 'longName' in info:
                 name = str(info['longName']).lower()
                 if 'banco' in name or 'financeira' in name or 'seguridade' in name:
                     sector = 'Financial Services'
                 elif 'energia' in name or 'elétrica' in name:
                     sector = 'Utilities'
            
            data.append({
                'ticker': t,
                'sector': sector,
                'currentPrice': info.get('currentPrice', info.get('previousClose', np.nan)),
                'forwardPE': info.get('forwardPE', info.get('trailingPE', np.nan)),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan)
            })
        except:
            pass
        
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).set_index('ticker')

def calculate_advanced_metrics(prices_series: pd.Series, risk_free_rate_annual: float = 0.10):
    """Calcula métricas de performance avançadas (CAGR, Sortino, Drawdown)."""
    if prices_series.empty or len(prices_series) < 2:
        return {}
    
    daily_rets = prices_series.pct_change().dropna()
    if daily_rets.empty: return {}
    
    # Retorno Total e CAGR
    total_ret = (prices_series.iloc[-1] / prices_series.iloc[0]) - 1
    days = (prices_series.index[-1] - prices_series.index[0]).days
    cagr = (1 + total_ret)**(365/days) - 1 if days > 0 else 0
    
    # Volatilidade
    vol_ann = daily_rets.std() * np.sqrt(252)
    
    # Sharpe e Sortino
    rf_daily = (1 + risk_free_rate_annual)**(1/252) - 1
    excess_rets = daily_rets - rf_daily
    sharpe = (excess_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    
    downside_rets = excess_rets[excess_rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (excess_rets.mean() * 252) / downside_std if (downside_std > 0 and not np.isnan(downside_std)) else 0
    
    # Drawdown
    cum_rets = (1 + daily_rets).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'Retorno Total': total_ret,
        'CAGR': cagr,
        'Volatilidade': vol_ann,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_dd
    }

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto (usa Mediana e MAD para evitar outliers extremos)."""
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: return series - median 
    z = (series - median) / (mad * 1.4826) 
    return z.clip(-3, 3) 

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Residual Momentum (Alpha vs BOVA11)."""
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    window = lookback + skip
    
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        
        y = rets[ticker].tail(window)
        x = market.tail(window)
        
        if len(y) < window: continue
            
        try:
            # Alinha índices
            common_idx = y.index.intersection(x.index)
            y, x = y.loc[common_idx], x.loc[common_idx]
            
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            resid = model.resid[:-skip] # Remove o mês mais recente (skip)
            
            std_resid = np.std(resid)
            if std_resid == 0 or len(resid) < 3:
                scores[ticker] = 0
            else:
                scores[ticker] = np.sum(resid) / std_resid
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    """Crescimento (Growth) combinado."""
    metrics = ['earningsGrowth', 'revenueGrowth']
    temp_df = pd.DataFrame(index=fund_df.index)
    
    for m in metrics:
        if m in fund_df.columns:
            # Preenche NaN com mediana para não punir severamente falta de dados
            s = fund_df[m].fillna(fund_df[m].median())
            temp_df[m] = robust_zscore(s)
            
    if temp_df.empty: return pd.Series(0, index=fund_df.index)
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Value: Yields de Lucro e Patrimônio."""
    scores = pd.DataFrame(index=fund_df.index)
    
    # Inverte métricas (menor P/L é melhor -> maior Earnings Yield é melhor)
    if 'forwardPE' in fund_df: 
        scores['Earnings_Yield'] = fund_df['forwardPE'].apply(lambda x: 1/x if x > 0 else 0)
    if 'priceToBook' in fund_df: 
        scores['Book_Yield'] = fund_df['priceToBook'].apply(lambda x: 1/x if x > 0 else 0)
    
    # Normaliza
    for col in scores.columns:
        scores[col] = robust_zscore(scores[col])
        
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Quality: ROE, Margem, Baixa Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: scores['Low_Debt'] = -1 * fund_df['debtToEquity'] # Menor dívida é melhor
    
    for col in scores.columns:
        # Trata NaNs
        scores[col] = scores[col].fillna(scores[col].median())
        scores[col] = robust_zscore(scores[col])
        
    return scores.mean(axis=1).rename("Quality_Score")

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 3: SIMULAÇÃO E PORTFÓLIO
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    if vol_target is not None:
        valid_sel = [s for s in selected if s in prices.columns]
        if not valid_sel: return pd.Series()
        
        # Volatilidade recente (3 meses)
        recent_rets = prices[valid_sel].pct_change().tail(63)
        vols = recent_rets.std() * np.sqrt(252)
        vols = vols.replace(0, 1e-6)
        
        # Inverse Volatility Weighting
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
    else:
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights.sort_values(ascending=False)

def run_monte_carlo(current_balance, monthly_contrib, mu_annual, sigma_annual, years, simulations=1000):
    """Simulação Monte Carlo para projeção de patrimônio."""
    if np.isnan(mu_annual) or np.isnan(sigma_annual): return pd.DataFrame()
        
    months = int(years * 12)
    dt = 1/12
    # Drift ajustado geometricamente
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    
    # Choques aleatórios
    shock = sigma_annual * np.sqrt(dt) * np.random.normal(0, 1, (months, simulations))
    monthly_returns = np.exp(drift + shock) - 1
    
    # Caminhos
    portfolio_paths = np.zeros((months + 1, simulations))
    portfolio_paths[0] = current_balance
    
    for t in range(1, months + 1):
        portfolio_paths[t] = portfolio_paths[t-1] * (1 + monthly_returns[t-1]) + monthly_contrib
        
    # Percentis
    percentiles = np.percentile(portfolio_paths, [5, 50, 95], axis=1)
    dates = [datetime.now() + timedelta(days=30*i) for i in range(months + 1)]
    
    return pd.DataFrame({
        'Pessimista (5%)': percentiles[0],
        'Base (50%)': percentiles[1],
        'Otimista (95%)': percentiles[2]
    }, index=dates)

def run_dca_backtest(
    all_prices: pd.DataFrame, 
    all_fundamentals: pd.DataFrame, 
    factor_weights: dict, 
    top_n: int, 
    dca_amount: float, 
    use_vol_target: bool,
    use_sector_neutrality: bool, 
    start_date: datetime,
    end_date: datetime
):
    """Backtest Walk-Forward com Aportes (DCA)."""
    all_prices = all_prices.ffill() 
    dca_start = start_date + timedelta(days=30) 
    dates = all_prices.loc[dca_start:end_date].resample('MS').first().index.tolist()
    
    if not dates or len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    portfolio_value = pd.Series(0.0, index=all_prices.index)
    benchmark_value = pd.Series(0.0, index=all_prices.index)
    
    portfolio_holdings = {} 
    benchmark_holdings = {'BOVA11.SA': 0.0}
    monthly_transactions = []
    
    # Cache de fatores estáticos (Fundamentos) - na vida real variariam, aqui usamos snapshot recente
    # para simular o "Fator" funcionando, mas o Momentum varia.
    fund_mom = compute_fundamental_momentum(all_fundamentals)
    val_score = compute_value_score(all_fundamentals)
    qual_score = compute_quality_score(all_fundamentals)
    
    for i, month_start in enumerate(dates):
        # Janelas de Dados
        eval_date = month_start - timedelta(days=1)
        mom_start = month_start - timedelta(days=400) 
        
        prices_for_mom = all_prices.loc[mom_start:eval_date]
        prices_for_risk = all_prices.loc[(month_start - timedelta(days=90)):eval_date]
        
        # Recalcula Momentum Dinamicamente
        res_mom = compute_residual_momentum(prices_for_mom) if not prices_for_mom.empty else pd.Series(dtype=float)
        
        # Monta DataFrame de Fatores
        df_master = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_master['Res_Mom'] = res_mom
        df_master['Fund_Mom'] = fund_mom
        df_master['Value'] = val_score
        df_master['Quality'] = qual_score
        if 'sector' in all_fundamentals.columns: 
             df_master['Sector'] = all_fundamentals['sector']
        
        # Limpeza
        df_master.dropna(thresh=2, inplace=True)
        
        # Normalização e Pesos (Sector Neutrality)
        norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
        w_keys = {}
        
        if use_sector_neutrality and 'Sector' in df_master.columns and df_master['Sector'].nunique() > 1:
            for c in norm_cols:
                if c in df_master.columns:
                    new_col = f"{c}_Z_Sec"
                    # Z-Score por setor
                    df_master[new_col] = df_master.groupby('Sector')[c].transform(
                        lambda x: robust_zscore(x) if len(x) > 2 else (x - x.mean())/x.std() if x.std()>0 else 0
                    )
                    w_keys[new_col] = factor_weights.get(c, 0.0)
        else:
            for c in norm_cols:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    w_keys[new_col] = factor_weights.get(c, 0.0)

        # Ranking Final
        ranked_df = build_composite_score(df_master, w_keys)
        
        # Alocação de Pesos
        target_weights = construct_portfolio(ranked_df, prices_for_risk, top_n, 0.15 if use_vol_target else None)
        
        # Execução (Preços do dia do aporte)
        try:
            # Encontra o próximo dia útil válido
            valid_days = all_prices.loc[month_start:]
            if valid_days.empty: break
            rebal_date = valid_days.index[0]
            current_prices = valid_days.iloc[0]
        except:
            break
            
        # 1. Benchmark (Compra Simples)
        bova_price = current_prices.get('BOVA11.SA', np.nan)
        if not np.isnan(bova_price) and bova_price > 0:
            qtd_bova = dca_amount / bova_price
            benchmark_holdings['BOVA11.SA'] += qtd_bova
            
        # 2. Estratégia (Rebalanceamento + Aporte)
        # Calcula valor total atual da carteira
        current_nav = dca_amount # Dinheiro novo
        for t, q in portfolio_holdings.items():
            if t in current_prices and not np.isnan(current_prices[t]):
                current_nav += q * current_prices[t]
                
        # Define novas quantidades alvo
        new_holdings = {}
        for ticker, w in target_weights.items():
            if ticker in current_prices and not np.isnan(current_prices[ticker]):
                p = current_prices[ticker]
                if p > 0:
                    val_alloc = current_nav * w
                    new_holdings[ticker] = val_alloc / p
                    
                    monthly_transactions.append({
                        'Date': rebal_date,
                        'Ticker': ticker,
                        'Price': p,
                        'Weight': w,
                        'Action': 'Rebal/Buy'
                    })
        
        portfolio_holdings = new_holdings
        
        # Marcação a Mercado (MTM) até o próximo mês
        next_date = dates[i+1] if i < len(dates)-1 else end_date
        date_range = all_prices.loc[rebal_date:next_date].index
        
        for d in date_range:
            # Strategy Valuation
            val_strat = 0
            for t, q in portfolio_holdings.items():
                if t in all_prices.columns:
                    val_strat += all_prices.at[d, t] * q
            portfolio_value[d] = val_strat
            
            # Benchmark Valuation
            if 'BOVA11.SA' in all_prices.columns:
                benchmark_value[d] = all_prices.at[d, 'BOVA11.SA'] * benchmark_holdings['BOVA11.SA']
    
    # Consolidação
    portfolio_value = portfolio_value[portfolio_value > 0].sort_index()
    benchmark_value = benchmark_value[benchmark_value > 0].sort_index()
    
    # Alinha datas
    common = portfolio_value.index.intersection(benchmark_value.index)
    equity_curve = pd.DataFrame({
        'Strategy_DCA': portfolio_value.loc[common], 
        'BOVA11.SA_DCA': benchmark_value.loc[common]
    })
    
    return equity_curve, pd.DataFrame(monthly_transactions), portfolio_holdings

# ==============================================================================
# APP PRINCIPAL (STREAMLIT)
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Enhanced Edition")
    st.markdown("""
    **Otimização de Portfólio Multifator (Long-Only)**
    * **Motor Híbrido:** Momentum Residual + Qualidade + Valor.
    * **Gestão de Risco:** Volatilidade Inversa e Neutralidade Setorial (Opcional).
    """)

    # --- SIDEBAR ---
    st.sidebar.header("1. Universo de Ativos")
    default_univ = "ITUB3.SA, TOTS3.SA, WEGE3.SA, PRIO3.SA, VALE3.SA, PETR4.SA, BBAS3.SA, BPAC11.SA, EQTL3.SA, RENT3.SA, RADL3.SA, LREN3.SA, B3SA3.SA, VIVT3.SA, CMIG4.SA, ELET3.SA, SUZB3.SA, JBSS3.SA, CSAN3.SA, RDOR3.SA"
    ticker_input = st.sidebar.text_area("Tickers (.SA)", default_univ, height=120)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40, help="Força da tendência excluindo o beta de mercado.")
    w_fm = st.sidebar.slider("Fundamental Growth", 0.0, 1.0, 0.20, help="Crescimento de Lucros e Receita.")
    w_val = st.sidebar.slider("Value (P/L, P/VP)", 0.0, 1.0, 0.20, help="Preço atrativo em relação aos fundamentos.")
    w_qual = st.sidebar.slider("Quality (ROE, Margin)", 0.0, 1.0, 0.20, help="Rentabilidade e baixa dívida.")

    st.sidebar.header("3. Gestão de Portfólio")
    top_n = st.sidebar.number_input("Ativos na Carteira", 3, 20, 8)
    use_vol_target = st.sidebar.checkbox("Risk Parity (Vol Inversa)", True, help="Ativos mais voláteis recebem menos peso.")
    use_sector_neutrality = st.sidebar.checkbox("Neutralidade Setorial", False, help="Compara ativos apenas com pares do mesmo setor.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("4. Parâmetros de Backtest")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 50000, 2000)
    dca_years = st.sidebar.slider("Histórico (Anos)", 1, 10, 5)
    mc_years = st.sidebar.slider("Projeção Futura (Anos)", 1, 20, 5)
    
    run_btn = st.sidebar.button("🚀 Executar Estratégia", type="primary")

    if run_btn:
        if not tickers:
            st.error("Insira pelo menos um ticker.")
            return

        # --- EXECUÇÃO ---
        with st.status("Processando Pipeline Quantitativo...", expanded=True) as status:
            end_date = datetime.now()
            start_date_total = end_date - timedelta(days=365 * (dca_years + 2)) 
            
            # 1. Dados
            status.write("📥 Baixando cotações de fechamento ajustado...")
            prices = fetch_price_data(tickers, start_date_total, end_date)
            
            status.write("🔍 Analisando fundamentos (YFinance)...")
            fundamentals = fetch_fundamentals(tickers)
            
            if prices.empty or fundamentals.empty:
                status.update(label="Erro: Dados insuficientes.", state="error")
                st.error("Falha na obtenção de dados. Verifique os tickers.")
                return

            # 2. Cálculo de Fatores Atuais (Para Ranking de Hoje)
            status.write("🧮 Calculando Scores Multifatoriais...")
            
            res_mom = compute_residual_momentum(prices)
            fund_mom = compute_fundamental_momentum(fundamentals)
            val_score = compute_value_score(fundamentals)
            qual_score = compute_quality_score(fundamentals)

            df_now = pd.DataFrame(index=tickers)
            df_now['Residual Momentum'] = res_mom
            df_now['Growth'] = fund_mom
            df_now['Value'] = val_score
            df_now['Quality'] = qual_score
            
            if 'sector' in fundamentals.columns: df_now['Setor'] = fundamentals['sector']
            
            # Limpeza de tickers sem dados
            df_now.dropna(thresh=2, inplace=True)

            # Normalização e Score Final
            weights_map = {'Residual Momentum': w_rm, 'Growth': w_fm, 'Value': w_val, 'Quality': w_qual}
            w_keys_now = {}
            
            if use_sector_neutrality and 'Setor' in df_now.columns:
                for col in weights_map.keys():
                    if col in df_now.columns:
                        new_c = f"{col}_Z"
                        df_now[new_c] = df_now.groupby('Setor')[col].transform(robust_zscore)
                        w_keys_now[new_c] = weights_map[col]
            else:
                for col in weights_map.keys():
                    if col in df_now.columns:
                        new_c = f"{col}_Z"
                        df_now[new_c] = robust_zscore(df_now[col])
                        w_keys_now[new_c] = weights_map[col]
            
            final_rank = build_composite_score(df_now, w_keys_now)
            
            # Pesos Sugeridos Hoje
            sug_weights = construct_portfolio(final_rank, prices, top_n, 0.15 if use_vol_target else None)

            # 3. Backtest
            status.write("⚙️ Executando Backtest DCA (Walk-Forward)...")
            dca_curve, dca_trans, dca_holdings = run_dca_backtest(
                prices, fundamentals, weights_map, top_n, dca_amount, 
                use_vol_target, use_sector_neutrality, 
                end_date - timedelta(days=365*dca_years), end_date
            )

            status.update(label="Análise Concluída!", state="complete", expanded=False)

        # --- DASHBOARD ---
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Carteira Recomendada", 
            "📈 Performance DCA", 
            "🔮 Projeção Monte Carlo",
            "📊 Detalhes da Custódia",
            "📋 Dados Brutos"
        ])

        # TAB 1: RECOMENDAÇÃO ATUAL
        with tab1:
            st.subheader("Sugestão de Alocação (Atual)")
            col_top, col_act = st.columns([2, 1])
            
            with col_top:
                # Prepara Tabela Bonita
                if not sug_weights.empty:
                    display_df = final_rank.loc[sug_weights.index].copy()
                    display_df['Peso Sugerido'] = sug_weights
                    
                    # Dados auxiliares
                    last_prices = prices.iloc[-1]
                    display_df['Preço Atual'] = last_prices.reindex(display_df.index)
                    
                    # Cálculo Financeiro
                    display_df['Aporte (R$)'] = display_df['Peso Sugerido'] * dca_amount
                    display_df['Qtd Aprox.'] = (display_df['Aporte (R$)'] / display_df['Preço Atual']).fillna(0).astype(int)
                    
                    # Formatação
                    cols_final = ['Setor', 'Preço Atual', 'Composite_Score', 'Peso Sugerido', 'Aporte (R$)', 'Qtd Aprox.', 'Value', 'Quality']
                    view_df = display_df[[c for c in cols_final if c in display_df.columns]]
                    
                    st.dataframe(
                        view_df.style.format({
                            'Preço Atual': 'R$ {:.2f}',
                            'Composite_Score': '{:.2f}',
                            'Peso Sugerido': '{:.1%}',
                            'Aporte (R$)': 'R$ {:,.2f}',
                            'Value': '{:.2f}',
                            'Quality': '{:.2f}'
                        }).background_gradient(subset=['Composite_Score'], cmap='Greens'),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.warning("Não foi possível gerar sugestões com os dados atuais.")

            with col_act:
                if not sug_weights.empty:
                    st.markdown("### Distribuição")
                    fig_pie = px.pie(names=sug_weights.index, values=sug_weights.values, hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    if 'Setor' in display_df.columns:
                        st.markdown("### Exposição Setorial")
                        sector_weights = display_df.groupby('Setor')['Peso Sugerido'].sum()
                        st.plotly_chart(px.bar(sector_weights, orientation='h', color=sector_weights.index), use_container_width=True)

        # TAB 2: BACKTEST DCA
        with tab2:
            if not dca_curve.empty:
                # KPIs Principais
                start_val = 0 # Considerando DCA
                end_val = dca_curve['Strategy_DCA'].iloc[-1]
                total_invested = len(dca_trans['Date'].unique()) * dca_amount
                profit = end_val - total_invested
                roi = (profit / total_invested) if total_invested > 0 else 0
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Patrimônio Final", f"R$ {end_val:,.2f}")
                kpi2.metric("Total Investido", f"R$ {total_invested:,.2f}")
                kpi3.metric("Lucro Líquido", f"R$ {profit:,.2f}", delta=f"{roi:.1%}")
                
                # Métricas Avançadas
                adv_metrics = calculate_advanced_metrics(dca_curve['Strategy_DCA'])
                bench_metrics = calculate_advanced_metrics(dca_curve['BOVA11.SA_DCA'])
                
                kpi4.metric("Sharpe (Estratégia)", f"{adv_metrics.get('Sharpe',0):.2f}", delta=f"vs {bench_metrics.get('Sharpe',0):.2f}")

                # Gráfico
                st.plotly_chart(px.line(dca_curve, title="Curva de Patrimônio Comparativa"), use_container_width=True)
                
                # Tabela de Comparação
                st.subheader("Análise de Risco e Retorno")
                comp_data = {
                    'Métrica': ['CAGR', 'Volatilidade', 'Sharpe', 'Sortino', 'Max Drawdown', 'Calmar'],
                    'Estratégia': [
                        adv_metrics.get('CAGR'), adv_metrics.get('Volatilidade'), adv_metrics.get('Sharpe'),
                        adv_metrics.get('Sortino'), adv_metrics.get('Max Drawdown'), adv_metrics.get('Calmar')
                    ],
                    'Benchmark (BOVA11)': [
                        bench_metrics.get('CAGR'), bench_metrics.get('Volatilidade'), bench_metrics.get('Sharpe'),
                        bench_metrics.get('Sortino'), bench_metrics.get('Max Drawdown'), bench_metrics.get('Calmar')
                    ]
                }
                df_comp = pd.DataFrame(comp_data).set_index('Métrica')
                st.table(df_comp.style.format("{:.2%}"))
                
                # Drawdown Chart
                peak = dca_curve['Strategy_DCA'].cummax()
                drawdown = (dca_curve['Strategy_DCA'] - peak) / peak
                st.plotly_chart(px.area(drawdown, title="Underwater Plot (Drawdown)", color_discrete_sequence=['red']), use_container_width=True)
                
            else:
                st.warning("Dados insuficientes para gerar curva DCA.")

        # TAB 3: MONTE CARLO
        with tab3:
            st.subheader("Projeção Probabilística de Patrimônio")
            st.caption(f"Simulação baseada nas estatísticas do backtest para os próximos {mc_years} anos.")
            
            if not dca_curve.empty:
                daily_rets = dca_curve['Strategy_DCA'].pct_change().dropna()
                mu = daily_rets.mean() * 252
                sigma = daily_rets.std() * np.sqrt(252)
                
                mc_df = run_monte_carlo(dca_curve['Strategy_DCA'].iloc[-1], dca_amount, mu, sigma, mc_years)
                
                if not mc_df.empty:
                    fig_mc = px.line(mc_df, labels={'index': 'Data', 'value': 'Patrimônio (R$)'}, color_discrete_map={
                        'Pessimista (5%)': 'red', 'Base (50%)': 'blue', 'Otimista (95%)': 'green'
                    })
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    mc_final = mc_df.iloc[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cenário Pessimista", f"R$ {mc_final['Pessimista (5%)']:,.2f}")
                    c2.metric("Cenário Base", f"R$ {mc_final['Base (50%)']:,.2f}")
                    c3.metric("Cenário Otimista", f"R$ {mc_final['Otimista (95%)']:,.2f}")
            else:
                st.info("Rode o backtest primeiro para gerar dados para a simulação.")

        # TAB 4: CUSTÓDIA
        with tab4:
            if dca_holdings:
                st.subheader("Posição Atual (Backtest)")
                # Cria DF de Holdings
                hold_df = pd.DataFrame.from_dict(dca_holdings, orient='index', columns=['Qtd'])
                hold_df = hold_df[hold_df['Qtd'] > 0]
                
                last_prices = prices.iloc[-1]
                hold_df['Preço'] = last_prices.reindex(hold_df.index)
                hold_df['Total (R$)'] = hold_df['Qtd'] * hold_df['Preço']
                hold_df['Peso (%)'] = hold_df['Total (R$)'] / hold_df['Total (R$)'].sum()
                
                col_chart, col_tab = st.columns([1, 1])
                
                with col_chart:
                    st.plotly_chart(px.sunburst(hold_df.reset_index(), path=['index'], values='Total (R$)', title="Alocação Financeira"), use_container_width=True)
                
                with col_tab:
                    st.dataframe(hold_df.sort_values('Peso (%)', ascending=False).style.format({
                        'Qtd': '{:.1f}', 'Preço': 'R$ {:.2f}', 'Total (R$)': 'R$ {:,.2f}', 'Peso (%)': '{:.1%}'
                    }), use_container_width=True)
                    
                st.divider()
                st.subheader("Histórico de Transações")
                st.dataframe(dca_trans.sort_values('Date', ascending=False), use_container_width=True)
            else:
                st.info("Nenhuma posição ativa.")

        # TAB 5: DADOS BRUTOS
        with tab5:
            st.subheader("Matriz de Fundamentos")
            st.dataframe(fundamentals)
            
            st.subheader("Correlação dos Fatores (Ranking Atual)")
            if not final_rank.empty:
                numeric_cols = final_rank.select_dtypes(include=np.number).columns
                corr = final_rank[numeric_cols].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

if __name__ == "__main__":
    main()
