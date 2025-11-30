import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from datetime import timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA - Regressão e Classificação", layout="wide")

# Função de cache para não baixar dados toda hora
@st.cache_data(ttl=3600)
def baixar_dados(ticker):
    # Baixa dados de 2 anos
    df = yf.download(ticker, period='2y')
    # Correção para o novo formato do yfinance (remove o MultiIndex)
    if df is None or df.empty:
        return None 

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

# --- INTERFACE ---
st.title('Previsão Híbrida: Regressão & Classificação')
st.markdown("""
Este sistema utiliza **dois modelos de IA** simultâneos:
1. **Classificação:** Para prever a direção (Alta/Baixa).
2. **Regressão:** Para prever o preço futuro exato.
""")

col1, col2 = st.columns([1, 3])
with col1:
    lista_tickers = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
        "ABEV3.SA", "MGLU3.SA", "WEGE3.SA", "RENT3.SA", "PRIO3.SA",
        "HAPV3.SA", "RDOR3.SA", "B3SA3.SA", "SUZB3.SA", "GGBR4.SA",
        "JBSS3.SA", "RADL3.SA", "RAIL3.SA", "CSNA3.SA", "ELET3.SA",
        "BBSE3.SA", "LREN3.SA", "VIVT3.SA", "EQTL3.SA", "BPAC11.SA",
        "UGPA3.SA", "CMIG4.SA", "CCRO3.SA", "SBSP3.SA", "TIMS3.SA",
        "EMBR3.SA", "ASAI3.SA", "CRFB3.SA", "CSAN3.SA", "CPLE6.SA",
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"
    ]

    ticker = st.selectbox(
        'Selecione o Ticker da ação:',
        options=lista_tickers,
        index=0
    )

    btn_predict = st.button('Executar')

# --- PROCESSAMENTO ---
if btn_predict:
    with st.spinner(f'Treinando modelos para {ticker}...'):
        df = baixar_dados(ticker)

        if len(df) > 100:
            # 1. Feature Engineering (Criar dados para a IA aprender)
            df['Media_Mov_9'] = df['Close'].rolling(window=9).mean()
            df['Media_Mov_21'] = df['Close'].rolling(window=21).mean()

            # LAGS (Atrasos): O preço de ontem ajuda a prever o de hoje
            df['Close_Ontem'] = df['Close'].shift(1)
            df['Close_Anteontem'] = df['Close'].shift(2)
            df['Retorno'] = df['Close'].pct_change()

            df.dropna(inplace=True)

            # 2. Definindo os Alvos (Targets)
            # Alvo da Classificação: 1 se subiu, 0 se caiu (comparado ao dia anterior)
            # Shift(-1) pega o dado de "Amanhã" e traz para a linha de "Hoje" para treinar
            df['Target_Class'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            # Alvo da Regressão: O PREÇO exato de amanhã
            df['Target_Reg'] = df['Close'].shift(-1)
            
            # Removemos a última linha pois ela não tem o "amanhã" (Target é NaN)
            dados_treino = df.iloc[:-1].copy()
            ultimo_dia_para_prever = df.iloc[[-1]].copy() # Usaremos este para prever o futuro real
            
            # 3. Separação das Features (X)
            features = ['Close', 'Close_Ontem', 'Media_Mov_9', 'Media_Mov_21', 'Retorno']
            
            X = dados_treino[features]
            y_class = dados_treino['Target_Class'] # Alvo Binário
            y_reg = dados_treino['Target_Reg']     # Alvo Numérico (Preço)
            
            # Split de Treino e Teste (Últimos 50 dias para validar)
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
                X, y_class, y_reg, test_size=50, shuffle=False
            )
            
            # 4. Treinamento dos Modelos
            
            # Modelo 1: Random Forest Classifier (Direção)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_class_train)
            acc = accuracy_score(y_class_test, clf.predict(X_test))
            
            # Modelo 2: Random Forest Regressor (Preço)
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, y_reg_train)
            preds_reg = reg.predict(X_test)
            erro_medio = mean_absolute_error(y_reg_test, preds_reg)
            
            # 5. PREVISÃO PARA O FUTURO (AMANHÃ)
            # Pegamos os dados de HOJE (ultimo_dia_para_prever) para prever AMANHÃ
            X_futuro = ultimo_dia_para_prever[features]
            
            previsao_tendencia = clf.predict(X_futuro)[0]
            previsao_preco = reg.predict(X_futuro)[0]
            
            data_atual = df.index[-1]
            data_futura = data_atual + timedelta(days=1)
            
            # --- EXIBIÇÃO DOS RESULTADOS ---
            with col2:
                # Métricas lado a lado
                m1, m2, m3 = st.columns(3)
                m1.metric("Preço Atual", f"R$ {df['Close'].iloc[-1]:.2f}")
                m2.metric("Previsão (Regressão)", f"R$ {previsao_preco:.2f}", 
                          delta=f"{((previsao_preco - 
                                     df['Close'].iloc[-1])/df['Close']
                                    .iloc[-1])*100:.2f}%")
                
                tendencia_txt = "ALTA!!!" if previsao_tendencia == 1 else "BAIXA!!!"
                cor_tendencia = "green" if previsao_tendencia == 1 else "red"
                m3.markdown(f"**Tendência:** :{cor_tendencia}[{tendencia_txt}]")
                
                st.info(f"Erro médio do modelo de preço nos testes: R$ {erro_medio:.2f}")

                # Lógica para verificar divergência
                preco_atual = df['Close'].iloc[-1]
                variacao_regressao = previsao_preco - preco_atual

                # Se Regressão diz que sobe (>0) e Classificação diz que cai (0), ou vice-versa
                direcao_regressao = 1 if variacao_regressao > 0 else 0

                if direcao_regressao != previsao_tendencia:
                    st.warning("**Atenção: Divergência Detectada!** Os modelos de Preço e Tendência discordam. Isso indica alta volatilidade ou indefinição do mercado para amanhã.")
                else:
                    st.success("**Sinal Forte:** Ambos os modelos concordam na direção do mercado!")

                # --- GRÁFICO FINAL (O que você pediu) ---
                fig = go.Figure()
                
                # Linha Histórica (Últimos 100 dias)
                hist_df = df.tail(100)
                fig.add_trace(go.Scatter(
                    x=hist_df.index, y=hist_df['Close'],
                    mode='lines', name='Histórico Real',
                    line=dict(color='blue')
                ))
                
                # Ponto da Previsão
                fig.add_trace(go.Scatter(
                    x=[data_atual, data_futura],
                    y=[df['Close'].iloc[-1], previsao_preco],
                    mode='lines+markers',
                    name='Previsão Futura',
                    line=dict(color='orange', width=3, dash='dot'),
                    marker=dict(size=10, color='orange')
                ))
                
                fig.update_layout(
                    title=f"Projeção de Preço para {ticker}",
                    xaxis_title="Data",
                    yaxis_title="Preço (R$)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("Dados insuficientes para análise.")
