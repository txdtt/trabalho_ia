import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)

st.set_page_config(page_title="IA - Regressão e Classificação", layout="wide")


@st.cache_data(ttl=3600)
def baixar_dados(ticker):
    # baixa dados de 2 anos do yahoo finance
    df = yf.download(ticker, period='2y')

    if df is None or df.empty:
        return None

    # correção para o novo formato do yfinance (remove o MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    return df


# --- INTERFACE ---
st.title('Previsão de valores de ações')
st.markdown("""
Este projeto utiliza dois modelos de IA para previsão do mercado financeiro:
1. **Classificação (RandomForestClassifier):**
 Para prever se irá subir ou descer.

2. **Regressão (RandomForestRegressor):** Para prever o preço futuro da ação.
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

if btn_predict:
    with st.spinner(f'Treinando modelos para {ticker}...'):
        df = baixar_dados(ticker)

        # verifica se há mais de 100 linhas na tabela (dataframe)
        # necessário para calcular as médias móveis longas
        if len(df) > 100:
            # média móvel dos últimos 9 dias
            df['Media_Mov_9'] = df['Close'].rolling(window=9).mean()

            # média móvel dos últimos 21 dias
            df['Media_Mov_21'] = df['Close'].rolling(window=21).mean()

            # LAGS (atrasos): o preço de ontem ajuda a prever o de hoje
            df['Close_Ontem'] = df['Close'].shift(1)
            df['Close_Anteontem'] = df['Close'].shift(2)
            df['Retorno'] = df['Close'].pct_change()

            df.dropna(inplace=True)

            # definindo alvos
            df['Target_Class'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            df['Target_Reg'] = df['Close'].shift(-1)

            dados_treino = df.iloc[:-1].copy()
            ultimo_dia_para_prever = df.iloc[[-1]].copy()

            features = ['Close', 'Close_Ontem', 'Media_Mov_9', 'Media_Mov_21',
                        'Retorno']

            X = dados_treino[features]
            y_class = dados_treino['Target_Class']
            y_reg = dados_treino['Target_Reg']

            # split de treino e teste (últimos 50 dias para validação)
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
                X, y_class, y_reg, test_size=50, shuffle=False
            )

            # treinamento dos modelos
            # direção
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_class_train)
            acc = accuracy_score(y_class_test, clf.predict(X_test))

            # preço
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, y_reg_train)
            preds_reg = reg.predict(X_test)
            erro_medio = mean_absolute_error(y_reg_test, preds_reg)

            # prevendo o futuro da ação
            X_futuro = ultimo_dia_para_prever[features]

            previsao_tendencia = clf.predict(X_futuro)[0]
            previsao_preco = reg.predict(X_futuro)[0]

            data_atual = df.index[-1]
            data_futura = data_atual + BDay(1)

            with col2:
                m1, m2, m3 = st.columns(3)

                if ".SA" in ticker:
                    m1.metric("Preço Atual", f"R$ {df['Close'].iloc[-1]:.2f}")
                    m2.metric("Previsão (Regressão)", f"R$ {previsao_preco:.2f}",
                              delta=f"{((previsao_preco -
                                         df['Close'].iloc[-1])/df['Close']
                                        .iloc[-1])*100:.2f}%")
                else:
                    m1.metric("Preço Atual", f"$ {df['Close'].iloc[-1]:.2f}")
                    m2.metric("Previsão (Regressão)", f"$ {previsao_preco:.2f}",
                              delta=f"{((previsao_preco -
                                         df['Close'].iloc[-1])/df['Close']
                                        .iloc[-1])*100:.2f}%")

                tendencia_txt = "ALTA!!!" if previsao_tendencia == 1 else "BAIXA!!!"
                cor_tendencia = "green" if previsao_tendencia == 1 else "red"
                m3.markdown(f"**Tendência:** :{cor_tendencia}[{tendencia_txt}]")

                mse = mean_squared_error(y_reg_test, preds_reg)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_reg_test, preds_reg)
                mape = np.mean(np.abs((y_reg_test - preds_reg) / y_reg_test)) * 100

                preds_class = clf.predict(X_test)

                precision = precision_score(y_class_test, preds_class)
                recall = recall_score(y_class_test, preds_class)
                f1 = f1_score(y_class_test, preds_class)

                st.write("### Relatório de Desempenho do Modelo")

                tab_reg, tab_class = st.tabs(["Métricas de Preço (Regressão)", "Métricas de Tendência (Classificação)"])

                with tab_reg:
                    st.info(f"""
                    **MAE (Erro Médio Absoluto):** R$ {erro_medio:.2f}

                    **RMSE (Raiz do Erro Quadrático):** R$ {rmse:.2f} (penaliza erros grandes)

                    **MAPE (Erro Percentual):** {mape:.2f}% (< 3% => ótimo)

                    **R² (Score de Ajuste):** {r2:.3f}
                    """)

                with tab_class:
                    st.info(f"""
                    **Acurácia Global:** {acc:.1%}

                    **Precisão (Precision):** {precision:.1%}

                    **Revocação (Recall):** {recall:.1%} (% que indica quanto o modelo detectou corretamente a subida da ação)

                    **F1-Score:** {f1:.3f} (média harmônica entre Precision e Recall)
                    """)

                # verificando divergência
                preco_atual = df['Close'].iloc[-1]
                variacao_regressao = previsao_preco - preco_atual

                direcao_regressao = 1 if variacao_regressao > 0 else 0

                if direcao_regressao != previsao_tendencia:
                    st.warning("**Divergência!**")
                else:
                    st.success("Sem divergências.")

                fig = go.Figure()

                # série histórica
                hist_df = df.tail(100)
                fig.add_trace(go.Scatter(
                    x=hist_df.index, y=hist_df['Close'],
                    mode='lines', name='Histórico Real',
                    line=dict(color='blue')
                ))

                # previsão
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
