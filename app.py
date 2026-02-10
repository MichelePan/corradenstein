import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ================================
# CONFIGURAZIONE PAGINA
# ================================
st.set_page_config(page_title="Dashboard Unica", layout="wide")

# ================================
# TAB
# ================================
tab1, tab2 = st.tabs(["📈 Stock Screener", "🧮 Calcolatore Variazioni"])

# ================================
# TAB 1: STOCK SCREENER
# ================================
with tab1:

    # ----- Tickers -----
    TICKERS = {
        "ALPHABET INC": "GOOGL",
        "AMAZON": "AMZN",
        "AMERICA AIRLINES": "AAL",
        "AMERICAN BATTERY TECHNOLOGY COMPANY": "ABAT",
        "ATOSSA THERAPEUTICS INC": "ATOS",
        "ALIBABA GROUP HOLDING": "BABA",
        "BANK OF AMERICA CORP": "BAC",
        "BEYOND MEAT": "BYND",
        "CENOVUS ENERGY INC": "CVE",
        "CERENCE": "CRNC",
        "CLEAN ENERGY FUELS CORP": "CLNE",
        "COMCAST CORPORATION": "CMCSA",
        "COTERRA ENERGY INC": "CTRA",
        "CRONOS GROUP INC": "CRON",
        "DELTA AIRLINES": "DAL",
        "DEVON ENERGY CORPORATION": "DVN",
        "EBAY INC": "EBAY",
        "FISERV": "FISV",
        "FORD MOTOR CO": "F",
        "HASBRO": "HAS",
        "HP INC": "HPQ",
        "HUNTINGTON BANCSHARES INC": "HBAN",
        "ICAHN ENTERPRISES LP": "IEP",
        "INCANNEX HEALTHCARE INC": "IXHL",
        "INTEL": "INTC",
        "IONIS PHARMACEUTICALS": "IONS",
        "KOSMOS ENERGY LTD": "KOS",
        "LYFT INC": "LYFT",
        "NETFLIX": "NFLX",
        "NEW FORTRESS ENERGY INC": "NFE",
        "NVIDIA": "NVDA",
        "PAYPAL HOLDINGS INC": "PYPL",
        "PELOTON INTERACTIVE": "PTON",
        "PINTEREST INC": "PINS",
        "REVIVA PHARMACEUTICALS HOLDING INC": "RVPH",
        "RIVIAN AUTOMOTIVE INC": "RIVN",
        "SNAP INC": "SNAP",
        "TARGET HOSPITAL CORP": "TH",
        "THE COCA-COLA COMPANY": "KO",
        "THE WALT DISNEY COMPANY": "DIS",
        "TESLA": "TSLA",
        "TILRAY BRANDS INC": "TLRY",
        "TRANSOCEAN LTD": "RIG",
        "TRAWS PHARMA INC": "TRAW",
        "UNIQURE NV": "QURE",
        "VITAL ENERGY INC": "VTLE"
    }

    # ----- CACHE -----
    @st.cache_data(ttl=3600)
    def load_data(ticker):
        try:
            df = yf.download(ticker, period="5y", interval="1d", progress=False)
            return df
        except:
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def run_arima(series, steps):
        model = ARIMA(series, order=(2, 0, 2)).fit()
        forecast = model.forecast(steps=steps)
        conf = model.get_forecast(steps=steps).conf_int()
        return forecast, conf

    # ----- UI PARAMETRI -----
    with st.sidebar:
        st.header("Parametri Stock Screener")
        historical_period = st.selectbox("Numero valori storici", [120, 360, 720])
        forecast_period = st.selectbox("Previsione futura (giorni)", [30, 60, 120])
        run = st.button("Applica Stock Screener")

    # ----- LOGICA -----
    if run:
        rows = []
        with st.spinner("Calcolo in corso..."):
            for name, ticker in TICKERS.items():
                row = {
                    "NAME": name,
                    "TICKER": ticker,
                    "ON MKT": np.nan,
                    "MIN": np.nan,
                    "AVG": np.nan,
                    "MAX": np.nan,
                    "FORECAST MIN": np.nan,
                    "FORECAST VALUE": np.nan,
                    "FORECAST MAX": np.nan,
                    "Δ % FORECAST": np.nan,
                    "STATUS": "OK"
                }

                try:
                    df_raw = load_data(ticker)
                    if df_raw.empty or "Close" not in df_raw.columns:
                        row["STATUS"] = "NO DATA"
                        rows.append(row)
                        continue

                    df_close = df_raw[["Close"]].tail(historical_period)
                    df_close = df_close[pd.to_numeric(df_close["Close"], errors="coerce").notna()]

                    if df_close.empty:
                        row["STATUS"] = "NO NUMERIC DATA"
                        rows.append(row)
                        continue

                    row["ON MKT"] = float(df_close["Close"].iloc[-1])
                    row["MIN"] = float(df_close["Close"].min().round(2))
                    row["AVG"] = float(df_close["Close"].mean().round(2))
                    row["MAX"] = float(df_close["Close"].max().round(2))

                    if len(df_close) < 10:
                        row["FORECAST MIN"] = row["MIN"]
                        row["FORECAST VALUE"] = row["ON MKT"]
                        row["FORECAST MAX"] = row["MAX"]
                        row["Δ % FORECAST"] = 0
                        row["STATUS"] = "INSUFFICIENT DATA"
                    else:
                        try:
                            forecast, conf = run_arima(df_close["Close"], forecast_period)
                            row["FORECAST MIN"] = float(conf.iloc[-1, 0].round(2))
                            row["FORECAST VALUE"] = float(forecast.iloc[-1].round(2))
                            row["FORECAST MAX"] = float(conf.iloc[-1, 1].round(2))
                            row["Δ % FORECAST"] = float(
                                ((row["FORECAST VALUE"] - row["ON MKT"]) / row["ON MKT"] * 100).round(2)
                            )
                        except:
                            row["FORECAST MIN"] = row["MIN"]
                            row["FORECAST VALUE"] = row["ON MKT"]
                            row["FORECAST MAX"] = row["MAX"]
                            row["Δ % FORECAST"] = 0
                            row["STATUS"] = "ARIMA FALLBACK"

                except:
                    row["STATUS"] = "ERROR"

                rows.append(row)

        df = pd.DataFrame(rows)

        def color_rows(row):
            styles = []
            for col in row.index:
                if col == "FORECAST VALUE" and not pd.isna(row[col]):
                    styles.append("color: blue; font-weight:bold" if row[col] > row["ON MKT"] else "color:red; font-weight:bold")
                elif col == "Δ % FORECAST" and not pd.isna(row[col]):
                    if row[col] > 20:
                        styles.append("color:green; font-weight:bold")
                    elif row[col] < 0:
                        styles.append("color:magenta; font-weight:bold")
                    else:
                        styles.append("")
                elif col == "STATUS" and row[col] != "OK":
                    styles.append("color:orange; font-weight:bold")
                else:
                    styles.append("")
            return styles

        row_height = 35
        header_height = 40
        table_height = header_height + row_height * len(df)

        st.dataframe(
            df.style.apply(color_rows, axis=1),
            use_container_width=True,
            height=table_height
        )
    else:
        st.info("👈 Imposta i parametri e premi **Applica**")

# ================================
# TAB 2: CALCOLATORE VARIAZIONI
# ================================
with tab2:

    st.set_page_config(page_title="Calcolatore Variazioni", layout="wide")

    # --- CSS PER STILIZZAZIONE ---
    st.markdown("""
    <style>
        .input-container {background: linear-gradient(135deg, #fff9c4 0%, #fff176 100%); padding:10px; border-radius:5px; border:1px solid #fbc02d;}
        .output-container {background-color:#e0f7fa; padding:15px 10px; border-radius:5px; border:1px solid #00bcd4; text-align:center; margin-bottom:10px;}
        .cell-label {font-size:0.8em;color:#555;margin-bottom:5px;font-weight:bold;text-transform:uppercase;}
        .cell-value {font-size:1.2em;font-weight:bold;}
        .text-red {color:red;}
        .text-black {color:black;}
    </style>
    """, unsafe_allow_html=True)

    def styled_input(label, key, value=0.0):
        with st.container():
            st.markdown(f'<div class="input-container"><div class="cell-label">{label}</div>', unsafe_allow_html=True)
            val = st.number_input("", value=value, key=key, label_visibility="collapsed", format="%f")
            st.markdown('</div>', unsafe_allow_html=True)
        return val

    def styled_output(label, value, is_out=False):
        val_str = f"{value:.2f}" if isinstance(value, float) else str(value)
        text_class = "text-red" if (is_out or (isinstance(value,float) and value <0)) else "text-black"
        st.markdown(f'<div class="output-container"><div class="cell-label">{label}</div><div class="cell-value {text_class}">{val_str}</div></div>', unsafe_allow_html=True)

    st.title("Calcolatore Aumento e Decremento Percentuale")

    # Esempio: puoi replicare qui il codice originale del calcolatore positivo/negativo
    st.info("📌 Inserisci i valori nella sezione input e premi CALCOLA")
