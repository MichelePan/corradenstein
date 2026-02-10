import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Dashboard Portfolio", layout="wide")

# ================================
# TABS
# ================================
tab1, tab2 = st.tabs(["📈 Calibra Surveillance", "🧮 Calcolatore Variazione"])

# ================================
# TICKERS
# ================================
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

# ================================
# TAB 1: CALIBRA SURVEILLANCE
# ================================
with tab1:
    st.title("📈 Calibra Surveillance – Stock Screener")

    with st.sidebar:
        st.header("Parametri")
        historical_period = st.selectbox("Numero valori storici", [120, 360, 720])
        forecast_period = st.selectbox("Previsione futura (giorni)", [30, 60, 120])
        run = st.button("Applica")

    @st.cache_data(ttl=3600)
    def load_all_data(tickers):
        tickers_str = " ".join(tickers.values())
        df = yf.download(tickers_str, period="5y", interval="1d", progress=False)
        return df

    @st.cache_data(ttl=3600)
    def run_arima_cached(series, steps):
        if len(series) < 20:
            return None, None
        try:
            model = ARIMA(series, order=(2,0,2)).fit()
            forecast = model.forecast(steps=steps)
            conf = model.get_forecast(steps=steps).conf_int()
            return forecast, conf
        except:
            return None, None

    def compute_row(name, ticker, historical_period, forecast_period, df_all):
        row = {"NAME": name, "TICKER": ticker, "STATUS": "OK",
               "ON MKT": np.nan, "MIN": np.nan, "AVG": np.nan, "MAX": np.nan,
               "FORECAST MIN": np.nan, "FORECAST VALUE": np.nan, "FORECAST MAX": np.nan,
               "Δ % FORECAST": np.nan}

        try:
            if 'Close' not in df_all or ticker not in df_all['Close'].columns:
                row["STATUS"] = "NO DATA"
                return row

            df_close = df_all['Close'][ticker].dropna().tail(historical_period)
            if len(df_close) < 20:
                row["STATUS"] = "INSUFFICIENT DATA"
                return row

            forecast, conf = run_arima_cached(df_close, forecast_period)
            if forecast is None:
                row["STATUS"] = "ARIMA ERROR"
                return row

            on_mkt = float(df_close.iloc[-1].round(2))
            row.update({
                "ON MKT": on_mkt,
                "MIN": float(df_close.min().round(2)),
                "AVG": float(df_close.mean().round(2)),
                "MAX": float(df_close.max().round(2)),
                "FORECAST MIN": float(conf.iloc[-1,0].round(2)),
                "FORECAST VALUE": float(forecast.iloc[-1].round(2)),
                "FORECAST MAX": float(conf.iloc[-1,1].round(2)),
                "Δ % FORECAST": float(((forecast.iloc[-1]-on_mkt)/on_mkt*100).round(2))
            })

        except:
            row["STATUS"] = "ARIMA ERROR"

        return row

    if run:
        st.info("Calcolo in corso...")
        df_all = load_all_data(TICKERS)
        rows = []

        with st.spinner("Elaborazione ticker..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(compute_row, name, ticker, historical_period, forecast_period, df_all)
                           for name, ticker in TICKERS.items()]
                for future in concurrent.futures.as_completed(futures):
                    rows.append(future.result())

        df = pd.DataFrame(rows).round(2)

        def color_rows(row):
            styles = []
            for col in row.index:
                if col == "FORECAST VALUE" and not pd.isna(row[col]):
                    styles.append("color: blue; font-weight:bold" if row[col] > row["ON MKT"] else "color: red; font-weight:bold")
                elif col == "Δ % FORECAST" and not pd.isna(row[col]):
                    if row[col] > 20: styles.append("color: green; font-weight:bold")
                    elif row[col] < 0: styles.append("color: magenta; font-weight:bold")
                    else: styles.append("")
                elif col == "STATUS" and row[col] != "OK":
                    styles.append("color: orange; font-weight:bold")
                else:
                    styles.append("")
            return styles

        row_height = 35
        header_height = 40
        table_height = header_height + row_height*len(df)

        st.dataframe(
            df.style.apply(color_rows, axis=1),
            use_container_width=True,
            height=table_height
        )
    else:
        st.info("👈 Imposta i parametri e premi **Applica**")

# ================================
# TAB 2: CALCOLATORE VARIAZIONE
# ================================
with tab2:
    st.title("🧮 Calcolatore Aumento/Decremento Percentuale")

    # --- Funzioni helper
    st.markdown("""
    <style>
        .input-container {background: linear-gradient(135deg,#fff9c4,#fff176);padding:10px;border-radius:5px;border:1px solid #fbc02d;height:100%;display:flex;flex-direction:column;justify-content:center;}
        .output-container {background-color:#e0f7fa;padding:15px 10px;border-radius:5px;border:1px solid #00bcd4;text-align:center;height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center;margin-bottom:10px;}
        .cell-label{font-size:0.8em;color:#555;margin-bottom:5px;text-transform:uppercase;font-weight:bold;}
        .cell-value{font-size:1.2em;font-weight:bold;}
        .text-red{color:red;}.text-black{color:black;}
    </style>
    """, unsafe_allow_html=True)

    def styled_input(label, key, value=0.0):
        val = st.number_input(label, value=float(value), key=key, format="%.2f")
        return val

    def styled_output(label, value, is_out=False):
        if isinstance(value,float):
            val_str = f"{value:.2f}"
        else:
            val_str = str(value)
        text_class = "text-red" if (value<0 or is_out) else "text-black"
        st.markdown(f'<div class="output-container"><div class="cell-label">{label}</div><div class="cell-value {text_class}">{val_str}</div></div>', unsafe_allow_html=True)

    st.subheader("Calcolatore Variazione Percentuale Positiva")

    # INPUTS
    start_pos = styled_input("START", "start_pos")
    end_pos = styled_input("END", "end_pos")
    qty_pos = styled_input("QTY", "qty_pos")
    hyp_pos = styled_input("HYP", "hyp_pos")
    out_f = styled_input("OUT/F", "out_f")
    atx = styled_input("ATX%", "atx")

    calculate = st.button("CALCOLA/RESET")

    if calculate:
        # --- LOGICA POSITIVA ---
        incr = end_pos - start_pos
        var = (incr/start_pos*100) if start_pos!=0 else 0
        lqy = start_pos*qty_pos
        pl = end_pos*qty_pos - lqy
        out = lqy/hyp_pos if hyp_pos!=0 else 0
        res = qty_pos - out
        val = hyp_pos*out
        cst = start_pos*out_f
        gr_inc = hyp_pos*out_f
        gr_pl = gr_inc - cst
        tx = gr_pl*atx/100
        n_pl = gr_pl - tx
        n_inc = gr_inc - tx
        diff = n_inc - lqy

        # OUTPUT
        styled_output("INCR", incr)
        styled_output("VAR", var)
        styled_output("LQY CMD", lqy)
        styled_output("P/L", pl)
        styled_output("OUT", out, is_out=True)
        styled_output("RES", res)
        styled_output("VAL", val)
        styled_output("CST", cst)
        styled_output("GR/INC", gr_inc)
        styled_output("GR/P/L", gr_pl)
        styled_output("TX", tx)
        styled_output("N/P/L", n_pl)
        styled_output("N/INC", n_inc)
        styled_output("DIFF", diff)
