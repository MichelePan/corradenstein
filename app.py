import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dashboard Unificata", layout="wide")

# ================================
# Tickers
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
# Tabs
# ================================
tab1, tab2 = st.tabs(["Calibra Surveillance", "Calcolatore Variazione"])

# ================================
# Tab 1: Calibra Surveillance
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
        progress_bar = st.progress(0)
        total = len(TICKERS)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(compute_row, name, ticker, historical_period, forecast_period, df_all): ticker
                       for name, ticker in TICKERS.items()}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                rows.append(future.result())
                progress_bar.progress((i+1)/total)

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
# Tab 2: Calcolatore Variazione (con layout, colori e calcolo negativo)
# ================================
with tab2:
    st.title("🧮 Calcolatore Aumento/Decremento Percentuale")

    # --- CSS per styling ---
    st.markdown("""
    <style>
        .output-container{background-color:#e0f7fa;padding:15px 10px;border-radius:5px;border:1px solid #00bcd4;text-align:center;margin-bottom:10px;}
        .cell-label{font-size:0.8em;color:#555;margin-bottom:5px;font-weight:bold;}
        .cell-value{font-size:1.2em;font-weight:bold;}
        .text-red{color:red;}.text-black{color:black;}
    </style>
    """, unsafe_allow_html=True)

    def styled_output(label,value,is_out=False):
        if isinstance(value,float): val_str=f"{value:.2f}"
        else: val_str=str(value)
        text_class="text-red" if (value<0 or is_out) else "text-black"
        st.markdown(f'<div class="output-container"><div class="cell-label">{label}</div><div class="cell-value {text_class}">{val_str}</div></div>', unsafe_allow_html=True)

    # --- Input numero ---
    def input_float(label,key,val=0.0):
        return st.number_input(label,value=float(val),format="%.2f",key=key)

    # --- Layout colonne e placeholder ---
    # Positivo
    c1,c2,c3,c4=st.columns(4)
    start_pos = input_float("START","start_pos")
    end_pos = input_float("END","end_pos")
    incr_display_pos=c3.empty()
    var_display_pos=c4.empty()

    c5,c6,c7,c8=st.columns(4)
    qty_pos=input_float("QTY","qty_pos")
    lqy_display_pos=c6.empty()
    pl_display_pos=c7.empty()
    hyp_pos=input_float("HYP","hyp_pos")

    c9,c10,c11=st.columns(3)
    out_display_pos=c9.empty()
    res_display_pos=c10.empty()
    val_display_pos=c11.empty()

    c12,c13,c14,c15,c16=st.columns(5)
    out_f_input=input_float("OUT/F","out_f")
    cst_display=c13.empty()
    gr_inc_display=c14.empty()
    gr_pl_display=c15.empty()
    atx_input=input_float("ATX%","atx")

    c17,c18,c19,c20=st.columns(4)
    tx_display=c17.empty()
    n_pl_display=c18.empty()
    n_inc_display=c19.empty()
    diff_display=c20.empty()

    # Negativo
    n1,n2,n3,n4=st.columns(4)
    start_neg=input_float("START","start_neg")
    end_neg=input_float("END","end_neg")
    incr_display_neg=n3.empty()
    var_display_neg=n4.empty()

    n5,n6,n7,n8=st.columns(4)
    qty_neg=input_float("QTY","qty_neg")
    lqy_display_neg=n6.empty()
    npl_display_neg=n7.empty()
