import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")

# ================================
# CONFIGURAZIONE PAGINA
# ================================
st.set_page_config(page_title="CORRADENSTEIN", layout="wide")

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
# CACHE FUNZIONI
# ================================
@st.cache_data(ttl=3600)
def load_data_multiple(tickers):
    """Scarica tutti i ticker in una volta sola, nascondendo output di yfinance."""
    tickers = tuple(tickers)  # hashable
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df = yf.download(tickers=list(tickers), period="5y", interval="1d", progress=False, group_by='ticker')
    return df

@st.cache_data(ttl=3600)
def run_arima(series, steps):
    """Esegue ARIMA sulla serie e ritorna forecast e conf_int."""
    model = ARIMA(series, order=(2,0,2)).fit()
    forecast = model.forecast(steps=steps)
    conf = model.get_forecast(steps=steps).conf_int()
    return forecast, conf

def extract_close_column(df):
    """Estrae la colonna Close dai dati scaricati."""
    if "Close" in df.columns:
        return df[["Close"]].copy()
    if df.shape[1]==1:
        return df.rename(columns={df.columns[0]:"Close"})
    for c in df.columns:
        if "close" in str(c).lower():
            return df[[c]].rename(columns={c:"Close"})
    raise ValueError("Colonna 'Close' non trovata")

# ================================
# CREAZIONE TAB
# ================================
tab1, tab2 = st.tabs(["Calibra", "Calcolatore"])

# ================================
# TAB 1 - SURVEILLANCE
# ================================
with tab1:
    st.title("📈 Calibra – Stock Screener")
    
    with st.sidebar:
        st.header("Parametri Calibra")
        historical_period = st.selectbox("Numero valori storici", [120,360,720])
        forecast_period = st.selectbox("Previsione futura (giorni)", [30,60,120])
        run_tab1 = st.button("Applica", key="tab1_run")
    
    if run_tab1:
        rows = []
        num_tickers = len(TICKERS)
        
        # download multi-ticker con cache e output nascosto
        all_data = load_data_multiple(TICKERS.values())
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i, (name, ticker) in enumerate(TICKERS.items(), start=1):
            row = {"NAME":name,"TICKER":ticker,"ON MKT":np.nan,
                   "MIN":np.nan,"AVG":np.nan,"MAX":np.nan,
                   "FORECAST MIN":np.nan,"FORECAST VALUE":np.nan,"FORECAST MAX":np.nan,
                   "Δ % FORECAST":np.nan,"STATUS":"OK"}
            try:
                if ticker not in all_data:
                    row["STATUS"] = "NO DATA"
                    rows.append(row)
                    continue
                df_raw = all_data[ticker].copy()
                if df_raw.empty or "Close" not in df_raw.columns:
                    row["STATUS"]="NO DATA"
                    rows.append(row)
                    continue
                row["ON MKT"]=float(df_raw["Close"].iloc[-1])
                df_close = extract_close_column(df_raw).dropna().tail(historical_period)
                if len(df_close)<20:
                    row["STATUS"]="INSUFFICIENT DATA"
                    rows.append(row)
                    continue
                forecast, conf = run_arima(df_close["Close"], forecast_period)
                row["MIN"]=float(df_close["Close"].min().round(2))
                row["AVG"]=float(df_close["Close"].mean().round(2))
                row["MAX"]=float(df_close["Close"].max().round(2))
                row["FORECAST MIN"]=float(conf.iloc[-1,0].round(2))
                row["FORECAST VALUE"]=float(forecast.iloc[-1].round(2))
                row["FORECAST MAX"]=float(conf.iloc[-1,1].round(2))
                row["Δ % FORECAST"]=float((row["FORECAST VALUE"]-row["ON MKT"])/row["ON MKT"]*100)
            except Exception:
                row["STATUS"]="ARIMA ERROR"
            rows.append(row)
            
            # aggiorna barra di progresso
            progress = i / num_tickers
            progress_bar.progress(progress)
            progress_text.text(f"Elaborazione ticker {i}/{num_tickers}: {ticker}")
        
        progress_bar.empty()
        progress_text.empty()
        
        df = pd.DataFrame(rows).round(2)
        
        def color_rows(row):
            styles=[]
            for col in row.index:
                if col=="FORECAST VALUE" and not pd.isna(row[col]):
                    styles.append("color: blue; font-weight:bold" if row[col]>row["ON MKT"] else "color:red;font-weight:bold")
                elif col=="Δ % FORECAST" and not pd.isna(row[col]):
                    styles.append("color:green;font-weight:bold" if row[col]>20 else ("color:magenta;font-weight:bold" if row[col]<0 else ""))
                elif col=="STATUS" and row[col]!="OK":
                    styles.append("color:orange;font-weight:bold")
                else:
                    styles.append("")
            return styles
        
        row_height=35
        header_height=40
        table_height=header_height+row_height*len(df)
        
        st.dataframe(df.style.apply(color_rows, axis=1), use_container_width=True, height=table_height)

# ================================
# TAB 2 - CALCOLATORE
# ================================
with tab2:
    st.title("Calcolatore Azionario")
    # [Il codice del calcolatore rimane invariato, copia quello già presente]
