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
    tickers = tuple(tickers)  # assicura hashable
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
    st.title("📈 SURVEILLANCE Portfolio – Stock Screener")
    
    with st.sidebar:
        st.header("Parametri CALIBRA")
        historical_period = st.selectbox("Numero valori storici", [120,360,720])
        forecast_period = st.selectbox("Previsione futura (giorni)", [30,60,120])
        run_tab1 = st.button("Applica", key="tab1_run")
    
    if run_tab1:
        rows = []
        num_tickers = len(TICKERS)
        
        # scarica tutti i dati insieme
        all_data = load_data_multiple(list(TICKERS.values()))
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        with st.spinner("Calcolo in corso..."):
            for i, (name, ticker) in enumerate(TICKERS.items(), start=1):
                row = {"NAME":name,"TICKER":ticker,"ON MKT":np.nan,
                       "MIN":np.nan,"AVG":np.nan,"MAX":np.nan,
                       "FORECAST MIN":np.nan,"FORECAST VALUE":np.nan,"FORECAST MAX":np.nan,
                       "Δ % FORECAST":np.nan,"STATUS":"OK"}
                try:
                    # prendi i dati dal DataFrame multi-ticker
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

    # --- Funzioni di styling ---
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

    def styled_input_int(label, key, value=0):
        with st.container():
            st.markdown(f'<div class="input-container"><div class="cell-label">{label}</div>', unsafe_allow_html=True)
            val = st.number_input("", value=int(value), key=key, label_visibility="collapsed", step=1)
            st.markdown('</div>', unsafe_allow_html=True)
        return val

    def styled_output(label, value, is_out=False):
        val_str = f"{value:.2f}" if isinstance(value,float) else str(value)
        text_class = "text-red" if (is_out or (isinstance(value,float) and value < 0)) else "text-black"
        st.markdown(f'<div class="output-container"><div class="cell-label">{label}</div><div class="cell-value {text_class}">{val_str}</div></div>', unsafe_allow_html=True)

    # ========================
    # INPUT POSITIVO
    # ========================
    st.markdown("### Calcolatore Variazione Percentuale Positiva")
    c1, c2, c3, c4 = st.columns(4)
    with c1: start_pos = styled_input("START", "start_pos")
    with c2: end_pos = styled_input("END", "end_pos")
    with c3: incr_display_pos = st.empty()
    with c4: var_display_pos = st.empty()
    
    c5, c6, c7, c8 = st.columns(4)
    with c5: qty_pos = styled_input_int("QTY", "qty_pos", 0)
    with c6: lqy_display_pos = st.empty()
    with c7: pl_display_pos = st.empty()
    with c8: hyp_pos = styled_input("HYP", "hyp_pos", 0)

    c9, c10, c11 = st.columns(3)
    with c9: out_display_pos = st.empty()
    with c10: res_display_pos = st.empty()
    with c11: val_display_pos = st.empty()

    c12, c13, c14, c15, c16 = st.columns(5)
    with c12: out_f_input = styled_input("OUT/F", "out_f")
    with c13: cst_display = st.empty()
    with c14: gr_inc_display = st.empty()
    with c15: gr_pl_display = st.empty()
    with c16: atx_input = styled_input("ATX%", "atx")

    c17, c18, c19, c20 = st.columns(4)
    with c17: tx_display = st.empty()
    with c18: n_pl_display = st.empty()
    with c19: n_inc_display = st.empty()
    with c20: diff_display = st.empty()

    # ========================
    # INPUT NEGATIVO
    # ========================
    st.markdown("### Calcolatore Variazione Percentuale Negativa")
    n1, n2, n3, n4 = st.columns(4)
    with n1: start_neg = styled_input("START", "start_neg")
    with n2: end_neg = styled_input("END", "end_neg")
    with n3: incr_display_neg = st.empty()
    with n4: var_display_neg = st.empty()

    n5, n6, n7, n8 = st.columns(4)
    with n5: qty_neg = styled_input("QTY", "qty_neg", 0)
    with n6: lqy_display_neg = st.empty()
    with n7: npl_display_neg = st.empty()

    # Pulsante CALCOLA
    col_btn = st.columns([1,2,1])[1]
    calculate = col_btn.button("CALCOLA/RESET", type="primary", use_container_width=True)

    # LOGICA CALCOLI
    if calculate:
        # POSITIVO
        val_incr_pos = end_pos - start_pos
        val_var_pos = (val_incr_pos/start_pos*100) if start_pos!=0 else 0
        val_lqy_pos = start_pos*qty_pos
        val_pl_pos = (end_pos*qty_pos)-val_lqy_pos
        val_out_pos = val_lqy_pos/hyp_pos if hyp_pos!=0 else 0
        val_res_pos = qty_pos - val_out_pos
        val_val_pos = hyp_pos * val_out_pos
        val_cst = start_pos*out_f_input
        val_gr_inc = hyp_pos*out_f_input
        val_gr_pl = val_gr_inc - val_cst
        val_tx = (val_gr_pl*atx_input)/100
        val_n_pl = val_gr_pl - val_tx
        val_n_inc = val_gr_inc - val_tx
        val_diff = val_n_inc - val_lqy_pos

        # Visualizzazione POSITIVO
        with c3: styled_output("INCR", val_incr_pos)
        with c4: styled_output("VAR", val_var_pos)
        with c6: styled_output("LQY CMD", val_lqy_pos)
        with c7: styled_output("P/L", val_pl_pos)
        with c9: styled_output("OUT", val_out_pos, True)
        with c10: styled_output("RES", val_res_pos)
        with c11: styled_output("VAL", val_val_pos)
        with c13: styled_output("CST", val_cst)
        with c14: styled_output("GR/INC", val_gr_inc)
        with c15: styled_output("GR/P/L", val_gr_pl)
        with c17: styled_output("TX", val_tx)
        with c18: styled_output("N/P/L", val_n_pl)
        with c19: styled_output("N/INC", val_n_inc)
        with c20: styled_output("DIFF", val_diff)

        # NEGATIVO
        val_incr_neg = end_neg - start_neg
        val_var_neg = (val_incr_neg/start_neg*100) if start_neg!=0 else 0
        val_lqy_neg = start_neg*qty_neg
        val_npl_neg = (end_neg*qty_neg)-val_lqy_neg
        with n3: styled_output("INCR", val_incr_neg)
        with n4: styled_output("VAR", val_var_neg)
        with n6: styled_output("LQY CMD", val_lqy_neg)
        with n7: styled_output("N/P/L", val_npl_neg)
