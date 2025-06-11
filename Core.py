# app.py (Versi Final: Alur Terintegrasi dengan Stabilitas Penuh)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
from urllib.parse import urlparse
from datetime import datetime, timedelta
import logging
import mpmath as mp
import matplotlib.pyplot as plt

# Asumsi Anda memiliki file ini atau sudah menginstal pustaka yang diperlukan
try:
    from streamlit_searchbox import st_searchbox
    from stocksymbol import StockSymbol
except ImportError:
    st.error("Beberapa pustaka belum terinstal. Jalankan: pip install streamlit-searchbox stocksymbol yfinance pandas numpy mpmath matplotlib")
    st.stop()

# Tambahkan fungsi ini di bagian atas file Anda, bersama fungsi lainnya

from io import BytesIO

@st.cache_data
def to_excel(df):
    """Mengonversi DataFrame ke file Excel di dalam memori."""
    output = BytesIO()
    # Gunakan 'with' statement untuk memastikan writer ditutup dengan benar
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
    # Ambil data biner dari output
    processed_data = output.getvalue()
    return processed_data

# --- 1. KONFIGURASI HALAMAN & FUNGSI HELPER ---
st.set_page_config(
    page_title="Aplikasi Saham Terintegrasi",
    page_icon="⚡",
    layout="wide"
)

# --- Fungsi-fungsi Pencarian ---
@st.cache_data
def load_stock_data():
    """Memuat daftar saham dari pasar AS dan Indonesia."""
    api_key = 'b42c90b3-651d-43ee-8dca-c7f8d68cc930' 
    if api_key == 'GANTI_DENGAN_API_KEY_ANDA' or not api_key:
        st.sidebar.warning("API Key untuk 'stocksymbol' tidak diisi. Daftar pencarian akan kosong.")
        return pd.DataFrame()
    try:
        ss = StockSymbol(api_key)
        with st.spinner("Mengunduh daftar saham (hanya sekali)..."):
            symbol_list_us = ss.get_symbol_list(market="US")
            symbol_list_id = ss.get_symbol_list(market="ID")
        all_symbols = symbol_list_us + symbol_list_id
        df = pd.DataFrame(all_symbols)
        df = df[['symbol', 'longName', 'market']].rename(columns={'symbol': 'Symbol', 'longName': 'Name', 'market': 'Market'})
        df.dropna(subset=['Name', 'Symbol'], inplace=True); df = df[df['Name'] != '']
        df['display_name'] = df['Name'] + " (" + df['Symbol'] + ") - " + df['Market']
        return df
    except Exception as e:
        st.sidebar.error(f"Gagal memuat daftar saham: {e}"); return pd.DataFrame()

def search_stocks(search_term: str, stock_df: pd.DataFrame):
    """Fungsi pencarian hibrida final."""
    if not search_term or stock_df.empty: return []
    search_term_lower = search_term.lower(); search_words = [word for word in search_term_lower.split() if word]
    if not search_words: return []
    results_df = stock_df.copy(); results_df['score'] = 0
    results_df['symbol_lower'] = results_df['Symbol'].str.lower()
    results_df['symbol_sanitized'] = results_df['symbol_lower'].str.split('.').str[0]
    for word in search_words:
        results_df.loc[results_df['symbol_sanitized'] == word, 'score'] += 50
        results_df.loc[results_df['Name'].str.lower().str.contains(word, na=False), 'score'] += 1
    results_df.loc[results_df['Name'].str.lower().str.startswith(search_term_lower), 'score'] += 10
    results_df.loc[results_df['symbol_sanitized'] == search_term_lower, 'score'] += 100
    results_df.drop(columns=['symbol_lower', 'symbol_sanitized'], inplace=True)
    final_results = results_df[results_df['score'] > 0].copy()
    final_results.sort_values(by='score', ascending=False, inplace=True)
    if not final_results.empty: return list(final_results['display_name'].head(15))
    if len(search_words) == 1 and len(search_term) <= 12:
        try:
            potential_ticker_symbol = search_term.upper()
            temp_ticker = yf.Ticker(potential_ticker_symbol)
            temp_info = temp_ticker.info
            if temp_info and (temp_info.get('shortName') or temp_info.get('longName')):
                name = temp_info.get('longName') or temp_info.get('shortName'); symbol = temp_info.get('symbol', potential_ticker_symbol)
                return [f"{name} ({symbol}) - LIVE"]
            elif potential_ticker_symbol.isalnum() and not potential_ticker_symbol.startswith('^'):
                potential_ticker_symbol_jk = f"{potential_ticker_symbol}.JK"; temp_ticker_jk = yf.Ticker(potential_ticker_symbol_jk); temp_info_jk = temp_ticker_jk.info
                if temp_info_jk and (temp_info_jk.get('shortName') or temp_info_jk.get('longName')):
                    name = temp_info_jk.get('longName') or temp_info_jk.get('shortName'); symbol = temp_info_jk.get('symbol', potential_ticker_symbol_jk)
                    return [f"{name} ({symbol}) - LIVE"]
        except Exception: return []
    return []

# --- Fungsi-fungsi Forecasting (Sesuai Logika Asli Anda dengan Perbaikan Stabilitas) ---
def get_data(stock_symbol, start_date, end_date):
    """Mengambil data harga penutupan dari Yahoo finance."""
    data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=False)
    if data.empty: return np.array([])
    return data['Close'].to_numpy()

mp.dps = 100
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler()])

def determine_v_n(Sn,Sn_1):
    v_n = (Sn - Sn_1) / 1 #delta_t = 1
    # Pemeriksaan untuk mencegah pembagian dengan nol 
    if abs(v_n) < 1e-12:
        return 1e-12 
    return v_n

def determine_alpha_n(Sn_minus_4, Sn_minus_3, Sn_minus_2, Sn_minus_1):
    AA = (Sn_minus_2 - 2 * Sn_minus_3 + Sn_minus_4)
    BB = (Sn_minus_1 - Sn_minus_2)
    CC = (Sn_minus_1 - 2 * Sn_minus_2 + Sn_minus_3)
    DD = (Sn_minus_2 - Sn_minus_3)

    alpha_pembilang = (AA * BB) - (CC * DD)
    alpha_penyebut = DD * BB * (DD - BB)

    # Pemeriksaan untuk mencegah pembagian dengan nol 
    if abs(alpha_penyebut) < 1e-12:
        return 1e-12  
    return (alpha_pembilang/alpha_penyebut)

def determine_beta_n(Sn_minus_3, Sn_minus_2, Sn_minus_1, alpha_n):
    CC = (Sn_minus_1 - 2 * Sn_minus_2 + Sn_minus_3)
    BB = (Sn_minus_1 - Sn_minus_2)

    # Pemeriksaan untuk mencegah pembagian dengan nol 
    if abs(BB) < 1e-12:
        return 1e-12 
    
    return (CC-(alpha_n * (BB**2)))/(BB * 1) #delta_t = 1
def determine_h_n(v_1, alpha_n, beta_n):
    if abs(alpha_n) < 1e-12:
        alpha_n = 1e-12
    if abs(v_1) < 1e-12:
        v_1 = 1e-12
    
    try:
        h_n = abs((v_1 + (beta_n / alpha_n) / v_1))
        return h_n
    except (ZeroDivisionError) as e:
        logging.warning(f"Error in determine_h_n: {e}. Using fallback value.")
        return 1.0

def determine_s_n(s1, alpha, beta, h, condition_1, s_n, v_n, v_1):
    # Pemeriksaan untuk mencegah pembagian dengan nol 
    if abs(alpha) < 1e-12:
        alpha = 1e-12
    if abs(beta) < 1e-12:
        beta = 1e-12

    condition_2 = v_n > v_1
    condition_3 = s_n > s1
 
    try:
        if condition_1 > 0 and condition_2 and condition_3:
            s_n = s1 - (1/alpha) * mp.log(mp.fabs((mp.exp(beta) - h) / (1 - h)))
        if condition_1 > 0 and condition_2 and not condition_3:
            s_n = s1 + mp.fabs(1/alpha) * (mp.fabs(beta)/beta) * mp.log(mp.fabs((mp.exp(beta) - h) / (1 - h)))
        if condition_1 < 0 and condition_2 and condition_3:
            s_n = s1 - (1/alpha) * mp.log(mp.fabs((mp.exp(beta) + h) / (1 + h)))
        if condition_1 < 0 and condition_2 and not condition_3:
            s_n = s1 - mp.fabs(1/alpha) * (mp.fabs(beta)/beta) * mp.log(mp.fabs((mp.exp(beta) + h) / (1 + h)))
        if condition_1 > 0 and not condition_2 and condition_3:
            s_n = s1 - (1/alpha) * (beta/mp.fabs(beta)) * mp.log(mp.fabs((mp.exp(beta) -h) / (1 - h)))
        if condition_1 > 0 and not condition_2 and not condition_3:
            s_n = s1 - mp.fabs(1/alpha) * mp.log(mp.fabs((mp.exp(-mp.fabs(beta)) - h) / (1 - h)))
        if condition_1 < 0 and not condition_2 and condition_3:
            s_n = s1 + (1/alpha) * (beta/mp.fabs(beta)) * mp.log(mp.fabs(mp.exp(-mp.fabs(beta)) + h) / (1 + h))
        if condition_1 < 0 and not condition_2 and not condition_3:
            s_n = s1 + mp.fabs(1/alpha) * mp.log(mp.fabs(mp.exp(-mp.fabs(beta)) + h) / (1 + h))
    except (ZeroDivisionError) as e:
        logging.error(f'Error in determine_s_n: {e}. Using fallback value.')
        s_n = s1  # menggunakan nilai sebelumnya sebagai nilai fallback

    logging.debug(f'determine_s_n result: s_n={s_n}')
    return s_n

def determine_MAPE_list(actual: list, predicted: list) -> list:
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    num_of_cases = len(actual)
    sum_of_percentage_error = 0
    mape_list = []
    for i in range(num_of_cases):
        if actual[i] == 0:
            continue  # Skip jika nilai aktual = 0 untuk menghindari pembagian 0
        abs_error = mp.fabs(actual[i] - predicted[i])
        percentage_error = abs_error / actual[i]
        sum_of_percentage_error += percentage_error
        MAPE = sum_of_percentage_error / (i + 1) * 100
        mape_list.append(float(MAPE))
    return mape_list


def fitting(closing_prices, stock_symbol):
    Fitting_S_n_list = []
    v_list = []
    first_run = True
    
    # Check if we have enough data points
    if len(closing_prices) < 4:
        st.error("Tidak cukup data untuk melakukan fitting. Minimal 4 data point diperlukan.")
        return [], []
    
    for i in range(3):
        Fitting_S_n_list.append(float(closing_prices[i]))

    for i in range(3, len(closing_prices)):
        S_minus_1 = closing_prices[i - 3]
        S_0 = closing_prices[i - 2]
        S_1 = closing_prices[i - 1]
        S_2 = closing_prices[i]
        
        v_0 = determine_v_n(S_0, S_minus_1)
        v_1 = determine_v_n(S_1, S_0)
        v_2 = determine_v_n(S_2, S_1)
        
        if first_run:
            v_list.append(v_0)
            v_list.append(v_1)
            first_run = False
        v_list.append(v_2)

        try:
            alpha_n = determine_alpha_n(S_minus_1,S_0, S_1, S_2)
            beta_n = determine_beta_n(S_minus_1,S_1, S_2, alpha_n)
            h_n = determine_h_n(v_0, alpha_n, beta_n)
            condition_1 = (v_2 + (beta_n / alpha_n)) * v_2
            S_n = determine_s_n(S_minus_1, alpha_n, beta_n, h_n, condition_1, S_2, v_2, v_0)
        except (ZeroDivisionError) as e:
            logging.warning(f"Error in calculation at index {i}: {e}. Using fallback.")
            S_n = S_2  # fallback, data tidak berubah

        Fitting_S_n_list.append(float(S_n))
        logging.debug(f'Appended S_n={S_n} to Fitting_S_n_list')
    
    return Fitting_S_n_list, v_list

def forecasting(Fitting_S_n_list, start_date_str, end_date_str, stock_symbol):
    if len(Fitting_S_n_list) < 4: return None, None
    fitting_S_last = Fitting_S_n_list.copy()
    try:
        closing_prices_full_np = get_data(stock_symbol, start_date_str, end_date_str)
        if closing_prices_full_np.size == 0: return None, None
        closing_prices_full = [p.item() for p in closing_prices_full_np]
        closing_prices_full = filter_prices_duplicates(closing_prices_full)
        forecast_days = len(closing_prices_full) - len(Fitting_S_n_list)
        if forecast_days <= 0: return [], []
    except Exception as e:
        st.error(f"Gagal mendapatkan data untuk periode forecast: {e}"); return None, None
    for _ in range(forecast_days):
        try:
            current_len = len(fitting_S_last)
            S_minus_1, S_0, S_1, S_2 = fitting_S_last[current_len-4 : current_len]
            v_0, v_2 = determine_v_n(S_0, S_minus_1), determine_v_n(S_2, S_1)
            alpha_n = determine_alpha_n(S_minus_1,S_0, S_1, S_2)
            beta_n = determine_beta_n(S_0, S_1, S_2, alpha_n)
            h_n = determine_h_n(v_0, alpha_n, beta_n)
            condition_1 = (v_2 + (beta_n / alpha_n)) * v_2
            S_n = determine_s_n(S_minus_1, alpha_n, beta_n, h_n, condition_1, S_2, v_2, v_0)
        except Exception as e:
            logging.warning(f"Error in forecast step: {e}. Using previous value.")
            S_n = fitting_S_last[-1]
        fitting_S_last.append(float(S_n))
    forecast_S_list = fitting_S_last[len(Fitting_S_n_list):]
    closing_forecast_actual = closing_prices_full[len(Fitting_S_n_list):]
    return forecast_S_list, closing_forecast_actual

def filter_prices_duplicates(closing_prices):
    if not closing_prices:
        return []
    
    filtered_prices = [closing_prices[0]]
    for i in range(1, len(closing_prices)):
        if closing_prices[i] != closing_prices[i-1]:
            filtered_prices.append(closing_prices[i])
    return filtered_prices

# --- 3. MEMUAT DATA UTAMA & TAMPILAN APLIKASI ---
stock_df = load_stock_data()

st.title("⚡ Aplikasi Analisis & Prediksi Saham")
st.write("Cari saham untuk melihat detail, lalu jalankan prediksi harga di bawahnya.")
st.markdown("---")

st.header("1. Cari Saham Pilihan Anda")
selected_stock_display = st_searchbox(
    lambda term: search_stocks(term, stock_df),
    key="main_searchbox",
    placeholder="Ketik nama atau simbol saham (misal: BBCA, ^JKSE, GOOGL)",
)

if selected_stock_display:
    yfinance_symbol = ""
    try:
        match = re.search(r'\(([^)]+)\)\s*-\s*(\S+)$', selected_stock_display)
        if match:
            selected_symbol, market = match.group(1), match.group(2)
            if market.upper() == 'LIVE': yfinance_symbol = selected_symbol
            elif market.upper() == 'ID': yfinance_symbol = f"{selected_symbol}.JK"
            else: yfinance_symbol = selected_symbol
            
            with st.spinner(f"Mengambil data untuk {yfinance_symbol}..."):
                stock_info = yf.Ticker(yfinance_symbol); info = stock_info.info

            if not info or 'symbol' not in info:
                st.error(f"Tidak dapat menemukan data untuk simbol `{yfinance_symbol}`.")
            else:
                st.subheader(f"Detail untuk: {info.get('longName', yfinance_symbol)}")
                logo_url = info.get('logo_url')
                if not logo_url and info.get('website'):
                    try: domain = urlparse(info['website']).netloc; logo_url = f"https://logo.clearbit.com/{domain}"
                    except Exception: pass
                if logo_url: st.image(logo_url, width=90)
                st.write(f"**Simbol Ticker:** `{info.get('symbol', yfinance_symbol)}` | **Sektor:** {info.get('sector', 'N/A')}")
                
                st.subheader("Grafik Harga Historis (1 Tahun)")
                hist_data = stock_info.history(period="1y")
                if not hist_data.empty: st.line_chart(hist_data['Close'])
                else: st.warning("Data historis tidak tersedia.")
                
                st.markdown("---")
                st.header(f"2. Jalankan Prediksi untuk {yfinance_symbol}")
                
                with st.expander("Buka/Tutup Parameter Prediksi", expanded=True):
                    col1_fc, col2_fc, col3_fc = st.columns(3)
                    with col1_fc:
                        fc_start_train = st.date_input("Mulai Training", value=datetime.now() - timedelta(days=365))
                    with col2_fc:
                        fc_end_train = st.date_input("Akhir Training", value=datetime.now())
                    with col3_fc:
                        forecast_days_count = st.number_input("Prediksi untuk (hari ke depan)", min_value=1, max_value=365, value=90, step=1)

                if fc_start_train >= fc_end_train: st.error("Tanggal mulai training harus sebelum akhir training.")
                else:
                    forecast_end_date = fc_end_train + timedelta(days=forecast_days_count)
                    if fc_end_train >= forecast_end_date:
                        st.error("Tanggal akhir training harus sebelum tanggal akhir prediksi.")
                    else:
                        if st.button("🚀 Jalankan Prediksi", key="run_forecast", use_container_width=True):
                            try:
                                with st.spinner("Mengambil data & menjalankan kalkulasi..."):
                                    stock_symbol_to_forecast = yfinance_symbol
                                    closing_prices_np = get_data(stock_symbol_to_forecast, fc_start_train, fc_end_train)
                                    closing_prices = [p.item() for p in closing_prices_np]
                                    closing_prices = filter_prices_duplicates(closing_prices)
                                    if len(closing_prices) < 4:
                                        st.error(f"Data unik tidak cukup (butuh > 4). Coba rentang tanggal training lebih panjang."); st.stop()
                                    
                                    Fitting_S_n_list, v_list = fitting(closing_prices, stock_symbol_to_forecast)
                                    if not Fitting_S_n_list:
                                        st.error("Gagal melakukan fitting data."); st.stop()
                                    
                                    S_forecast, closing_forecast = forecasting(Fitting_S_n_list, fc_start_train.strftime("%Y-%m-%d"), forecast_end_date.strftime("%Y-%m-%d"), stock_symbol_to_forecast)
                                    
                                    mape_fit = determine_MAPE_list(closing_prices, Fitting_S_n_list)
                                    mape_forecast = determine_MAPE_list(closing_forecast, S_forecast) if S_forecast and closing_forecast else []
                                st.success("Prediksi selesai! Menampilkan hasil...")
                                
                                if Fitting_S_n_list:
                                    st.subheader(f"📊 Grafik Fitting vs Actual ({stock_symbol_to_forecast})")
                                    fig_fit, ax_fit = plt.subplots(figsize=(10, 6)); ax_fit.plot(closing_prices, label="Actual", color='black', linewidth=2); ax_fit.plot(Fitting_S_n_list, label="Fitted", color='blue', linewidth=2, linestyle='--'); ax_fit.set_title(f"Fitting Data Harga Saham ({stock_symbol_to_forecast})"); ax_fit.set_xlabel("Hari"); ax_fit.set_ylabel("Harga"); ax_fit.legend(); ax_fit.grid(True, alpha=0.3); st.pyplot(fig_fit)
                                    if S_forecast and closing_forecast:
                                        st.subheader(f"📈 Grafik Fitting + Forecast vs Actual ({stock_symbol_to_forecast})")
                                        fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6)); all_actual = closing_prices + closing_forecast; ax_forecast.plot(all_actual, label="Actual", color='black', linewidth=2); ax_forecast.plot(range(len(Fitting_S_n_list)), Fitting_S_n_list, label="Fitted", color='blue', linewidth=2); forecast_start_idx = len(Fitting_S_n_list); forecast_end_idx = forecast_start_idx + len(S_forecast); ax_forecast.plot(range(forecast_start_idx, forecast_end_idx), S_forecast, label="Forecast", color='orange', linewidth=2); ax_forecast.axvline(x=len(closing_prices)-1, color='red', linestyle='--', label='Forecast Start', alpha=0.7); ax_forecast.set_title(f"Fitting dan Forecast Harga Saham ({stock_symbol_to_forecast})"); ax_forecast.set_xlabel("Hari"); ax_forecast.set_ylabel("Harga"); ax_forecast.legend(); ax_forecast.grid(True, alpha=0.3); st.pyplot(fig_forecast)
                                    if mape_fit:
                                        st.subheader(f"📉 Hasil MAPE Fitting - Rata-rata: {np.mean(mape_fit):.2f}%")
                                        fig_mape_fit, ax_mape_fit = plt.subplots(figsize=(10, 6)); ax_mape_fit.plot(mape_fit, color='purple', label='MAPE Fitting (%)', linewidth=2); ax_mape_fit.set_title(f"Grafik MAPE Selama Fitting ({stock_symbol_to_forecast})"); ax_mape_fit.set_xlabel("Hari"); ax_mape_fit.set_ylabel("MAPE (%)"); ax_mape_fit.legend(); ax_mape_fit.grid(True, alpha=0.3); st.pyplot(fig_mape_fit)
                                    if mape_forecast:
                                        st.subheader(f"📉 Hasil MAPE Forecast - Rata-rata: {np.mean(mape_forecast):.2f}%")
                                        fig_mape_forecast, ax_mape_forecast = plt.subplots(figsize=(10, 6)); ax_mape_forecast.plot(mape_forecast, color='orange', label='MAPE Forecast (%)', linewidth=2); ax_mape_forecast.set_title(f"Grafik MAPE Selama Forecasting ({stock_symbol_to_forecast})"); ax_mape_forecast.set_xlabel("Hari"); ax_mape_forecast.set_ylabel("MAPE (%)"); ax_mape_forecast.legend(); ax_mape_forecast.grid(True, alpha=0.3); st.pyplot(fig_mape_forecast)
                                    st.subheader("📋 Tabel Ringkasan Data")
                                    if S_forecast and closing_forecast:
                                        df_result = pd.DataFrame({"Actual": pd.Series(closing_prices + closing_forecast),"Forecast": pd.Series(Fitting_S_n_list + S_forecast)})
                                    else:
                                        df_result = pd.DataFrame({"Actual": pd.Series(closing_prices), "Fitted": pd.Series(Fitting_S_n_list)})
                                    st.dataframe(df_result, use_container_width=True)
                                    # --- BAGIAN BARU: TOMBOL DOWNLOAD ---
                                    # Siapkan data excel dari DataFrame hasil
                                    excel_data = to_excel(df_result)

                                    # Buat nama file yang dinamis
                                    file_name_excel = f"hasil_prediksi_{stock_symbol_to_forecast}_{datetime.now().strftime('%Y%m%d')}.xlsx"

                                    st.download_button(
                                        label="📥 Unduh Data ke Excel",
                                        data=excel_data,
                                        file_name=file_name_excel,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                    # --- AKHIR BAGIAN BARU ---
                                    st.subheader("📊 Statistik Ringkasan")
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Jumlah Data Training", f"{len(closing_prices)} hari")
                                    if mape_fit: col2.metric("MAPE Fitting", f"{np.mean(mape_fit):.2f}%")
                                    if mape_forecast: col3.metric("MAPE Prediksi", f"{np.mean(mape_forecast):.2f}%")
                                    col4.metric("Periode Prediksi", f"{forecast_days_count} hari")
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat menjalankan prediksi: {e}")
                                logging.error(f"Forecast execution error: {e}", exc_info=True)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses pilihan Anda: {e}")
else:
    st.info("Selamat datang! Ketik nama saham di atas untuk memulai.")
