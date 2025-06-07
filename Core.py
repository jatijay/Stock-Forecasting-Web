import streamlit as st
from io import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import getting_data as gd
import logging
import mpmath as mp
import matplotlib.pyplot as plt


st.title("Stocks Forecasting Web [PALING DASAR]")
st.write("Ini adalah website untuk memprediksi harga saham")

mp.dps = 100

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler()]
)

def calculate_v_n(Sn,Sn_1):
    v_n = (Sn - Sn_1) / 1 #delta_t = 1

    if abs(v_n) < 1e-12:
        return 1e-12  # Return small value instead of 1 to avoid division issues

    return v_n

def calculate_alpha_n(Sn_minus_4, Sn_minus_3, Sn_minus_2, Sn_minus_1):
    AA = (Sn_minus_2 - 2 * Sn_minus_3 + Sn_minus_4)
    BB = (Sn_minus_1 - Sn_minus_2)
    CC = (Sn_minus_1 - 2 * Sn_minus_2 + Sn_minus_3)
    DD = (Sn_minus_2 - Sn_minus_3)

    alpha_pembilang = (AA * BB) - (CC * DD)
    alpha_penyebut = DD * BB * (DD - BB) 

    if abs(alpha_penyebut) < 1e-12:
        return 1e-12  # Return small value instead of 1
    return (alpha_pembilang/alpha_penyebut)

def calculate_beta_n(Sn_minus_3, Sn_minus_2, Sn_minus_1, alpha_n):
    CC = (Sn_minus_1 - 2 * Sn_minus_2 + Sn_minus_3)
    BB = (Sn_minus_1 - Sn_minus_2)

    if abs(BB) < 1e-12:
        return 1e-12  # Return small value instead of 1
    
    return (CC-(alpha_n * (BB**2)))/(BB * 1) #delta_t = 1

def calculate_h_n(v_1, alpha_n, beta_n):
    # Add safety checks to prevent division by zero
    if abs(alpha_n) < 1e-12:
        alpha_n = 1e-12
    if abs(v_1) < 1e-12:
        v_1 = 1e-12
    
    try:
        h_n = abs((v_1 + (beta_n / alpha_n) / v_1))
        return h_n
    except (ZeroDivisionError, OverflowError) as e:
        logging.warning(f"Error in calculate_h_n: {e}. Using fallback value.")
        return 1.0

def calculate_s_n(s1, alpha, beta, h, condition_1, s_n, v_n, v_1):
    logging.debug(f"calculate_s_n called with: s1={s1}, alpha={alpha}, beta={beta}, h={h}, condition_1={condition_1}, s_n={s_n}, v_n={v_n}, v_1={v_1}")

    # Safety checks
    if abs(alpha) < 1e-12:
        alpha = 1e-12
    if abs(beta) < 1e-12:
        beta = 1e-12

    condition_2 = v_n > v_1
    condition_3 = s_n > s1
 
    try:
        if condition_1 > 0 and condition_2 and condition_3:
            s_n = s1 - (1/alpha) * mp.log(mp.fabs((mp.exp(beta) - h) / (1 - h)))
        elif condition_1 > 0 and condition_2 and not condition_3:
            s_n = s1 + mp.fabs(1/alpha) * (mp.fabs(beta)/beta) * mp.log(mp.fabs((mp.exp(beta) - h) / (1 - h)))
        elif condition_1 < 0 and condition_2 and condition_3:
            s_n = s1 - (1/alpha) * mp.log(mp.fabs((mp.exp(beta) + h) / (1 + h)))
        elif condition_1 < 0 and condition_2 and not condition_3:
            s_n = s1 - mp.fabs(1/alpha) * (mp.fabs(beta)/beta) * mp.log(mp.fabs((mp.exp(beta) + h) / (1 + h)))
        elif condition_1 > 0 and not condition_2 and condition_3:
            s_n = s1 - (1/alpha) * (beta/mp.fabs(beta)) * mp.log(mp.fabs((mp.exp(beta) -h) / (1 - h)))
        elif condition_1 > 0 and not condition_2 and not condition_3:
            s_n = s1 - mp.fabs(1/alpha) * mp.log(mp.fabs((mp.exp(-mp.fabs(beta)) - h) / (1 - h)))
        elif condition_1 < 0 and not condition_2 and condition_3:
            s_n = s1 + (1/alpha) * (beta/mp.fabs(beta)) * mp.log(mp.fabs(mp.exp(-mp.fabs(beta)) + h) / (1 + h))
        else:  # condition_1 < 0 and not condition_2 and not condition_3
            s_n = s1 + mp.fabs(1/alpha) * mp.log(mp.fabs(mp.exp(-mp.fabs(beta)) + h) / (1 + h))
    except (ZeroDivisionError) as e:
        logging.error(f'Error in calculate_s_n: {e}. Using fallback value.')
        s_n = s1  # Use previous value as fallback

    logging.debug(f'calculate_s_n result: s_n={s_n}')
    return s_n

def calculate_MAPE_list(actual: list, predicted: list) -> list:
    logging.debug(f'actual: {actual}, len {len(actual)}')
    logging.debug(f"predicted: {predicted}, len {len(predicted)}")
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    num_of_cases = len(actual)
    sum_of_percentage_error = 0
    mape_list = []
    for i in range(num_of_cases):
        if actual[i] == 0:
            continue  # Skip if actual value is zero to avoid division by zero
        abs_error = mp.fabs(actual[i] - predicted[i])
        percentage_error = abs_error / actual[i]
        sum_of_percentage_error += percentage_error
        MAPE = sum_of_percentage_error / (i + 1) * 100
        mape_list.append(float(MAPE))
    return mape_list

def fit(closing_prices, stock_name):
    logging.debug(f'fit called with closing_prices={closing_prices}, stock_name={stock_name}')
    S_n_list = []
    v_list = []
    first_run = True
    
    # Check if we have enough data points
    if len(closing_prices) < 5:
        st.error("Tidak cukup data untuk melakukan fitting. Minimal 5 data point diperlukan.")
        return [], []
    
    for i in range(4):
        S_n_list.append(float(closing_prices[i]))

    for i in range(4, len(closing_prices)):
        S_minus_1 = closing_prices[i - 4]
        S_0 = closing_prices[i - 3]
        S_1 = closing_prices[i - 2]
        S_2 = closing_prices[i - 1]
        S_3 = closing_prices[i]
        
        v_0 = calculate_v_n(S_0, S_minus_1)
        v_1 = calculate_v_n(S_1, S_0)
        v_2 = calculate_v_n(S_2, S_1)
        v_3 = calculate_v_n(S_3, S_2)
        
        if first_run:
            v_list.append(v_0)
            v_list.append(v_1)
            v_list.append(v_2)
            first_run = False
        v_list.append(v_3)

        try:
            alpha_n = calculate_alpha_n(S_0, S_1, S_2, S_3)
            beta_n = calculate_beta_n(S_1, S_2, S_3, alpha_n)
            h_n = calculate_h_n(v_0, alpha_n, beta_n)
            condition_1 = (v_3 + (beta_n / alpha_n)) * v_3
            S_n = calculate_s_n(S_0, alpha_n, beta_n, h_n, condition_1, S_3, v_3, v_0)
        except (ZeroDivisionError) as e:
            logging.warning(f"Error in calculation at index {i}: {e}. Using fallback.")
            S_n = S_3  # fallback, data tidak berubah

        S_n_list.append(float(S_n))
        logging.debug(f'Appended S_n={S_n} to S_n_list')
    
    return S_n_list, v_list

def forecast(S_n_list, closing_prices, start_date, end_date, stock_name):
    if len(S_n_list) < 5:
        st.error("Tidak cukup data fitting untuk melakukan forecast.")
        return [], []
        
    S_last = S_n_list[-5:].copy()  # Make a copy to avoid modifying original
    
    try:
        closing_prices_full = gd.get_data(stock_name, start_date, end_date)
        closing_prices_full = [price.item() for price in closing_prices_full]
        closing_prices_full = filter_prices_duplicates(closing_prices_full)
    except Exception as e:
        st.error(f"Error getting forecast data: {e}")
        return [], []
    
    forecast_days = len(closing_prices_full) - len(S_n_list)
    
    if forecast_days <= 0:
        st.warning("Tidak ada data baru untuk di-forecast.")
        return [], closing_prices_full[len(S_n_list)-1:]

    for i in range(4, forecast_days + 4):
        if i >= len(S_last):
            break
            
        S_minus_1 = S_last[i - 4]
        S_0 = S_last[i - 3]
        S_1 = S_last[i - 2]
        S_2 = S_last[i - 1]
        S_3 = S_last[i]
        
        v_0 = calculate_v_n(S_0, S_minus_1)
        v_3 = calculate_v_n(S_3, S_2)
        
        try:
            alpha_n = calculate_alpha_n(S_0, S_1, S_2, S_3)
            beta_n = calculate_beta_n(S_1, S_2, S_3, alpha_n)
            h_n = calculate_h_n(v_0, alpha_n, beta_n)
            condition_1 = (v_3 + (beta_n / alpha_n)) * v_3
            S_n = calculate_s_n(S_0, alpha_n, beta_n, h_n, condition_1, S_3, v_3, v_0)
        except (ZeroDivisionError) as e:
            logging.warning(f"Error in forecast at step {i}: {e}. Using previous value.")
            S_n = S_3
            
        S_last.append(float(S_n))
    
    S_last = S_last[4:]
    closing_forecast = closing_prices_full[len(S_n_list)-1:]
    return S_last, closing_forecast

def filter_prices_duplicates(closing_prices):
    if not closing_prices:
        return []
    
    filtered_prices = [closing_prices[0]]
    for i in range(1, len(closing_prices)):
        if closing_prices[i] != closing_prices[i-1]:
            filtered_prices.append(closing_prices[i])
    return filtered_prices

# Streamlit UI
st.title("📈 Stock Forecasting Web App")

# Sidebar
st.sidebar.header("🔍 Pilih Parameter")
stock_name = st.sidebar.text_input("Stock Symbol", value="^JKSE")
start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 1, 1))

# End Date Option with Radio Button
st.sidebar.subheader("📅 End Date Options")
end_date_option = st.sidebar.radio(
    "Pilih metode input End Date:",
    ("Tanggal Spesifik", "Jumlah Hari"),
    help="Pilih apakah ingin input tanggal langsung atau berdasarkan jumlah hari dari start date"
)

if end_date_option == "Tanggal Spesifik":
    end_date = st.sidebar.date_input("End Date", value=datetime(2024, 5, 1))
else:
    days_count = st.sidebar.number_input(
        "Jumlah hari dari Start Date:",
        min_value=1,
        max_value=365,
        value=120,
        step=1,
        help="Masukkan jumlah hari yang ingin ditambahkan dari start date"
    )
    end_date = start_date + timedelta(days=days_count)
    st.sidebar.info(f"End Date akan menjadi: {end_date.strftime('%Y-%m-%d')}")

# Forecast End Date Option with Radio Button
st.sidebar.subheader("🔮 Forecast End Date Options")
forecast_end_date_option = st.sidebar.radio(
    "Pilih metode input Forecast End Date:",
    ("Tanggal Spesifik", "Jumlah Hari"),
    help="Pilih apakah ingin input tanggal langsung atau berdasarkan jumlah hari dari end date"
)

if forecast_end_date_option == "Tanggal Spesifik":
    forecast_end_date = st.sidebar.date_input("Forecast Until", value=datetime(2024, 11, 27))
else:
    forecast_days_count = st.sidebar.number_input(
        "Jumlah hari dari End Date:",
        min_value=1,
        max_value=365,
        value=200,
        step=1,
        help="Masukkan jumlah hari yang ingin ditambahkan dari end date untuk forecasting"
    )
    forecast_end_date = end_date + timedelta(days=forecast_days_count)
    st.sidebar.info(f"Forecast End Date akan menjadi: {forecast_end_date.strftime('%Y-%m-%d')}")

# Validation
if start_date >= end_date:
    st.sidebar.error("Start date harus lebih kecil dari end date!")
elif end_date >= forecast_end_date:
    st.sidebar.error("End date harus lebih kecil dari forecast end date!")

# Display selected dates summary
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Ringkasan Tanggal")
st.sidebar.write(f"**Start Date:** {start_date.strftime('%Y-%m-%d')}")
st.sidebar.write(f"**End Date:** {end_date.strftime('%Y-%m-%d')}")
st.sidebar.write(f"**Forecast Until:** {forecast_end_date.strftime('%Y-%m-%d')}")
st.sidebar.write(f"**Training Period:** {(end_date - start_date).days} hari")
st.sidebar.write(f"**Forecast Period:** {(forecast_end_date - end_date).days} hari")

if st.sidebar.button("🔮 Jalankan Forecast"):
    try:
        with st.spinner("Mengambil dan memproses data..."):
            # Get initial data
            closing_prices = gd.get_data(stock_name, start_date, end_date)
            closing_prices = [price.item() for price in closing_prices]
            closing_prices = filter_prices_duplicates(closing_prices)
            
            if len(closing_prices) < 5:
                st.error("Data tidak cukup untuk melakukan forecasting. Minimal 5 data point diperlukan.")
                st.stop()

            # FITTING
            S_n_list, v_list = fit(closing_prices, stock_name)
            
            if not S_n_list:
                st.error("Gagal melakukan fitting data.")
                st.stop()
                
            mape_fit = calculate_MAPE_list(closing_prices, S_n_list)

            # FORECASTING
            S_forecast, closing_forecast = forecast(
                S_n_list,
                closing_prices,
                start_date.strftime("%Y-%m-%d"),
                forecast_end_date.strftime("%Y-%m-%d"),
                stock_name
            )
            
            if S_forecast and closing_forecast:
                mape_forecast = calculate_MAPE_list(closing_forecast, S_forecast)
            else:
                mape_forecast = []

        st.success("Selesai!")

        # Display results only if we have valid data
        if S_n_list:
            # Grafik 1: Fitting vs Actual
            st.subheader(f"📊 Grafik Fitting vs Actual ({stock_name})")
            fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
            ax_fit.plot(closing_prices, label="Actual", color='black', linewidth=2)
            ax_fit.plot(S_n_list, label="Fitted", color='blue', linewidth=2)
            ax_fit.set_title(f"Fitting Data Harga Saham ({stock_name})")
            ax_fit.set_xlabel("Hari")
            ax_fit.set_ylabel("Harga")
            ax_fit.legend()
            ax_fit.grid(True, alpha=0.3)
            st.pyplot(fig_fit)

            # Grafik 2: Fitting + Forecast vs Actual (only if forecast data exists)
            if S_forecast and closing_forecast:
                st.subheader(f"📈 Grafik Fitting + Forecast vs Actual ({stock_name})")
                fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
                
                # Plot actual data
                all_actual = closing_prices + closing_forecast
                ax_forecast.plot(all_actual, label="Actual", color='black', linewidth=2)
                
                # Plot fitted data
                ax_forecast.plot(range(len(S_n_list)), S_n_list, label="Fitted", color='blue', linewidth=2)
                
                # Plot forecast data
                forecast_start_idx = len(S_n_list)
                forecast_end_idx = forecast_start_idx + len(S_forecast)
                ax_forecast.plot(range(forecast_start_idx, forecast_end_idx), S_forecast, 
                               label="Forecast", color='orange', linewidth=2)
                
                # Add vertical line to show where forecast starts
                ax_forecast.axvline(x=len(closing_prices), color='red', linestyle='--', 
                                  label='Forecast Start', alpha=0.7)
                
                ax_forecast.set_title(f"Fitting dan Forecast Harga Saham ({stock_name})")
                ax_forecast.set_xlabel("Hari")
                ax_forecast.set_ylabel("Harga")
                ax_forecast.legend()
                ax_forecast.grid(True, alpha=0.3)
                st.pyplot(fig_forecast)

            # MAPE Fitting
            if mape_fit:
                st.subheader(f"📉 Hasil MAPE Fitting - Rata-rata: {np.mean(mape_fit):.2f}%")
                fig_mape_fit, ax_mape_fit = plt.subplots(figsize=(10, 6))
                ax_mape_fit.plot(mape_fit, color='purple', label='MAPE Fitting (%)', linewidth=2)
                ax_mape_fit.set_title(f"Grafik MAPE Selama Fitting ({stock_name})")
                ax_mape_fit.set_xlabel("Hari")
                ax_mape_fit.set_ylabel("MAPE (%)")
                ax_mape_fit.legend()
                ax_mape_fit.grid(True, alpha=0.3)
                st.pyplot(fig_mape_fit)

            # MAPE Forecast
            if mape_forecast:
                st.subheader(f"📉 Hasil MAPE Forecast - Rata-rata: {np.mean(mape_forecast):.2f}%")
                fig_mape_forecast, ax_mape_forecast = plt.subplots(figsize=(10, 6))
                ax_mape_forecast.plot(mape_forecast, color='orange', label='MAPE Forecast (%)', linewidth=2)
                ax_mape_forecast.set_title(f"Grafik MAPE Selama Forecasting ({stock_name})")
                ax_mape_forecast.set_xlabel("Hari")
                ax_mape_forecast.set_ylabel("MAPE (%)")
                ax_mape_forecast.legend()
                ax_mape_forecast.grid(True, alpha=0.3)
                st.pyplot(fig_mape_forecast)

            # Tabel hasil
            st.subheader("📋 Tabel Ringkasan Data")
            if S_forecast and closing_forecast:
                df_result = pd.DataFrame({
                    "Actual": closing_prices + closing_forecast,
                    "Forecast": S_n_list + S_forecast
                })
            else:
                df_result = pd.DataFrame({
                    "Actual": closing_prices,
                    "Fitted": S_n_list
                })
            st.dataframe(df_result)
            
            # Display summary statistics
            st.subheader("📊 Statistik Ringkasan")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Jumlah Data Training", len(closing_prices))
            with col2:
                if mape_fit:
                    st.metric("MAPE Fitting", f"{np.mean(mape_fit):.2f}%")
            with col3:
                if mape_forecast:
                    st.metric("MAPE Forecast", f"{np.mean(mape_forecast):.2f}%")
            with col4:
                st.metric("Periode Forecast", f"{(forecast_end_date - end_date).days} hari")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        logging.error(f"Main execution error: {e}")
        st.info("Silakan coba dengan parameter yang berbeda atau periksa koneksi data.")
        st.info("Silakan coba dengan parameter yang berbeda atau periksa koneksi data.")