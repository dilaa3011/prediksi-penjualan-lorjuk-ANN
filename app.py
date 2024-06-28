import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from component.nav import navbar

st.set_page_config(page_title="Prediksi Penjualan Lorjuk", layout="wide")

navbar()

st.title('Aplikasi Prediksi Penjualan Lorjuk')
st.markdown("""
    Aplikasi ini menggunakan model Artificial Neural Network (ANN) untuk memprediksi penjualan lorjuk di bulan berikutnya 
    berdasarkan data historis penjualan.
""")

# tab
tab1, tab2 = st.tabs(['Data', 'Prediksi'])

with tab1:
    st.subheader('Data Historis Penjualan')
    df = pd.read_excel('dataset.xlsx')
    st.dataframe(df)

with tab2:
    # Sidebar input
    st.sidebar.header('Input Data')
    data = st.sidebar.number_input('Masukkan Hasil Penjualan Sebelumnya:')
    submit_button = st.sidebar.button('Prediksi')

    if 'prediksi' not in st.session_state:
        st.session_state['prediksi'] = False

    if not submit_button and not st.session_state['prediksi']:
        st.warning("Silakan masukkan hasil penjualan sebelumnya dan klik 'Prediksi' untuk melihat hasil prediksi.")
    elif submit_button:
        st.write("## Hasil Prediksi")

        # Simpan data historis sebelum ada input baru
        df_historis = df.copy()

        # Add data baru ke dataframe
        new_sales = data
        last_month = df.index[-1]
        df.loc[last_month + 1] = new_sales

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[['hasil']])  # Only scale the 'hasil' column

        # bagi data
        X, y = [], []

        for i in range(1, len(data_scaled)): #prediksi berdasarkan data bulan kemarin
            X.append(data_scaled[i-1])
            y.append(data_scaled[i])

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # model ann
        model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
        model.fit(X_train, y_train.ravel())

        y_pred = model.predict(X_test)

        # eval model
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Display  eval
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.4f}")
        col2.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")

        # prediksi bulan depan
        last_month_sales = np.array([data_scaled[-1]])
        next_month_sales = model.predict(last_month_sales)
        next_month_sales = scaler.inverse_transform(next_month_sales.reshape(-1, 1))

        st.success(f"Prediksi Penjualan Bulan Berikutnya: {next_month_sales[0][0]:.2f}")

        # grafik
        st.subheader('Visualisasi Penjualan')

        # Pisahkan dataframe historis dan prediksi
        df_pred = df.copy()
        df_pred['Prediksi'] = np.nan
        df_pred.at[last_month + 1, 'Prediksi'] = next_month_sales[0][0]

        # Grafik Penjualan Historis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('Penjualan Historis')
            st.line_chart(df_historis[['hasil']])
        
        # Grafik Prediksi Penjualan
        with col2:
            st.write('Prediksi Penjualan')
            st.line_chart(df_pred[['hasil', 'Prediksi']])

        st.session_state['prediksi'] = True
