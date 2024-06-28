import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from component.nav import navbar

navbar()

# Header aplikasi
st.title('Prediksi Penjualan Keripik Lorjuk Menggunakan Metode ANN')

st.image('lorjuk.jpg', use_column_width=True)


# Markdown untuk gaya teks header dan subheader
st.markdown("""
    <style>
    .header {
        font-size: 24px;
        color: #FF9A00;
        padding-bottom: 10px;
        border-bottom: 1px solid #ccc;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 18px;
        color: #FFBF00;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Tujuan Analisa
st.markdown('<p class="header">Tujuan Analisa</p>', unsafe_allow_html=True)
st.write('Project ini bertujuan untuk melakukan analisa dan memprediksi terkait penjualan kripik yang dilakukan dengan metode ANN, dimana metode ini digunakan untuk mengklasifikasikan informasi, memprediksi hasil, dan juga data cluster. Nantinya dari data yang sudah tersedia akan di klasifikasikan untuk memprediksi nilai penjualan pada bulan berikutnya.')

# Data Understanding
st.markdown('<p class="header">Data Understanding</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Mengetahui Jumlah Penjualan Kripik Lorjuk</p>', unsafe_allow_html=True)
st.write('Pada proses ini kami akan menganalisa jumlah penjualan kripik pada bulan berikutnya berdasarkan dari data-data dari tahun sebelumnya. Dari data-data yang tersedia dari tahun sebelumnya data tersebut nantinya akan diolah menggunakan metode ANN (Artificial Neural Network). Nantinya metode ANN akan bekerja dimana Lapisan input menerima berbagai bentuk informasi dari dunia luar. Ini adalah data yang ingin diproses atau dipelajari oleh jaringan. Dari unit input, data melewati satu atau lebih unit tersembunyi. Tugas unit tersembunyi adalah mengubah input menjadi sesuatu yang dapat digunakan unit output.')

# Mengambil dan Menampilkan Data
st.markdown('<p class="subheader">Mengambil dan Menampilkan Data</p>', unsafe_allow_html=True)
st.write('Untuk mengambil data kita bisa menggunakan banyak cara salah satunya menggunakan query sql seperti dibawah ini:')
st.code('SELECT * FROM st_read("dataset penjualan.xlsx")')
st.write('atau bisa juga menggunakan code pandas berikut:')
st.code('''
import streamlit as st
import pandas as pd

df = pd.read_excel('dataset.xlsx')
st.dataframe(df)
''')
st.write('Code di atas digunakan untuk menampilkan data yang sudah disiapkan dalam file excel sebelumnya. Nantinya data tersebut akan diolah lagi menggunakan metode ANN untuk menentukan prediksi penjualan pada tahun berikutnya. Berikut hasil eksekusi dari code di atas:')
df = pd.read_excel('dataset.xlsx')
st.dataframe(df)

# Penjelasan fitur
st.markdown('<p class="header">Penjelasan fitur</p>', unsafe_allow_html=True)
st.code('''
data.info()

# atau bisa juga menggunakan
dtypes = pd.DataFrame(df.dtypes, columns=["Tipe Data"])
st.dataframe(dtypes)
''')
st.write('Dengan menggunakan code tersebut kita bisa mengetahui ada berapa attribut dan tipedata yang digunakan dalam data kita.')
dtypes = pd.DataFrame(df.dtypes, columns=["Tipe Data"])
st.dataframe(dtypes)
st.write('1. Fitur tanggal menunjukkan tanggal dari setiap data penjualan yang memiliki value dengan tipe data datetime')
st.write('2. Fitur hasil merupakan perolehan penjualan kripik di setiap tanggalnya dengan tipe data integer')

# Cek Missing Value
st.markdown('<p class="header">Cek Missing Value</p>', unsafe_allow_html=True)
st.code('''
df = df.isnull().sum()
print(df)
''')
st.write('Code di atas digunakan untuk menampilkan apakah ada missing value dari data yang akan digunakan. Berdasarkan output di atas dalam data yang akan digunakan tidak terdapat missing value.')
missing_values = df.isnull().sum()
st.write(missing_values)

# Memisahkan dataset menjadi data training dan data testing
st.markdown('<p class="header">Memisahkan dataset</p>', unsafe_allow_html=True)
st.code(
    '''
from sklearn.model_selection import train_test_split


X = data['tanggal']
Y = data[['hasil']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    '''
)
st.write('Code di atas berfungsi untuk memisahkan dataset menjadi data training dan juga data testing, dimana persentase dari kedua data tersebut yaitu 10% data testing dan juga 90% data training. Nantinya kedua data tersebut akan digunakan untuk melakukan perhitungan untuk mengetahui prediksi penjualan pada tahun berikutnya menggunakan ANN.')

# Normalisasi data
st.markdown('<p class="header">Normalisasi data</p>', unsafe_allow_html=True)
st.code(
    '''
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Normalisasi Y_train dan Y_test
Y_train_scaled = scaler.fit_transform(Y_train)
Y_test_scaled = scaler.transform(Y_test)

# Mengonversi hasil normalisasi menjadi DataFrame
Y_train = pd.DataFrame(Y_train_scaled, columns=['hasil'])
Y_test = pd.DataFrame(Y_test_scaled, columns=['hasil'])

# Menggabungkan hasil normalisasi dengan fitur tanggal untuk membuat DataFrame akhir
train = pd.concat([X_train.reset_index(drop=True), Y_train], axis=1)
test = pd.concat([X_test.reset_index(drop=True), Y_test], axis=1)
    '''
)
st.write('Code di atas digunakan untuk melakukan normalisasi pada dataset dengan menggunakan MinMaxScaler. Setelah mengimport modul yang dibutuhkan langkah selanjutnya yaitu melakukan normalisasi pada data Y_train dan juga Y_test. Setelah dilakukan normalisasi pada kedua data tersebut selanjutnya data tersebut dikonversi ke dalam dataframe. Setelah itu buat dataframe akhir dengan menggunakan fitur tanggal.')

# Melakukan prediksi
st.markdown('<p class="header">Melakukan prediksi</p>', unsafe_allow_html=True)
st.code(
    '''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Data penjualan historis
df = data
new_sales = 600


# Tambahkan data baru ke dataframe
last_month = len(df) - 1
df.loc[last_month + 1] = new_sales

# Skalakan data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['hasil']])  # Hanya kolom 'hasil' yang diskalakan

# Persiapkan data untuk pelatihan
X = []
y = []

# Menggunakan 1 bulan sebelumnya untuk memprediksi bulan berikutnya
for i in range(1, len(data_scaled)):
    X.append(data_scaled[i-1])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Membangun model ANN
model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, verbose=True)
model.fit(X_train, y_train.ravel())

# Prediksi di data testing
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Hitung Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Prediksi penjualan bulan berikutnya
last_month_sales = np.array([data_scaled[-1]])
next_month_sales = model.predict(last_month_sales)
next_month_sales = scaler.inverse_transform(next_month_sales.reshape(-1, 1))

print(f"Prediksi Penjualan Bulan Berikutnya: {next_month_sales[0][0]}")
    '''
)
st.write('Code di atas digunakan untuk melakukan prediksi penjualan berdasarkan model ANN yang telah dibangun.')

# Data penjualan historis
new_sales = 600
last_month = len(df) - 1
df.loc[last_month + 1] = new_sales

# Skalakan data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['hasil']])

# Persiapkan data untuk pelatihan
X = []
y = []

for i in range(1, len(data_scaled)):
    X.append(data_scaled[i-1])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Membangun model ANN
model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, verbose=True)
model.fit(X_train, y_train.ravel())

# Prediksi di data testing
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Hitung Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Tampilkan loss error selama pelatihan
st.markdown('<p class="header">Loss Error Selama Pelatihan</p>', unsafe_allow_html=True)
loss_df = pd.DataFrame(model.loss_curve_, columns=['Loss'])

col1, col2 = st.columns(2)
with col1 :
    # Tampilkan grafik loss
    st.line_chart(loss_df)
with col2:
    # Tampilkan dataframe loss
    st.dataframe(loss_df)

# Prediksi penjualan bulan berikutnya
last_month_sales = np.array([data_scaled[-1]])
next_month_sales = model.predict(last_month_sales)
next_month_sales = scaler.inverse_transform(next_month_sales.reshape(-1, 1))

st.write(f"Prediksi Penjualan Bulan Berikutnya: {next_month_sales[0][0]}")

st.write(f'code diatas digunakan untuk mendapatkan prediksi pejualan. Dengan mengimport data yang dibutuhkan, lalu menambahkan data baru ke dalam dataframe. Setelah data baru ditambahkan ke dataframe, data pada kolom hasil akan di normalisasikan menggunakan MinMaxScaler supaya berada pada rentang 0-1. Untuk memprediksi penjualan bulan berikutnya yaitu dengan menggunakan data 1 bulan sebelumnya. Kemudian data akan dibagi menjadi 2 menjadi data training dan data testing selanjutnya akan dibuat model ANN dengan 2 lapisan tersembunyi yang terdiri dari 10 neuron dengan iterasi maksimal 1000 iterasi. Prediksi dilakukan pada data testing lalu akan dievaluasi dengan Mean Squared Error(MSE) dan Mean Absolute Percentage Error (MAPE) semakin rendah nilai mape maka semakin baik model yang dipakai. Memprediksi penjualan untuk bulan berikutnya dengan menggunakan data terakhir yang dinormalisasi dan hasil diubah kembali ke skala asli. Sehingga didapatkan hasil prediksi penjualan pada bulan berikutnya sebesar {next_month_sales[0][0]}')