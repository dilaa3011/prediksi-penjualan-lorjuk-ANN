import streamlit as st
from component.nav import navbar

st.set_page_config(page_title="Prediksi Penjualan Lorjuk", layout="wide")

# css
st.markdown("""
    <style>
        .profile-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin: 10px;
        }   
        .profile-card h2 {
            color: #2E86C1;
        }
        .profile-card h3 {
            color: #117A65;
        }
        .profile-card img {
            border-radius: 50%;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

navbar()
st.title("Kelompok 1 Kecerdasan Komputasional B")

row1 = st.columns(3)

grid = [col.container() for col in row1]

with grid[0]:
    st.markdown(f"""
            <div class="profile-card">
                <h2>Kun Fadhilah Aini</h2>
                <h3>220411100025</h3>
            </div>
        """, unsafe_allow_html=True)
with grid[1]:
    st.markdown(f"""
            <div class="profile-card">
                <h2>Rizqiya Ivada</h2>
                <h3>220411100090</h3>
            </div>
        """, unsafe_allow_html=True)
with grid[2]:
    st.markdown(f"""
            <div class="profile-card">
                <h2>Cahya Fadhilah Yasmin</h2>
                <h3>220411100106</h3>
            </div>
        """, unsafe_allow_html=True)


