import streamlit as st 

def navbar():
    # Membuat 4 kolom
    col1, col2, col3 = st.columns(3, gap="small")

    # Menambahkan konten di dalam kolom pertama
    with col1:
        st.page_link("app.py", label="Aplikasi")
    with col2:
        st.page_link("pages/data.py", label="Data")
    with col3:
        st.page_link("pages/about.py", label="About")


    
    