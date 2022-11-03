import streamlit as st

st.header("About Page")
text = "KTH-BIOX LAB POWER FACTOR CALCULATION GUI INTERFACE"



def example(color1, color2, color3, content):
    st.markdown(
        f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:24px;border-radius:2%;">{content}</p>',
        unsafe_allow_html=True)

color1 = '#1aa3ff'
color2 = '#00ff00'
color3 = '#ffffff'
example(color1, color2, color3, text)
st.subheader("Purpose:")
st.write("Purpose of this software is to help researchers to run, explore and visualize their experiment through this GUI.")
st.subheader('Author: ')
st.write("@adembjorn / @borekson & @ahembal")
st.subheader('Contact: ')
st.write("bjorn@borekson.com & balsever@kth.se")
st.subheader('Usage: ')
st.write("**To run with sample data:**")
st.write("First, push the *Load Sample Data* button at the bottom of sidebar. "
        "Then you can select activities from drow-down menu at the top of sidebar. "
        "Plot input data through Plot section. Tune paramaters through Tune section. "
        "Analyse and see results through Analysis section.")
st.write("**To run with uploaded data:**")
st.write("First, upload the defined data files through *drag and drop section*. "
        "Then you can select activities from drow-down menu at the top of sidebar. "
        "Plot input data through *Plot* section. Tune paramaters through *Tune* section. "
        "Analyse and see results through *Analysis* section. "
         "Download or delete results through *Session Info* section.")
st.subheader('License: ')
st.write("All rights reserved.®")


# TODO Show real graph on results page not the pictures
