import datetime

import streamlit as st
import requests

st.set_page_config(
    layout='wide'
)

if 'token' not in st.session_state: st.session_state['token'] = { 'timestamp':datetime.datetime.now(), 'token': None}

def request_token(username, password):
    endpoint = "http://127.0.0.1:8001/api/auth/"
    data = {
        'username': username,
        'password': password,
    }
    token = requests.post(endpoint, json=data)
    if token:
        return token


col1, col2, col3, _, _, _, _ = st.columns(7)
with col1:
    username = st.text_input('Username:')
with col2:
    password = st.text_input('Password:')
col1, col2, _ = st.columns(3)
with col1:
    if st.button("Authorize"):
        get_response = request_token(username, password)
        if get_response != None and get_response.status_code == 200:
            st.session_state['token']['timestamp'] = datetime.datetime.now()
            st.session_state['token']['token'] = get_response.json()['token']
        else:
            st.session_state['token']['token'] = None
            st.error(f"Something went wrong!")
            st.error(f"Please double check your credentials!")

if st.session_state['token']['token'] != None:
    st.success("Token is received!")
    keyword = st.text_input("Search keyword:")