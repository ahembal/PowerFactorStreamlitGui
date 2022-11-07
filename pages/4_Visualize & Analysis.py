import datetime

import pandas as pd
import streamlit as st
import requests

st.set_page_config(
    layout='wide'
)

if 'token' not in st.session_state: st.session_state['token'] = { 'timestamp':datetime.datetime.now(), 'token': None}


@st.cache
def get_experiments_list():
    endpoint = "https://bioxapi.balsever.com/experiments/"
    get_response = requests.get(endpoint)
    return get_response
def request_token(username, password):
    endpoint = "https://bioxapi.balsever.com/api/auth/"
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
            st.session_state['token']['id'] = get_response.json()['id']
        else:
            st.session_state['token']['token'] = None
            st.error(f"Something went wrong!")
            st.error(f"Please double check your credentials!")

if st.session_state['token']['token'] != None:
    st.success("Token is received! You're authorized!")

    keyword = st.text_input("Search in comments:")

    experiments_lst = get_experiments_list()
    exp_df = pd.DataFrame(experiments_lst.json())
    show_df = exp_df
    if st.checkbox("Only show my experiments"):
        show_df = show_df[show_df['user_id'] == st.session_state['token']['id']]
    if keyword:
        show_df = show_df[show_df['comments'].str.contains(keyword, case=False)]

    st.write(show_df)