import json
import os
import streamlit as st
import pandas as pd
import requests
import datetime

cur_dir = os.getcwd()
st.set_page_config(
    layout='wide'
)

if 'input_variables' not in st.session_state: st.session_state['input_variables'] = {
    'iv_len':1,
    'zero_bias_point': 0,
    'interp_len': 100,
    'smooth_param_iv': 1,
    'smooth_param_temp': 1,
    'dvdt_polyfit_order': 1,
    'seebeck_polyfit_order': 0,
    'film_thickness_micrometer': 50,
    'film_width_micrometer': 50,
    'film_length_micrometer': 50,
    'img_dpi': 300,
    'img_show': False,
    'show_summary': False,
    'delimiter_csv_file': '\t',
    'fig_no': 0,
    'delimiter_type_meas': '\t',
    'Time_index': 0,
    'Voltage_index': 1,
    'Current_index': 2,
    'Resistance_index': 3,
    'skip_meas': 23,
    'delimiter_type_temp': ',',
    'T_time_index': 0,
    'T_low_index': 1,
    'T_high_index': 2,
    'skip_temp': 1

}
if 'experiment_parameters' not in st.session_state: st.session_state['experiment_parameters'] = {
    'exp_date': '',
    'exp_number': '',
    'sample_name': '',
    'powder_composition': '',
    'synthesis_method': '',
    'deposition': '',
    'substrate': '',
    'nanoparticle_percentage': '',
    'polymer': '',
    'polymer_percentage': '',
    'linker_composite': '',
    'linker_percentage': '',
    'geometry': '',
    'thickness': '',
    'comments': '',
}
if 'available_experiment_parameters' not in st.session_state: st.session_state['available_experiment_parameters'] = {
    'powder_composition': ['Bi2Te3', 'Sb2Te3', 'BiSbTe'],
    'synthesis_method': ['Hydrothermal', 'Polyol', 'Thermolysis- Oleic'],
    'polymer': ['PMMA', 'PVDF'],
    'substrate': ['Glass', 'Lines - 10μm', 'Wafer'],
    'deposition': ['Dr. Blading', 'Spin Coating', 'EPD'],
    'nanoparticle_percentage': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'polymer_percentage': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'linker_composite': ['DDT', 'HDT'],
    'linker_percentage': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
if 'execute' not in st.session_state: st.session_state['execute'] = False
if 'token' not in st.session_state: st.session_state['token'] = { 'timestamp':datetime.datetime.now(), 'token': None}
if 'temp' not in st.session_state: st.session_state['temp'] = {}

abbrv = {
    'aep': 'available_experiment_parameters',
    'expar': 'experiment_parameters'
}


if os.path.exists(f'{cur_dir}/data/results/DATABASE_8.csv'):
    st.session_state['execute'] = True
    st.success("Results found! You can set the parameters and save the result into DB.")
    res_vol_temp = pd.read_csv(f'{cur_dir}/data/results/DATABASE_8.csv')
else:
    st.error("No results found")

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.session_state[abbrv['expar']]['exp_date'] = st.date_input('Experiment Date:')
with col2:
    st.session_state[abbrv['expar']]['exp_number'] = st.text_input('Experiment Number:', placeholder='NXXX')
with col3:
    st.session_state[abbrv['expar']]['sample_name'] = st.text_input('Sample Name:')

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.session_state[abbrv['expar']]['powder_composition'] = st.selectbox('Powder Composition:', options=st.session_state[abbrv['aep']]['powder_composition'])
with col2:
    st.session_state[abbrv['expar']]['synthesis_method'] = st.selectbox('Synthesis Method:', options=st.session_state[abbrv['aep']]['synthesis_method'], help="Available options are 'Hydrothermal', 'Polyol', and 'Thermolysis- Oleic'")
with col3:
    st.session_state[abbrv['expar']]['deposition'] = st.selectbox('Deposition:', options=st.session_state[abbrv['aep']]['deposition'])
with col4:
    st.session_state[abbrv['expar']]['substrate'] = st.selectbox('Substrate:', options=st.session_state[abbrv['aep']]['substrate'])


col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.session_state[abbrv['expar']]['nanoparticle_percentage'] = st.selectbox('Nanopartical Percentage(%):', options=st.session_state[abbrv['aep']]['nanoparticle_percentage'])
with col2:
    st.session_state[abbrv['expar']]['polymer'] = st.selectbox('Polymer:', options=st.session_state[abbrv['aep']]['polymer'])
with col3:
    st.session_state[abbrv['expar']]['polymer_percentage'] = st.selectbox('Polymer Percentage(%):', options=st.session_state[abbrv['aep']]['polymer_percentage'])
with col4:
    st.session_state[abbrv['expar']]['linker_composite'] = st.selectbox('Linker Composite:', options=st.session_state[abbrv['aep']]['linker_composite'])
with col5:
    st.session_state[abbrv['expar']]['linker_percentage'] = st.selectbox('Linker Percentage(%):', options=st.session_state[abbrv['aep']]['linker_percentage'])
with col6:
    st.session_state[abbrv['expar']]['geometry'] = st.text_input('Geometry:')
with col7:
    st.session_state['input_variables']['film_thickness_micrometer'] = st.number_input("Film Thickness(micrometer):", value=50)



st.session_state[abbrv['expar']]['comments'] = st.text_input('Comment:')

columns = ["Date", "Number", "Sample Name", "Powder Comp",	"Synth. Method", "Deposition", "Substrate",
           "Nanoparticle Percentage", "Polymer", "Polymer Percentage",
          "Linker Comp.", "Linker Percentage",
        "Geometry", "Thickness", "Comments"]

row_data = [
    st.session_state[abbrv['expar']]['exp_date'],
    st.session_state[abbrv['expar']]['exp_number'],
    st.session_state[abbrv['expar']]['sample_name'],
    st.session_state[abbrv['expar']]['powder_composition'],
    st.session_state[abbrv['expar']]['synthesis_method'],
    st.session_state[abbrv['expar']]['deposition'],
    st.session_state[abbrv['expar']]['substrate'],
    st.session_state[abbrv['expar']]['nanoparticle_percentage'],
    st.session_state[abbrv['expar']]['polymer'],
    st.session_state[abbrv['expar']]['polymer_percentage'],
    st.session_state[abbrv['expar']]['linker_composite'],
    st.session_state[abbrv['expar']]['linker_percentage'],
    st.session_state[abbrv['expar']]['geometry'],
    st.session_state['input_variables']['film_thickness_micrometer'],
    st.session_state[abbrv['expar']]['comments'],
]

database_row = {}
for k, v in zip(columns, row_data):
    database_row[k] = v
df = pd.DataFrame([database_row])
st.write(df)


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
    if st.button("Get Token"):
        get_response = request_token(username, password)
        if get_response != None and get_response.status_code == 200:
            st.session_state['token']['timestamp'] = datetime.datetime.now()
            st.session_state['token']['token'] = get_response.json()['token']
            st.session_state['token']['id'] = get_response.json()['id']
        else:
            st.session_state['token']['token'] = None
            st.error(f"Something went wrong!")
            st.error(f"Please, check your credentials again!")

def records_to_db_serializer(endpoint, headers, experiment_id, res_vol_temp):
    responses = {}
    for i in range(100):

        res = res_vol_temp.iloc[i][0]
        vol = res_vol_temp.iloc[i+100][0]
        temp = res_vol_temp.iloc[i+200][0]

        post_record_data = {
            'experiment_id': experiment_id,
            'measurement_order': i+1,
            'resistance': res,
            'voltage': vol,
            'temp': temp,

        }
        response = requests.post(endpoint, json=post_record_data, headers=headers)
        responses[i] = response.json()
    return responses

if st.session_state['token']['token'] != None:
    st.success("Token is received!")
if not st.session_state['execute']:
    st.error("Nothing to save into DB!")
if st.session_state['execute']:
    # st.write(st.session_state['token'])
    post_experiment_data = {
        "user_id": st.session_state['token']['id'],
        "experiment_number": st.session_state[abbrv['expar']]['exp_number'],
        "experiment_date": str(st.session_state[abbrv['expar']]['exp_date']),
        "sample_name": st.session_state[abbrv['expar']]['sample_name'],
        "powder_comp": st.session_state[abbrv['expar']]['powder_composition'],
        "synthesis_method": st.session_state[abbrv['expar']]['synthesis_method'],
        "polymer": st.session_state[abbrv['expar']]['polymer'],
        "substrate": st.session_state[abbrv['expar']]['substrate'],
        "deposition": st.session_state[abbrv['expar']]['deposition'],
        "nanoparticle_percentage": st.session_state[abbrv['expar']]['nanoparticle_percentage'],
        "polymer_percentage": st.session_state[abbrv['expar']]['polymer_percentage'],
        "linker_composite": st.session_state[abbrv['expar']]['linker_composite'],
        "linker_percentage": st.session_state[abbrv['expar']]['linker_percentage'],
        "geometry": st.session_state[abbrv['expar']]['geometry'],
        "thickness": st.session_state['input_variables']['film_thickness_micrometer'],
        "comments": st.session_state[abbrv['expar']]['comments']
    }
    headers = {}

    if st.button("Save data to DB"):
        endpoint = "https://bioxapi.balsever.com/experiments/"
        headers['Authorization'] = f"Token {st.session_state['token']['token']}"
        response = requests.post(endpoint, json=post_experiment_data, headers=headers)
        if response.status_code == 400:
            st.error("Please check below for error!")
        elif response.status_code == 201:
            st.success("Data saved to DB, please check below for details!")
            st.session_state['temp']['pk'] = response.json()['pk']

            endpoint = "https://bioxapi.balsever.com/experiments/records/"
            experiment_id = st.session_state['temp']['pk']
            responses =  records_to_db_serializer(endpoint, headers, experiment_id, res_vol_temp)
            # st.write(responses)

        st.write(response.json())



# database_csv = pd.read_csv(f'{cur_dir}/data/results/DATABASE_8.csv')
# with open(f'{cur_dir}/data/results/RESULTS.csv') as f:
#     results_csv = f.readlines()
#     results_csv = results_csv[17:]
# st.write(database_csv)
# st.write(results_csv)


def filter_record_in_db(df):
    pass


def add_record_to_db(df):
    pass


def check_uniqueness_of_result(df):
    query_result = filter_record_in_db(df)
    if query_result:
        return query_result
    else:
        return 'unique'


# if st.session_state['execute']:
#     if st.button('Add record to DB'):
#         result = check_uniqueness_of_result(df)
#         if result == 'unique':
#             response = add_record_to_db(df)
#             if response == 'success':
#                 st.success('Result successfully recorded in the DB.')
#             else:
#                 st.error('Something went wrong!')
#         else:
#             st.error('The results already exist in the DB!')
#             st.write(result)
#
#
