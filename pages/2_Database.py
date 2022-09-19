import os
import streamlit as st
import pandas as pd

cur_dir = os.getcwd()
st.set_page_config(
    layout='wide'
)

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

abbrv = {
    'aep': 'available_experiment_parameters',
    'expar': 'experiment_parameters'
}

if os.path.exists(f'{cur_dir}/data/results/RESULTS.csv'):
    st.success("Results found! You can set the parameters and record the result into DB.")
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
for k,v in zip(columns, row_data):
    database_row[k] = v

df = pd.DataFrame([database_row])
st.write(df)

database_csv = pd.read_csv(f'{cur_dir}/data/results/DATABASE_8.csv')
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


if st.button('Add record to DB'):
    result = check_uniqueness_of_result(df)
    if result == 'unique':
        response = add_record_to_db(df)
        if response == 'success':
            st.success('Result successfully recorded in the DB.')
        else:
            st.error('Something went wrong!')
    else:
        st.error('The results already exist in the DB!')
        st.write(result)


