import csv
import os
# Core Pkgs
from collections import defaultdict

import pandas
import streamlit as st

# EDA Pkgs
import pandas as pd
# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
from calculation import main_run

matplotlib.use("Agg")

#  Instantiating parameters
if 'results' not in st.session_state: st.session_state['results'] = None
if 'execute' not in st.session_state: st.session_state['execute'] = None
if 'loaded_sample_data' not in st.session_state: st.session_state['loaded_sample_data'] = False
if 'uploaded_input_data' not in st.session_state: st.session_state['uploaded_input_data'] = False
if 'experiment_parameters' not in st.session_state: st.session_state['experiment_parameters'] = {}
if 'meas_path' not in st.session_state: st.session_state['meas_path'] = None
if 'csv_path' not in st.session_state: st.session_state['csv_path'] = None
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
if 'plot_returns' not in st.session_state: st.session_state['plot_returns'] = {}


def reload_def_params():
    st.session_state['input_variables'] = {
        'iv_len': 1,
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


def main():
    cur_dir = os.getcwd()
    required_paths = ['data', 'data/input', 'data/results']
    check_list = [(os.path.exists(i), i) for i in [cur_dir+'/'+i for i in required_paths]]
    if False in check_list:
        false_list = [i for i in check_list if i[0] is False]
        for i in false_list:
            os.mkdir(i[1])

    st.sidebar.title("Power Factor Calculation")
    activities = ["About", "Plot Input Data", "Tune The Parameters", "Analysis & Results", "Session Info"]
    choice = st.sidebar.selectbox("Please Select Activities", activities)

    def clean_upload_dir():
        if os.path.exists(f'{cur_dir}/data/input/uploaded'):
            files_input = os.listdir(f'{cur_dir}/data/input/uploaded')
            files_input = [f'{cur_dir}/data/input/uploaded/{file}' for file in files_input]
            tobe_deleted_files = files_input
            for file in tobe_deleted_files:
                if os.path.exists(file):
                    os.remove(file)
        st.session_state['uploaded_input_data'] = False

    def return_format(delimeter):
        if delimeter == "\t":
            return "\\t"
        elif delimeter == ",":
            return ","

    st.sidebar.subheader("File Upload")
    data_meas = st.sidebar.file_uploader("Upload a meas file")
    data_csv = st.sidebar.file_uploader("Upload a csv file", type=["csv", "txt"])

    if data_meas and data_csv:
        st.sidebar.success("Now you can select \n"
                        "Analysis Result section \n"
                        "from drow-down-menu and \n"
                        "execute the code!")
        st.session_state['meas_path'] = f"{cur_dir}/data/input/uploaded/meas"
        st.session_state['csv_path'] = f"{cur_dir}/data/input/uploaded/Temp.csv"
        st.session_state['uploaded_input_data'] = True
        st.session_state['loaded_sample_data'] = False
        if st.session_state['results'] and not st.session_state['execute']:
            st.warning("You uploaded new data, but your results are old, if you would like to re execute the code, first you need to delete results from Session info section!")
        with open(f"{cur_dir}/data/input/uploaded/meas", "wb") as f:
            f.write(data_meas.getbuffer())
        with open(f"{cur_dir}/data/input/uploaded/Temp.csv", "wb") as f:
            f.write(data_csv.getbuffer())
    else:
        clean_upload_dir()

    if not st.session_state['loaded_sample_data'] and not st.session_state['uploaded_input_data']:
        if st.sidebar.button("Load Sample Data"):
            st.session_state['meas_path'] = f'{cur_dir}/data/input/sample/meas'
            st.session_state['csv_path'] = f'{cur_dir}/data/input/sample/Temp.csv'
            st.session_state['loaded_sample_data'] = True


    if choice == 'Plot Input Data':
        st.header("Visualize Input Data")
        if st.session_state['uploaded_input_data'] or st.session_state['loaded_sample_data']:
            st.subheader("temp csv data")
            if st.session_state['uploaded_input_data']:
                df = pd.read_csv(st.session_state['csv_path'])
                meas_columns = defaultdict(list)
                with open(st.session_state['meas_path']) as f:
                    meas_data = csv.reader(f, delimiter=st.session_state['input_variables']['delimiter_type_meas'])
                    for skip in range(23):
                        next(meas_data)
                    for row in meas_data:
                        for (i, v) in enumerate(row):
                            meas_columns[i].append(v)
                df_meas = pd.DataFrame(meas_columns)
            elif st.session_state['loaded_sample_data']:
                df = pd.read_csv(st.session_state['csv_path'])
                meas_columns = defaultdict(list)
                with open(st.session_state['meas_path']) as f:
                    meas_data = csv.reader(f, delimiter=st.session_state['input_variables']['delimiter_type_meas'])
                    for skip in range(23):
                        next(meas_data)
                    for row in meas_data:
                        for (i, v) in enumerate(row):
                            meas_columns[i].append(v)
                df_meas = pd.DataFrame(meas_columns)
            else:
                st.warning("Something went wrong with file upload/load!")
            df1 = df_meas[1].apply(pd.to_numeric)
            df2 = df_meas[2].apply(pd.to_numeric)
            df3 = df_meas[3].apply(pd.to_numeric)
            fig, ax = plt.subplots(4, 1, figsize=(11, 22))
            matplotlib.pyplot.subplots_adjust(wspace=0.5, hspace=0.5)
            ax1 = plt.subplot(411)
            ax2 = plt.subplot(412)
            ax3 = plt.subplot(413)
            ax4 = plt.subplot(414)
            df.plot(ax=ax1)
            df1.plot(ax=ax2, title='Voltage')
            df2.plot(ax=ax3, title='Current')
            df3.plot(ax=ax4, title='Resistance')
            st.pyplot(fig=plt, )
        else:
            st.subheader("First you need to upload data or you can click to Load Sample Data button!")

    elif choice == 'Tune The Parameters':
        st.header("Tune Parameters")
        st.button('ReLoad default parameters!', on_click=reload_def_params)

        ######################### INPUT VARIABLES  ######################################
        st.subheader("Input Variables")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state['input_variables']['film_thickness_micrometer'] = st.number_input('Film Thickness micrometer', 0, 100, value=st.session_state['input_variables']['film_thickness_micrometer'], help="Film thickness in the unit of micrometer, for Power Factor calculations")
        with col2:
            st.session_state['input_variables']['film_width_micrometer'] = st.number_input('Film Width micrometer', 0, 100, help="Film thickness in the unit of micrometer, for Power Factor calculations")
        with col3:
            st.session_state['input_variables']['film_length_micrometer'] = st.number_input('Film Length micrometer', 0, 100, help="Film thickness in the unit of micrometer, for Power Factor calculations")

        # st.session_state['input_variables']['iv_len'] = st.selectbox('IV length', [0, 1, 3, 5], index=[0, 1, 3, 5].index(st.session_state['input_variables']['iv_len']), help="number of data points in one IVC, 0-1-3-5")
        # st.session_state['input_variables']['zero_bias_point'] = st.radio('Zero Bias Point', [0, 1], index=[0, 1].index(st.session_state['input_variables']['zero_bias_point']), horizontal=True, help="1 or 0 depending on if there is an extra point")
        # st.session_state['input_variables']['interp_len'] = st.slider('Interpolation Length', 50, 150, value=st.session_state['input_variables']['interp_len'], help="interpolation length - rec value around the number of measured IVCs")
        # st.session_state['input_variables']['smooth_param_iv'] = st.number_input('Smooth Param IV', 0, 10, value=st.session_state['input_variables']['smooth_param_iv'], help="gaussian smoothing parameter for the IV data , min. rec value 4 - 10 or interp_len/10")
        # st.session_state['input_variables']['smooth_param_temp'] = st.number_input('Smooth Param Temp', 0, 10, value=st.session_state['input_variables']['smooth_param_temp'], help="gaussian smoothing parameter for the Temp data , min. rec value 4 - 10 or interp_len/10")
        # st.session_state['input_variables']['dvdt_polyfit_order'] = st.selectbox('Order of the Polynomial fit', [0, 1, 2], index=[0, 1, 2].index(st.session_state['input_variables']['dvdt_polyfit_order']), help="Order of the polynomial fit for DV/DT plot, 0 for average, 1 for linear fit, 2 for 2nd degree poly fit")
        # st.session_state['input_variables']['seebeck_polyfit_order'] = st.selectbox('Order of the Polynomial fit', [0, 1, 2], index=[0, 1, 2].index(st.session_state['input_variables']['seebeck_polyfit_order']), help="Order of the polynomil fit for Seebeck plot, 0 for average, 1 for linear fit, 2 for 2nd degree poly fit")
        # st.session_state['input_variables']['img_dpi'] = st.number_input('Img Dpi', 200, 400, value=st.session_state['input_variables']['img_dpi'], help="Resolution of the saved images")
        # st.session_state['input_variables']['img_show'] = st.radio('Img Show', [False, True], index=[False, True].index(st.session_state['input_variables']['img_show']), horizontal=True, help="Show images before saving, True or False")
        # st.session_state['input_variables']['show_summary'] = st.radio('Img Show', [False, True], index=[False, True].index(st.session_state['input_variables']['show_summary']), horizontal=True, help="Save summary")
        # st.session_state['input_variables']['delimiter_csv_file'] = st.selectbox('Delimiter type', ["\t", ","], index=["\t", ","].index(st.session_state['input_variables']['delimiter_csv_file']), format_func=return_format, help="Delimiter type for created text files (not for the IVC or Temp files)")
        # st.session_state['input_variables']['fig_no'] = st.number_input('Fig no', 0, 100, value=st.session_state['input_variables']['fig_no'], help="Starting value for the figure num, rec value 0")

        ######################### OTHER VARIABLES  ######################################
        # st.subheader("Other Variables")
        # st.write("Meas File variables:")
        # st.session_state['input_variables']['delimiter_type_meas'] = st.selectbox('Delimiter type meas', ["\t", ","], index=["\t", ","].index(st.session_state['input_variables']['delimiter_type_meas']), format_func=return_format, help="Delimiter type the IVC data file")
        # st.session_state['input_variables']['Time_index'] = st.number_input('Time Index', 0, 5, value=st.session_state['input_variables']['Time_index'], help="Column number of the Time Index in the  IVC file (0 means column 1, 1 means column 2, etc ...)")
        # st.session_state['input_variables']['Voltage_index'] = st.number_input('Voltage Index', 0, 5, value=st.session_state['input_variables']['Voltage_index'], help="Column number of the Voltage data in the  IVC file (0 means column 1, 1 means column 2, etc ...)")
        # st.session_state['input_variables']['Current_index'] = st.number_input('Current Index', 0, 5, value=st.session_state['input_variables']['Current_index'], help="Column number of the Current data in the  IVC file (0 means column 1, 1 means column 2, etc ...)")
        # st.session_state['input_variables']['Resistance_index'] = st.number_input('Resistance Index', 0, 5, value=st.session_state['input_variables']['Resistance_index'], help="Column number of the Resistance data in the  IVC file (0 means column 1, 1 means column 2, etc ...)")
        # st.session_state['input_variables']['skip_meas'] = st.number_input('Number of Rows to skip in meas file', 0, 30, value=st.session_state['input_variables']['skip_meas'], help="Number of rows, that will be skipped at the beginning of IVC data file")
        # st.write("-------------------------------------------------------------------------")
        # st.write("Temp File variables:")
        # st.session_state['input_variables']['delimiter_type_temp'] = st.selectbox('Delimiter type temp', ["\t", ","], index=["\t", ","].index(st.session_state['input_variables']['delimiter_type_temp']), format_func=return_format, help="Delimiter type the Temperature data file")
        # st.session_state['input_variables']['T_time_index'] = st.number_input('T time Index', 0, 5, value=st.session_state['input_variables']['T_time_index'], help="Column number of the Time Index in the Temperature file (0 means column 1, 1 means column 2, etc ...)")
        # st.session_state['input_variables']['T_low_index'] = st.number_input('T low Index', 0, 5, value=st.session_state['input_variables']['T_low_index'], help="Column number of the Cold-side measurement in the Temperature file (0 means column 1, 1 means column 2, etc ...)")
        # st.session_state['input_variables']['T_high_index'] = st.number_input('T high Index', 0, 5, value=st.session_state['input_variables']['T_high_index'], help="Column number of the Hot-side measurement in the Temperature file (0 means column 1, 1 means column 2, etc")
        # st.session_state['input_variables']['skip_temp'] = st.number_input('Number of Rows to skip in temp file', 0, 30, value=st.session_state['input_variables']['skip_temp'], help="Number of rows, that will be skipped at the beginning of Temperature data file")

    elif choice == 'Analysis & Results':
        st.subheader("Exploratory Data Analysis")

        if st.session_state['results']:
            fig_list = [i for i in os.listdir(f"{cur_dir}/data/results") if ".png" in i]
            csv_list = [i for i in os.listdir(f"{cur_dir}/data/results") if "Figure" in i]
            fig_list = sorted(fig_list, key=lambda x: int(x.split("_")[1].split('.')[0]))

            for i in fig_list:
                st.title(i.split('.')[0])
                fig = plt.imread(f"{cur_dir}/data/results/{i}")
                number = i.split('.')[0].split('_')[1]
                search_text = f"Figure_{number}.csv"
                search_text_2 = f"Figure_{number}_part"

                csvs = [c for c in csv_list if search_text in c or search_text_2 in c]
                # st.write(csvs)

                if len(csvs) == 1:
                    with open(f"{cur_dir}/data/results/{csvs[0]}") as f:
                            st.download_button(f'{csvs[0]}', f, file_name=csvs[0])  # Defaults to 'text/plain'
                elif len(csvs) == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(f"{cur_dir}/data/results/{csvs[0]}") as f:
                            st.download_button(f'{csvs[0]}', f, file_name=csvs[0])
                    with col2:
                        with open(f"{cur_dir}/data/results/{csvs[1]}") as f:
                            st.download_button(f'{csvs[1]}', f, file_name=csvs[1])

                elif len(csvs) == 3:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with open(f"{cur_dir}/data/results/{csvs[0]}") as f:
                            st.download_button(f'{csvs[0]}', f, file_name=csvs[0])
                    with col2:
                        with open(f"{cur_dir}/data/results/{csvs[1]}") as f:
                            st.download_button(f'{csvs[1]}', f, file_name=csvs[1])
                    with col3:
                        with open(f"{cur_dir}/data/results/{csvs[2]}") as f:
                            st.download_button(f'{csvs[2]}', f, file_name=csvs[2])
                # col1, col2 = st.columns(2)
                # with col1:
                #     st.subheader(i.split('.')[0])
                # with col2:
                #     for c in csvs:
                #         with open(f"{cur_dir}/data/results/{c}") as f:
                #             st.download_button(f'Download CSV - {c}', f,
                #                                file_name='RESULTS.csv')  # Defaults to 'text/plain'
                st.image(fig)

        elif st.session_state['uploaded_input_data'] or st.session_state['loaded_sample_data']:
            if (
                    st.button('EXECUTE', args=('execute', True))
            ):
                st.warning('Are you sure you want to execute this?')

                st.button('YES', on_click=main_run)
                st.session_state['results'] = True
                st.session_state['execute'] = True
        else:
            st.warning('You have not executed any analysis and/or uploaded any data!')

    elif choice == 'About':
        st.header("About Page")
        text = "KTH-BIOX LAB POWER FACTOR CALCULATION GUI INTERFACE!"

        def example(color1, color2, color3, content):
            st.markdown(
                f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:24px;border-radius:2%;">{content}</p>',
                unsafe_allow_html=True)

        color1 = '#1aa3ff'
        color2 = '#00ff00'
        color3 = '#ffffff'
        example(color1, color2, color3, text)
        st.subheader("Purpose:")
        st.write("Purpose of this software is to help researchers to run explore and visualize their experiment through this GUI.")
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

    elif choice == 'Session Info':
        st.subheader("Session Info")

        def clean_results_dir():
            files = os.listdir(f'{cur_dir}/data/results')
            for file in files:
                os.remove(f'{cur_dir}/data/results/{file}')
            st.session_state['results'] = False
            st.session_state['loaded_sample_data'] = False
            st.session_state['uploaded_input_data'] = False

        # "st.session_state object: ", st.session_state
        if os.path.exists(f'{cur_dir}/data/results/RESULTS.csv'):

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""<style> 
                            div.stButton > button:first-child {} </style>""",
                            unsafe_allow_html=True)
                with open('data/results/RESULTS.csv') as f:
                    st.download_button('Download RESULTS as CSV', f, file_name='RESULTS.csv')  # Defaults to 'text/plain'
            with col2:
                st.markdown("""<style> 
                            div.stButton > button:first-child {} </style>""",
                            unsafe_allow_html=True)
                with open('data/results/DATABASE_8.csv') as f:
                    st.download_button('Download DATABASE_8', f, file_name='DATABASE_8.csv')  # Defaults to 'text/plain'
            with col3:
                st.markdown("""<style> 
                            div.stButton > button:first-child { background-color: rgb(211, 77, 77); } </style>""",
                            unsafe_allow_html=True)
                if st.button('Delete Results!'):
                    st.warning("All results will be deleted!")
                    st.button('Yes, Delete!', on_click=clean_results_dir)

            st.subheader("CSV context:")
            # text, csv = split_result_csv()
            with open('data/results/RESULTS.csv') as f:
                meas_data = csv.reader(f, delimiter=st.session_state['input_variables']['delimiter_type_meas'])
                for row in meas_data:
                    row = str(row).replace("[", " ").replace("]", " ").replace("'", " ").replace('"', ' ')
                    st.text(row)
            # df_res = pd.read_csv('data/results/RESULTS.csv', skiprows=23)
            # st.write(df_res)
        else:
            st.warning("If you would like to see and download *RESULTS.csv*:"
                       "\n\nFirst, you should upload files and run the analysis in Analysis & Results page!")


if __name__ == '__main__':
    main()
