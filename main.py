import os
import time

import streamlit as st
# Core Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd
# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Agg")
import seaborn as sns

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

if 'results' not in st.session_state:
    st.session_state['results'] = None
    st.session_state['execute'] = None

from calculation import main_run
def main():
    """Semi Automated ML App with Streamlit """
    cur_dir = os.getcwd()
    if not os.path.exists(f"{cur_dir}/data/input/uploaded"):
        os.mkdir(f"{cur_dir}/data/input/uploaded")
    global data_meas
    global data_csv

    def callback(state, object):
        st.session_state[f'{state}'] = object

    def calculate():
        st.sidebar.selectbox()
        print(data_meas)
        print(data_csv)

    st.sidebar.title("Power Factor Calculation")
    activities = ["About", "Plot Input Data", "Tune The Parameters", "Analysis Result", "Session Info"]
    choice = st.sidebar.selectbox("Please Select Activities", activities)

    st.sidebar.subheader("File Upload")
    data_meas = st.sidebar.file_uploader("Upload a meas file")
    data_csv = st.sidebar.file_uploader("Upload a csv file", type=["csv", "txt"])
    if data_meas and data_csv:
        st.sidebar.text("Now you can select \n"
                        "Analysis Result section \n"
                        "from drow-down-menu and \n"
                        "execute the code!")

        with open(f"{cur_dir}/data/input/uploaded/meas", "wb") as f:
            f.write(data_meas.getbuffer())
        with open(f"{cur_dir}/data/input/uploaded/Temp.csv", "wb") as f:
            f.write(data_csv.getbuffer())
    else:
        if st.sidebar.button("Load Sample Data"):
            data_meas = open(f'{cur_dir}/data/input/sample/meas', "r")
            data_csv = open(f'{cur_dir}/data/input/sample/Temp.csv', "r")

    global df_data_mess
    global df_data_csv

    if choice == 'Plot Input Data':
        st.header("Visualize Input Data")
        if data_csv:
            st.subheader("temp csv data")
            df = pd.read_csv(data_csv)
            df.plot()
            st.pyplot(fig=plt)
        # if data_meas:
        #     st.subheader("meas data")
        #     with open(data_meas) as f:
        #         print(f.readline())
        else:
            st.subheader("First you need to upload data or you can click to Load Sample Data button!")

    elif choice == 'Tune The Parameters':
        st.subheader("Tune Parameters")

        ######################### INPUT VARIABLES  ######################################
        iv_len = 1  # number of data points in one IVC, 0-1-3-5
        zero_bias_point = 0  # 1 or 0 depending on if there is an extra point
        interp_len = 100  # interpolation length - rec value around the number of measured IVCs
        smooth_param_iv = 1  # gaussian smoothing parameter for the IV data , min. rec value 4 - 10 or interp_len/10
        smooth_param_temp = 1  # gaussian smoothing parameter for the Temp data , min. rec value 4 - 10 or interp_len/10
        dvdt_polyfit_order = 1  # Order of the polynomil fit for DV/DT plot, 0 for average, 1 for linear fit, 2 for 2nd degree poly fit
        seebeck_polyfit_order = 0  # Order of the polynomil fit for Seebeck plot, 0 for average, 1 for linear fit, 2 for 2nd degree poly fit
        film_thickness_micrometer = 50  # Film thickness in the unit of micrometer, for Power Factor calculations
        img_dpi = 300  # Resolution of the saved images
        img_show = False  # Show images before saving, True or False
        show_summary = False  # Save summary
        delimiter_csv_file = '\t'  # Delimiter type for created text files (not for the IVC or Temp files)
        fig_no = 0  # Starting value for the figure num, rec value 0

        ######################### OTHER VARIABLES  ######################################
        delimiter_type_meas = '\t'  # Delimiter type the IVC data file
        Time_index = 0  # Column number of the Time Index in the  IVC file (0 means column 1, 1 means column 2, etc ...)
        Voltage_index = 1  # Column number of the Voltage data in the  IVC file (0 means column 1, 1 means column 2, etc ...)
        Current_index = 2  # Column number of the Current data in the  IVC file (0 means column 1, 1 means column 2, etc ...)
        Resistance_index = 3  # Column number of the Resistance data in the  IVC file (0 means column 1, 1 means column 2, etc ...)
        skip_meas = 23  # Number of rows, that will be skipped at the beginning of IVC data file

        delimiter_type_temp = ','  # Delimiter type the Temperature data file
        T_time_index = 0  # Column number of the Time Index in the Temperature file (0 means column 1, 1 means column 2, etc ...)
        T_low_index = 1  # Column number of the Cold-side measurement in the Temperature file (0 means column 1, 1 means column 2, etc ...)
        T_high_index = 2  # Column number of the Hot-side measurement in the Temperature file (0 means column 1, 1 means column 2, etc
        skip_temp = 1  # Number of rows, that will be skipped at the beginning of Temperature data file

    elif choice == 'Analysis Result':
        st.subheader("Exploratory Data Analysis")
        if (
                st.button('EXECUTE', on_click=callback, args=('execute', True))
            ):
            st.warning('Are you sure you want to execute this?')

            st.button('YES', on_click=main_run)

        if st.session_state['results']:

            fig_list = [i for i in os.listdir(f"{cur_dir}/data/results") if ".png" in i]
            fig_list = sorted(fig_list, key=lambda x: int(x.split("_")[1].split('.')[0]))
            for i in fig_list:
                fig = plt.imread(f"{cur_dir}/data/results/{i}")
                st.subheader(i.split('.')[0])
                st.image(fig)
        # if (
        #         st.button('EXECUTE', on_click=callback, args=('execute', True))
        #         or st.session_state.execute == True
        # ):
        #     st.warning('Are you sure you want to execute this?')
        #     if st.button('YES'):
        #         print('test11')
        #         st.write('You did it!')
        #         st.session_state.execute = False


    elif choice == 'About':
        st.subheader("About Page")
        text = "This page will contain information about the software!"

        def example(color1, color2, color3, content):
            st.markdown(
                f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:24px;border-radius:2%;">{content}</p>',
                unsafe_allow_html=True)

        color1 = '#1aa3ff'
        color2 = '#00ff00'
        color3 = '#ffffff'
        example(color1, color2, color3, text)

    elif choice == 'Session Info':
        st.subheader("Session Info")

        "st.session_state object: ", st.session_state
        st.info


if __name__ == '__main__':
    main()

# col1, col2, col3 = st.beta_columns(3)
# with col1:
#     color1 = st.color_picker('col1', '#1aa3ff', key=1)
# st.write(f"col1{color1}")
# with col2:
#     color2 = st.color_picker('col2', '#00ff00', key=2)
# st.write(f"col2{color2}")
# with col3:
#     color3 = st.color_picker('col3', '#ffffff', key=3)
# st.write(f"col3{color3}")
# text = st.text_input("text input")

