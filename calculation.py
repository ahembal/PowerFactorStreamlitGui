import os



def main_run():
    import streamlit as st
    import csv
    import numpy
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    import math
    import sys
    from numpy import diff

    ######################### INPUT VARIABLES  ######################################

    iv_len = st.session_state['input_variables']['iv_len']   # number of data points in one IVC, 0-1-3-5
    zero_bias_point = st.session_state['input_variables']['zero_bias_point']  # 1 or 0 depending on if there is an extra point
    interp_len = st.session_state['input_variables']['interp_len']  # interpolation length - rec value around the number of measured IVCs
    smooth_param_iv = st.session_state['input_variables']['smooth_param_iv']  # gaussian smoothing parameter for the IV data , min. rec value 4 - 10 or interp_len/10
    smooth_param_temp = st.session_state['input_variables']['smooth_param_temp']  # gaussian smoothing parameter for the Temp data , min. rec value 4 - 10 or interp_len/10
    dvdt_polyfit_order = st.session_state['input_variables']['dvdt_polyfit_order']  # Order of the polynomil fit for DV/DT plot, 0 for average, 1 for linear fit, 2 for 2nd degree poly fit
    seebeck_polyfit_order = st.session_state['input_variables']['seebeck_polyfit_order']  # Order of the polynomil fit for Seebeck plot, 0 for average, 1 for linear fit, 2 for 2nd degree poly fit
    film_thickness_micrometer = st.session_state['input_variables']['film_thickness_micrometer']  # Film thickness in the unit of micrometer, for Power Factor calculations
    img_dpi = st.session_state['input_variables']['img_dpi']  # Resolution of the saved images
    img_show = st.session_state['input_variables']['img_show']  # Show images before saving, True or False
    show_summary = st.session_state['input_variables']['show_summary']  # Save summary
    delimiter_csv_file = st.session_state['input_variables']['delimiter_csv_file']  # Delimiter type for created text files (not for the IVC or Temp files)
    fig_no = st.session_state['input_variables']['fig_no']  # Starting value for the figure num, rec value 0

    ######################## INPUT FILES STRUCTURE  ################################
    cur_dir = os.getcwd()

    File_name_meas = st.session_state['meas_path']  # Name of the IVC data file
    File_name_temp = st.session_state['csv_path']  # Name of the Temperature data file

    # File_name_meas='meas_neg.txt'
    # File_name_meas='meas_only_iv'
    # File_name_meas='meas_single_point'

    # File_name_meas='meas_long.txt'
    # File_name_temp='Temp_long.csv'
    ####################################

    delimiter_type_meas = st.session_state['input_variables']['delimiter_type_meas']  # Delimiter type the IVC data file
    Time_index = st.session_state['input_variables']['Time_index']  # Column number of the Time Index in the  IVC file (0 means column 1, 1 means column 2, etc ...)
    Voltage_index = st.session_state['input_variables']['Voltage_index']  # Column number of the Voltage data in the  IVC file (0 means column 1, 1 means column 2, etc ...)
    Current_index = st.session_state['input_variables']['Current_index']  # Column number of the Current data in the  IVC file (0 means column 1, 1 means column 2, etc ...)
    Resistance_index = st.session_state['input_variables']['Resistance_index']  # Column number of the Resistance data in the  IVC file (0 means column 1, 1 means column 2, etc ...)
    skip_meas = st.session_state['input_variables']['skip_meas']  # Number of rows, that will be skipped at the beginning of IVC data file

    delimiter_type_temp = st.session_state['input_variables']['delimiter_type_temp']  # Delimiter type the Temperature data file
    T_time_index = st.session_state['input_variables']['T_time_index']  # Column number of the Time Index in the Temperature file (0 means column 1, 1 means column 2, etc ...)
    T_low_index = st.session_state['input_variables']['T_low_index']  # Column number of the Cold-side measurement in the Temperature file (0 means column 1, 1 means column 2, etc ...)
    T_high_index = st.session_state['input_variables']['T_high_index']  # Column number of the Hot-side measurement in the Temperature file (0 means column 1, 1 means column 2, etc
    skip_temp = st.session_state['input_variables']['skip_temp']  # Number of rows, that will be skipped at the beginning of Temperature data file

    ###########  PRINTING INPUT VARIABLES ##############################
    if not show_summary:
        if not os.path.exists(f'{cur_dir}/data/results/RESULTS.csv'):
            os.mkdir(f'{cur_dir}/data/results')
        sys.stdout = open(f'{cur_dir}/data/results/RESULTS.csv', "w")
    print('------------   INPUT VARIABLES   ----------------')
    print('iv_len=', iv_len)
    print('zero_bias_point=', zero_bias_point)
    print('interp_len=', interp_len)
    print('smooth_param_iv=', smooth_param_iv)
    print('smooth_param_temp=', smooth_param_temp)
    print('dvdt_polyfit_order= ', dvdt_polyfit_order)
    print('seebeck_polyfit_order= ', seebeck_polyfit_order)
    print('film_thickness_micrometer = ', film_thickness_micrometer)
    print('img_dpi=', img_dpi)
    print('img_show = ', img_show)
    print('show_summary = ', show_summary)
    ########### LOADING MEAS DATA FILE (IVC) ############################
    meas_columns = defaultdict(list)  # each value in each column is appended to a list
    with open(File_name_meas) as f:
        meas_data = csv.reader(f, delimiter=delimiter_type_meas)
        for skip in range(skip_meas):
            next(meas_data)
        for row in meas_data:
            for (i, v) in enumerate(row):
                meas_columns[i].append(v)
    print('--------------------------------------------------')
    print('There are %s columns in IVC data' % len(meas_columns))

    Time = np.array(list(np.float_(meas_columns[Time_index])))
    Voltage = np.array(list(np.float_(meas_columns[Voltage_index])))
    Current = np.array(list(np.float_(meas_columns[Current_index])))
    Resistance = np.array(list(np.float_(meas_columns[Resistance_index])))
    sum_dif_Voltage = sum(diff(Voltage))
    # print('Time:',Time)

    print('sum_dif_Voltage:', sum_dif_Voltage)
    if (sum_dif_Voltage > 0):
        print('Positive Seebeck factor')
    if (sum_dif_Voltage < 0):
        print('Negative Seebeck factor')

    ########## Creating Table ############
    # headers = ['Voltage', 'Current', 'Resistance']
    # data= [Voltage,Current,Resistance]
    # list_1 = data.tolist()
    # table = columnar(list_1, headers, no_borders=True)
    # print(table)

    # print('Voltage:',Voltage)
    # print('Current:',Current)
    # print('Resistance:',Resistance)

    ####### RAW MEAS DATA PLOTTING ###################
    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(17.5, 5))
    plt.subplot(131)
    plt.plot(Voltage, '-o')
    plt.title("Voltage (V)")
    plt.subplot(132)
    plt.plot(Current, '-o')
    plt.title("Current (A)")
    plt.subplot(133)
    plt.plot(Resistance, '-o')
    plt.title("Resistance ($\Omega$)")
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    csv_data = numpy.asarray([Voltage, Current, Resistance])
    csv_data_transpose = csv_data.transpose()
    variables = ['Voltage', 'Current', 'Resistance']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ##### Restructure Meas Data #####
    count = -1
    total_iv_len = round(iv_len + zero_bias_point)

    if (total_iv_len < 1):
        print('Problem with input parameters: iv_len + zero_bias_point can not be smaller than 1.')
        sys.exit()

    iv_number = math.floor(len(Voltage) / total_iv_len)
    # print('total_iv_len=',total_iv_len)
    print('There are %d different IVC lines.' % total_iv_len)
    print('IVCs consist of %d data points.' % iv_number)

    Voltage_2d = np.zeros((total_iv_len, iv_number))
    Current_2d = np.zeros((total_iv_len, iv_number))
    Resistance_2d = np.zeros((total_iv_len, iv_number))

    for k in range(iv_number):
        count = count + 1
        for i in range(total_iv_len):
            # print('count=',count)
            # print('total_iv_len=',total_iv_len)
            # print('index=',total_iv_len*count+i)
            Voltage_2d[i][count] = Voltage[total_iv_len * count + i]
            Current_2d[i][count] = Current[total_iv_len * count + i]
            Resistance_2d[i][count] = Resistance[total_iv_len * count + i]

            # Voltage_2d[i][count]=0
            # print(i)
    # print('Voltage_2d=', Voltage_2d)

    ######################   IVC PLOTS #################################

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(17.5, 5))
    plt.subplot(131)
    for i in range(len(Voltage_2d)):
        plt.plot(Voltage_2d[i,], '-o', label='IV_point %d' % i)
    plt.title("Voltage (V)")
    plt.subplot(132)
    for i in range(len(Voltage_2d)):
        plt.plot(Current_2d[i,], '-o', label='IV_point %d' % i)
    plt.title("Current (A)")
    plt.subplot(133)
    for i in range(len(Voltage_2d)):
        plt.plot(Resistance_2d[i,], '-o', label='IV_point %d' % i)
    plt.title("Resistance ($\Omega$)")
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    Voltage_2d_flat = Voltage_2d.flatten()
    Current_2d_flat = Current_2d.flatten()
    Resistance_2d_flat = Resistance_2d.flatten()

    csv_data = numpy.asarray([Voltage_2d_flat, Current_2d_flat, Resistance_2d_flat])
    csv_data_transpose = csv_data.transpose()
    variables = ['Voltage_2d', 'Current_2d', 'Resistance_2d']
    with open(f'{cur_dir}/data/results/figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)

    print(variables)
    print(csv_data_transpose)
    #####################################################

    ##############  IVC INTERPOLATION ################################
    Voltage_2d_interp = np.zeros((total_iv_len, interp_len))
    Voltage_2d_interp_norm = np.zeros((total_iv_len, interp_len))
    Current_2d_interp = np.zeros((total_iv_len, interp_len))
    Resistance_2d_interp = np.zeros((total_iv_len, interp_len))
    # columns = len(an_array[0])
    # print('len_v_x=',len(Voltage_2d))
    # print('len_v_y=',len(Voltage_2d[0]))

    x_old = np.linspace(0, len(Voltage_2d[0]) - 1, num=len(Voltage_2d[0]))
    x_new = np.linspace(0, len(Voltage_2d[0]) - 1, num=len(Voltage_2d_interp[0]))

    print('Interpolation length is: ', len(x_new))
    # print('len_Voltage_2d_interp[0]=',len(Voltage_2d_interp[0]))

    # print('x_old=',x_old)
    # print('Voltage_2d_interp=',Voltage_2d_interp)
    for i in range(len(Voltage_2d)):
        interp_func_voltage = interp1d(x_old, Voltage_2d[i, :], kind='cubic')
        Voltage_2d_interp[i, :] = interp_func_voltage(x_new)
        Voltage_2d_interp_norm[i, :] = Voltage_2d_interp[i, :] - min(Voltage_2d_interp[i, :])
        interp_func_current = interp1d(x_old, Current_2d[i, :], kind='cubic')
        Current_2d_interp[i, :] = interp_func_current(x_new)
        interp_func_resistance = interp1d(x_old, Resistance_2d[i, :], kind='cubic')
        Resistance_2d_interp[i, :] = interp_func_resistance(x_new)
        if (sum_dif_Voltage < 0):
            Voltage_2d_interp_norm[i, :] = Voltage_2d_interp_norm[i, :] - max(Voltage_2d_interp_norm[i, :])

        # print('interp_func_voltage',interp_func_voltage)
        # Voltage_2d_interp[i,:] = interp_func_voltage(x_new)
        # Current_2d_interp[i,:] = interp_func_current(x_new)
        # Resistance_2d_interp[i,:] = interp_func_resistance(x_new)

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(17.5, 5))
    plt.subplot(131)
    for i in range(len(Voltage_2d_interp[:, 1])):
        plt.plot(Voltage_2d_interp[i,], '-o', label='IV_point %d' % i)
    plt.title("Voltage_interp (V)")
    plt.subplot(132)
    for i in range(len(Voltage_2d_interp[:, 1])):
        plt.plot(Current_2d_interp[i,], '-o', label='IV_point %d' % i)
    plt.title("Current_interp (A)")
    plt.subplot(133)
    for i in range(len(Voltage_2d_interp[:, 1])):
        plt.plot(Resistance_2d_interp[i,], '-o', label='IV_point %d' % i)
    plt.title("Resistance_interp ($\Omega$)")
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    Voltage_2d_interp_flat = Voltage_2d_interp.flatten()
    Current_2d_interp_flat = Current_2d_interp.flatten()
    Resistance_2d_interp_flat = Resistance_2d_interp.flatten()

    csv_data = numpy.asarray([Voltage_2d_interp_flat, Current_2d_interp_flat, Resistance_2d_interp_flat])
    csv_data_transpose = csv_data.transpose()
    variables = ['Voltage_2d_interp', 'Current_2d_interp', 'Resistance_2d_interp']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)

    print(variables)
    print(csv_data_transpose)
    #####################################################

    ########################################################################
    ##### CALCULATING R_FIT ##############
    Resistance_fit = np.zeros(iv_number)
    Voltage_fit = np.zeros(iv_number)

    if (iv_len > 1):
        print("Resistance_fit from IVCs:")
        for k in range(iv_number):
            fit = np.polyfit(Current_2d[:, k], Voltage_2d[:, k], 1)
            f_fit = np.poly1d(fit)
            print('IVC_%d = ' % k, f_fit)
            Resistance_fit[k] = abs(fit[0])
            Voltage_fit[k] = fit[1]

    ######## FIT DATA INTERPOLATION #############
    Resistance_fit_interp = np.zeros(interp_len)
    x_Resistance_fit_old = np.linspace(0, len(Resistance_fit) - 1, num=len(Resistance_fit))
    x_Resistance_fit_new = np.linspace(0, len(Resistance_fit) - 1, num=len(Resistance_fit_interp))
    interp_func_fit = interp1d(x_Resistance_fit_old, Resistance_fit, kind='cubic')
    Resistance_fit_interp = interp_func_fit(x_Resistance_fit_new)

    Voltage_fit_interp = np.zeros(interp_len)
    x_Voltage_fit_old = np.linspace(0, len(Voltage_fit) - 1, num=len(Voltage_fit))
    x_Voltage_fit_new = np.linspace(0, len(Voltage_fit) - 1, num=len(Voltage_fit_interp))
    interp_func_fit = interp1d(x_Voltage_fit_old, Voltage_fit, kind='cubic')
    Voltage_fit_interp = interp_func_fit(x_Voltage_fit_new)
    Voltage_fit_interp_norm = Voltage_fit_interp - min(Voltage_fit_interp)
    if (sum_dif_Voltage < 0):
        Voltage_fit_interp_norm = Voltage_fit_interp_norm - max(Voltage_fit_interp_norm)

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(10.5, 5))
    plt.subplot(121)
    plt.plot(Voltage_fit, '-o')
    plt.title("Voltage_fit (V)")
    plt.subplot(122)
    plt.plot(Resistance_fit, '-o')
    plt.title("Resistance_fit ($\Omega$)")
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    csv_data = numpy.asarray([Voltage_fit, Resistance_fit])
    csv_data_transpose = csv_data.transpose()
    variables = ['Voltage_fit', 'Resistance_fit']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)

    print(variables)
    print(csv_data_transpose)
    #####################################################

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(10.5, 5))
    plt.subplot(121)
    plt.plot(Voltage_fit_interp, '-o')
    plt.title("Voltage_fit_interp (V)")
    plt.subplot(122)
    plt.plot(Resistance_fit_interp, '-o')
    plt.title("Resistance_fit_interp ($\Omega$)")
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    csv_data = numpy.asarray([Voltage_fit_interp, Resistance_fit_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['Voltage_fit_interp', 'Resistance_fit_interp']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ###############################################################
    ########### LOADING TEMP DATA FILE ############################
    temp_columns = defaultdict(list)  # each value in each column is appended to a list
    with open(File_name_temp) as f:
        temp_data = csv.reader(f, delimiter=delimiter_type_temp)
        for skip in range(skip_temp):
            next(temp_data)
        for row in temp_data:
            for (i, v) in enumerate(row):
                temp_columns[i].append(v)
    print('--------------------------------------------------')
    print('There are %s columns in TEMP data' % len(temp_columns))
    # print(temp_columns)

    # T_time = list(np.float_(temp_columns[T_time_index]))
    T_low = np.array(list(np.float_(temp_columns[T_low_index])))
    T_high = np.array(list(np.float_(temp_columns[T_high_index])))
    Delta_T = abs(T_high - T_low)

    # print('T_low:',T_low)
    # print('T_high:',T_high)
    # print('Delta_T:',Delta_T)

    ######## TEMP DATA INTERPOLATION #############
    Delta_T_interp = np.zeros(interp_len)
    x_temp_old = np.linspace(0, len(Delta_T) - 1, num=len(Delta_T))
    x_temp_new = np.linspace(0, len(Delta_T) - 1, num=len(Delta_T_interp))

    interp_func_temp = interp1d(x_temp_old, Delta_T, kind='cubic')
    Delta_T_interp = interp_func_temp(x_temp_new)
    T_high_interp = Delta_T_interp + min(T_low)
    ####### TEMP DATA PLOTTING ###################
    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(10, 12))
    plt.subplot(221)
    plt.plot(T_high, '-ro', label='T_high')
    plt.plot(T_low, '-bo', label='T_low')
    plt.ylabel('$T~(C)$')
    plt.xlabel('$Meas~Point$')
    plt.legend()
    # plt.title("T_low (C)")
    plt.subplot(222)
    plt.plot(T_low, '-o', label='T_low')
    # plt.title("T_low (C)")
    plt.ylabel('$T~(C)$')
    plt.xlabel('$Meas~Point$')
    plt.legend()
    plt.subplot(223)
    plt.plot(Delta_T, '-o', label='$Delta T$')
    # plt.title("$\Delta T$ (K)")
    plt.ylabel('$Delta T~(K)$')
    plt.xlabel('$Meas~Point$')
    plt.legend()
    plt.subplot(224)
    plt.plot(Delta_T_interp, '-o', label='$Delta T_{interp}$')
    plt.ylabel('$Delta T~(K)$')
    plt.xlabel('$Meas~Point$')
    plt.legend()
    # plt.title("$\Delta T_{interp}$ (K)")
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    csv_data = numpy.asarray([T_low, T_high, Delta_T])
    csv_data_transpose = csv_data.transpose()
    variables = ['T_low', 'T_high', 'Delta_T']
    with open(f'{cur_dir}/data/results/Figure_%d_part1.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d_part1.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################
    #######  SAVE DATA TO A CSV FILE #######################
    csv_data = numpy.asarray([Delta_T_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['Delta_T_interp']
    with open(f'{cur_dir}/data/results/Figure_%d_part2.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d_part2.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ######################################

    ########  dvdt PLOTS CALCULATIONS   ########
    Voltage_fit_interp_norm = Voltage_fit_interp_norm * 10 ** 6
    Voltage_2d_interp_norm = Voltage_2d_interp_norm * 10 ** 6

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        plt.plot(Delta_T_interp, Voltage_fit_interp_norm, 'o-', label='IV_fit')
    for i in range(len(Voltage_2d_interp)):
        plt.plot(Delta_T_interp, Voltage_2d_interp_norm[i,], 'o-', label='IV_point %d' % i)
    plt.title("$Delta V$=f($\ T$)")
    plt.xlabel('$Delta T (K)$')
    plt.ylabel('$Delta V (mu V)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    Voltage_2d_interp_norm_flat = Voltage_2d_interp_norm.flatten()
    csv_data = numpy.concatenate([Voltage_fit_interp_norm, Voltage_2d_interp_norm_flat, Delta_T_interp])
    # csv_data=numpy.asarray([Voltage_2d_interp_norm_flat,Voltage_fit_interp_norm,Delta_T_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['Voltage_fit_interp_norm and Voltage_2d_interp_norm and Delta_T_interp']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)

    print('----------------------------------------------------')
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ####### SMOOTHING DATA ##########################
    Voltage_2d_interp_norm_smooth = np.zeros((total_iv_len, interp_len))
    Voltage_fit_interp_norm_smooth = np.zeros(interp_len)
    Delta_T_interp_smooth = np.zeros(len(Delta_T_interp))
    dvdt_Voltage_fit_interp_norm_smooth = np.zeros(dvdt_polyfit_order + 1)
    dvdt_Voltage_2d_interp_norm_smooth = np.zeros((total_iv_len, dvdt_polyfit_order + 1))
    f_dvdt_Voltage_2d_interp_norm_smooth = np.zeros((total_iv_len, dvdt_polyfit_order + 1))

    if (smooth_param_iv > 0):
        Voltage_fit_interp_norm_smooth = gaussian_filter1d(Voltage_fit_interp_norm, smooth_param_iv)
        for i in range(len(Voltage_2d_interp)):
            Voltage_2d_interp_norm_smooth[i, :] = gaussian_filter1d(Voltage_2d_interp_norm[i, :], smooth_param_iv)
    else:
        Voltage_fit_interp_norm_smooth = Voltage_fit_interp_norm
        Voltage_2d_interp_norm_smooth = Voltage_2d_interp_norm

    if (smooth_param_temp > 0):
        Delta_T_interp_smooth = gaussian_filter1d(Delta_T_interp, smooth_param_temp)
    else:
        Delta_T_interp_smooth = Delta_T_interp

    ###### FIT - Delta_V vs. Delta_T Plot #########################
    dvdt_Voltage_fit_interp_norm_smooth = np.polyfit(Delta_T_interp_smooth, Voltage_fit_interp_norm_smooth,
                                                     dvdt_polyfit_order)
    f_dvdt_Voltage_fit_interp_norm_smooth = np.poly1d(dvdt_Voltage_fit_interp_norm_smooth)
    print('Linear fit dv/dt (from IVC) = ', f_dvdt_Voltage_fit_interp_norm_smooth)

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        plt.plot(Delta_T_interp_smooth, Voltage_fit_interp_norm_smooth, 'o')
        plt.plot(Delta_T_interp_smooth, f_dvdt_Voltage_fit_interp_norm_smooth(Delta_T_interp_smooth), '-',
                 label=f_dvdt_Voltage_fit_interp_norm_smooth)
    for i in range(len(Voltage_2d_interp)):
        plt.plot(Delta_T_interp_smooth, Voltage_2d_interp_norm_smooth[i, :], 'o')
        dvdt_Voltage_2d_interp_norm_smooth = np.polyfit(Delta_T_interp_smooth, Voltage_2d_interp_norm_smooth[i, :],
                                                        dvdt_polyfit_order)
        f_dvdt_Voltage_2d_interp_norm_smooth = np.poly1d(dvdt_Voltage_2d_interp_norm_smooth)
        # print('dvdt_Voltage_2d_interp_norm_smooth = ',dvdt_Voltage_2d_interp_norm_smooth)
        print('Linear fit dv/dt (from IVC_%d) = ' % i, f_dvdt_Voltage_2d_interp_norm_smooth)
        plt.plot(Delta_T_interp_smooth, f_dvdt_Voltage_2d_interp_norm_smooth(Delta_T_interp_smooth), '-',
                 label=f_dvdt_Voltage_2d_interp_norm_smooth)
    plt.title("$Delta V_{sm}$ = f($Delta T_{sm}$)")
    plt.xlabel('$Delta T (K)$')
    plt.ylabel('$Delta V (mu V)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    Voltage_2d_interp_norm_smooth_flat = Voltage_2d_interp_norm_smooth.flatten()
    csv_data = numpy.concatenate(
        [Voltage_fit_interp_norm_smooth, Voltage_2d_interp_norm_smooth_flat, Delta_T_interp_smooth])
    csv_data_transpose = csv_data.transpose()
    variables = ['Voltage_fit_interp_norm_smooth and Voltage_2d_interp_norm_smooth and Delta_T_smooth_interp']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ###################  CREATING A DATABASE FILE #############
    #######  SAVE DATA TO A CSV FILE #######################
    Voltage_2d_interp_norm_smooth_flat = Voltage_2d_interp_norm_smooth.flatten()
    csv_data = numpy.concatenate([Resistance_2d_interp_flat, Voltage_2d_interp_norm_smooth_flat, Delta_T_interp_smooth])
    csv_data_transpose = csv_data.transpose()
    variables = ['Res Vol T']
    with open(f'{cur_dir}/data/results/DATABASE_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/DATABASE_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ###### Delta_V/Delta_T Plot vs T_HIGH #########################

    Delta_V_vs_Delta_T_2d = np.zeros((total_iv_len, interp_len))
    Delta_V_vs_Delta_T_interp = np.zeros(interp_len)
    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        Delta_V_vs_Delta_T_interp = Voltage_fit_interp_norm_smooth / Delta_T_interp_smooth
        plt.plot(T_high_interp, Voltage_fit_interp_norm_smooth / Delta_T_interp_smooth, 'o')
    for i in range(len(Voltage_2d_interp)):
        Delta_V_vs_Delta_T_2d[i, :] = Voltage_2d_interp_norm_smooth[i, :] / Delta_T_interp_smooth
        plt.plot(T_high_interp, Voltage_2d_interp_norm_smooth[i, :] / Delta_T_interp_smooth, 'o')
    plt.title("$Delta V_{sm}/Delta T_{sm}$")
    plt.xlabel('$T_{High} (C)$')
    plt.ylabel('$Cumulative~Seebeck~Coeff.~(mu V/K) $')
    # plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    Delta_V_vs_Delta_T_2d_flat = Delta_V_vs_Delta_T_2d.flatten()
    csv_data = numpy.concatenate([Delta_V_vs_Delta_T_interp, Delta_V_vs_Delta_T_2d_flat, T_high_interp])

    csv_data_transpose = csv_data.transpose()
    variables = [
        'Delta_Voltage_fit_interp_norm_smooth_BY_Delta_T_smooth_interp_T and Voltage_2d_interp_norm_smooth_BY_Delta_T_smooth_interp']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ################ SEEBECK & IVC & TEMP PlOT ######################################
    diff_Voltage_2d_interp_norm_smooth = np.zeros((total_iv_len, interp_len - 1))
    seebeck_Voltage_2d_interp_norm_smooth = np.zeros((total_iv_len, interp_len - 1))
    seebeck_Voltage_fit_interp_norm_smooth = diff(Voltage_fit_interp_norm_smooth) / diff(Delta_T_interp_smooth)

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 15))
    plt.subplot(311)
    if (total_iv_len > 1):
        diff_Voltage_fit_interp_norm_smooth = diff(Voltage_fit_interp_norm_smooth)
        plt.plot(diff_Voltage_fit_interp_norm_smooth, 'o-', label='IV_fit')
    for i in range(len(Voltage_2d_interp)):
        diff_Voltage_2d_interp_norm_smooth[i, :] = diff(Voltage_2d_interp_norm_smooth[i, :])
        plt.plot(diff_Voltage_2d_interp_norm_smooth[i, :], 'o-', label='IV_point %d' % i)
    plt.legend()
    plt.xlabel('$Meas. Point$')
    plt.ylabel('$d V_{INTERP} (mu V)$')

    plt.subplot(312)
    diff_Delta_T_interp_smooth = diff(Delta_T_interp_smooth)
    plt.plot(diff_Delta_T_interp_smooth, '-o')
    plt.xlabel('$Meas. Point$')
    plt.ylabel('$d T_{High} (C)$')

    plt.subplot(313)
    if (total_iv_len > 1):
        plt.plot(T_high_interp[0:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_fit_interp_norm_smooth[0:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_fit_data')
    for i in range(len(Voltage_2d_interp)):
        seebeck_Voltage_2d_interp_norm_smooth[i, :] = diff(Voltage_2d_interp_norm_smooth[i, :]) / diff(
            Delta_T_interp_smooth)
        plt.plot(T_high_interp[0:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_2d_interp_norm_smooth[i, 0:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_point %d' % i)
    plt.title("$Seebeck~Coefficient = d V_{SMOOTH} / d T_{SMOOTH}$")
    plt.ylabel('$Seebeck~Coeff.~(mu V/K)$')
    plt.xlabel('$T_{High} (C)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    if (total_iv_len < 2):
        diff_Voltage_fit_interp_norm_smooth = np.zeros(interp_len - 1)

    diff_Voltage_2d_interp_norm_smooth_flat = diff_Voltage_2d_interp_norm_smooth.flatten()
    part11 = numpy.concatenate([diff_Voltage_fit_interp_norm_smooth, diff_Voltage_2d_interp_norm_smooth_flat])

    seebeck_Voltage_2d_interp_norm_smooth_flat = seebeck_Voltage_2d_interp_norm_smooth.flatten()
    part12 = numpy.concatenate([seebeck_Voltage_fit_interp_norm_smooth, seebeck_Voltage_2d_interp_norm_smooth_flat])

    csv_data = numpy.asarray([part11, part12])
    csv_data_transpose = csv_data.transpose()
    variables = ['d_V', 'Seebeck']
    with open(f'{cur_dir}/data/results/Figure_%d_part1.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d_part1.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    #######  SAVE DATA TO A CSV FILE #######################
    csv_data = numpy.asarray([diff_Delta_T_interp_smooth])
    csv_data_transpose = csv_data.transpose()
    variables = ['diff_Delta_T_smooth']
    with open(f'{cur_dir}/data/results/Figure_%d_part2.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d_part2.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    #######  SAVE DATA TO A CSV FILE #######################
    csv_data = numpy.asarray([T_high_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['T_high_interp']
    with open(f'{cur_dir}/data/results/Figure_%d_part3.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d_part3.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ################ SEEBECK & IVC & TEMP PlOT NORMALIZED ######################################
    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 15))
    plt.subplot(211)
    if (total_iv_len > 1):
        diff_Voltage_fit_interp_norm_smooth = diff(Voltage_fit_interp_norm_smooth)
        plt.plot(diff_Voltage_fit_interp_norm_smooth / max(diff_Voltage_fit_interp_norm_smooth), 'o-', label='IV_fit')
    for i in range(len(Voltage_2d_interp)):
        diff_Voltage_2d_interp_norm_smooth[i, :] = diff(Voltage_2d_interp_norm_smooth[i, :])
        plt.plot(diff_Voltage_2d_interp_norm_smooth[i, :] / max(diff_Voltage_2d_interp_norm_smooth[i, :]), 'o-',
                 label='IV_point %d' % i)
    # plt.legend()
    # plt.xlabel('$Meas. Point$')
    # plt.ylabel('$\Delta V_{INTERP} (\mu V)$')

    diff_Delta_T_interp_smooth = diff(Delta_T_interp_smooth)
    plt.plot(diff_Delta_T_interp_smooth / max(diff_Delta_T_interp_smooth), '-o', label='Delta_T')
    plt.xlabel('$Meas. Point$')
    plt.ylabel('$Normalized~d T_{High}~and~d V_{INTERP}$')
    plt.legend()

    plt.subplot(212)
    if (total_iv_len > 1):
        plt.plot(T_high_interp[0:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_fit_interp_norm_smooth[0:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_fit_data')
    for i in range(len(Voltage_2d_interp)):
        seebeck_Voltage_2d_interp_norm_smooth[i, :] = diff(Voltage_2d_interp_norm_smooth[i, :]) / diff(
            Delta_T_interp_smooth)
        plt.plot(T_high_interp[0:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_2d_interp_norm_smooth[i, 0:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_point %d' % i)
    plt.title("$Seebeck~Coefficient = d V_{SMOOTH} / d T_{SMOOTH}$")
    plt.ylabel('$Seebeck~Coeff.~(mu V/K)$')
    plt.xlabel('$T_{High} (C)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    ################ SEEBECK vs TEMP PLOT ######################################
    seebeck_Voltage_2d_interp_norm_smooth = np.zeros((total_iv_len, interp_len - 1))
    seebeck_Voltage_fit_interp_norm_smooth = diff(Voltage_fit_interp_norm_smooth) / diff(Delta_T_interp_smooth)

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        plt.plot(T_high_interp[0:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_fit_interp_norm_smooth[0:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_fit_data')
    for i in range(len(Voltage_2d_interp)):
        seebeck_Voltage_2d_interp_norm_smooth[i, :] = diff(Voltage_2d_interp_norm_smooth[i, :]) / diff(
            Delta_T_interp_smooth)
        plt.plot(T_high_interp[0:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_2d_interp_norm_smooth[i, 0:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_point %d' % i)

    plt.title("$Seebeck~Coefficient$")
    plt.ylabel('$Seebeck~Coeff.~(mu V/K)$')
    plt.xlabel('$T_{High} (C)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    seebeck_Voltage_2d_interp_norm_smooth_flat = seebeck_Voltage_2d_interp_norm_smooth.flatten()
    csv_data = numpy.concatenate(
        [seebeck_Voltage_fit_interp_norm_smooth, seebeck_Voltage_2d_interp_norm_smooth_flat, T_high_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['Seebeck_fit and Seebeck_2d and T_high']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ###### Seebeck FIT #########################
    polyfit_seebeck_Voltage_fit_interp_norm_smooth = np.polyfit(
        T_high_interp[0:len(seebeck_Voltage_fit_interp_norm_smooth)], seebeck_Voltage_fit_interp_norm_smooth,
        seebeck_polyfit_order)
    f_polyfit_seebeck_Voltage_fit_interp_norm_smooth = np.poly1d(polyfit_seebeck_Voltage_fit_interp_norm_smooth)
    print('Poly fit to Seebeck (from IVC) = ', f_polyfit_seebeck_Voltage_fit_interp_norm_smooth)

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        plt.plot(T_high_interp[1:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_fit_interp_norm_smooth[1:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_fit_data')
        plt.plot(T_high_interp, f_polyfit_seebeck_Voltage_fit_interp_norm_smooth(T_high_interp), '-',
                 label=f_polyfit_seebeck_Voltage_fit_interp_norm_smooth)
    for i in range(len(Voltage_2d_interp)):
        seebeck_Voltage_2d_interp_norm_smooth[i, :] = diff(Voltage_2d_interp_norm_smooth[i, :]) / diff(
            Delta_T_interp_smooth)
        plt.plot(T_high_interp[1:len(seebeck_Voltage_fit_interp_norm_smooth)],
                 seebeck_Voltage_2d_interp_norm_smooth[i, 1:len(seebeck_Voltage_fit_interp_norm_smooth)], '-o',
                 label='IV_point %d' % i)

        polyfit_seebeck_Voltage_2d_interp_norm_smooth = np.polyfit(
            T_high_interp[0:len(seebeck_Voltage_2d_interp_norm_smooth[i, :])],
            seebeck_Voltage_2d_interp_norm_smooth[i, :],
            seebeck_polyfit_order)
        f_polyfit_seebeck_Voltage_2d_interp_norm_smooth = np.poly1d(polyfit_seebeck_Voltage_2d_interp_norm_smooth)
        print('Poly fit to Seebeck (from IVC_%d) = ' % i, f_polyfit_seebeck_Voltage_2d_interp_norm_smooth)
        plt.plot(T_high_interp, f_polyfit_seebeck_Voltage_2d_interp_norm_smooth(T_high_interp), '-',
                 label=f_polyfit_seebeck_Voltage_2d_interp_norm_smooth)
    plt.title("$Seebeck~Coefficient$")
    plt.ylabel('$Seebeck~Coeff.~(mu V/K)$')
    plt.xlabel('$T_{High} (C)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    seebeck_Voltage_2d_interp_norm_smooth_flat = seebeck_Voltage_2d_interp_norm_smooth.flatten()
    csv_data = numpy.concatenate(
        [seebeck_Voltage_fit_interp_norm_smooth, seebeck_Voltage_2d_interp_norm_smooth_flat, T_high_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['Seebeck_fit and Seebeck_2d and T_high']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ################ Resistance vs TEMP PLOT ######################################
    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        # plt.plot(T_high_interp, Resistance_fit_interp,'-o',label='IV_fit_data')
        print('Resistance_fit_interp=', Resistance_fit_interp)
    for i in range(len(Voltage_2d_interp)):
        plt.plot(T_high_interp, Resistance_2d_interp[i,], '-o', label='IV_point %d' % i)
    plt.title("$Resistance$")
    plt.ylabel('$R~(\Omega)$')
    plt.xlabel('$T_{High} (C)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    Resistance_2d_interp_flat = Resistance_2d_interp.flatten()
    csv_data = numpy.concatenate([Resistance_fit_interp, Resistance_2d_interp_flat, T_high_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['Resistance_fit and Resitance_2d and T_high']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ########### CONDUCTIVITY  #################################
    Conductivity_2d_interp = np.zeros((total_iv_len, interp_len))
    film_thickness_cm = film_thickness_micrometer / 10 ** (4)

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        Conductivity_fit_interp = 1 / (Resistance_fit_interp * film_thickness_cm)
        # plt.plot(T_high_interp, Conductivity_fit_interp,'-o',label='IV_fit')
        print('Conductivity_fit_interp =', Conductivity_fit_interp)
    for i in range(len(Voltage_2d_interp)):
        Conductivity_2d_interp[i,] = 1 / (Resistance_2d_interp[i,] * film_thickness_cm)
        plt.plot(T_high_interp, Conductivity_2d_interp[i,], '-o', label='IV_point %d' % i)
    plt.title("$Conductivity$")
    plt.ylabel('$\sigma~(Siemens/cm)$')
    plt.xlabel('$T_{High} (C)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    if (total_iv_len < 2):
        Conductivity_fit_interp = np.zeros(interp_len - 1)
    Conductivity_2d_interp_flat = Conductivity_2d_interp.flatten()
    csv_data = numpy.concatenate([Conductivity_fit_interp, Conductivity_2d_interp_flat, T_high_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['Conductivity_fit and Conductivity_2d and T_high']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################

    ########### POWER FACTOR  #################################
    PowerFactor_2d_interp = np.zeros((total_iv_len, interp_len - 1))

    fig_no = fig_no + 1
    plt.figure(fig_no, figsize=(7.5, 5))
    if (total_iv_len > 1):
        PowerFactor_fit_interp = seebeck_Voltage_fit_interp_norm_smooth ** 2 * Conductivity_fit_interp[0:len(
            seebeck_Voltage_fit_interp_norm_smooth)] / 10 ** (4)
        plt.plot(T_high_interp[0:len(PowerFactor_fit_interp)], PowerFactor_fit_interp, '-o', label='IV_fit')
        print('PowerFactor_fit_interp =', PowerFactor_fit_interp)
    for i in range(len(Voltage_2d_interp)):
        PowerFactor_2d_interp[i, 0:len(seebeck_Voltage_fit_interp_norm_smooth)] = seebeck_Voltage_2d_interp_norm_smooth[
                                                                                  i, :] ** 2 * Conductivity_2d_interp[i,
                                                                                               0:len(
                                                                                                   seebeck_Voltage_fit_interp_norm_smooth)] / 10 ** (
                                                                                      4)
        plt.plot(T_high_interp[0:len(PowerFactor_2d_interp[i,])], PowerFactor_2d_interp[i,], '-o',
                 label='IV_point %d' % i)
    plt.title("$Power~Factor$")
    plt.ylabel('$S^2\sigma~(mu W/(mK^2))$')
    plt.xlabel('$T_{High} (C)$')
    plt.legend()
    plt.savefig(f'{cur_dir}/data/results/Fig_%d.png' % fig_no, dpi=img_dpi)
    if img_show == True:
        plt.show()

    #######  SAVE DATA TO A CSV FILE #######################
    if (total_iv_len < 2):
        PowerFactor_fit_interp = np.zeros(interp_len - 1)
    PowerFactor_2d_interp_flat = PowerFactor_2d_interp.flatten()
    csv_data = numpy.concatenate([PowerFactor_fit_interp, PowerFactor_2d_interp_flat, T_high_interp])
    csv_data_transpose = csv_data.transpose()
    variables = ['PowerFactor_fit and PowerFactor_2d and T_high']
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'a+') as f:
        header = csv.writer(f, delimiter=delimiter_csv_file)
        header.writerow(variables)
    with open(f'{cur_dir}/data/results/Figure_%d.csv' % fig_no, 'ab') as f:
        np.savetxt(f, csv_data_transpose, delimiter=delimiter_csv_file)
    print(variables)
    print(csv_data_transpose)
    #####################################################
    if show_summary == False:
        sys.stdout.close()