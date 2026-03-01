# ВИЗУАЛИЗАЦИЯ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(df_imu: pd.DataFrame, df_gnss_noisy: pd.DataFrame):
    """
    Строит графики:
    1. 3D траектория (сравнение Истины и ГНСС-измерений)
    2. Координаты и Скорости (по компонентам)
    3. Данные ИМУ (Удельная сила, Угловая скорость, Ориентация)
    """
    plt.style.use('bmh') # Красивый стиль графиков
    
    # --- РИСУНОК 1: 3D Траектория ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Истинная траектория (ИМУ - сплошная линия)
    ax.plot(df_imu['E_ant'], df_imu['N_ant'], df_imu['U_ant'], 
            label='Истинная траектория (IMU 50Hz)', color='black', linewidth=1.5, alpha=0.8)
    
    # ГНСС измерения (Точки с шумом)
    ax.scatter(df_gnss_noisy['E'], df_gnss_noisy['N'], df_gnss_noisy['U'], 
               label='ГНСС измерения (1Hz)', color='red', s=10, alpha=0.6)

    # Старт/Финиш
    ax.scatter(df_imu['E_ant'].iloc[0], df_imu['N_ant'].iloc[0], df_imu['U_ant'].iloc[0], 
               color='green', s=50, marker='^', label='Старт')
    ax.scatter(df_imu['E_ant'].iloc[-1], df_imu['N_ant'].iloc[-1], df_imu['U_ant'].iloc[-1], 
               color='blue', s=50, marker='s', label='Финиш')

    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('3D Траектория движения (ИНС)')
    ax.legend()
    
    # Выравнивание осей для корректного восприятия масштаба
    # (Matplotlib 3D по умолчанию искажает масштаб)
    try:
        ax.set_box_aspect([np.ptp(df_imu['E_ant']), np.ptp(df_imu['N_ant']), np.ptp(df_imu['U_ant'])])
    except:
        pass # Если старая версия matplotlib

    # --- РИСУНОК 2: Навигация (Координаты и Скорость) ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig2.suptitle('Навигационные параметры (ENU)', fontsize=14)

    t_imu = df_imu['t']
    t_gnss = df_gnss_noisy['t']

    # Координаты (Истина vs Шум)
    comps = ['E', 'N', 'U']
    for i, comp in enumerate(comps):
        ax = axes2[0, i]
        ax.plot(t_imu, df_imu[f'{comp}_ant'], 'k-', label='Истина')
        ax.plot(t_gnss, df_gnss_noisy[comp], 'r.', markersize=4, label='ГНСС', alpha=0.5)
        ax.set_ylabel(f'{comp} [m]')
        ax.set_title(f'Позиция {comp}')
        if i == 0: ax.legend()
        ax.grid(True)

    # Скорости (Только истина, т.к. ГНСС скорость мы не шумели, но можно добавить)
    v_comps = ['vE', 'vN', 'vU']
    for i, comp in enumerate(v_comps):
        ax = axes2[1, i]
        ax.plot(t_imu, df_imu[comp], 'b-', label='Истина')
        ax.set_ylabel(f'{comp} [m/s]')
        ax.set_xlabel('Время [с]')
        ax.set_title(f'Скорость {comp}')
        ax.grid(True)

    # --- РИСУНОК 3: Сенсоры ИМУ и Ориентация ---
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig3.suptitle('Данные ИМУ (Ideal / Noise-free)', fontsize=14)

    # 1. Ориентация (Euler)
    ax = axes3[0]
    ax.plot(t_imu, np.degrees(df_imu['roll']), label='Roll')
    ax.plot(t_imu, np.degrees(df_imu['pitch']), label='Pitch')
    ax.plot(t_imu, np.degrees(df_imu['yaw']), label='Yaw')
    ax.set_ylabel('Углы [град]')
    ax.set_title('Ориентация (Roll, Pitch, Yaw)')
    ax.legend(loc='right')
    ax.grid(True)

    # 2. Акселерометры (Specific Force)
    ax = axes3[1]
    ax.plot(t_imu, df_imu['f_x'], label='f_x')
    ax.plot(t_imu, df_imu['f_y'], label='f_y')
    ax.plot(t_imu, df_imu['f_z'], label='f_z')
    ax.set_ylabel('Уд. сила [m/s²]')
    ax.set_title('Акселерометр (Specific Force Body)')
    ax.legend(loc='right')
    ax.grid(True)

    # 3. Гироскопы (Angular Rate)
    ax = axes3[2]
    ax.plot(t_imu, np.degrees(df_imu['w_x']), label='w_x')
    ax.plot(t_imu, np.degrees(df_imu['w_y']), label='w_y')
    ax.plot(t_imu, np.degrees(df_imu['w_z']), label='w_z')
    ax.set_ylabel('Угл. скорость [deg/s]')
    ax.set_xlabel('Время [с]')
    ax.set_title('Гироскоп (Angular Rate Body)')
    ax.legend(loc='right')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_kf_comparison(df_true: pd.DataFrame, df_meas: pd.DataFrame, df_kf: pd.DataFrame, title_suffix: str = ""):
    """
    Отображает три окна с результатами фильтра Калмана:
    1. 3D траектория.
    2. Координаты E, N, U от времени.
    3. Скорости vE, vN, vU от времени.
    """
    import matplotlib.pyplot as plt

    # --- Окно 1: 3D Траектория ---
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(df_true['E'], df_true['N'], df_true['U'], label='Истина', color='black', linewidth=2)
    ax_3d.scatter(df_meas['E'], df_meas['N'], df_meas['U'], label='GNSS Измерения', color='red', s=10, alpha=0.5)
    
    # Обрабатываем названия колонок (если они имеют суффикс _est)
    kf_E = df_kf['E_est'] if 'E_est' in df_kf.columns else df_kf['E']
    kf_N = df_kf['N_est'] if 'N_est' in df_kf.columns else df_kf['N']
    kf_U = df_kf['U_est'] if 'U_est' in df_kf.columns else df_kf['U']
    
    ax_3d.plot(kf_E, kf_N, kf_U, label='Оценка ФК', color='blue', linewidth=2)
    ax_3d.set_xlabel('East (м)')
    ax_3d.set_ylabel('North (м)')
    ax_3d.set_zlabel('Up (м)')
    ax_3d.set_title(f'3D Траектория {title_suffix}')
    ax_3d.legend()
    ax_3d.grid(True)

    # --- Окно 2: Сравнение координат в проекциях E, N, U ---
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig_pos.suptitle(f'Координаты (E, N, U) {title_suffix}', fontsize=16)
    
    axes_labels = ['E', 'N', 'U']
    kf_labels = ['E_est', 'N_est', 'U_est'] if 'E_est' in df_kf.columns else axes_labels
    
    for i, (axis, kf_axis) in enumerate(zip(axes_labels, kf_labels)):
        axs_pos[i].plot(df_true['t'], df_true[axis], label='Истина', color='black', linewidth=2)
        axs_pos[i].scatter(df_meas['t'], df_meas[axis], label='GNSS', color='red', s=10, alpha=0.5)
        axs_pos[i].plot(df_kf['t'], df_kf[kf_axis], label='ФК', color='blue', linewidth=2)
        axs_pos[i].set_ylabel(f'{axis} (м)')
        axs_pos[i].legend(loc='upper right')
        axs_pos[i].grid(True)
    axs_pos[2].set_xlabel('Время (с)')
    plt.tight_layout()

    # --- Окно 3: Сравнение скоростей vE, vN, vU ---
    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig_vel.suptitle(f'Скорости (vE, vN, vU) {title_suffix}', fontsize=16)
    
    vel_axes_labels = ['vE', 'vN', 'vU']
    vel_kf_labels = ['vE_est', 'vN_est', 'vU_est'] if 'vE_est' in df_kf.columns else vel_axes_labels
    
    for i, (axis, kf_axis) in enumerate(zip(vel_axes_labels, vel_kf_labels)):
        axs_vel[i].plot(df_true['t'], df_true[axis], label='Истина', color='black', linewidth=2)
        axs_vel[i].plot(df_kf['t'], df_kf[kf_axis], label='ФК', color='blue', linewidth=2)
        axs_vel[i].set_ylabel(f'{axis} (м/с)')
        axs_vel[i].legend(loc='upper right')
        axs_vel[i].grid(True)
    axs_vel[2].set_xlabel('Время (с)')
    plt.tight_layout()

    plt.show()

# ... (предыдущий код файла graph.py остается без изменений)

def plot_wls_results(df_true: pd.DataFrame, df_wls: pd.DataFrame):
    """
    Визуализация результатов Взвешенного МНК (WLS) vs Истина.
    Строит графики ошибок по координатам (ENU).
    """
    # Убедимся, что данные отсортированы
    df_true = df_true.sort_values('t')
    df_wls = df_wls.sort_values('t')
    
    # Объединяем датафреймы по ближайшей временной метке (на случай рассинхрона частот)
    # suffixes: _est для оценки (WLS), _true для истины
    df_merged = pd.merge_asof(
        df_wls, 
        df_true[['t', 'E', 'N', 'U']], 
        on='t', 
        suffixes=('_est', '_true'), 
        direction='nearest'
    )
    
    # Вычисляем ошибки позиционирования
    df_merged['err_E'] = df_merged['E_est'] - df_merged['E_true']
    df_merged['err_N'] = df_merged['N_est'] - df_merged['N_true']
    df_merged['err_U'] = df_merged['U_est'] - df_merged['U_true']
    
    # Считаем 3D ошибку и HDOP/VDOP (из ковариации, если она есть в df_wls)
    # Но пока просто построим ошибки координат
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Ошибки решения Взвешенного МНК (WLS Errors)', fontsize=16)
    
    cols = ['err_E', 'err_N', 'err_U']
    titles = ['East Error', 'North Error', 'Up Error']
    colors = ['r', 'g', 'b']
    
    for i, (col, title, color) in enumerate(zip(cols, titles, colors)):
        rmse = np.sqrt((df_merged[col]**2).mean())
        max_err = df_merged[col].abs().max()
        
        ax = axes[i]
        ax.plot(df_merged['t'], df_merged[col], color=color, label=f'Error', linewidth=1)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        
        # Добавляем статистику в легенду
        ax.legend([f'RMSE: {rmse:.2f} m\nMax: {max_err:.2f} m'], loc='upper right')
        
        ax.set_ylabel('Ошибка [м]')
        ax.set_title(title)
        ax.grid(True)
        
    axes[2].set_xlabel('Время [с]')
    plt.tight_layout()
    plt.show()
