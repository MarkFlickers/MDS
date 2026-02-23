# ВИЗУАЛИЗАЦИЯ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(df_imu: pd.DataFrame, df_gnss_clean: pd.DataFrame, df_gnss_noisy: pd.DataFrame):
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
    ax.set_title('3D Траектория движения (Антенна)')
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