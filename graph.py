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
    import matplotlib.pyplot as plt
    import numpy as np
    
    kf_E = 'E_est' if 'E_est' in df_kf.columns else 'E'
    kf_N = 'N_est' if 'N_est' in df_kf.columns else 'N'
    kf_U = 'U_est' if 'U_est' in df_kf.columns else 'U'
    
    kf_vE = 'vE_est' if 'vE_est' in df_kf.columns else 'vE'
    kf_vN = 'vN_est' if 'vN_est' in df_kf.columns else 'vN'
    kf_vU = 'vU_est' if 'vU_est' in df_kf.columns else 'vU'

    # Окно 1: 3D
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(df_true['E'], df_true['N'], df_true['U'], label='Истина', color='black', linewidth=2)
    ax_3d.scatter(df_meas['E'], df_meas['N'], df_meas['U'], label='GNSS Измерения', color='red', s=10, alpha=0.5)
    ax_3d.plot(df_kf[kf_E], df_kf[kf_N], df_kf[kf_U], label='Оценка ФК', color='blue', linewidth=2)
    ax_3d.set_xlabel('East (м)')
    ax_3d.set_ylabel('North (м)')
    ax_3d.set_zlabel('Up (м)')
    ax_3d.set_title(f'3D Траектория {title_suffix}')
    ax_3d.legend()
    
    # Для расчета ошибок сделаем простую интерполяцию истины к частоте фильтра
    df_true_interp = pd.DataFrame({'t': df_kf['t']})
    for col in ['E', 'N', 'U', 'vE', 'vN', 'vU']:
        df_true_interp[col] = np.interp(df_kf['t'], df_true['t'], df_true[col])
        
    df_meas_interp = pd.DataFrame({'t': df_kf['t']})
    for col in ['E', 'N', 'U']:
        if col in df_meas.columns:
            df_meas_interp[col] = np.interp(df_kf['t'], df_meas['t'], df_meas[col])

    # Окно 2: Координаты и Ошибки (2 колонки)
    fig_pos, axs_pos = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig_pos.suptitle(f'Координаты и Ошибки {title_suffix}', fontsize=16)
    
    axes = ['E', 'N', 'U']
    kf_axes = [kf_E, kf_N, kf_U]
    colors = ['r', 'g', 'b']
    
    for i, (ax_name, kf_col, c) in enumerate(zip(axes, kf_axes, colors)):
        # Слева: Координаты
        axs_pos[i, 0].plot(df_true['t'], df_true[ax_name], label='Истина', color='black', linewidth=2)
        axs_pos[i, 0].scatter(df_meas['t'], df_meas[ax_name], label='GNSS', color='red', s=10, alpha=0.3)
        axs_pos[i, 0].plot(df_kf['t'], df_kf[kf_col], label='ФК', color='blue', linewidth=2)
        axs_pos[i, 0].set_ylabel(f'{ax_name} (м)')
        axs_pos[i, 0].legend(loc='upper right')
        axs_pos[i, 0].grid(True)
        
        # Справа: Ошибки
        meas_err = df_meas_interp[ax_name] - df_true_interp[ax_name]
        kf_err = df_kf[kf_col] - df_true_interp[ax_name]
        
        axs_pos[i, 1].scatter(df_kf['t'], meas_err, label='Ошибка GNSS', color='red', s=5, alpha=0.3)
        axs_pos[i, 1].plot(df_kf['t'], kf_err, label='Ошибка ФК', color='blue', linewidth=1.5)
        axs_pos[i, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axs_pos[i, 1].set_ylabel('Ошибка (м)')
        axs_pos[i, 1].legend(loc='upper right')
        axs_pos[i, 1].grid(True)
        
    axs_pos[2, 0].set_xlabel('Время (с)')
    axs_pos[2, 1].set_xlabel('Время (с)')
    plt.tight_layout()
    
    # Окно 3: Скорости и Ошибки (2 колонки)
    fig_vel, axs_vel = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig_vel.suptitle(f'Скорости и Ошибки {title_suffix}', fontsize=16)
    
    v_axes = ['vE', 'vN', 'vU']
    kf_v_axes = [kf_vE, kf_vN, kf_vU]
    
    for i, (ax_name, kf_col, c) in enumerate(zip(v_axes, kf_v_axes, colors)):
        # Слева: Значения скоростей
        axs_vel[i, 0].plot(df_true['t'], df_true[ax_name], label='Истина', color='black', linewidth=2)
        axs_vel[i, 0].plot(df_kf['t'], df_kf[kf_col], label='ФК', color='blue', linewidth=2)
        axs_vel[i, 0].set_ylabel(f'{ax_name} (м/с)')
        axs_vel[i, 0].legend(loc='upper right')
        axs_vel[i, 0].grid(True)
        
        # Справа: Ошибки скоростей
        kf_err = df_kf[kf_col] - df_true_interp[ax_name]
        
        axs_vel[i, 1].plot(df_kf['t'], kf_err, label='Ошибка ФК', color='blue', linewidth=1.5)
        axs_vel[i, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axs_vel[i, 1].set_ylabel('Ошибка (м/с)')
        axs_vel[i, 1].legend(loc='upper right')
        axs_vel[i, 1].grid(True)

    axs_vel[2, 0].set_xlabel('Время (с)')
    axs_vel[2, 1].set_xlabel('Время (с)')
    plt.tight_layout()
    
    plt.show()

def plot_wls_results(df_true: pd.DataFrame, df_wls: pd.DataFrame):
    """
    Рисует сравнение результатов МНК (WLS) с эталонной траекторией:
    1. 3D график траектории.
    2. Графики проекций по осям E, N, U.
    """
    import matplotlib.pyplot as plt

    # --- 1. 3D Траектория ---
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(df_true['E'], df_true['N'], df_true['U'], label='Истина', color='black', linewidth=2)
    ax3d.scatter(df_wls['E'], df_wls['N'], df_wls['U'], label='Оценка МНК', color='green', s=15, alpha=0.7)
    ax3d.set_xlabel('East (м)')
    ax3d.set_ylabel('North (м)')
    ax3d.set_zlabel('Up (м)')
    ax3d.set_title('Сравнение траекторий в 3D (WLS)')
    ax3d.legend()
    
    # --- 2. Графики проекций и ошибок ---
    fig_pos, axs_pos = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig_pos.suptitle('Координаты E, N, U: МНК vs Истина и Ошибки', fontsize=16)
    
    df_merged = pd.merge_asof(df_wls, df_true, on='t', direction='nearest', suffixes=('_est', '_true'))
    
    axes_labels = ['E', 'N', 'U']
    colors = ['r', 'g', 'b']
    
    for i, (axis, color) in enumerate(zip(axes_labels, colors)):
        # Слева: Координаты
        axs_pos[i, 0].plot(df_merged['t'], df_merged[f'{axis}_true'], label='Истина', color='black', linewidth=2)
        axs_pos[i, 0].scatter(df_merged['t'], df_merged[f'{axis}_est'], label='МНК', color=color, s=15, alpha=0.6)
        axs_pos[i, 0].set_ylabel(f'{axis} (м)')
        axs_pos[i, 0].legend(loc='upper right')
        axs_pos[i, 0].grid(True)
        
        # Справа: Ошибки
        error = df_merged[f'{axis}_est'] - df_merged[f'{axis}_true']
        axs_pos[i, 1].plot(df_merged['t'], error, label=f'Ошибка {axis}', color=color, linewidth=1.5)
        axs_pos[i, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axs_pos[i, 1].set_ylabel(f'Ошибка (м)')
        axs_pos[i, 1].legend(loc='upper right')
        axs_pos[i, 1].grid(True)

    axs_pos[2, 0].set_xlabel('Время (с)')
    axs_pos[2, 1].set_xlabel('Время (с)')
    plt.tight_layout()
    
    # --- 3. График DOP ---
    # if 'HDOP' in df_wls.columns:
    #     fig_dop, ax_dop = plt.subplots(figsize=(12, 4))
    #     ax_dop.plot(df_wls['t'], df_wls['HDOP'], label='HDOP (Горизонт)', color='blue', linewidth=2)
    #     ax_dop.plot(df_wls['t'], df_wls['VDOP'], label='VDOP (Вертикаль)', color='red', linewidth=2)
    #     ax_dop.plot(df_wls['t'], df_wls['PDOP'], label='PDOP (Пространство)', color='green', linestyle='--', linewidth=2)
    #     ax_dop.set_xlabel('Время (с)')
    #     ax_dop.set_ylabel('Значение DOP')
    #     ax_dop.set_title('Геометрический фактор потери точности (DOP)')
    #     ax_dop.legend()
    #     ax_dop.grid(True)
    #     plt.tight_layout()

    plt.show()

def plot_earth_and_satellites(sat_positions: np.ndarray, receiver_pos: np.ndarray, orbit_lines: list = None):
    """
    Рисует 3D модель Земли, орбиты и спутники.
    Внутри себя проверяет, какие спутники видимы (над горизонтом), 
    рисует их красным и строит к ним линии визирования (LOS).
    sat_positions: Массив (N, 3) координат спутников (ECEF)
    receiver_pos: Массив (3,) координат приемника (ECEF)
    orbit_lines: Список массивов (M, 3) с точками орбит для отрисовки колец
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from gnss import _is_satellite_visible

    R_EARTH = 6371000.0

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. Отрисовка Земли ---
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x_sphere = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y_sphere = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z_sphere = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightblue', linewidth=0.5, alpha=0.5)

    # --- 2. Экватор и Меридиан ---
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R_EARTH * np.cos(theta), R_EARTH * np.sin(theta), np.zeros_like(theta), color='blue', alpha=0.9, label='Экватор')
    ax.plot(R_EARTH * np.cos(theta), np.zeros_like(theta), R_EARTH * np.sin(theta), color='green', alpha=0.9, label='Нулевой меридиан')

    # --- 3. Отрисовка колец орбит ---
    if orbit_lines is not None:
        for i, orb_pts in enumerate(orbit_lines):
            label = 'Орбиты ГНСС' if i == 0 else ""
            ax.plot(orb_pts[:, 0], orb_pts[:, 1], orb_pts[:, 2], color='gray', linewidth=1.0, alpha=0.4, label=label)

    # --- 4. Фильтрация и Отрисовка Спутников ---
    visible_sats = []
    invisible_sats = []
    
    # Распределяем спутники на видимые и невидимые (угол места > 5 градусов)
    for sat in sat_positions:
        if _is_satellite_visible(sat, receiver_pos, mask_angle_deg=5.0):
            visible_sats.append(sat)
        else:
            invisible_sats.append(sat)
            
    visible_sats = np.array(visible_sats) if len(visible_sats) > 0 else np.empty((0, 3))
    invisible_sats = np.array(invisible_sats) if len(invisible_sats) > 0 else np.empty((0, 3))

    # Рисуем невидимые спутники (за горизонтом)
    if len(invisible_sats) > 0:
        ax.scatter(invisible_sats[:, 0], invisible_sats[:, 1], invisible_sats[:, 2], 
                   color='gray', s=20, label='Спутники (за горизонтом)', alpha=0.5, depthshade=False, zorder=4)

    # Рисуем видимые спутники и линии визирования (LOS)
    if len(visible_sats) > 0:
        ax.scatter(visible_sats[:, 0], visible_sats[:, 1], visible_sats[:, 2], 
                   color='red', s=50, label='Видимые спутники', depthshade=False, zorder=5)

        for sat in visible_sats:
            ax.plot([receiver_pos[0], sat[0]], [receiver_pos[1], sat[1]], [receiver_pos[2], sat[2]], 
                    color='orange', linestyle=':', linewidth=1.0, alpha=0.8)

    # --- 5. Отрисовка Приемника ---
    ax.scatter(*receiver_pos, color='magenta', s=150, marker='*', label='Приемник (Машина)', depthshade=False, zorder=10)
    ax.plot([0, receiver_pos[0]], [0, receiver_pos[1]], [0, receiver_pos[2]], color='magenta', linewidth=1.5, alpha=0.8)

    # --- Настройка осей ---
    max_val = np.max(np.abs(sat_positions)) * 1.05 if len(sat_positions) > 0 else R_EARTH * 4
    
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    ax.set_xlabel('X (ECEF) [м]')
    ax.set_ylabel('Y (ECEF) [м]')
    ax.set_zlabel('Z (ECEF) [м]')
    ax.set_title('Орбитальная группировка ГНСС (Walker Constellation)', fontsize=14)
    
    # Космический фон
    ax.set_facecolor('black')
    ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    
    ax.legend(loc='upper right', facecolor='white', framealpha=0.8)

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    plt.tight_layout()
    plt.show()