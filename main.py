import pandas as pd
import numpy as np
from configuration import TrajectoryConfig, stages_scenario
from trajectory import generate_trajectory, simulate_imu_errors
from gnss import process_gnss, simulate_gnss_raw
from graph import plot_results, plot_kf_comparison, plot_wls_results
from metrics import calculate_rmse
from kalman import LinearKalmanFilter, ExtendedKalmanFilter
from coord_conversion import ecef_to_enu, enu_to_ecef
from wls import WlsConfig, wls_epoch
from data_io import save_trajectories, save_metadata

def show_satellite_positions(df_true: pd.DataFrame, config: TrajectoryConfig):
    """
    Генерирует положения всех спутников группировки на момент t0
    и передает их на отрисовку 3D модели.
    """
    from coord_conversion import enu_to_ecef
    from gnss import _compute_satellite_state, GM_EARTH
    
    print("\n  -> Отрисовка 3D модели Земли и орбит (все спутники)...")
    
    # 1. Получаем время и ECEF координаты приемника (машины)
    t0 = df_true['t'].iloc[0]
    row0 = df_true.iloc[0]
    rec_X, rec_Y, rec_Z = enu_to_ecef(
        row0['E'], row0['N'], row0['U'], 
        config.ref_lat, config.ref_lon, config.ref_alt
    )
    receiver_pos = np.array([rec_X, rec_Y, rec_Z])
    
    # 2. Генерируем положение ВСЕХ спутников на момент t0
    total_sim_satellites = config.num_satellites
    R_orb = config.satellite_radius
    
    all_sats = []
    for svid in range(total_sim_satellites):
        r_sat, _ = _compute_satellite_state(t0, svid, total_sim_satellites, R_orb)
        all_sats.append(r_sat)
            
    all_sats = np.array(all_sats)
    
    # 3. Генерируем кольца орбит
    period = 2 * np.pi * np.sqrt((R_orb**3) / GM_EARTH)
    time_steps = np.linspace(0, period, 100)
    
    orbit_lines = []
    num_planes = 6
    for svid in range(num_planes):
        pts = []
        for t_orb in time_steps:
            r_sat, _ = _compute_satellite_state(t_orb, svid, total_sim_satellites, R_orb)
            pts.append(r_sat)
        orbit_lines.append(np.array(pts))
    
    # 4. Вызов отрисовки
    from graph import plot_earth_and_satellites
    plot_earth_and_satellites(all_sats, receiver_pos, orbit_lines=orbit_lines)


def get_Q_piecewise_white_noise(dt: float, sigma_a: float) -> np.ndarray:
    Q = np.zeros((6, 6))
    dt2 = dt**2
    dt3 = dt**3 / 2.0
    dt4 = dt**4 / 4.0
    
    q_block = np.array([
        [dt4, dt3],
        [dt3, dt2]
    ]) * (sigma_a ** 2)

    for i in range(3):
        Q[i, i] = q_block[0, 0]          # pos-pos
        Q[i, i+3] = q_block[0, 1]        # pos-vel
        Q[i+3, i] = q_block[1, 0]        # vel-pos
        Q[i+3, i+3] = q_block[1, 1]      # vel-vel

    return Q

def get_Q_continuous_white_noise(dt: float, sigma_a: float) -> np.ndarray:
    Q = np.zeros((6, 6))
    dt2 = dt**2 / 2.0
    dt3 = dt**3 / 3.0
    
    q_block = np.array([
        [dt3, dt2],
        [dt2, dt]
    ]) * (sigma_a ** 2)

    for i in range(3):
        Q[i, i] = q_block[0, 0]          # pos-pos
        Q[i, i+3] = q_block[0, 1]        # pos-vel
        Q[i+3, i] = q_block[1, 0]        # vel-pos
        Q[i+3, i+3] = q_block[1, 1]      # vel-vel
        
    return Q

def get_R(sigma_measurement: float) -> np.ndarray:
    """ Матрица ковариации шума измерений R (3x3) """
    R = np.eye(3) * sigma_measurement**2
    return R

def run_wls_solver(df_raw: pd.DataFrame, config: TrajectoryConfig) -> pd.DataFrame:
    print("  -> Запуск взвешенного МНК (WLS)...")
    wls_cfg = WlsConfig()

    x_ref, y_ref, z_ref = enu_to_ecef(0.0, 0.0, 0.0, config.ref_lat, config.ref_lon, config.ref_alt)
    x0 = np.array([x_ref, y_ref, z_ref, 0.0], dtype=float)
    x_prev = x0.copy()

    wls_rows = []

    for t, g in df_raw.groupby('t'):
        g = g.reset_index(drop=True)
        x_hat, P_hat, dops = wls_epoch(g, x_prev, wls_cfg)  # Забираем DOP

        wls_rows.append({
            't': float(t),
            'X': x_hat[0], 'Y': x_hat[1], 'Z': x_hat[2], 'cb': x_hat[3],
            'P00': P_hat[0, 0], 'P11': P_hat[1, 1], 'P22': P_hat[2, 2], 'P33': P_hat[3, 3],
            'HDOP': dops['HDOP'], 'VDOP': dops['VDOP'], 'PDOP': dops['PDOP'] # Добавляем в датафрейм
        })
        x_prev = x_hat 

    df_wls = pd.DataFrame(wls_rows)
    enu_coords = [ecef_to_enu(r.X, r.Y, r.Z, config.ref_lat, config.ref_lon, config.ref_alt) for r in df_wls.itertuples()]
    df_wls['E'], df_wls['N'], df_wls['U'] = zip(*enu_coords)

    return df_wls

def run_linear_kf(df_wls: pd.DataFrame, config: TrajectoryConfig, sigma_a: float = 0.73) -> pd.DataFrame:
    """Запускает линейный ФК из ЛР1, используя ENU координаты от МНК в качестве измерений."""
    
    Q = get_Q_continuous_white_noise(config.dt_gnss, sigma_a)
    R = get_R(config.gnss_pos_sigma)
    
    # Инициализация
    z0 = df_wls.iloc[0][['E', 'N', 'U']].values
    x0 = np.array([z0[0], z0[1], z0[2], 0.0, 0.0, 0.0])
    P0 = np.eye(6)
    P0[0:3, 0:3] *= (config.gnss_pos_sigma ** 2)
    P0[3:6, 3:6] *= 100.0
    
    kf = LinearKalmanFilter(dt=config.dt_gnss, Q=Q, R=R, x0=x0, P0=P0)
    kf_results = []
    
    for _, row in df_wls.iterrows():
        z = np.array([row['E'], row['N'], row['U']])
        x_est, P_est = kf.step(z)
        
        kf_results.append({
            't': row['t'],
            'E_est': x_est[0], 'N_est': x_est[1], 'U_est': x_est[2],
            'vE_est': x_est[3], 'vN_est': x_est[4], 'vU_est': x_est[5]
        })
        
    return pd.DataFrame(kf_results)

def run_extended_gnss_kf(df_raw: pd.DataFrame, df_wls: pd.DataFrame, config: TrajectoryConfig, sigma_a: float = 1.0) -> pd.DataFrame:
    sigma_cb = 0.5 
    sigma_cd = 0.05 

    row0 = df_wls.iloc[0]
    x0 = np.array([
        row0['X'], row0['Y'], row0['Z'], 
        0.0, 0.0, 0.0, 
        row0['cb'], 0.0 
    ])

    P0 = np.eye(8) * 100.0 

    ekf = ExtendedKalmanFilter(
        dt=config.dt_gnss,
        sigma_a=sigma_a, sigma_cb=sigma_cb, sigma_cd=sigma_cd,
        sigma_doppler=config.raw_doppler_sigma,
        x0=x0, P0=P0
    )

    ekf_results = []
    for t in df_wls['t'].values:
        ekf.predict_step()

        wls_row = df_wls[df_wls['t'] == t].iloc[0]
        x_wls = np.array([wls_row['X'], wls_row['Y'], wls_row['Z'], wls_row['cb']])
        P_wls = np.diag([wls_row['P00'], wls_row['P11'], wls_row['P22'], wls_row['P33']])
        ekf.update_wls(x_wls, P_wls)

        raw_epoch = df_raw[df_raw['t'] == t]
        if len(raw_epoch) > 0:
            doppler = raw_epoch['doppler'].values
            sat_pos = raw_epoch[['sat_X', 'sat_Y', 'sat_Z']].values
            sat_vel = raw_epoch[['sat_vX', 'sat_vY', 'sat_vZ']].values
            ekf.update_doppler(doppler, sat_pos, sat_vel)

        x_est = ekf.x.flatten()
        E, N, U = ecef_to_enu(x_est[0], x_est[1], x_est[2], config.ref_lat, config.ref_lon, config.ref_alt)

        lat, lon = config.ref_lat, config.ref_lon
        R_ecef2enu = np.array([
            [-np.sin(lon),              np.cos(lon),             0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
            [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
        ])
        v_ecef = np.array([x_est[3], x_est[4], x_est[5]])
        v_enu = R_ecef2enu @ v_ecef

        ekf_results.append({
            't': t,
            'E_est': E, 'N_est': N, 'U_est': U,
            'vE_est': v_enu[0], 'vN_est': v_enu[1], 'vU_est': v_enu[2]
        })

    return pd.DataFrame(ekf_results)


# ==========================================
# Общая генерация данных
# ==========================================
def generate_all_data():
    config = TrajectoryConfig()
    df_imu_clean = generate_trajectory(config, stages_scenario)
    df_imu_noisy = simulate_imu_errors(df_imu_clean, config)
    df_gnss_clean, df_gnss_noisy = process_gnss(df_imu_clean, config)
    df_gnss_raw = simulate_gnss_raw(df_gnss_clean, config)
    save_trajectories(df_imu_clean, df_imu_noisy, df_gnss_clean, df_gnss_noisy, df_gnss_raw)
    save_metadata(config)
    return config, df_imu_clean, df_imu_noisy, df_gnss_clean, df_gnss_noisy, df_gnss_raw

# ==========================================
# Лабораторная работа №1: Линейный КФ (ENU)
# ==========================================
def run_lab01(df_true, df_meas, config):
    print("\n--- Lab 01: Linear Kalman Filter ---")
    # Настройка и запуск линейного фильтра Калмана
    test_sigmas = [0.01, 1.0, 10.0, 0.69, 0.7, 0.73]
    # test_sigmas = [0.69]
    
    df_true_renamed = df_true[['t', 'E', 'N', 'U', 'vE', 'vN', 'vU']].rename(
        columns={'E': 'E_true', 'N': 'N_true', 'U': 'U_true',
                 'vE': 'vE_true', 'vN': 'vN_true', 'vU': 'vU_true'}
    )
    

    z0 = df_meas.iloc[0][['E', 'N', 'U']].values
    x0 = np.array([z0[0], z0[1], z0[2], 0.0, 0.0, 0.0])
    P0 = np.eye(6)
    P0[0:3, 0:3] *= (config.gnss_pos_sigma ** 2)
    P0[3:6, 3:6] *= 100.0
    R = get_R(config.gnss_pos_sigma)
    for sigma_a in test_sigmas:
        for Q in [get_Q_continuous_white_noise(config.dt_gnss, sigma_a), get_Q_piecewise_white_noise(config.dt_gnss, sigma_a)]:
            kf = LinearKalmanFilter(
                dt=config.dt_gnss,
                Q=Q,
                R=R,
                x0=x0,
                P0=P0
            )

            kf_results = []
            for _, row in df_meas.iterrows():
                z = np.array([row['E'], row['N'], row['U']])
                x_est, P_est = kf.step(z)
                kf_results.append({
                    't': row['t'],
                    'E_est': x_est[0], 'N_est': x_est[1], 'U_est': x_est[2],
                    'vE_est': x_est[3], 'vN_est': x_est[4], 'vU_est': x_est[5]
                })

            df_kf = pd.DataFrame(kf_results)
            metrics = calculate_rmse(df_true_renamed, df_kf)
            if np.array_equal(Q, get_Q_continuous_white_noise(config.dt_gnss, sigma_a)) :
                print("Модель непрерывного белого шума, ", end="")
            elif np.array_equal(Q, get_Q_piecewise_white_noise(config.dt_gnss, sigma_a)):
                print("Модель \"кусочного\" белого шума, ", end="")
            print(f"СКО неучтённого ускорения = {sigma_a}: Ошибка Координаты = {metrics['pos_rmse_3d']:.3f} м, ошибка скорости = {metrics['vel_rmse_3d']:.3f} м/с")

            # plot_kf_comparison(df_true, df_meas, df_kf)
        plot_kf_comparison(df_true, df_meas, df_kf)

# ==========================================
# Лабораторная работа №2: Псевдодальности и WLS
# ==========================================
def run_lab02(df_raw: pd.DataFrame, df_true: pd.DataFrame, config: TrajectoryConfig):
    print("\n--- Lab 02: GNSS WLS & Extended KF ---")
    
    df_true_renamed = df_true[['t', 'E', 'N', 'U', 'vE', 'vN', 'vU']].rename(columns={
        'E': 'E_true', 'N': 'N_true', 'U': 'U_true',
        'vE': 'vE_true', 'vN': 'vN_true', 'vU': 'vU_true'
    })
    
    show_satellite_positions(df_true, config)
    
    # --- Шаг 1: МНК ---
    df_wls = run_wls_solver(df_raw, config)
    metrics_wls = calculate_rmse(df_true_renamed, df_wls.rename(columns={'E':'E_est', 'N':'N_est', 'U':'U_est'}))
    print(f" [WLS] RMSE Позиции (3D): {metrics_wls['pos_rmse_3d']:.3f} м")
    print(f" [WLS] Средний HDOP: {df_wls['HDOP'].mean():.2f}, VDOP: {df_wls['VDOP'].mean():.2f}")
    
    plot_wls_results(df_true, df_wls)
    
    # --- Шаг 2: Линейный ФК ---
    print("\n [Linear KF] Подбор параметра sigma_a (доверие модели):")
    test_sigmas = [0.01, 0.1, 0.5, 0.7, 0.8, 0.9, 1.0, 5.0, 10.0]
    best_df_kf_lin = None
    best_rmse = 999999.9
    
    for sa in test_sigmas:
        df_kf_linear = run_linear_kf(df_wls, config, sigma_a=sa)
        met = calculate_rmse(df_true_renamed, df_kf_linear)
        print(f"LKF -> sigma_a = {sa:4.2f} | RMSE Позиции: {met['pos_rmse_3d']:.3f} м | Скорости: {met['vel_rmse_3d']:.3f} м/с")
        
        if met['pos_rmse_3d'] < best_rmse:
            best_rmse = met['pos_rmse_3d']
            best_df_kf_lin = df_kf_linear
    
    plot_kf_comparison(df_true, df_wls, best_df_kf_lin, title_suffix="(Линейный ФК по МНК)")
    
    # --- Шаг 3: Расширенный ФК с перебором параметров ---
    print("\n [Extended KF] Подбор параметра sigma_a (доверие модели):")
    test_sigmas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.73, 1.0, 5.0, 10.0]
    best_df_kf_ext = None
    best_rmse = 999999.9
    
    for sa in test_sigmas:
        df_kf_ext_temp = run_extended_gnss_kf(df_raw, df_wls, config, sigma_a=sa)
        met = calculate_rmse(df_true_renamed, df_kf_ext_temp)
        print(f"EKF -> sigma_a = {sa:4.2f} | RMSE Позиции: {met['pos_rmse_3d']:.3f} м | Скорости: {met['vel_rmse_3d']:.3f} м/с")
        
        if met['pos_rmse_3d'] < best_rmse:
            best_rmse = met['pos_rmse_3d']
            best_df_kf_ext = df_kf_ext_temp
            
    plot_kf_comparison(df_true, df_wls, best_df_kf_ext, title_suffix="(Расширенный ФК: МНК + Доплер)")

# ==========================================
# Лабораторная работа №3: Слабосвязанная ИНС/ГНСС
# ==========================================
def run_lab03():
    print("\n--- Lab 03: Loosely Coupled INS/GNSS ---")
    # 1. Механизация ИНС (Strapdown)
    # 2. 15-мерный фильтр Калмана в пространстве ошибок (15-state Error KF)
    # 3. Обновление по позиции и скорости от GNSS, учет lever arm
    print("Структура для ЛР3 готова. Ожидает реализации механизации и 15-мерного состояния.")

# ==========================================
# Лабораторная работа №4: Сильносвязанная ИНС/ГНСС
# ==========================================
def run_lab04():
    print("\n--- Lab 04: Tightly Coupled INS/GNSS ---")
    # 1. Расширенное состояние (15 ошибок ИНС + часы cb, cd)
    # 2. Обновление напрямую по сырым псевдодальностям и доплерам через LOS
    print("Структура для ЛР4 готова. Ожидает построения матрицы измерений.")

if __name__ == "__main__":
    config, df_imu_clean, df_imu_noisy, df_gnss_clean, df_gnss_noisy, df_gnss_raw = generate_all_data()
    
    plot_results(df_imu_clean, df_gnss_noisy)
    run_lab01(df_gnss_clean, df_gnss_noisy, config)
    run_lab02(df_gnss_raw, df_gnss_clean, config)
    #run_lab03()
    #run_lab04()
