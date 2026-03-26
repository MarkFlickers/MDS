import pandas as pd
import numpy as np
from configuration import TrajectoryConfig, stages_scenario, stages_scenario_hard, stages_scenario_extreme, stages_scenario_city
from trajectory import generate_trajectory, simulate_imu_errors
from gnss import process_gnss, simulate_gnss_raw, get_wave_length
from graph import plot_results, plot_kf_comparison, plot_wls_results, plot_nav_solution_comparison
from metrics import calculate_rmse
from kalman import LinearKalmanFilter, ExtendedKalmanFilter
from ins import mechanize_ins, run_loosely_coupled_ins_gnss
from coord_conversion import ecef_to_enu, enu_to_ecef
from wls import WlsConfig, wls_epoch
from data_io import save_trajectories, save_metadata
from graph import plot_earth_and_satellites

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
        x_hat, P_hat, dops = wls_epoch(g, x_prev, wls_cfg, config)  # Забираем DOP

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
    row1 = df_wls.iloc[1]
    init_coords = row0
    init_speed = (row1[['X', 'Y', 'Z']] - row0[['X', 'Y', 'Z']]) / (row1['t'] - row0['t'])
    x0 = np.array([
        init_coords['X'], init_coords['Y'], init_coords['Z'], 
        init_speed['X'], init_speed['Y'], init_speed['Z'], 
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
    df_imu_clean = generate_trajectory(config, stages_scenario_hard)
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
    best_df_kf = None
    best_rmse = 999999.9
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
            if metrics['pos_rmse_3d'] < best_rmse:
                best_rmse = metrics['pos_rmse_3d']
                best_df_kf = df_kf
            if np.array_equal(Q, get_Q_continuous_white_noise(config.dt_gnss, sigma_a)) :
                print("Модель непрерывного белого шума, ", end="")
            elif np.array_equal(Q, get_Q_piecewise_white_noise(config.dt_gnss, sigma_a)):
                print("Модель \"кусочного\" белого шума, ", end="")
            print(f"СКО неучтённого ускорения = {sigma_a}: Ошибка Координаты = {metrics['pos_rmse_3d']:.3f} м, ошибка скорости = {metrics['vel_rmse_3d']:.3f} м/с")

            # plot_kf_comparison(df_true, df_meas, df_kf)
    plot_kf_comparison(df_true, df_meas, best_df_kf)

# ==========================================
# Лабораторная работа №2: Псевдодальности и WLS
# ==========================================
def run_lab02(df_raw: pd.DataFrame, df_true: pd.DataFrame, config: TrajectoryConfig):
    print("\n--- Lab 02: GNSS WLS & Extended KF ---")
    
    df_true_renamed = df_true[['t', 'E', 'N', 'U', 'vE', 'vN', 'vU']].rename(columns={
        'E': 'E_true', 'N': 'N_true', 'U': 'U_true',
        'vE': 'vE_true', 'vN': 'vN_true', 'vU': 'vU_true'
    })
    
    #show_satellite_positions(df_true, config)
    sat_positions = df_raw[['sat_X', 'sat_Y', 'sat_Z']][df_raw['t'] == 0.0].values
    rec_position = df_true[['X_ecef', 'Y_ecef', 'Z_ecef']].values[0]
    plot_earth_and_satellites(sat_positions, rec_position)
    
    # --- Шаг 1: МНК ---
    df_wls = run_wls_solver(df_raw, config)
    metrics_wls = calculate_rmse(df_true_renamed, df_wls.rename(columns={'E':'E_est', 'N':'N_est', 'U':'U_est'}))
    print(f" [WLS] RMSE Позиции (3D): {metrics_wls['pos_rmse_3d']:.3f} м")
    print(f" [WLS] Средний HDOP: {df_wls['HDOP'].mean():.2f}, VDOP: {df_wls['VDOP'].mean():.2f}")
    
    plot_wls_results(df_true, df_wls)
    
    # --- Шаг 2: Линейный ФК ---
    print("\n [Linear KF] Подбор параметра sigma_a (доверие модели):")
    test_sigmas = [0.01, 0.1, 0.5, 0.7, 0.8, 0.9, 1.0, 5.0, 10.0, 100.0]
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
    test_sigmas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.73, 1.0, 5.0, 10.0, 100.0]
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
def run_lab03(
    df_ref: pd.DataFrame,
    df_gnss_meas: pd.DataFrame,
    df_imu_noisy: pd.DataFrame,
    config: TrajectoryConfig,
    gnss_pos_sigma: float | None = None,
    accel_bias_rw_sigma: float = 1e-3,
    gyro_bias_rw_sigma: float = 1e-4,
    gnss_pos_sigma_grid: list[float] | None = None,
    accel_bias_rw_sigma_grid: list[float] | None = None,
    gyro_bias_rw_sigma_grid: list[float] | None = None,
    tune_filter: bool = True,
    plot_enabled: bool = True,
):
    print("\n--- Lab 03: Loosely Coupled INS/GNSS ---")

    gnss_pos_sigma = config.gnss_pos_sigma if gnss_pos_sigma is None else gnss_pos_sigma
    gnss_pos_sigma_grid = [gnss_pos_sigma] if gnss_pos_sigma_grid is None else gnss_pos_sigma_grid
    accel_bias_rw_sigma_grid = [accel_bias_rw_sigma] if accel_bias_rw_sigma_grid is None else accel_bias_rw_sigma_grid
    gyro_bias_rw_sigma_grid = [gyro_bias_rw_sigma] if gyro_bias_rw_sigma_grid is None else gyro_bias_rw_sigma_grid

    if tune_filter:
        if gnss_pos_sigma_grid == [gnss_pos_sigma]:
            gnss_pos_sigma_grid = [4.0, 5.0, 7.0, 8.0]
        if accel_bias_rw_sigma_grid == [accel_bias_rw_sigma]:
            accel_bias_rw_sigma_grid = [1e-6, 1e-5, 1e-4]
        if gyro_bias_rw_sigma_grid == [gyro_bias_rw_sigma]:
            gyro_bias_rw_sigma_grid = [1e-6, 1e-5, 1e-4]

    df_ref_nav = df_ref[[
        't', 'E_imu', 'N_imu', 'U_imu',
        'vE', 'vN', 'vU',
        'roll', 'pitch', 'yaw',
        'q_w', 'q_x', 'q_y', 'q_z',
        'E_ant', 'N_ant', 'U_ant'
    ]].copy()

    row_gnss0 = df_gnss_meas.iloc[0]
    row_ref0 = df_ref.iloc[0]
    init_pos = row_gnss0[['E', 'N', 'U']].to_numpy(dtype=float)
    init_vel = (
        row_gnss0[['vE', 'vN', 'vU']].to_numpy(dtype=float)
        if all(col in df_gnss_meas.columns for col in ['vE', 'vN', 'vU'])
        else np.zeros(3)
    )
    init_quat = row_ref0[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)

    print(" -> Автономная ИНС-механизация...")
    df_ins = mechanize_ins(
        df_imu=df_imu_noisy,
        cfg=config,
        init_pos=init_pos,
        init_vel=init_vel,
        init_quat=init_quat,
    )

    df_true_rmse = df_ref_nav[['t', 'E_imu', 'N_imu', 'U_imu', 'vE', 'vN', 'vU']].rename(columns={
        'E_imu': 'E_true',
        'N_imu': 'N_true',
        'U_imu': 'U_true',
        'vE': 'vE_true',
        'vN': 'vN_true',
        'vU': 'vU_true',
    })

    df_ins_rmse = df_ins[['t', 'E_est', 'N_est', 'U_est', 'vE_est', 'vN_est', 'vU_est']].copy()
    metrics_ins = calculate_rmse(df_true_rmse, df_ins_rmse)
    print(
        f"Автономная ИНС: RMSE позиции = {metrics_ins['pos_rmse_3d']:.3f} м, "
        f"RMSE скорости = {metrics_ins['vel_rmse_3d']:.3f} м/с"
    )

    if plot_enabled:
        plot_nav_solution_comparison(
            df_ref_nav, df_ins, df_gnss=df_gnss_meas,
            title_suffix='(автономная ИНС)'
        )

    def run_single_filter(use_lever_arm: bool, pos_sigma: float, accel_rw: float, gyro_rw: float):
        return run_loosely_coupled_ins_gnss(
            df_imu=df_imu_noisy,
            df_gnss=df_gnss_meas,
            df_ref=df_ref,
            cfg=config,
            use_lever_arm=use_lever_arm,
            gnss_pos_sigma=pos_sigma,
            accel_bias_rw_sigma=accel_rw,
            gyro_bias_rw_sigma=gyro_rw,
        )

    def select_best_result(use_lever_arm: bool, label: str):
        best_df = None
        best_metrics = None
        best_params = None
        best_rmse = np.inf

        print(f" -> Подбор параметров для {label}...")
        for pos_sigma in gnss_pos_sigma_grid:
            for accel_rw in accel_bias_rw_sigma_grid:
                for gyro_rw in gyro_bias_rw_sigma_grid:
                    df_candidate = run_single_filter(use_lever_arm, pos_sigma, accel_rw, gyro_rw)
                    df_candidate_rmse = df_candidate[[
                        't', 'E_est', 'N_est', 'U_est', 'vE_est', 'vN_est', 'vU_est'
                    ]].copy()
                    metrics_candidate = calculate_rmse(df_true_rmse, df_candidate_rmse)

                    print(
                        f"   {label}: R={pos_sigma:.3f}, "
                        f"q_ba={accel_rw:.1e}, q_bg={gyro_rw:.1e} -> "
                        f"RMSE pos={metrics_candidate['pos_rmse_3d']:.3f} м, "
                        f"vel={metrics_candidate['vel_rmse_3d']:.3f} м/с"
                    )

                    if metrics_candidate['pos_rmse_3d'] < best_rmse:
                        best_rmse = metrics_candidate['pos_rmse_3d']
                        best_df = df_candidate
                        best_metrics = metrics_candidate
                        best_params = {
                            'gnss_pos_sigma': pos_sigma,
                            'accel_bias_rw_sigma': accel_rw,
                            'gyro_bias_rw_sigma': gyro_rw,
                        }

        return best_df, best_metrics, best_params

    df_eskf_noarm, metrics_eskf_noarm, best_params_noarm = select_best_result(
        False, 'ESKF без lever arm'
    )
    print(
        f"Лучший ESKF без lever arm: "
        f"R={best_params_noarm['gnss_pos_sigma']:.3f}, "
        f"q_ba={best_params_noarm['accel_bias_rw_sigma']:.1e}, "
        f"q_bg={best_params_noarm['gyro_bias_rw_sigma']:.1e}, "
        f"RMSE позиции = {metrics_eskf_noarm['pos_rmse_3d']:.3f} м, "
        f"RMSE скорости = {metrics_eskf_noarm['vel_rmse_3d']:.3f} м/с"
    )
    if plot_enabled:
        plot_nav_solution_comparison(
            df_ref_nav, df_eskf_noarm, df_gnss=df_gnss_meas,
            title_suffix='(лучший ESKF без lever arm)'
        )

    df_eskf_arm, metrics_eskf_arm, best_params_arm = select_best_result(
        True, 'ESKF с lever arm'
    )
    print(
        f"Лучший ESKF с lever arm: "
        f"R={best_params_arm['gnss_pos_sigma']:.3f}, "
        f"q_ba={best_params_arm['accel_bias_rw_sigma']:.1e}, "
        f"q_bg={best_params_arm['gyro_bias_rw_sigma']:.1e}, "
        f"RMSE позиции = {metrics_eskf_arm['pos_rmse_3d']:.3f} м, "
        f"RMSE скорости = {metrics_eskf_arm['vel_rmse_3d']:.3f} м/с"
    )
    if plot_enabled:
        plot_nav_solution_comparison(
            df_ref_nav, df_eskf_arm, df_gnss=df_gnss_meas,
            title_suffix='(лучший ESKF с lever arm)'
        )

    return {
        'df_ins': df_ins,
        'df_eskf_noarm': df_eskf_noarm,
        'df_eskf_arm': df_eskf_arm,
        'metrics_ins': metrics_ins,
        'metrics_eskf_noarm': metrics_eskf_noarm,
        'metrics_eskf_arm': metrics_eskf_arm,
        'best_params_noarm': best_params_noarm,
        'best_params_arm': best_params_arm,
    }

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
    df_true = df_imu_clean.rename(
        columns={'E_imu': 'E', 'N_imu': 'N', 'U_imu': 'U',
                 'vE_imu': 'vE', 'vN_imu': 'vN', 'vU_imu': 'vU'}
    )
    df_true = pd.concat([df_true, df_gnss_clean[['X_ecef', 'Y_ecef', 'Z_ecef']]], axis=1)
    df_gnss_raw = pd.read_csv("output/SignalSimTrajectory.obs.csv", header=0)
    lam = df_gnss_raw['sv_id'].astype(str).str[0].map(get_wave_length)
    df_gnss_raw['doppler'] = -lam * df_gnss_raw['doppler']

    #plot_results(df_imu_clean, df_gnss_noisy)
    #run_lab01(df_true, df_gnss_noisy, config)
    run_lab02(df_gnss_raw, df_true, config)
    #run_lab03(df_imu_clean, df_gnss_noisy, df_imu_noisy, config, gnss_pos_sigma_grid=[7.0], accel_bias_rw_sigma_grid=[1e-5], gyro_bias_rw_sigma_grid=[1e-5],)
    # run_lab03(df_imu_clean, df_gnss_noisy, df_imu_noisy, config)
    # run_lab04()