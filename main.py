import pandas as pd
import numpy as np
from configuration import TrajectoryConfig, stages_scenario
from trajectory import generate_trajectory, simulate_imu_errors
from gnss import process_gnss, simulate_gnss_raw
from graph import plot_results, plot_kf_comparison, plot_wls_results
from metrics import calculate_rmse
from kalman import LinearKalmanFilter, GNSSKalmanFilter
from coord_conversion import ecef_to_enu, enu_to_ecef
from wls import WlsConfig, wls_epoch
from data_io import save_trajectories, save_metadata

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
    
    # 1. Часть: Моделирование траектории и шума 
    
    # 2. Часть: Настройка и запуск линейного фильтра Калмана
    test_sigmas = [0.01, 1.0, 10.0, 0.61]
    
    df_true_renamed = df_true[['t', 'E', 'N', 'U', 'vE', 'vN', 'vU']].rename(
        columns={'E': 'E_true', 'N': 'N_true', 'U': 'U_true',
                 'vE': 'vE_true', 'vN': 'vN_true', 'vU': 'vU_true'}
    )
    
    for sigma_a in test_sigmas:
        z0 = df_meas.iloc[0][['E', 'N', 'U']].values
        x0 = np.array([z0[0], z0[1], z0[2], 0.0, 0.0, 0.0])
        P0 = np.eye(6)
        P0[0:3, 0:3] *= (config.gnss_pos_sigma ** 2)
        P0[3:6, 3:6] *= 100.0
        
        kf = LinearKalmanFilter(
            dt=config.dt_gnss,
            sigma_a=sigma_a,
            sigma_gnss=config.gnss_pos_sigma,
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
        print(f"Sigma_a = {sigma_a} (Дисперсия Q): Ошибка Pos = {metrics['pos_rmse_3d']:.2f} м, Скорость = {metrics['vel_rmse_3d']:.2f} м/с")

        plot_kf_comparison(df_true, df_meas, df_kf)
        
    # 3. Часть: Анализ результатов и подготовка к ответам на вопросы к защите
    # Анализируется влияние матрицы R (шум измерений) и Q (шум процесса)

# ==========================================
# Лабораторная работа №2: Псевдодальности и WLS
# ==========================================
def run_lab02(df_raw, df_true, config):
    print("\n--- Lab 02: GNSS WLS & Extended KF ---")
    
    wls_cfg = WlsConfig()
    
    # Опорная точка для перевода в ENU
    x_ref, y_ref, z_ref = enu_to_ecef(0.0, 0.0, 0.0, config.ref_lat, config.ref_lon, config.ref_alt)
    x0 = np.array([x_ref, y_ref, z_ref, 0.0], dtype=float)
    
    wls_rows = []
    x_prev = x0.copy()
    
    # 1. Часть: Решение навигационной задачи методом взвешенных МНК (WLS)
    for t, g in df_raw.groupby('t'):
        g = g.reset_index(drop=True)
        x_hat, P_hat = wls_epoch(g, x_prev, wls_cfg)
        
        wls_rows.append({
            't': float(t),
            'X': x_hat[0], 'Y': x_hat[1], 'Z': x_hat[2], 'cb': x_hat[3],
            'P00': P_hat[0, 0], 'P11': P_hat[1, 1], 'P22': P_hat[2, 2], 'P33': P_hat[3, 3],
        })
        x_prev = x_hat # Использование прошлого решения для старта (Warm start)
        
    df_wls = pd.DataFrame(wls_rows)
    
    # Перевод в ENU для оценки ошибки
    enu = np.array([ecef_to_enu(r.X, r.Y, r.Z, config.ref_lat, config.ref_lon, config.ref_alt) for r in df_wls.itertuples()])
    df_wls['E'] = enu[:, 0]
    df_wls['N'] = enu[:, 1]
    df_wls['U'] = enu[:, 2]
    
    print("WLS посчитан: проверьте точность df_wls (E,N,U) относительно истины.")
    plot_wls_results(df_true, df_wls)
    # 2. Часть: Расширенный КФ с оценкой часов (cb, cd) и обновлением по доплерам
    print("Место для эксперимента: GNSSKalmanFilter с обновлением по Доплеру.")
    return df_wls

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
    run_lab03()
    run_lab04()
