from configuration import TrajectoryConfig, stages_scenario
from trajectory import generate_trajectory, simulate_imu_errors
from gnss import process_gnss, simulate_gnss_raw
from graph import plot_results, plot_kf_comparison
from dataclasses import asdict
from data_io import save_trajectories, save_metadata
from metrics import calculate_rmse
from kalman import LinearKalmanFilter, GNSSKalmanFilter
from coord_conversion import ecef_to_enu, enu_to_ecef
from wls import WlsConfig, wls_epoch
import pandas as pd
import numpy as np

def test_kalman_filter(df_true: pd.DataFrame, df_meas: pd.DataFrame, config: TrajectoryConfig):
    """
    Тестирование линейного фильтра Калмана с разными параметрами шума процесса.
    """
    print("\n--- ТЕСТИРОВАНИЕ ФИЛЬТРА КАЛМАНА (ЛР1) ---")
    
    # 1. Параметры для перебора. 
    # sigma_a - ожидаемое фильтром ускорение объекта (м/с^2). 
    # Малое = верим модели равномерного движения. Большое = верим сырым измерениям ГНСС.
    test_sigmas = [0.01, 1.0, 10.0] 
    
    # 2. Подготовка эталонного датафрейма для функции метрик (переименуем колонки для удобства)
    df_true_renamed = df_true[['t', 'E', 'N', 'U', 'vE', 'vN', 'vU']].rename(columns={
        'E': 'E_true', 'N': 'N_true', 'U': 'U_true',
        'vE': 'vE_true', 'vN': 'vN_true', 'vU': 'vU_true'
    })

    for sigma_a in test_sigmas:
        # Инициализация ФК
        # x0 берем по первому измерению ГНСС, скорости = 0
        z0 = df_meas.iloc[0][['E', 'N', 'U']].values
        x0 = np.array([z0[0], z0[1], z0[2], 0.0, 0.0, 0.0])
        
        # P0 делаем большой, т.к. мы не знаем точную начальную скорость
        P0 = np.eye(6)
        P0[0:3, 0:3] *= (config.gnss_pos_sigma ** 2) # Дисперсия позиции равна точности ГНСС
        P0[3:6, 3:6] *= 100.0                        # Высокая неопределенность по начальной скорости
        
        kf = LinearKalmanFilter(
            dt=config.dt_gnss, 
            sigma_a=sigma_a, 
            sigma_gnss=config.gnss_pos_sigma, 
            x0=x0, P0=P0
        )
        
        # Массив для сохранения результатов
        kf_results = []
        
        # Цикл фильтрации по эпохам ГНСС
        for _, row in df_meas.iterrows():
            z = np.array([row['E'], row['N'], row['U']])
            x_est, P_est = kf.step(z)
            
            kf_results.append({
                't': row['t'],
                'E_est': x_est[0], 'N_est': x_est[1], 'U_est': x_est[2],
                'vE_est': x_est[3], 'vN_est': x_est[4], 'vU_est': x_est[5]
            })
            
        df_kf = pd.DataFrame(kf_results)
        
        # Считаем метрики
        metrics = calculate_rmse(df_true_renamed, df_kf)
        print(f"\nРезультаты для sigma_a = {sigma_a} м/с^2:")
        print(f"  СКО 3D Позиции: {metrics['pos_rmse_3d']:.2f} м (Сам ГНСС шумит на ~{config.gnss_pos_sigma*np.sqrt(3):.2f} м)")
        print(f"  СКО 3D Скорости: {metrics['vel_rmse_3d']:.2f} м/с")
        
        # Строим графики с переименованными колонками для совместимости
        df_kf_for_plot = df_kf.rename(columns={'E_est': 'E', 'N_est': 'N', 'U_est': 'U', 'vE_est': 'vE', 'vN_est': 'vN', 'vU_est': 'vU'})
        plot_kf_comparison(df_true, df_meas, df_kf_for_plot, sigma_a_label=sigma_a)

def test_wls(config: TrajectoryConfig) -> pd.DataFrame:
    config = TrajectoryConfig()

    df_raw = pd.read_csv("output/gnss_raw_observables.csv")
    df_true = pd.read_csv("output/trajectory_gnss_true.csv")

    wls_cfg = WlsConfig(
        max_iter=10,
        tol=1e-3,
        sigma_pr=config.raw_pr_sigma,  # можно оставить cfg.gnss_pos_sigma, но правильнее sigma псевдодальности
        use_elevation_weights=True,
        el_mask_deg=0.0
    )

    # Начальное приближение: опорная точка ENU=(0,0,0) в ECEF + cb=0
    x_ref, y_ref, z_ref = enu_to_ecef(0.0, 0.0, 0.0, config.ref_lat, config.ref_lon, config.ref_alt)
    x0 = np.array([x_ref, y_ref, z_ref, 0.0], dtype=float)

    wls_rows = []
    x_prev = x0.copy()

    for t, g in df_raw.groupby("t"):
        g = g.reset_index(drop=True)

        sol = wls_epoch(g, x_prev, wls_cfg)
        x_hat = sol["x_hat"]
        P_hat = sol["P_hat"]

        wls_rows.append({
            "t": float(t),
            "X": x_hat[0], "Y": x_hat[1], "Z": x_hat[2],
            "cb": x_hat[3],
            "P00": P_hat[0, 0], "P11": P_hat[1, 1], "P22": P_hat[2, 2], "P33": P_hat[3, 3],
        })

        x_prev = x_hat  # warm start для следующей эпохи

    df_wls = pd.DataFrame(wls_rows)

    # Для сравнения и графиков переведём WLS в ENU
    enu = np.array([ecef_to_enu(r.X, r.Y, r.Z, config.ref_lat, config.ref_lon, config.ref_alt) for r in df_wls.itertuples()])
    df_wls["E"], df_wls["N"], df_wls["U"] = enu[:, 0], enu[:, 1], enu[:, 2]

    return df_wls

# 1. Инициализация конфигурации
config = TrajectoryConfig()

# 2. Интегрирование кинематики на частоте ИМУ
print("Генерация ИМУ-траектории (Идеал)...")
df_imu_clean = generate_trajectory(config, stages_scenario)

# 3. Генерируем зашумленные ИМУ данные ---
print("Генерация ошибок ИМУ (ЛР3)...")
df_imu_noisy = simulate_imu_errors(df_imu_clean, config)

# 4. Формирование ГНСС-измерений
print("Генерация ГНСС-измерений (ЛР1)...")
df_gnss_clean, df_gnss_noisy = process_gnss(df_imu_clean, config)

# 5. Генерируем сырые наблюдения ГНСС ---
print("Генерация сырых ГНСС наблюдений (ЛР4)...")
df_gnss_raw = simulate_gnss_raw(df_gnss_clean, config)

# 6. Сохранение результатов
save_trajectories(df_imu_clean, df_imu_noisy, df_gnss_clean, df_gnss_noisy, df_gnss_raw)
save_metadata(config)
    
print("Генерация завершена. Файлы сохранены:")
print(" - trajectory_imu_noisy.csv (Зашумленные ИМУ 50Гц для ЛР3/4)")
print(" - trajectory_imu_ideal.csv (Идеальные ИМУ 50Гц)")
print(" - trajectory_gnss_true.csv (Эталон 1Гц для ЛР1-4)")
print(" - gnss_measurements.csv (Координаты с шумом для ЛР1/2)")
print(" - gnss_raw_observables.csv (Псевдодальности и доплер для ЛР4)")
print(" - trajectory_metadata.json")

print("Построение графиков...")
#plot_results(df_imu_clean, df_gnss_clean, df_gnss_noisy)

#test_kalman_filter(df_gnss_clean, df_gnss_noisy, config)
df_wls = test_wls(config)