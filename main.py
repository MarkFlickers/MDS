from configuration import TrajectoryConfig, stages_scenario
from trajectory import generate_trajectory, simulate_imu_errors
from gnss import process_gnss, simulate_gnss_raw
from graph import plot_results, plot_kf_comparison
from dataclasses import asdict
from data_io import save_trajectories, save_metadata
from metrics import calculate_rmse
from kalman import LinearKalmanFilter
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
        df_kf_for_plot = df_kf.rename(columns={'E_est': 'E', 'N_est': 'N', 'U_est': 'U'})
        plot_kf_comparison(df_true, df_meas, df_kf_for_plot, sigma_a_label=sigma_a)

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