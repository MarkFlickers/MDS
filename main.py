from configuration import TrajectoryConfig, stages_scenario
from trajectory import generate_trajectory, simulate_imu_errors
from gnss import process_gnss, simulate_gnss_raw
from graph import plot_results
from dataclasses import asdict
import json

config = TrajectoryConfig()

print("Генерация ИМУ-траектории (Идеал)...")
df_imu_clean = generate_trajectory(config, stages_scenario)

# --- ЛР3: Генерируем зашумленные ИМУ данные ---
print("Генерация ошибок ИМУ (ЛР3)...")
df_imu_noisy = simulate_imu_errors(df_imu_clean, config)

print("Генерация ГНСС-измерений (ЛР1)...")
df_gnss_clean, df_gnss_noisy = process_gnss(df_imu_clean, config)

# --- ЛР4: Генерируем сырые наблюдения ГНСС ---
print("Генерация сырых ГНСС наблюдений (ЛР4)...")
df_gnss_raw = simulate_gnss_raw(df_gnss_clean, config)

# Сохранение результатов
df_imu_noisy.to_csv("output/trajectory_imu.csv", index=False) # Тут реалистичные измерения с шумом
df_imu_clean.to_csv("output/trajectory_imu_ideal.csv", index=False) # Cохраняем идеал для отладки

df_gnss_clean.to_csv("output/trajectory_gnss_true.csv", index=False)

df_gnss_meas = df_gnss_noisy[['t', 'E', 'N', 'U', 'lat', 'lon', 'alt', 'X_ecef', 'Y_ecef', 'Z_ecef']]
df_gnss_meas.to_csv("output/gnss_measurements.csv", index=False)

# Сохраняем сырые наблюдения (ЛР4)
df_gnss_raw.to_csv("output/gnss_raw_observables.csv", index=False)

meta = asdict(config)
meta['lever_arm'] = config.lever_arm.tolist()
meta['init_pos_enu'] = config.init_pos_enu.tolist()
meta['init_vel_body'] = config.init_vel_body.tolist()
meta['init_euler'] = config.init_euler.tolist()
meta['accel_bias'] = config.accel_bias.tolist()
meta['gyro_bias'] = config.gyro_bias.tolist()

with open("output/trajectory_metadata.json", "w") as f:
    json.dump(meta, f, indent=4)
    
print("Генерация завершена. Файлы сохранены:")
print(" - trajectory_imu.csv (Зашумленные ИМУ 50Гц для ЛР3/4)")
print(" - trajectory_imu_ideal.csv (Идеальные ИМУ 50Гц)")
print(" - trajectory_gnss_true.csv (Эталон 1Гц для ЛР1-4)")
print(" - gnss_measurements.csv (Координаты с шумом для ЛР1/2)")
print(" - gnss_raw_observables.csv (Псевдодальности и доплер для ЛР4)")
print(" - trajectory_metadata.json")

print("Построение графиков...")
plot_results(df_imu_clean, df_gnss_clean, df_gnss_noisy)