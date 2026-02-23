# ГНСС

import numpy as np
import pandas as pd
from typing import Tuple
from configuration import TrajectoryConfig
from coord_conversion import c_light, ecef_to_llh, enu_to_ecef

# ==========================================
# СЫРЫЕ НАБЛЮДЕНИЯ ГНСС
# ==========================================
def simulate_gnss_raw(df_gnss_clean: pd.DataFrame, cfg: TrajectoryConfig) -> pd.DataFrame:
    """Генерирует сырые псевдодальности и доплеровские измерения для виртуальных спутников."""
    np.random.seed(cfg.seed_gnss + 1) # Чтобы шум сырых измерений отличался от шума координат ЛР1
    
    # Моделируем дрейф часов приемника (простая интеграция случайного блуждания)
    dt = cfg.dt_gnss
    n_epochs = len(df_gnss_clean)
    
    clk_drift = np.zeros(n_epochs)
    clk_bias = np.zeros(n_epochs)
    
    clk_drift[0] = cfg.clock_drift_init
    clk_bias[0] = cfg.clock_bias_init
    for i in range(1, n_epochs):
        clk_drift[i] = clk_drift[i-1] + np.random.normal(0, cfg.clock_bias_noise)
        clk_bias[i] = clk_bias[i-1] + clk_drift[i-1] * dt
        
    # Размещаем спутники на орбитах (упрощенная модель круговых орбит для геометрии)
    # Распределяем их равномерно по сфере
    sat_positions = []
    sat_velocities = []
    for i in range(cfg.num_satellites):
        theta = 2 * np.pi * i / cfg.num_satellites
        phi = np.pi/4 # Угол возвышения 45 градусов
        # ECEF координаты спутников
        x = cfg.satellite_radius * np.cos(theta) * np.cos(phi)
        y = cfg.satellite_radius * np.sin(theta) * np.cos(phi)
        z = cfg.satellite_radius * np.sin(phi)
        sat_positions.append(np.array([x, y, z]))
        
        # Упрощенная скорость спутника (движение по кругу)
        v_mag = 3870.0 # Примерная скорость GPS спутника м/с
        vx = -v_mag * np.sin(theta)
        vy = v_mag * np.cos(theta)
        vz = 0.0
        sat_velocities.append(np.array([vx, vy, vz]))
        
    # Собираем измерения
    raw_obs_list = []
    
    for i, row in df_gnss_clean.iterrows():
        t = row['t']
        r_rec = np.array([row['X_ecef'], row['Y_ecef'], row['Z_ecef']])
        
        # В ЛР4: нужно получить скорости в ECEF. Для упрощения аппроксимируем:
        # v_ecef \approx v_enu, переведенная в ECEF (очень грубо, но для моделирования сойдет)
        # В реальности нужно применять матрицу поворота ENU->ECEF к вектору скорости.
        lat, lon = cfg.ref_lat, cfg.ref_lon
        R_enu_ecef = np.array([
            [-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
            [ np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
            [ 0,           np.cos(lat),              np.sin(lat)            ]
        ])
        v_rec_enu = np.array([row['vE'], row['vN'], row['vU']])
        v_rec = R_enu_ecef @ v_rec_enu
        
        c_b = clk_bias[i] * c_light
        c_d = clk_drift[i] * c_light
        
        for sv_id in range(cfg.num_satellites):
            r_sat = sat_positions[sv_id]
            v_sat = sat_velocities[sv_id]
            
            # Геометрическая дальность
            delta_r = r_sat - r_rec
            rho = np.linalg.norm(delta_r)
            line_of_sight = delta_r / rho
            
            # Относительная скорость (проекция на линию визирования)
            delta_v = v_sat - v_rec
            doppler_true = np.dot(delta_v, line_of_sight)
            
            # Добавляем шумы и часы
            noise_pr = np.random.normal(0, cfg.raw_pr_sigma)
            noise_dop = np.random.normal(0, cfg.raw_doppler_sigma)
            
            pseudorange = rho + c_b + noise_pr
            doppler = doppler_true + c_d + noise_dop
            
            raw_obs_list.append({
                't': t,
                'sv_id': sv_id + 1,
                'sat_X': r_sat[0], 'sat_Y': r_sat[1], 'sat_Z': r_sat[2],
                'sat_vX': v_sat[0], 'sat_vY': v_sat[1], 'sat_vZ': v_sat[2],
                'pseudorange': pseudorange,
                'doppler': doppler
            })
            
    df_raw = pd.DataFrame(raw_obs_list)
    return df_raw


# ==========================================
# ГНСС (КООРДИНАТЫ) И ЛР1
# ==========================================
def process_gnss(df_imu: pd.DataFrame, cfg: TrajectoryConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Прореживает до ГНСС частоты, переводит координаты в WGS84 и добавляет шум."""
    # Прореживание
    step = max(1, int(round(cfg.dt_gnss / cfg.dt_imu)))
    df_gnss_true = df_imu.iloc[::step].copy().reset_index(drop=True)
    
    # 1. Формируем "чистую" ГНСС траекторию (истина)
    # Используем координаты антенны, т.к. именно она принимает сигнал
    df_clean = pd.DataFrame()
    df_clean['t'] = df_gnss_true['t']
    df_clean['E'] = df_gnss_true['E_ant']
    df_clean['N'] = df_gnss_true['N_ant']
    df_clean['U'] = df_gnss_true['U_ant']
    df_clean['vE'] = df_gnss_true['vE']
    df_clean['vN'] = df_gnss_true['vN']
    df_clean['vU'] = df_gnss_true['vU']

    # Конвертация истинной позиции ENU -> ECEF -> LLH
    ecef_coords = [enu_to_ecef(r['E'], r['N'], r['U'], cfg.ref_lat, cfg.ref_lon, cfg.ref_alt) 
                   for _, r in df_clean.iterrows()]
    df_clean['X_ecef'], df_clean['Y_ecef'], df_clean['Z_ecef'] = zip(*ecef_coords)
    
    llh_coords = [ecef_to_llh(x, y, z) for x, y, z in ecef_coords]
    df_clean['lat'], df_clean['lon'], df_clean['alt'] = zip(*llh_coords)
    
    # 2. Формируем "зашумлённую" ГНСС траекторию (измерения)
    df_noisy = df_clean.copy()
    np.random.seed(cfg.seed_gnss)
    
    noise_e = np.random.normal(0, cfg.gnss_pos_sigma, len(df_noisy))
    noise_n = np.random.normal(0, cfg.gnss_pos_sigma, len(df_noisy))
    noise_u = np.random.normal(0, cfg.gnss_pos_sigma, len(df_noisy))
    
    df_noisy['E'] += noise_e
    df_noisy['N'] += noise_n
    df_noisy['U'] += noise_u
    
    # Пересчёт зашумленных ECEF/LLH (т.к. мы шумим позицию ENU)
    noisy_ecef = [enu_to_ecef(r['E'], r['N'], r['U'], cfg.ref_lat, cfg.ref_lon, cfg.ref_alt) 
                  for _, r in df_noisy.iterrows()]
    df_noisy['X_ecef'], df_noisy['Y_ecef'], df_noisy['Z_ecef'] = zip(*noisy_ecef)
    
    noisy_llh = [ecef_to_llh(x, y, z) for x, y, z in noisy_ecef]
    df_noisy['lat'], df_noisy['lon'], df_noisy['alt'] = zip(*noisy_llh)
    
    return df_clean, df_noisy
