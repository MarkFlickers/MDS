# ГНСС

import numpy as np
import pandas as pd
from typing import Tuple
from configuration import TrajectoryConfig
from coord_conversion import c_light, ecef_to_llh, enu_to_ecef, R_ecef_to_enu

# Физические константы
GM_EARTH = 3.986004418e14  # Гравитационный параметр Земли (м^3/с^2)

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ОРБИТЫ И ВИДИМОСТЬ)
# ==========================================

def _simulate_clock_errors(n_epochs: int, dt: float, cfg: TrajectoryConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Генерирует профиль дрейфа и смещения часов приемника на всю симуляцию (модель случайного блуждания)."""
    clk_drift = np.zeros(n_epochs)
    clk_bias = np.zeros(n_epochs)
    
    clk_drift[0] = cfg.clock_drift_init
    clk_bias[0] = cfg.clock_bias_init
    
    for i in range(1, n_epochs):
        # Дрейф часов гуляет (Random Walk)
        clk_drift[i] = clk_drift[i-1] + np.random.normal(0, cfg.clock_bias_noise)
        # Смещение часов - это интеграл от дрейфа
        clk_bias[i] = clk_bias[i-1] + clk_drift[i-1] * dt
        
    return clk_bias, clk_drift

def _compute_satellite_state(t: float, svid: int, total_sats: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет истинную позицию и скорость спутника в координатах ECEF для круговой орбиты.
    Генерируется структура, похожая на группировку GPS (Walker constellation).
    """
    # 1. Скалярные параметры орбиты
    v_mag = np.sqrt(GM_EARTH / radius)
    w_orb = v_mag / radius
    
    # 2. Формируем "Walker Constellation"
    num_planes = 6  # 6 орбитальных плоскостей (как у GPS)
    plane_idx = svid % num_planes
    sat_idx = svid // num_planes
    sats_per_plane = max(1, total_sats // num_planes)
    
    # Наклон орбиты (~55 градусов для GPS)
    inclination = np.radians(55.0)
    
    # Долгота восходящего узла (RAAN) - разносим плоскости равномерно
    RAAN = (2 * np.pi * plane_idx) / num_planes
    
    # Начальная фаза (истинная аномалия) на орбите
    # Разносим спутники внутри плоскости + делаем небольшой фазовый сдвиг между плоскостями
    phase_offset = plane_idx * (2 * np.pi / total_sats)
    theta0 = (2 * np.pi * sat_idx) / sats_per_plane + phase_offset
    
    # Текущая фаза (угол)
    theta = w_orb * t + theta0
    
    # 3. Позиция и вектор скорости в 2D плоскости орбиты
    r_orb = np.array([
        radius * np.cos(theta),
        radius * np.sin(theta),
        0.0
    ])
    
    v_orb = np.array([
        -v_mag * np.sin(theta),
         v_mag * np.cos(theta),
         0.0
    ])
    
    # 4. Матрицы поворота (из орбитальной плоскости в ECEF)
    # Поворот на наклон вокруг оси X
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(inclination), -np.sin(inclination)],
        [0, np.sin(inclination),  np.cos(inclination)]
    ])
    
    # Поворот RAAN вокруг оси Z
    Rz = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN),  np.cos(RAAN), 0],
        [0, 0, 1]
    ])
    
    # Итоговая матрица поворота
    R_total = Rz @ Rx
    
    # 5. Перевод векторов в ECEF
    r_ecef = R_total @ r_orb
    v_ecef = R_total @ v_orb
    
    return r_ecef, v_ecef

def _is_satellite_visible(r_sat: np.ndarray, r_rec: np.ndarray, mask_angle_deg: float = 5.0) -> bool:
    """
    Проверяет, находится ли спутник в зоне видимости приемника (над горизонтом).
    mask_angle_deg - угол отсечения (маска угла места).
    """
    # Получаем широту и долготу приемника
    lat, lon, _ = ecef_to_llh(r_rec[0], r_rec[1], r_rec[2])
    
    # Матрица поворота ECEF -> ENU
    R_ecef2enu = R_ecef_to_enu(lat, lon)
    
    # Вектор от приемника к спутнику в системе ECEF
    dr_ecef = r_sat - r_rec
    
    # Переводим вектор визирования в локальную систему ENU
    dr_enu = R_ecef2enu @ dr_ecef
    
    # Считаем угол места (Elevation angle)
    horiz_dist = np.sqrt(dr_enu[0]**2 + dr_enu[1]**2)
    el_rad = np.arctan2(dr_enu[2], horiz_dist)
    
    return el_rad >= np.radians(mask_angle_deg)


# ==========================================
# СЫРЫЕ НАБЛЮДЕНИЯ ГНСС (ГЛАВНАЯ ФУНКЦИЯ)
# ==========================================

def simulate_gnss_raw(df_gnss_clean: pd.DataFrame, cfg: TrajectoryConfig) -> pd.DataFrame:
    """
    Генерирует сырые псевдодальности и доплеровские измерения.
    Реализует круговые орбиты спутников, отсечение по углу места и добавление шумов.
    """
    np.random.seed(cfg.seed_gnss + 1) # Чтобы шум сырых измерений отличался от шума координат ЛР1
    
    # 1. Генерируем ошибки часов на весь пролет
    n_epochs = len(df_gnss_clean)
    clk_bias_arr, clk_drift_arr = _simulate_clock_errors(n_epochs, cfg.dt_gnss, cfg)
    
    raw_obs_list = []
    
    # Увеличим число спутников до 24 аппаратно, чтобы после отсечения невидимых
    # всегда оставалось достаточно спутников (4+) для решения МНК.
    # Если в конфиге указано больше, используем значение из конфига.
    total_sim_satellites = max(24, cfg.num_satellites)
    
    # 2. Итерация по эпохам (времени)
    for i, row in df_gnss_clean.iterrows():
        t = row['t']
        
        # Истинное положение и скорость антенны приемника в ECEF
        r_rec = np.array([row['X_ecef'], row['Y_ecef'], row['Z_ecef']])
        
        # Перевод скорости приемника из ENU в ECEF
        # (В df_gnss_clean скорость vE, vN, vU хранится в ENU относительно локального референса)
        lat, lon = cfg.ref_lat, cfg.ref_lon
        R_enu2ecef = np.array([
            [-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
            [ np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
            [ 0,            np.cos(lat),             np.sin(lat)            ]
        ])
        v_rec_enu = np.array([row['vE'], row['vN'], row['vU']])
        v_rec_ecef = R_enu2ecef @ v_rec_enu
    
        # Ошибка часов в метрах и метрах в секунду
        cb = clk_bias_arr[i] * c_light
        cd = clk_drift_arr[i] * c_light
        
        # 3. Итерация по всем спутникам группировки
        visible_sats_count = 0
        for sv_id in range(total_sim_satellites):
            
            # Кинематика спутника
            r_sat, v_sat = _compute_satellite_state(t, sv_id, total_sim_satellites, cfg.satellite_radius)
            
            # Проверка видимости (Угол места > 5 градусов)
            if not _is_satellite_visible(r_sat, r_rec, mask_angle_deg=5.0):
                continue # Спутник под горизонтом, пропускаем
                
            visible_sats_count += 1
            
            # Вектор от приемника к спутнику (Line of Sight)
            delta_r = r_sat - r_rec
            rho = np.linalg.norm(delta_r)
            line_of_sight = delta_r / rho
            
            # Истинный Доплер (проекция относительной скорости на линию визирования)
            delta_v = v_sat - v_rec_ecef
            doppler_true = np.dot(delta_v, line_of_sight)
            
            # Добавляем Гауссовский шум
            noise_pr = np.random.normal(0, cfg.raw_pr_sigma)
            noise_dop = np.random.normal(0, cfg.raw_doppler_sigma)
            
            # Итоговые измерения
            pseudorange = rho + cb + noise_pr
            doppler = doppler_true + cd + noise_dop
            
            # Сохраняем в датасет
            raw_obs_list.append({
                't': t,
                'sv_id': sv_id + 1,
                'sat_X': r_sat[0], 'sat_Y': r_sat[1], 'sat_Z': r_sat[2],
                'sat_vX': v_sat[0], 'sat_vY': v_sat[1], 'sat_vZ': v_sat[2],
                'pseudorange': pseudorange,
                'doppler': doppler
            })
            
        # Защита от потери решения (если вдруг маневр или плохая геометрия отсекли все спутники)
        if visible_sats_count < 4:
            print(f"[Warning] Эпоха {t}s: Видимо только {visible_sats_count} спутников! (Нужно >=4)")
            
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
