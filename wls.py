import numpy as np
from dataclasses import dataclass
from coord_conversion import ecef_to_llh, R_ecef_to_enu, ecef_to_enu
from configuration import TrajectoryConfig
from metrics import calculate_dop

LIGHT_SPEED = 299792458.
WGS_OMEGDOTE = 7.2921151467e-5	# earth rotate rate
WGS_F_GTR = -4.442807633e-10	# factor of general theory of relativity

@dataclass
class WlsConfig:
    max_iter: int = 10      # Максимальное число итераций
    tol: float = 1e-3       # Критерий остановки (м)
    sigma_pr: float = 3.0   # СКО псевдодальности в зените (м)

def week_time_rounding(time):
    if (time > 302400.0):
        time -= 604800
    if (time < -302400.0):
        time += 604800
    return time

def gps_clock_correction(df_eph):

    TimeDiff = df_eph['satellite_time'].values - df_eph['toc'].values

    # Возврат в времени в диапазон недели
    TimeDiff = [week_time_rounding(time) for time in TimeDiff]
    

    ClockAdj = df_eph['af0'].values + (df_eph['af1'].values + df_eph['af2'].values * TimeDiff) * TimeDiff
    ClockAdj *= (1 - df_eph['af1'].values);	# подстройка времени

    return ClockAdj*LIGHT_SPEED

def TropoDelay(Lat, Altitude, Elevation):
    REL_HUMI = 0.7
    t0 = 273.16 + 15.0; # средняя температура на уровне моря
        
    if (Altitude < -100.0 or Altitude > 1e4 or Elevation <= 0):
        return 0.0
    if Altitude < 0:
        Altitude = 0
        
    Pressure = 1013.25 * pow(1.0 - 2.2557E-5 * Altitude, 5.2568)
    t = t0 - 6.5e-3 * Altitude
    e = 6.108 * REL_HUMI * np.exp((17.15 * t - 4684.0) / (t - 38.45))
        
    z = np.pi / 2.0 - Elevation
    trph = 0.0022767 * Pressure / (1.0 - 0.00266 * np.cos(2.0 * Lat) - 0.00028 * Altitude / 1E3) / np.cos(z)
    trpw = 0.002277 * (1255.0 / t + 0.05) * e / np.cos(z)

    return (trph + trpw)

def relativity_correction(df_eph):
    time_shift = WGS_F_GTR * df_eph['ecc'].values * df_eph['sqrtA'].values * np.sin(df_eph['Ek'].values)
    distance_shift = time_shift * LIGHT_SPEED
    return distance_shift

def wls_epoch(df_epoch, x_prev, wls_config: WlsConfig, trajectory_config: TrajectoryConfig):
    """
    Решает навигационную задачу для одной эпохи методом Взвешенного МНК (WLS).
    Веса наблюдений зависят от угла места спутника.
    """
    x = x_prev.copy() # Вектор состояния: [X, Y, Z, c*dt] (ECEF)
    m = len(df_epoch) # Количество спутников
    
    # Координаты спутников (ECEF)
    sat_pos = df_epoch[['sat_X', 'sat_Y', 'sat_Z']].values
    # Измеренные псевдодальности
    pr = df_epoch['pseudorange'].values
    
    # Параметры эфемерида
    if 'af1' in df_epoch:
        df_ehp = df_epoch[['af0', 'af1', 'af2', 'travel_time', 'satellite_time', 'toc', 'ecc', 'sqrtA', 'Ek', 'Ek_dot']]
        clock_correction = gps_clock_correction(df_ehp)
    else:
        df_ehp = None
        clock_correction = 0
    
    iono_delay = df_epoch['iono_delay'].values if 'iono_delay' in df_epoch else np.zeros_like(pr)
    group_delay = df_epoch['gd'].values * LIGHT_SPEED if 'gd' in df_epoch else np.zeros_like(pr)
    relativity = relativity_correction(df_ehp) if df_ehp is not None else np.zeros_like(pr)

    for _ in range(wls_config.max_iter):
        rx = x[0:3] # Текущая оценка координат приемника
        cb = x[3]   # Текущая оценка смещения часов
        
        # 1. Вектор дальностей и невязок
        dr = sat_pos - rx
        rho = np.linalg.norm(dr, axis=1) # Геометрическая дальность
        # Добавляем компенсацию вращения Земли
        earth_rot_dr = sat_pos[:,0] * rx[1] - sat_pos[:,1] * rx[0]
        rho += earth_rot_dr * (WGS_OMEGDOTE / LIGHT_SPEED)

        # Переводим текущие координаты приемника в Geodetic (Lat, Lon)
        lat, lon, alt = ecef_to_llh(rx[0], rx[1], rx[2])
        # Получаем матрицу поворота из ECEF в локальную ENU в точке приемника
        # R_ecef2enu = R_ecef_to_enu(trajectory_config.ref_lat, trajectory_config.ref_lon)
        R_ecef2enu = R_ecef_to_enu(lat, lon)
        #Переводим векторы "Приемник->Спутник" (dr) в ENU
        dr_enu = (R_ecef2enu @ dr.T).T 
        # Считаем угол места: el = atan2(Up, sqrt(East^2 + North^2))
        dist_hor = np.sqrt(dr_enu[:, 0]**2 + dr_enu[:, 1]**2)
        el = np.arctan2(dr_enu[:, 2], dist_hor)
        
        tropo_delay = [TropoDelay(lat, alt, e) for e in el] if df_ehp is not None else np.zeros_like(el)

        # travel_time = (d + dtrop)/c + tgd - dts - trel + diono
        z_pred = rho + cb - clock_correction + tropo_delay + iono_delay + group_delay - relativity                # Предсказанное измерение
        dz = pr - z_pred                 # Вектор невязок (Innovation)
        
        # 2. Матрица производных H (Design Matrix)
        # H = [-LOS_x, -LOS_y, -LOS_z, 1]
        H = np.zeros((m, 4))
        H[:, 0:3] = -dr / rho.reshape(-1, 1) # Line-of-sight векторы
        H[:, 3] = 1.0
        
        # 3. Формирование весовой матрицы W (Weight Matrix)
        
        # Ограничиваем угол места снизу (1 градус), чтобы веса не улетали в бесконечность
        el = np.maximum(el, np.radians(1.0))
        
        # 3.5 Расчет весов
        # Модель дисперсии: sigma_i^2 = sigma_zenith^2 / sin(el)^2
        # Вес: w_ii = 1 / sigma_i^2 = sin(el)^2 / sigma_zenith^2
        weights = (np.sin(el) ** 2) / (wls_config.sigma_pr ** 2)
        W = np.diag(weights)
        
        # 4. Решение Взвешенного МНК: dx = (H^T W H)^(-1) H^T W dz
        # Используем pinv для устойчивости, если матрица близка к вырожденной
        try:
            HtW = H.T @ W
            HtWH = HtW @ H
            Cov = np.linalg.inv(HtWH)
            dx = Cov @ HtW @ dz
        except np.linalg.LinAlgError:
            # Если геометрия плохая (мало спутников или они на одной линии)
            return x, np.eye(4) * 1000.0 
            
        x += dx
        
        # Условие выхода
        if np.linalg.norm(dx) < wls_config.tol:
            break
            
    # Результирующая ковариационная матрица оценки (Error Covariance)
    # P = (H^T W H)^-1
    P = np.linalg.inv(H.T @ W @ H)

    dops = calculate_dop(H, lat, lon)  # Вычисляем DOP

    return x, P, dops