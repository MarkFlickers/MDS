import numpy as np
from dataclasses import dataclass
from coord_conversion import ecef_to_llh, _R_ecef_to_enu

@dataclass
class WlsConfig:
    max_iter: int = 10      # Максимальное число итераций
    tol: float = 1e-3       # Критерий остановки (м)
    sigma_pr: float = 3.0   # СКО псевдодальности в зените (м)
    
def wls_epoch(df_epoch, x_prev, config: WlsConfig):
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
    
    for _ in range(config.max_iter):
        rx = x[0:3] # Текущая оценка координат приемника
        cb = x[3]   # Текущая оценка смещения часов
        
        # 1. Вектор дальностей и невязок
        dr = sat_pos - rx
        rho = np.linalg.norm(dr, axis=1) # Геометрическая дальность
        
        z_pred = rho + cb                # Предсказанное измерение
        dz = pr - z_pred                 # Вектор невязок (Innovation)
        
        # 2. Матрица производных H (Design Matrix)
        # H = [-LOS_x, -LOS_y, -LOS_z, 1]
        H = np.zeros((m, 4))
        H[:, 0:3] = -dr / rho.reshape(-1, 1) # Line-of-sight векторы
        H[:, 3] = 1.0
        
        # 3. Формирование весовой матрицы W (Weight Matrix)
        # Нам нужно оценить угол места (Elevation) для каждого спутника
        
        # 3.1 Переводим текущие координаты приемника в Geodetic (Lat, Lon)
        lat, lon, _ = ecef_to_llh(rx[0], rx[1], rx[2])
        
        # 3.2 Получаем матрицу поворота из ECEF в локальную ENU в точке приемника
        R_ecef2enu = _R_ecef_to_enu(lat, lon)
        
        # 3.3 Переводим векторы "Приемник->Спутник" (dr) в ENU
        # dr имеет размер (N, 3), R - (3, 3). Транспонируем для умножения.
        dr_enu = (R_ecef2enu @ dr.T).T 
        
        # 3.4 Считаем угол места: el = atan2(Up, sqrt(East^2 + North^2))
        dist_hor = np.sqrt(dr_enu[:, 0]**2 + dr_enu[:, 1]**2)
        el = np.arctan2(dr_enu[:, 2], dist_hor)
        
        # Ограничиваем угол места снизу (5 градусов), чтобы веса не улетали в бесконечность
        el = np.maximum(el, np.radians(5.0))
        
        # 3.5 Расчет весов
        # Модель дисперсии: sigma_i^2 = sigma_zenith^2 / sin(el)^2
        # Вес: w_ii = 1 / sigma_i^2 = sin(el)^2 / sigma_zenith^2
        weights = (np.sin(el) ** 2) / (config.sigma_pr ** 2)
        W = np.diag(weights)
        
        # 4. Решение Взвешенного МНК: dx = (H^T W H)^(-1) H^T W dz
        # Используем pinv для устойчивости, если матрица близка к вырожденной
        try:
            HtW = H.T @ W
            Cov = np.linalg.inv(HtW @ H)
            dx = Cov @ HtW @ dz
        except np.linalg.LinAlgError:
            # Если геометрия плохая (мало спутников или они на одной линии)
            return x, np.eye(4) * 1000.0 
            
        x += dx
        
        # Условие выхода
        if np.linalg.norm(dx) < config.tol:
            break
            
    # Результирующая ковариационная матрица оценки (Error Covariance)
    # P = (H^T W H)^-1
    P = np.linalg.inv(H.T @ W @ H)
    
    return x, P
