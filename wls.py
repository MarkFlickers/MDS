import numpy as np
from dataclasses import dataclass

@dataclass
class WlsConfig:
    max_iter: int = 10
    tol: float = 1e-3
    sigma_pr: float = 3.0
    
def wls_epoch(df_epoch, x_prev, config: WlsConfig):
    x = x_prev.copy()
    m = len(df_epoch)
    
    sat_pos = df_epoch[['sat_X', 'sat_Y', 'sat_Z']].values
    pr = df_epoch['pseudorange'].values
    
    for _ in range(config.max_iter):
        rx = x[0:3]
        cb = x[3]
        
        # Вычисление вектора дальностей и невязок
        dr = sat_pos - rx
        rho = np.linalg.norm(dr, axis=1)
        z_pred = rho + cb
        dz = pr - z_pred
        
        # Матрица производных (направляющие косинусы)
        H = np.zeros((m, 4))
        H[:, 0:3] = -dr / rho.reshape(-1, 1)
        H[:, 3] = 1.0
        
        # Весовая матрица (предполагаем одинаковую дисперсию)
        W = np.eye(m) / (config.sigma_pr ** 2)
        
        # Решение МНК: (H^T W H)^(-1) H^T W dz
        dx = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ dz
        x += dx
        
        if np.linalg.norm(dx) < config.tol:
            break
            
    P = np.linalg.inv(H.T @ W @ H)
    return x, P
