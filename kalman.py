import numpy as np

class BaseKalmanFilter:
    """
    Базовый линейный фильтр Калмана:
      x_{k+1} = F x_k + w
      z_k     = H x_k + v
    """
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.I = np.eye(self.x.shape[0], dtype=float)

    def predict(self, F: np.ndarray, Q: np.ndarray):
        """Этап экстраполяции (прогноза)."""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        y = z - H @ self.x                     # Вектор невязки (Инновация)
        S = H @ self.P @ H.T + R               # Ковариация невязки
        K = self.P @ H.T @ np.linalg.inv(S)    # Усиление Калмана
        self.x = self.x + K @ y                # Обновление состояния
        self.P = (self.I - K @ H) @ self.P     # Обновление ковариации

class LinearKalmanFilter(BaseKalmanFilter):
    """
    Линейный фильтр Калмана для ЛР1 (модель постоянной скорости - Constant Velocity).
    Вектор состояния (6x1): [E, N, U, vE, vN, vU]^T
    Вектор измерений (3x1): [E_gnss, N_gnss, U_gnss]^T
    """
    def __init__(self, dt: float, sigma_a: float, sigma_gnss: float, x0: np.ndarray, P0: np.ndarray):
        super().__init__(x0, P0)
        self.dt = dt
        
        # Матрица перехода F (6x6) (Модель постоянной скорости - CV)
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        
        # Матрица наблюдений H (3x6) - мы измеряем только первые 3 компоненты (координаты)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        
        # Матрица ковариации шума измерений R (3x3)
        self.R = np.eye(3) * (sigma_gnss ** 2)
        
        # Матрица ковариации шума процесса Q (6x6) для дискретной модели Continuous White Noise Acceleration
        dt2 = dt**2
        dt3 = dt**3 / 2.0
        dt4 = dt**4 / 4.0
        
        q_block = np.array([
            [dt4, dt3],
            [dt3, dt2]
        ]) * (sigma_a ** 2)
        
        self.Q = np.zeros((6, 6))
        # Разносим блоки для осей E, N, U
        for i in range(3):
            self.Q[i, i] = q_block[0, 0]          # pos-pos
            self.Q[i, i+3] = q_block[0, 1]        # pos-vel
            self.Q[i+3, i] = q_block[1, 0]        # vel-pos
            self.Q[i+3, i+3] = q_block[1, 1]      # vel-vel

    def step(self, z: np.ndarray):
        self.predict(self.F, self.Q)
        self.update(z, self.H, self.R)
        return self.x.copy(), self.P.copy()

class GNSSKalmanFilter(BaseKalmanFilter):
    def __init__(self, dt: float, sigma_a: float, sigma_cb: float, sigma_cd: float, sigma_doppler: float, x0: np.ndarray, P0: np.ndarray):
        super().__init__(x0, P0)
        self.dt = dt
        self.sigma_a = sigma_a
        self.sigma_cb = sigma_cb
        self.sigma_cd = sigma_cd
        self.sigma_doppler = sigma_doppler
        
        self.F = self._build_F(dt)
        self.Q = self._build_Q(dt)

    def _build_F(self, dt: float) -> np.ndarray:
        F = np.eye(8, dtype=float)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        F[6, 7] = dt  # Интегрирование дрейфа часов (cd) в смещение (cb)
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        Q = np.zeros((8, 8), dtype=float)
        q_block = np.array([
            [(dt**4)/4, (dt**3)/2],
            [(dt**3)/2, dt**2]
        ]) * (self.sigma_a ** 2)
        
        for i in range(3):
            Q[i, i] = q_block[0, 0]
            Q[i, i+3] = q_block[0, 1]
            Q[i+3, i] = q_block[1, 0]
            Q[i+3, i+3] = q_block[1, 1]
            
        Q[6, 6] = (self.sigma_cb ** 2) * dt
        Q[7, 7] = (self.sigma_cd ** 2) * dt
        return Q

    def predict_step(self):
        self.predict(self.F, self.Q)

    def update_wls(self, x_wls: np.ndarray, P_wls: np.ndarray):
        z = np.asarray(x_wls, dtype=float).reshape(4, 1)
        H = np.zeros((4, 8), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 6] = 1.0
        R = np.asarray(P_wls, dtype=float).copy() + np.eye(4)*1e-6
        self.update(z, H, R)

    def update_doppler(self, doppler: np.ndarray, sat_pos: np.ndarray, sat_vel: np.ndarray):
        m = doppler.shape[0]
        if m == 0: return
            
        rx, v = self.x[0:3].flatten(), self.x[3:6].flatten()
        cd = self.x[7].item()
        
        dr = sat_pos - rx
        rho = np.linalg.norm(dr, axis=1)
        los = dr / rho.reshape(-1, 1) # Line of sight векторы
        
        dop_pred = np.sum(los * (sat_vel - v), axis=1) + cd
        z = doppler.reshape(-1, 1)
        y = z - dop_pred.reshape(-1, 1)
        
        H = np.zeros((m, 8), dtype=float)
        H[:, 3:6] = -los
        H[:, 7] = 1.0
        R = np.eye(m, dtype=float) * (self.sigma_doppler ** 2)
        
        # Стандартное обновление для доплера без повторного вызова базового update для удобства формирования y
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P
