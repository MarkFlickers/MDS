import numpy as np
from gnss import c_light

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
    def __init__(self, dt: float, Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        super().__init__(x0, P0)
        self.dt = dt
        
        # Матрица перехода F (6x6) (Модель постоянной скорости - CV)
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        
        # Матрица наблюдений H (3x6) - мы измеряем только первые 3 компоненты (координаты)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        
        self.R = R
        self.Q = Q

    def step(self, z: np.ndarray):
        self.predict(self.F, self.Q)
        self.update(z, self.H, self.R)
        return self.x.copy(), self.P.copy()

class ExtendedKalmanFilter(BaseKalmanFilter):
    """
    Расширенный фильтр Калмана (EKF) для ЛР2.
    Работает в системе координат ECEF.
    Вектор состояния (8x1): [X, Y, Z, vX, vY, vZ, cb, cd]^T
    """
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
        F[6, 7] = dt # Интегрирование дрейфа часов (cd) в смещение (cb)
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
        """Линейный апдейт по координатам от МНК (WLS)."""
        z = np.asarray(x_wls, dtype=float).reshape(4,)
        H = np.zeros((4, 8), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 6] = 1.0
        R = np.asarray(P_wls, dtype=float).copy() + np.eye(4)*1e-6
        self.update(z, H, R)

    def update_doppler(self, doppler: np.ndarray, sat_pos: np.ndarray, sat_vel: np.ndarray):
        """Нелинейный (EKF) апдейт по сырым доплеровским измерениям."""
        m = doppler.shape[0]
        if m == 0: return

        rx, v = self.x[0:3].flatten(), self.x[3:6].flatten()
        cd = self.x[7].item()

        dr = sat_pos - rx
        rho = np.linalg.norm(dr, axis=1)
        los = dr / rho.reshape(-1, 1) # Line of sight векторы
        rel_vel = sat_vel - v

        # Предсказанный доплер: h(x)
        dop_pred = np.sum(los * rel_vel, axis=1) + cd
        z = doppler
        y = z - dop_pred

        proj = np.sum(los * rel_vel, axis=1)[:, None]
        # Матрица Якоби H
        H = np.zeros((m, 8), dtype=float)
        H[:, 0:3] = (proj * los - rel_vel) / rho[:, None]
        H[:, 3:6] = -los
        H[:, 7] = 1.0
        R = np.eye(m, dtype=float) * (self.sigma_doppler ** 2)

        # Обновление
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

class InsErrorStateKalmanFilter(BaseKalmanFilter):
    """
    15-состояний фильтр Калмана в пространстве ошибок для ЛР3.
    Вектор состояния:
      [dr, dv, dtheta, dba, dbg]^T
    где
      dr     - ошибки положения ENU,
      dv     - ошибки скорости ENU,
      dtheta - малые ошибки ориентации,
      dba    - ошибки смещений акселерометров,
      dbg    - ошибки дрейфов гироскопов.
    """
    def __init__(
        self,
        sigma_accel: float,
        sigma_gyro: float,
        sigma_accel_bias: float,
        sigma_gyro_bias: float,
        P0: np.ndarray,
    ):
        super().__init__(x0=np.zeros(15, dtype=float), P0=P0)
        self.sigma_accel = float(sigma_accel)
        self.sigma_gyro = float(sigma_gyro)
        self.sigma_accel_bias = float(sigma_accel_bias)
        self.sigma_gyro_bias = float(sigma_gyro_bias)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        vx, vy, vz = np.asarray(v, dtype=float)
        return np.array([
            [0.0, -vz,  vy],
            [vz,  0.0, -vx],
            [-vy, vx,  0.0],
        ], dtype=float)

    def _build_F(self, C_nb: np.ndarray, f_b_corr: np.ndarray, w_b_corr: np.ndarray, dt: float) -> np.ndarray:
        F_c = np.zeros((15, 15), dtype=float)
        F_c[0:3, 3:6] = np.eye(3)
        F_c[3:6, 6:9] = -C_nb @ self._skew(f_b_corr)
        F_c[3:6, 9:12] = -C_nb
        F_c[6:9, 6:9] = -self._skew(w_b_corr)
        F_c[6:9, 12:15] = -np.eye(3)
        return np.eye(15, dtype=float) + F_c * dt

    def _build_Q(self, C_nb: np.ndarray, dt: float) -> np.ndarray:
        G = np.zeros((15, 12), dtype=float)
        G[3:6, 0:3] = C_nb
        G[6:9, 3:6] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        q_diag = np.concatenate([
            np.full(3, self.sigma_accel ** 2),
            np.full(3, self.sigma_gyro ** 2),
            np.full(3, self.sigma_accel_bias ** 2),
            np.full(3, self.sigma_gyro_bias ** 2),
        ])
        Q_c = np.diag(q_diag)
        return G @ Q_c @ G.T * dt

    def predict_ins(self, C_nb: np.ndarray, f_b_corr: np.ndarray, w_b_corr: np.ndarray, dt: float):
        F = self._build_F(C_nb=C_nb, f_b_corr=f_b_corr, w_b_corr=w_b_corr, dt=dt)
        Q = self._build_Q(C_nb=C_nb, dt=dt)
        self.predict(F, Q)

    def update_position(self, z: np.ndarray, R: np.ndarray, C_nb: np.ndarray | None = None, lever_arm: np.ndarray | None = None):
        H = np.zeros((3, 15), dtype=float)
        H[:, 0:3] = np.eye(3)

        if lever_arm is not None and C_nb is not None:
            H[:, 6:9] = -C_nb @ self._skew(lever_arm)

        self.update(np.asarray(z, dtype=float).reshape(3,), H, np.asarray(R, dtype=float))
        return H

    def reset_error_state(self):
        self.x[:] = 0.0
        self.P = 0.5 * (self.P + self.P.T)
