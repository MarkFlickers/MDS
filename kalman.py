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
        """
        Этап коррекции (обновления).
        Joseph form:
          P = (I-KH)P(I-KH)^T + K R K^T
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        y = z - (H @ self.x)              # innovation
        S = H @ self.P @ H.T + R          # innovation covariance

        # solve instead of inverse
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        self.x = self.x + K @ y

        IKH = self.I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

class LinearKalmanFilter(BaseKalmanFilter):
    """
    Линейный фильтр Калмана для ЛР1 (модель постоянной скорости - Constant Velocity).
    Вектор состояния (6x1): [E, N, U, vE, vN, vU]^T
    Вектор измерений (3x1): [E_gnss, N_gnss, U_gnss]^T
    """
    def __init__(self, dt: float, sigma_a: float, sigma_gnss: float, x0: np.ndarray, P0: np.ndarray):
        super().__init__(x0, P0)
        self.dt = dt
        
        # Матрица перехода состояния F (6x6)
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

    def step(self, z: np.ndarray, F=None, Q=None, H=None, R=None):
        """Один полный шаг: предсказание + коррекция."""
        self.predict(F if F is not None else self.F, Q if Q is not None else self.Q)
        self.update(z, H if H is not None else self.H, R if R is not None else self.R)
        return self.x.copy(), self.P.copy()

class GNSSKalmanFilter(BaseKalmanFilter):
    """
    ЛР2: ФК на 8 состояниях (ECEF):
      x = [X, Y, Z, vX, vY, vZ, cb, cd]^T
        cb = c*dt     (м)
        cd = c*dt_dot (м/с)

    Обновления:
      1) по WLS: z = [X,Y,Z,cb] (линейно)
      2) по доплеру: z_i = los_i^T (v_sat_i - v_rec) + cd   (м/с)
    """
    def __init__(
        self,
        dt: float,
        sigma_a: float,
        sigma_cb: float,
        sigma_cd: float,
        sigma_doppler: float,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        super().__init__(x0, P0)
        self.dt = float(dt)

        self.sigma_a = float(sigma_a)
        self.sigma_cb = float(sigma_cb)
        self.sigma_cd = float(sigma_cd)
        self.sigma_doppler = float(sigma_doppler)

        self.F = self._build_F(self.dt)
        self.Q = self._build_Q(self.dt)

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        F = np.eye(8, dtype=float)
        # позиция <- скорость
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # часы: cb <- cd
        F[6, 7] = dt
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Q для модели:
          - CV по (pos, vel) с шумом ускорения sigma_a
          - RW по (cb, cd) с дисперсиями sigma_cb, sigma_cd (в дискретном виде)
        """
        Q = np.zeros((8, 8), dtype=float)

        # Блоки CV (как в ЛР1), но в ECEF
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0

        q_block = (self.sigma_a ** 2) * np.array([[dt4, dt3], [dt3, dt2]], dtype=float)

        for i in range(3):
            Q[i, i] = q_block[0, 0]
            Q[i, i + 3] = q_block[0, 1]
            Q[i + 3, i] = q_block[1, 0]
            Q[i + 3, i + 3] = q_block[1, 1]

        # Часы (простая диагональная дискретная модель)
        Q[6, 6] = (self.sigma_cb ** 2) * dt
        Q[7, 7] = (self.sigma_cd ** 2) * dt

        return Q

    def predict_step(self, dt: float | None = None):
        if dt is not None and float(dt) != self.dt:
            self.dt = float(dt)
            self.F = self._build_F(self.dt)
            self.Q = self._build_Q(self.dt)
        self.predict(self.F, self.Q)

    def update_wls(self, x_wls: np.ndarray, P_wls: np.ndarray):
        """
        Обновление по WLS решению.
        x_wls: (4,) [X,Y,Z,cb]
        P_wls: (4,4) ковариация WLS
        """
        z = np.asarray(x_wls, dtype=float).reshape(4)

        H = np.zeros((4, 8), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 6] = 1.0  # cb

        # Небольшая "подпорка", чтобы R не был вырожденным
        R = np.asarray(P_wls, dtype=float).copy()
        R = R + np.eye(4) * 1e-6

        self.update(z, H, R)

    def update_doppler(self, doppler: np.ndarray, sat_pos: np.ndarray, sat_vel: np.ndarray):
        """
        doppler: (m,) измерения (м/с), как в вашем simulate_gnss_raw()
        sat_pos: (m,3) ECEF (м)
        sat_vel: (m,3) ECEF (м/с)
        """
        doppler = np.asarray(doppler, dtype=float).reshape(-1)
        sat_pos = np.asarray(sat_pos, dtype=float)
        sat_vel = np.asarray(sat_vel, dtype=float)

        m = doppler.shape[0]
        if m == 0:
            return

        rx = self.x[0:3]
        v = self.x[3:6]
        cd = self.x[7]

        # LOS = (r_sat - r_rx)/rho
        dr = sat_pos - rx.reshape(1, 3)
        rho = np.linalg.norm(dr, axis=1)
        los = dr / rho.reshape(-1, 1)

        # Предсказание доплера:
        # dop_pred = los^T (v_sat - v) + cd
        dop_pred = np.sum(los * (sat_vel - v.reshape(1, 3)), axis=1) + cd

        z = doppler
        y = z - dop_pred

        # H: производные по [pos, vel, cb, cd]
        # Зависимость от позиции через los есть, но в базовой ЛР2 можно
        # считать геометрию фиксированной на шаге обновления (линеаризация).
        H = np.zeros((m, 8), dtype=float)
        H[:, 3:6] = -los
        H[:, 7] = 1.0

        R = np.eye(m, dtype=float) * (self.sigma_doppler ** 2)

        # Для совместимости с BaseKalmanFilter.update() передадим "псевдо-измерение":
        # update ожидает z и Hx. Мы хотим использовать инновацию y напрямую, поэтому
        # формируем z_eff = y + Hx, чтобы (z_eff - Hx) = y.
        z_eff = y + (H @ self.x)
        self.update(z_eff, H, R)