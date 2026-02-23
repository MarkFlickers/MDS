import numpy as np

class BaseKalmanFilter:
    """Базовый класс фильтра Калмана."""
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = x0.astype(float)
        self.P = P0.astype(float)
        self.I = np.eye(len(x0))

    def predict(self, F: np.ndarray, Q: np.ndarray):
        """Этап экстраполяции (прогноза)."""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Этап коррекции (обновления)."""
        y = z - H @ self.x              # Невязка (инновация)
        S = H @ self.P @ H.T + R        # Ковариация невязки
        K = self.P @ H.T @ np.linalg.inv(S) # Матрица усиления Калмана

        self.x = self.x + K @ y
        
        # Обновление матрицы P в форме Джозефа (численно более устойчиво)
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
