import numpy as np
from quaternion import quat_mul, quat_from_rotvec, quat_to_dcm, quat_to_euler, euler_to_quat

class INSMechanization:
    """
    Алгоритм механизации БИНС в локальной системе координат (ENU).
    """
    def __init__(self, dt: float, init_pos: np.ndarray, init_vel: np.ndarray, init_euler: np.ndarray, g: float = 9.81):
        self.dt = dt
        self.g_n = np.array([0.0, 0.0, -g])
        
        self.pos = np.asarray(init_pos, dtype=float).copy()
        self.vel = np.asarray(init_vel, dtype=float).copy()
        
        # Кватернион задает переход от Body к Nav
        self.q = euler_to_quat(init_euler[0], init_euler[1], init_euler[2])
        self.C_bn = quat_to_dcm(self.q)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

    def step(self, fb: np.ndarray, omegab: np.ndarray):
        """Шаг интегрирования БИНС."""
        # 1. Компенсация погрешностей перед использованием сырых измерений
        fb_corr = fb - self.ba
        omegab_corr = omegab - self.bg
        
        # 2. Интегрирование ориентации (Body -> Nav) с учетом скорректированной угл. скорости
        dq = quat_from_rotvec(omegab_corr * self.dt)
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        self.Cbn = quat_to_dcm(self.q)
        
        # 3. Интегрирование скорости и координат с учетом скорректированного ускорения
        fn = self.Cbn @ fb_corr
        an = fn + self.g_n
        self.vel += an * self.dt
        self.pos += self.vel * self.dt
        
    def get_euler(self):
        return quat_to_euler(self.q)
        
    def correct(self, err_state: np.ndarray):
        # Коррекция координат и скорости
        self.pos -= err_state[0:3]
        self.vel -= err_state[3:6]
        
        # Коррекция ориентации
        delta_euler = err_state[6:9]
        dq_err = quat_from_rotvec(delta_euler)
        dq_err_inv = np.array([dq_err[0], -dq_err[1], -dq_err[2], -dq_err[3]])
        self.q = quat_mul(dq_err_inv, self.q)
        self.q /= np.linalg.norm(self.q)
        self.Cbn = quat_to_dcm(self.q)
        
        # Накопление оценок погрешностей (ba и bg)
        self.ba += err_state[9:12]
        self.bg += err_state[12:15]
