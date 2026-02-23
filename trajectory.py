import numpy as np
import pandas as pd
from typing import List
from configuration import TrajectoryConfig, Stage
from quaternion import euler_to_quat, quat_from_rotvec, quat_to_dcm, quat_to_euler, quat_mul

# ==========================================
# ИНТЕГРАТОР ТРАЕКТОРИИ (ИДЕАЛЬНОЙ)
# ==========================================
def generate_trajectory(cfg: TrajectoryConfig, stages: List[Stage]) -> pd.DataFrame:
    """Интегрирует уравнения кинематики, генерирует истинную траекторию ИМУ и антенны."""
    dt = cfg.dt_imu
    
    # Разворачиваем стадии в массивы
    a_cmd_list, w_dot_cmd_list = [], []
    for s in stages:
        n_steps = max(1, int(round(s.duration_s / dt)))
        a_cmd_list.append(np.tile(s.a_body, (n_steps, 1)))
        w_dot_cmd_list.append(np.tile(s.w_dot_body, (n_steps, 1)))
        
    a_b_arr = np.vstack(a_cmd_list)
    w_dot_arr = np.vstack(w_dot_cmd_list)
    total_steps = len(a_b_arr)
    
    # Предвыделение памяти
    t_arr = np.arange(total_steps) * dt
    r_imu_enu = np.zeros((total_steps, 3))
    v_imu_enu = np.zeros((total_steps, 3))
    r_ant_enu = np.zeros((total_steps, 3))
    
    q_nb_hist = np.zeros((total_steps, 4))
    euler_hist = np.zeros((total_steps, 3))
    w_b_hist = np.zeros((total_steps, 3))
    f_b_hist = np.zeros((total_steps, 3))
    
    # Начальные состояния
    r_n = cfg.init_pos_enu.copy()
    v_b = cfg.init_vel_body.copy()
    w_b = np.array([0.0, 0.0, 0.0])
    q_nb = euler_to_quat(*cfg.init_euler)
    
    # Вектор гравитации в ENU
    g_n = np.array([0.0, 0.0, -cfg.g])
    
    for k in range(total_steps):
        a_b = a_b_arr[k]
        w_dot = w_dot_arr[k]
        
        # 1. Угловая кинематика
        w_b = w_b + w_dot * dt
        dq = quat_from_rotvec(w_b * dt)
        q_nb = quat_mul(q_nb, dq)
        q_nb = q_nb / np.linalg.norm(q_nb) # Нормализация
        C_nb = quat_to_dcm(q_nb)
        
        # 2. Линейная кинематика
        v_b = v_b + a_b * dt
        v_n = C_nb @ v_b
        r_n = r_n + v_n * dt
        
        # 3. Удельная сила (измерения акселерометра ИМУ)
        # f^b = a^b - C_nb^T * g^n
        f_b = a_b - C_nb.T @ g_n

        # 4. Позиция антенны ГНСС
        r_ant = r_n + C_nb @ cfg.lever_arm
        
        # Сохранение истории
        r_imu_enu[k] = r_n
        v_imu_enu[k] = v_n
        r_ant_enu[k] = r_ant
        q_nb_hist[k] = q_nb
        euler_hist[k] = quat_to_euler(q_nb)
        w_b_hist[k] = w_b
        f_b_hist[k] = f_b
        
    # Упаковка в DataFrame
    df = pd.DataFrame({
        't': t_arr,
        'E_imu': r_imu_enu[:, 0], 'N_imu': r_imu_enu[:, 1], 'U_imu': r_imu_enu[:, 2],
        'vE': v_imu_enu[:, 0], 'vN': v_imu_enu[:, 1], 'vU': v_imu_enu[:, 2],
        'q_w': q_nb_hist[:, 0], 'q_x': q_nb_hist[:, 1], 'q_y': q_nb_hist[:, 2], 'q_z': q_nb_hist[:, 3],
        'roll': euler_hist[:, 0], 'pitch': euler_hist[:, 1], 'yaw': euler_hist[:, 2],
        'f_x': f_b_hist[:, 0], 'f_y': f_b_hist[:, 1], 'f_z': f_b_hist[:, 2],
        'w_x': w_b_hist[:, 0], 'w_y': w_b_hist[:, 1], 'w_z': w_b_hist[:, 2],
        'E_ant': r_ant_enu[:, 0], 'N_ant': r_ant_enu[:, 1], 'U_ant': r_ant_enu[:, 2],
    })
    return df

# ==========================================
# МОДЕЛИРОВАНИЕ ОШИБОК ИМУ
# ==========================================
def simulate_imu_errors(df_imu_clean: pd.DataFrame, cfg: TrajectoryConfig) -> pd.DataFrame:
    """Добавляет смещение и белый шум к идеальным данным ИМУ."""
    np.random.seed(cfg.seed_imu)
    df_noisy = df_imu_clean.copy()
    n_samples = len(df_noisy)
    dt = cfg.dt_imu
    
    # Шум (White Noise). σ = RandomWalk / sqrt(dt)
    accel_noise_std = cfg.accel_vrw / np.sqrt(dt)
    gyro_noise_std = cfg.gyro_arw / np.sqrt(dt)
    
    noise_fx = np.random.normal(0, accel_noise_std, n_samples)
    noise_fy = np.random.normal(0, accel_noise_std, n_samples)
    noise_fz = np.random.normal(0, accel_noise_std, n_samples)
    
    noise_wx = np.random.normal(0, gyro_noise_std, n_samples)
    noise_wy = np.random.normal(0, gyro_noise_std, n_samples)
    noise_wz = np.random.normal(0, gyro_noise_std, n_samples)
    
    # Добавляем Bias и Шум
    df_noisy['f_x'] += cfg.accel_bias[0] + noise_fx
    df_noisy['f_y'] += cfg.accel_bias[1] + noise_fy
    df_noisy['f_z'] += cfg.accel_bias[2] + noise_fz
    
    df_noisy['w_x'] += cfg.gyro_bias[0] + noise_wx
    df_noisy['w_y'] += cfg.gyro_bias[1] + noise_wy
    df_noisy['w_z'] += cfg.gyro_bias[2] + noise_wz
    
    return df_noisy