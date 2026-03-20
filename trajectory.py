import numpy as np
import pandas as pd
from typing import List
from configuration import TrajectoryConfig, Stage
from quaternion import euler_to_quat, quat_from_rotvec, quat_to_dcm, quat_to_euler, quat_mul


# ==========================================
# ЭТАЛОННАЯ ТРАЕКТОРИЯ ПО СЦЕНАРИЮ
# ==========================================

def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError('Zero-norm quaternion')
    return q / n


def _expand_stage_commands(cfg: TrajectoryConfig, stages: List[Stage]) -> tuple[np.ndarray, np.ndarray]:
    dt = cfg.dt_imu
    a_cmd_list, w_dot_cmd_list = [], []
    for s in stages:
        n_steps = max(1, int(round(s.duration_s / dt)))
        a_cmd_list.append(np.tile(np.asarray(s.a_body, dtype=float), (n_steps, 1)))
        w_dot_cmd_list.append(np.tile(np.asarray(s.w_dot_body, dtype=float), (n_steps, 1)))
    if not a_cmd_list:
        raise ValueError('stages must not be empty')
    return np.vstack(a_cmd_list), np.vstack(w_dot_cmd_list)


def generate_reference_trajectory(cfg: TrajectoryConfig, stages: List[Stage]) -> pd.DataFrame:
    """Генерирует эталонную траекторию по сценарию манёвров.

    Принятая семантика сценария:
    - a_body задаёт производную скорости в ССК, то есть dv_b/dt;
    - w_dot_body задаёт производную угловой скорости в ССК, то есть dw_b/dt.
    """
    dt = cfg.dt_imu
    a_b_cmd, w_dot_cmd = _expand_stage_commands(cfg, stages)
    n_steps = a_b_cmd.shape[0]
    n_samples = n_steps + 1

    t_arr = np.arange(n_samples, dtype=float) * dt
    r_imu_enu = np.zeros((n_samples, 3))
    v_imu_enu = np.zeros((n_samples, 3))
    v_body_hist = np.zeros((n_samples, 3))
    r_ant_enu = np.zeros((n_samples, 3))
    q_nb_hist = np.zeros((n_samples, 4))
    euler_hist = np.zeros((n_samples, 3))
    w_b_hist = np.zeros((n_samples, 3))
    a_nav_hist = np.zeros((n_samples, 3))
    a_body_cmd_hist = np.zeros((n_samples, 3))
    a_body_total_hist = np.zeros((n_samples, 3))
    w_dot_cmd_hist = np.zeros((n_samples, 3))

    r_n = np.asarray(cfg.init_pos_enu, dtype=float).copy()
    v_b = np.asarray(cfg.init_vel_body, dtype=float).copy()
    q_nb = _normalize_quat(euler_to_quat(*cfg.init_euler))
    w_b = np.zeros(3, dtype=float)

    C_nb = quat_to_dcm(q_nb)
    v_n = C_nb @ v_b

    def save_state(k: int, a_n: np.ndarray, a_b_cmd_k: np.ndarray, a_b_total_k: np.ndarray, w_dot_cmd_k: np.ndarray):
        C_nb_local = quat_to_dcm(q_nb)
        r_ant = r_n + C_nb_local @ cfg.lever_arm
        r_imu_enu[k] = r_n
        v_imu_enu[k] = v_n
        v_body_hist[k] = v_b
        r_ant_enu[k] = r_ant
        q_nb_hist[k] = q_nb
        euler_hist[k] = quat_to_euler(q_nb)
        w_b_hist[k] = w_b
        a_nav_hist[k] = a_n
        a_body_cmd_hist[k] = a_b_cmd_k
        a_body_total_hist[k] = a_b_total_k
        w_dot_cmd_hist[k] = w_dot_cmd_k

    save_state(0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

    for k in range(1, n_samples):
        a_b_k = a_b_cmd[k - 1]
        w_dot_k = w_dot_cmd[k - 1]

        v_b_prev = v_b.copy()
        v_n_prev = v_n.copy()

        w_b = w_b + w_dot_k * dt
        q_nb = _normalize_quat(quat_mul(q_nb, quat_from_rotvec(w_b * dt)))
        C_nb = quat_to_dcm(q_nb)

        v_b = v_b + a_b_k * dt
        v_n = C_nb @ v_b
        a_n = (v_n - v_n_prev) / dt
        r_n = r_n + v_n * dt

        a_b_total = a_b_k + np.cross(w_b, v_b_prev)
        save_state(k, a_n, a_b_k, a_b_total, w_dot_k)

    df = pd.DataFrame({
        't': t_arr,
        'E_imu': r_imu_enu[:, 0], 'N_imu': r_imu_enu[:, 1], 'U_imu': r_imu_enu[:, 2],
        'vE': v_imu_enu[:, 0], 'vN': v_imu_enu[:, 1], 'vU': v_imu_enu[:, 2],
        'vbx': v_body_hist[:, 0], 'vby': v_body_hist[:, 1], 'vbz': v_body_hist[:, 2],
        'q_w': q_nb_hist[:, 0], 'q_x': q_nb_hist[:, 1], 'q_y': q_nb_hist[:, 2], 'q_z': q_nb_hist[:, 3],
        'roll': euler_hist[:, 0], 'pitch': euler_hist[:, 1], 'yaw': euler_hist[:, 2],
        'w_x': w_b_hist[:, 0], 'w_y': w_b_hist[:, 1], 'w_z': w_b_hist[:, 2],
        'aE': a_nav_hist[:, 0], 'aN': a_nav_hist[:, 1], 'aU': a_nav_hist[:, 2],
        'a_cmd_x': a_body_cmd_hist[:, 0], 'a_cmd_y': a_body_cmd_hist[:, 1], 'a_cmd_z': a_body_cmd_hist[:, 2],
        'a_body_total_x': a_body_total_hist[:, 0], 'a_body_total_y': a_body_total_hist[:, 1], 'a_body_total_z': a_body_total_hist[:, 2],
        'w_dot_cmd_x': w_dot_cmd_hist[:, 0], 'w_dot_cmd_y': w_dot_cmd_hist[:, 1], 'w_dot_cmd_z': w_dot_cmd_hist[:, 2],
        'E_ant': r_ant_enu[:, 0], 'N_ant': r_ant_enu[:, 1], 'U_ant': r_ant_enu[:, 2],
    })
    return df


# ==========================================
# ИДЕАЛЬНЫЕ ИМУ-ДАННЫЕ ПО ЭТАЛОНУ
# ==========================================

def generate_imu_from_reference(df_ref: pd.DataFrame, cfg: TrajectoryConfig) -> pd.DataFrame:
    """Формирует идеальные показания ИМУ из эталонной траектории.

    Важно: f_b вычисляется из дискретного приращения навигационной скорости,
    а не напрямую из a_body команды. Это автоматически учитывает вклад вращения
    (в том числе центростремительную составляющую) и делает данные согласованными
    с strapdown-механизацией, которая затем интегрирует f_b и w_b.
    """
    g_n = np.array([0.0, 0.0, -cfg.g], dtype=float)
    df = df_ref.copy()

    f_b_hist = np.zeros((len(df), 3))
    for k, row in enumerate(df.itertuples(index=False)):
        q_nb = np.array([row.q_w, row.q_x, row.q_y, row.q_z], dtype=float)
        C_nb = quat_to_dcm(q_nb)
        a_n = np.array([row.aE, row.aN, row.aU], dtype=float)
        f_b_hist[k] = C_nb.T @ (a_n - g_n)

    df['f_x'] = f_b_hist[:, 0]
    df['f_y'] = f_b_hist[:, 1]
    df['f_z'] = f_b_hist[:, 2]
    return df


# ==========================================
# ОСНОВНОЙ ГЕНЕРАТОР
# ==========================================

def generate_trajectory(cfg: TrajectoryConfig, stages: List[Stage]) -> pd.DataFrame:
    """Строит эталонную траекторию по сценарию и формирует по ней идеальные ИМУ-данные."""
    df_ref = generate_reference_trajectory(cfg, stages)
    df_imu_clean = generate_imu_from_reference(df_ref, cfg)
    return df_imu_clean


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
