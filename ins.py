# МЕХАНИЗАЦИЯ ИНС И СЛАБОСВЯЗАННАЯ ИНТЕГРАЦИЯ

import numpy as np
import pandas as pd

from configuration import TrajectoryConfig
from kalman import InsErrorStateKalmanFilter
from quaternion import quat_from_rotvec, quat_mul, quat_to_dcm, quat_to_euler, euler_to_quat


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def mechanize_ins(
    df_imu: pd.DataFrame,
    cfg: TrajectoryConfig,
    init_pos: np.ndarray,
    init_vel: np.ndarray,
    init_quat: np.ndarray,
    accel_bias: np.ndarray | None = None,
    gyro_bias: np.ndarray | None = None,
) -> pd.DataFrame:
    """Автономная ИНС-механизация в локальной системе ENU."""
    accel_bias = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=float).copy()
    gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float).copy()

    t = df_imu['t'].to_numpy(dtype=float)
    n = len(df_imu)

    r_n = np.asarray(init_pos, dtype=float).copy()
    v_n = np.asarray(init_vel, dtype=float).copy()
    q_nb = _normalize_quat(np.asarray(init_quat, dtype=float).copy())

    g_n = np.array([0.0, 0.0, -cfg.g], dtype=float)

    hist = {
        't': np.zeros(n),
        'E_est': np.zeros(n), 'N_est': np.zeros(n), 'U_est': np.zeros(n),
        'vE_est': np.zeros(n), 'vN_est': np.zeros(n), 'vU_est': np.zeros(n),
        'q_w': np.zeros(n), 'q_x': np.zeros(n), 'q_y': np.zeros(n), 'q_z': np.zeros(n),
        'roll_est': np.zeros(n), 'pitch_est': np.zeros(n), 'yaw_est': np.zeros(n),
        'E_ant_est': np.zeros(n), 'N_ant_est': np.zeros(n), 'U_ant_est': np.zeros(n),
    }

    def save_state(k: int):
        C_nb = quat_to_dcm(q_nb)
        r_ant = r_n + C_nb @ cfg.lever_arm
        roll, pitch, yaw = quat_to_euler(q_nb)

        hist['t'][k] = t[k]
        hist['E_est'][k], hist['N_est'][k], hist['U_est'][k] = r_n
        hist['vE_est'][k], hist['vN_est'][k], hist['vU_est'][k] = v_n
        hist['q_w'][k], hist['q_x'][k], hist['q_y'][k], hist['q_z'][k] = q_nb
        hist['roll_est'][k], hist['pitch_est'][k], hist['yaw_est'][k] = roll, pitch, yaw
        hist['E_ant_est'][k], hist['N_ant_est'][k], hist['U_ant_est'][k] = r_ant

    save_state(0)

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        f_b_meas = df_imu.loc[k, ['f_x', 'f_y', 'f_z']].to_numpy(dtype=float)
        w_b_meas = df_imu.loc[k, ['w_x', 'w_y', 'w_z']].to_numpy(dtype=float)

        w_b_corr = w_b_meas - gyro_bias
        dq = quat_from_rotvec(w_b_corr * dt)
        q_nb = _normalize_quat(quat_mul(q_nb, dq))

        C_nb = quat_to_dcm(q_nb)
        f_b_corr = f_b_meas - accel_bias
        a_n = C_nb @ f_b_corr + g_n

        v_n = v_n + a_n * dt
        r_n = r_n + v_n * dt
        save_state(k)

    return pd.DataFrame(hist)


def run_loosely_coupled_ins_gnss(
    df_imu: pd.DataFrame,
    df_gnss: pd.DataFrame,
    df_ref: pd.DataFrame,
    cfg: TrajectoryConfig,
    use_lever_arm: bool = False,
    gnss_pos_sigma: float | None = None,
    accel_bias_rw_sigma: float = 1e-3,
    gyro_bias_rw_sigma: float = 1e-4,
    p0_pos_sigma: float = 5.0,
    p0_vel_sigma: float = 1.0,
    p0_att_sigma_deg: float = 5.0,
    p0_accel_bias_sigma: float = 0.05,
    p0_gyro_bias_sigma: float = 0.01,
) -> pd.DataFrame:
    """Слабосвязанная схема ГНСС/ИНС на основе 15-состояний ESKF."""
    gnss_pos_sigma = cfg.gnss_pos_sigma if gnss_pos_sigma is None else gnss_pos_sigma

    t = df_imu['t'].to_numpy(dtype=float)
    n = len(df_imu)

    gnss_row0 = df_gnss.iloc[0]
    ref_row0 = df_ref.iloc[0]

    init_vel = gnss_row0[['vE', 'vN', 'vU']].to_numpy(dtype=float) if all(col in df_gnss.columns for col in ['vE', 'vN', 'vU']) else np.zeros(3)
    if all(col in df_ref.columns for col in ['q_w', 'q_x', 'q_y', 'q_z']):
        init_quat = ref_row0[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    else:
        init_quat = euler_to_quat(ref_row0['roll'], ref_row0['pitch'], ref_row0['yaw'])

    init_quat = _normalize_quat(init_quat)
    C0 = quat_to_dcm(init_quat)
    init_pos = gnss_row0[['E', 'N', 'U']].to_numpy(dtype=float)
    if use_lever_arm:
        init_pos = init_pos - C0 @ cfg.lever_arm

    p0_diag = np.array([
        p0_pos_sigma, p0_pos_sigma, p0_pos_sigma,
        p0_vel_sigma, p0_vel_sigma, p0_vel_sigma,
        np.deg2rad(p0_att_sigma_deg), np.deg2rad(p0_att_sigma_deg), np.deg2rad(p0_att_sigma_deg),
        p0_accel_bias_sigma, p0_accel_bias_sigma, p0_accel_bias_sigma,
        p0_gyro_bias_sigma, p0_gyro_bias_sigma, p0_gyro_bias_sigma,
    ], dtype=float)
    P0 = np.diag(p0_diag ** 2)

    kf = InsErrorStateKalmanFilter(
        sigma_accel=cfg.accel_vrw,
        sigma_gyro=cfg.gyro_arw,
        sigma_accel_bias=accel_bias_rw_sigma,
        sigma_gyro_bias=gyro_bias_rw_sigma,
        P0=P0,
    )

    r_n = init_pos.copy()
    v_n = init_vel.copy()
    q_nb = init_quat.copy()
    b_a = np.zeros(3, dtype=float)
    b_g = np.zeros(3, dtype=float)
    g_n = np.array([0.0, 0.0, -cfg.g], dtype=float)
    R = np.eye(3, dtype=float) * (gnss_pos_sigma ** 2)

    gnss_lookup = {round(float(row.t), 9): row for row in df_gnss.itertuples(index=False)}

    hist = {
        't': np.zeros(n),
        'E_est': np.zeros(n), 'N_est': np.zeros(n), 'U_est': np.zeros(n),
        'vE_est': np.zeros(n), 'vN_est': np.zeros(n), 'vU_est': np.zeros(n),
        'q_w': np.zeros(n), 'q_x': np.zeros(n), 'q_y': np.zeros(n), 'q_z': np.zeros(n),
        'roll_est': np.zeros(n), 'pitch_est': np.zeros(n), 'yaw_est': np.zeros(n),
        'E_ant_est': np.zeros(n), 'N_ant_est': np.zeros(n), 'U_ant_est': np.zeros(n),
        'ba_x': np.zeros(n), 'ba_y': np.zeros(n), 'ba_z': np.zeros(n),
        'bg_x': np.zeros(n), 'bg_y': np.zeros(n), 'bg_z': np.zeros(n),
        'gnss_update': np.zeros(n, dtype=int),
    }

    def save_state(k: int, gnss_update: bool = False):
        C_nb = quat_to_dcm(q_nb)
        r_ant = r_n + C_nb @ cfg.lever_arm
        roll, pitch, yaw = quat_to_euler(q_nb)

        hist['t'][k] = t[k]
        hist['E_est'][k], hist['N_est'][k], hist['U_est'][k] = r_n
        hist['vE_est'][k], hist['vN_est'][k], hist['vU_est'][k] = v_n
        hist['q_w'][k], hist['q_x'][k], hist['q_y'][k], hist['q_z'][k] = q_nb
        hist['roll_est'][k], hist['pitch_est'][k], hist['yaw_est'][k] = roll, pitch, yaw
        hist['E_ant_est'][k], hist['N_ant_est'][k], hist['U_ant_est'][k] = r_ant
        hist['ba_x'][k], hist['ba_y'][k], hist['ba_z'][k] = b_a
        hist['bg_x'][k], hist['bg_y'][k], hist['bg_z'][k] = b_g
        hist['gnss_update'][k] = int(gnss_update)

    save_state(0, gnss_update=True)

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        f_b_meas = df_imu.loc[k, ['f_x', 'f_y', 'f_z']].to_numpy(dtype=float)
        w_b_meas = df_imu.loc[k, ['w_x', 'w_y', 'w_z']].to_numpy(dtype=float)

        w_b_corr = w_b_meas - b_g
        dq = quat_from_rotvec(w_b_corr * dt)
        q_nb = _normalize_quat(quat_mul(q_nb, dq))

        C_nb = quat_to_dcm(q_nb)
        f_b_corr = f_b_meas - b_a
        a_n = C_nb @ f_b_corr + g_n

        v_n = v_n + a_n * dt
        r_n = r_n + v_n * dt

        kf.predict_ins(C_nb=C_nb, f_b_corr=f_b_corr, w_b_corr=w_b_corr, dt=dt)

        gnss_update = False
        gnss_row = gnss_lookup.get(round(t[k], 9))
        if gnss_row is not None:
            gnss_update = True
            r_gnss = np.array([gnss_row.E, gnss_row.N, gnss_row.U], dtype=float)

            if use_lever_arm:
                r_ant_pred = r_n + C_nb @ cfg.lever_arm
                z = r_ant_pred - r_gnss
                kf.update_position(z=z, R=R, C_nb=C_nb, lever_arm=cfg.lever_arm)
            else:
                z = r_n - r_gnss
                kf.update_position(z=z, R=R)

            dx = kf.x.copy()
            r_n = r_n - dx[0:3]
            v_n = v_n - dx[3:6]
            q_nb = _normalize_quat(quat_mul(q_nb, quat_from_rotvec(-dx[6:9])))
            b_a = b_a - dx[9:12]
            b_g = b_g - dx[12:15]
            kf.reset_error_state()

        save_state(k, gnss_update=gnss_update)

    return pd.DataFrame(hist)
