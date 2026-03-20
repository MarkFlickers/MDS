import numpy as np
import pandas as pd
from coord_conversion import R_ecef_to_enu

def calculate_rmse(df_true: pd.DataFrame, df_est: pd.DataFrame) -> dict:
    """Вычисляет СКО (RMSE) между истинной траекторией и оценкой."""
    # Синхронизируем данные по времени на случай разных частот
    df_merged = pd.merge_asof(df_est, df_true, on='t', direction='nearest', suffixes=('_est', '_true'))
    
    # Ошибки по позиции
    err_E = df_merged['E_est'] - df_merged['E_true']
    err_N = df_merged['N_est'] - df_merged['N_true']
    err_U = df_merged['U_est'] - df_merged['U_true']
    pos_rmse = np.sqrt(np.mean(err_E**2 + err_N**2 + err_U**2))

    vel_cols = ['vE_est', 'vN_est', 'vU_est', 'vE_true', 'vN_true', 'vU_true']
    # Ошибки по скорости
    if all(col in df_merged.columns for col in vel_cols):
        err_vE = df_merged['vE_est'] - df_merged['vE_true']
        err_vN = df_merged['vN_est'] - df_merged['vN_true']
        err_vU = df_merged['vU_est'] - df_merged['vU_true']
        vel_rmse = np.sqrt(np.mean(err_vE**2 + err_vN**2 + err_vU**2))
    else:
        vel_rmse = 0.0

    return {
        'pos_rmse_3d': pos_rmse,
        'vel_rmse_3d': vel_rmse,
        'err_E': np.sqrt(np.mean(err_E**2)),
        'err_N': np.sqrt(np.mean(err_N**2)),
        'err_U': np.sqrt(np.mean(err_U**2)),
        'n_samples': int(len(df_merged)),
    }


def calculate_dop(H_ecef: np.ndarray, lat: float, lon: float) -> dict:
    """
    Вычисляет факторы потери точности (DOP) на основе матрицы направляющих косинусов.
    H_ecef: Матрица Якоби (M x 4) в ECEF.
    """
    try:
        # Для DOP используется не взвешенная матрица H
        Q_ecef = np.linalg.inv(H_ecef.T @ H_ecef)
    except np.linalg.LinAlgError:
        return {'GDOP': 99.9, 'PDOP': 99.9, 'HDOP': 99.9, 'VDOP': 99.9, 'TDOP': 99.9}
        
    # Ковариация позиций в ECEF
    P_xyz = Q_ecef[0:3, 0:3]
    
    # Переход в ENU
    R = R_ecef_to_enu(lat, lon)
    P_enu = R @ P_xyz @ R.T

    pdop = np.sqrt(np.trace(P_enu))
    hdop = np.sqrt(P_enu[0, 0] + P_enu[1, 1])
    vdop = np.sqrt(P_enu[2, 2])
    tdop = np.sqrt(max(0, Q_ecef[3, 3]))
    gdop = np.sqrt(pdop**2 + tdop**2)

    return {'GDOP': gdop, 'PDOP': pdop, 'HDOP': hdop, 'VDOP': vdop, 'TDOP': tdop}
