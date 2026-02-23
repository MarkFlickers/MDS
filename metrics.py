import numpy as np
import pandas as pd

def calculate_rmse(df_true: pd.DataFrame, df_est: pd.DataFrame) -> dict:
    """Вычисляет СКО (RMSE) между истинной траекторией и оценкой."""
    
    # Синхронизируем данные по времени на случай разных частот
    df_merged = pd.merge_asof(df_est, df_true, on='t', direction='nearest', suffixes=('_est', '_true'))
    
    # Ошибки по позиции
    err_E = df_merged['E_est'] - df_merged['E_true']  # Заменим '_ant' на '_true' при вызове, чтобы унифицировать
    err_N = df_merged['N_est'] - df_merged['N_true']
    err_U = df_merged['U_est'] - df_merged['U_true']
    pos_rmse = np.sqrt(np.mean(err_E**2 + err_N**2 + err_U**2))
    
    # Ошибки по скорости
    err_vE = df_merged['vE_est'] - df_merged['vE_true']
    err_vN = df_merged['vN_est'] - df_merged['vN_true']
    err_vU = df_merged['vU_est'] - df_merged['vU_true']
    vel_rmse = np.sqrt(np.mean(err_vE**2 + err_vN**2 + err_vU**2))
    
    return {
        'pos_rmse_3d': pos_rmse,
        'vel_rmse_3d': vel_rmse,
        'err_E': np.sqrt(np.mean(err_E**2)),
        'err_N': np.sqrt(np.mean(err_N**2)),
        'err_U': np.sqrt(np.mean(err_U**2)),
    }
