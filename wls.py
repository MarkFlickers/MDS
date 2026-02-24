import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

from coord_conversion import ecef_to_llh, ecef_delta_to_enu


@dataclass
class WlsConfig:
    max_iter: int = 10
    tol: float = 1e-4  # критерий остановки по норме приращения состояния (в метрах)
    sigma_pr: float = 3.0  # базовое СКО псевдодальности (м)
    use_elevation_weights: bool = True
    sin_el_floor: float = 0.2  # ограничение для весов, чтобы не взрывались при малых углах
    el_mask_deg: float = 5.0   # отсечка по углу места (град); 0 = не отсекать


def _elevation_rad(sat_ecef: np.ndarray, rx_ecef: np.ndarray) -> float:
    """
    Угол места (elevation) в радианах.
    Считаем в локальной ENU в точке текущей оценки приёмника.
    """
    lat, lon, _ = ecef_to_llh(rx_ecef[0], rx_ecef[1], rx_ecef[2])
    d = sat_ecef - rx_ecef
    e, n, u = ecef_delta_to_enu(d[0], d[1], d[2], lat, lon)
    horiz = np.sqrt(e*e + n*n)
    return np.arctan2(u, horiz)


def build_weights(sat_pos: np.ndarray, rx_ecef: np.ndarray, cfg: WlsConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строит диагональную матрицу весов W и маску используемых спутников.
    Веса ~ sin(el)^2 / sigma0^2, как базовый вариант.
    """
    m = sat_pos.shape[0]
    used = np.ones(m, dtype=bool)

    if cfg.el_mask_deg > 0:
        el_mask = np.deg2rad(cfg.el_mask_deg)
    else:
        el_mask = -np.inf

    w = np.zeros(m, dtype=float)

    for i in range(m):
        el = _elevation_rad(sat_pos[i], rx_ecef)
        if el < el_mask:
            used[i] = False
            w[i] = 0.0
            continue

        if not cfg.use_elevation_weights:
            w[i] = 1.0 / (cfg.sigma_pr ** 2)
        else:
            s = max(np.sin(el), cfg.sin_el_floor)
            # Эквивалентно sigma_i = sigma0 / s, W = 1/sigma_i^2 = s^2/sigma0^2
            w[i] = (s * s) / (cfg.sigma_pr ** 2)

    # Если веса отключены/обнулены — оставляем хотя бы единичные
    if np.sum(used) > 0 and np.all(w[used] == 0):
        w[used] = 1.0 / (cfg.sigma_pr ** 2)

    W = np.diag(w[used])
    return W, used


def wls_epoch(
    df_epoch: pd.DataFrame,
    x0: np.ndarray,
    cfg: WlsConfig,
) -> Dict:
    """
    Взвешенный МНК по псевдодальностям для одной эпохи.

    Состояние x = [X, Y, Z, cb]^T, где cb = c*dt (в метрах).
    Возвращает:
      - x_hat (4,)
      - P_hat (4,4) ковариация оценки (по линейной аппроксимации)
      - residuals (m_used,)
      - used_mask (m,)
      - A (m_used,4) дизайн-матрица последней итерации
    """
    sat_pos_all = df_epoch[['sat_X', 'sat_Y', 'sat_Z']].to_numpy(dtype=float)
    pr_all = df_epoch['pseudorange'].to_numpy(dtype=float)

    if sat_pos_all.shape[0] < 4:
        raise ValueError("WLS требует минимум 4 спутника для оценки [X,Y,Z,cb].")

    x = x0.astype(float).copy()

    for it in range(cfg.max_iter):
        rx = x[0:3]
        cb = x[3]

        # Веса и отсев по углу места считаем относительно текущей оценки rx
        W, used = build_weights(sat_pos_all, rx, cfg)
        sat_pos = sat_pos_all[used]
        pr = pr_all[used]
        m = sat_pos.shape[0]

        if m < 4:
            raise ValueError("После маски по углу места осталось <4 спутников — WLS невозможен.")

        # Геометрические дальности и LOS
        d = rx.reshape(1, 3) - sat_pos  # (m,3) = r_rx - r_sat
        rho = np.linalg.norm(d, axis=1)  # (m,)

        # Защита от деления на ноль
        if np.any(rho < 1.0):
            raise ValueError("Некорректная геометрия: слишком малая дальность до спутника.")

        # Предсказание псевдодальности: rho + cb
        pr_pred = rho + cb

        # Невязка (инновация) = измерение - предсказание
        v = pr - pr_pred  # (m,)

        # Дизайн-матрица A = d(pr_pred)/d(x)
        # drho/dr = (r_rx - r_sat)/rho
        A = np.zeros((m, 4), dtype=float)
        A[:, 0:3] = (d / rho.reshape(-1, 1))
        A[:, 3] = 1.0

        # Решаем нормальные уравнения WLS:
        # dx = (A^T W A)^-1 A^T W v
        N = A.T @ W @ A
        n = A.T @ W @ v

        # На случай плохой обусловленности — псевдообратная
        try:
            dx = np.linalg.solve(N, n)
        except np.linalg.LinAlgError:
            dx = np.linalg.pinv(N) @ n

        x = x + dx

        # Критерий остановки
        if np.linalg.norm(dx) < cfg.tol:
            break

    # Финальная ковариация оценки
    rx = x[0:3]
    W, used = build_weights(sat_pos_all, rx, cfg)
    sat_pos = sat_pos_all[used]
    pr = pr_all[used]
    d = rx.reshape(1, 3) - sat_pos
    rho = np.linalg.norm(d, axis=1)
    pr_pred = rho + x[3]
    v = pr - pr_pred

    A = np.zeros((sat_pos.shape[0], 4), dtype=float)
    A[:, 0:3] = (d / rho.reshape(-1, 1))
    A[:, 3] = 1.0

    N = A.T @ W @ A
    try:
        P = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(N)

    return {
        "x_hat": x,
        "P_hat": P,
        "residuals": v,
        "used_mask": used,
        "A": A
    }


def extract_epochs(df_raw: pd.DataFrame) -> Dict[float, pd.DataFrame]:
    """Удобный хелпер: разбиение raw наблюдений по эпохам t."""
    epochs = {}
    for t, g in df_raw.groupby('t'):
        epochs[float(t)] = g.reset_index(drop=True)
    return epochs
