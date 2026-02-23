# МАТЕМАТИКА ВРАЩЕНИЙ

import numpy as np
from typing import Tuple

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half_angle = norm_v / 2.0
    sin_half = np.sin(half_angle)
    return np.array([np.cos(half_angle), 
                     v[0]/norm_v * sin_half, 
                     v[1]/norm_v * sin_half, 
                     v[2]/norm_v * sin_half])

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*w*z,         2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x**2 - 2*z**2,     2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,         1 - 2*x**2 - 2*y**2]
    ])

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw