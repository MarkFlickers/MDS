# ПРЕОБРАЗОВАНИЯ КООРДИНАТ WGS84

import numpy as np
from typing import Tuple

a_wgs84 = 6378137.0
e2_wgs84 = 0.00669437999014
c_light = 299792458.0 # Скорость света для ЛР4

def enu_to_ecef(e: float, n: float, u: float, lat0: float, lon0: float, alt0: float) -> Tuple[float, float, float]:
    """Локальные ENU в ECEF."""
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)
    
    # ECEF опорной точки
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * sin_lat**2)
    x0 = (N + alt0) * cos_lat * cos_lon
    y0 = (N + alt0) * cos_lat * sin_lon
    z0 = (N * (1 - e2_wgs84) + alt0) * sin_lat
    
    # Матрица ENU -> ECEF
    R = np.array([
        [-sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon],
        [ cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon],
        [ 0,        cos_lat,         sin_lat        ]
    ])
    
    enu = np.array([e, n, u])
    dx, dy, dz = R @ enu
    return x0 + dx, y0 + dy, z0 + dz

def ecef_to_llh(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF в LLH (итерационный метод Bowring)."""
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2_wgs84))
    
    for _ in range(5):
        N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2_wgs84 * N / (N + alt)))
        
    return lat, lon, alt