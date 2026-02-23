
# КОНФИГУРАЦИЯ И СЦЕНАРИЙ МАНЁВРОВ

import numpy as np
from dataclasses import dataclass, field

@dataclass
class Stage:
    duration_s: float
    a_body: np.ndarray
    w_dot_body: np.ndarray
    name: str = ""

@dataclass
class TrajectoryConfig:
    dt_imu: float = 1/50
    dt_gnss: float = 1.0
    
    seed_gnss: int = 42
    seed_imu: int = 42
    
    ref_lat: float = 0.0
    ref_lon: float = 0.0
    ref_alt: float = 0.0
    
    g: float = 9.81
    gnss_pos_sigma: float = 3.0
    
    lever_arm: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.0]))
    init_pos_enu: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    init_vel_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    init_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # --- НОВЫЕ ПАРАМЕТРЫ ДЛЯ ЛР3 (Модели ошибок ИМУ) ---
    # Акселерометр
    accel_bias: np.ndarray = field(default_factory=lambda: np.array([0.02, -0.015, 0.03])) # м/с^2
    accel_vrw: float = 0.001  # Velocity Random Walk (м/с / sqrt(с))
    # Гироскоп
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.001, -0.002, 0.0015])) # рад/с
    gyro_arw: float = 0.0001 # Angular Random Walk (рад / sqrt(с))
    
    # --- НОВЫЕ ПАРАМЕТРЫ ДЛЯ ЛР4 (Сырые наблюдения ГНСС) ---
    # Модель часов приемника
    clock_bias_init: float = 1e-4      # Секунды (ок. 30 км ошибки дальности)
    clock_drift_init: float = 1e-7     # Сек/сек (ок. 30 м/с ошибки доплера)
    clock_bias_noise: float = 1e-9     # Шум дрейфа часов
    # Спутники
    num_satellites: int = 8            # Количество видимых спутников
    satellite_radius: float = 26600000.0 # Радиус орбиты GPS (м)
    raw_pr_sigma: float = 2.0          # СКО шума псевдодальности (м)
    raw_doppler_sigma: float = 0.1     # СКО шума доплера (м/с)

ZERO_A = np.array([0.0, 0.0, 0.0])

stages_scenario = [
    Stage(5.0,  np.array([ 1.0, 0.0, 0.0]), ZERO_A,                      "Разгон вперед"),
    Stage(10.0, ZERO_A,                     ZERO_A,                      "Крейсерская скорость 1"),
    Stage(5.0,  ZERO_A,                     np.array([0.0, 0.0,  0.05]), "Угловое ускорение (поворот влево)"),
    Stage(5.0,  ZERO_A,                     np.array([0.0, 0.0, -0.05]), "Угловое торможение (остановка вращения)"),
    Stage(10.0, ZERO_A,                     ZERO_A,                      "Крейсерская скорость 2"),
    Stage(5.0,  np.array([ 0.0, 0.0, 0.5]), ZERO_A,                      "Набор высоты"),
    Stage(10.0, ZERO_A,                     ZERO_A,                      "Горизонтальный полет на высоте"),
    Stage(5.0,  np.array([-1.0, 0.0, -0.5]), ZERO_A,                     "Торможение"),
    Stage(5.0,  ZERO_A,                     ZERO_A,                      "Покой")
]