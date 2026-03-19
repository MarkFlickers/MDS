
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
    # Временная сетка
    dt_imu: float = 1/100    # Шаг ИМУ (100 Гц)
    dt_gnss: float = 2    # Шаг ГНСС (0.5 Гц)
    
    # Генераторы случайных чисел (для воспроизводимости)
    seed_gnss: int = 42
    seed_imu: int = 42
    
    # Опорная точка (нейтральная, экватор/нулевой меридиан для простоты)
    ref_lat: float = 0.0    # [рад]
    ref_lon: float = 0.0    # [рад]
    ref_alt: float = 0.0    # [м]
    
    # Физика и геометрия
    g: float = 9.81                 # Ускорение свободного падения [м/с^2]
    gnss_pos_sigma: float = 3.0     # СКО шума ГНСС [м]
    lever_arm: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.0]))    # Вынос ИМУ -> Антенна ГНСС в ССК [м]

    # Начальные условия
    init_pos_enu: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))     # [E, N, U]
    init_vel_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))    # Скорость в ССК
    init_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))       # [roll, pitch, yaw] в радианах

    # --- Модели ошибок ИМУ ---
    # Акселерометр
    accel_bias: np.ndarray = field(default_factory=lambda: np.array([0.02, -0.015, 0.03])) # м/с^2
    accel_vrw: float = 0.001  # Velocity Random Walk (м/с / sqrt(с))
    # Гироскоп
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.001, -0.002, 0.0015])) # рад/с
    gyro_arw: float = 0.0001 # Angular Random Walk (рад / sqrt(с))
    
    # --- Сырые наблюдения ГНСС ---
    # Модель часов приемника
    clock_bias_init: float = 1e-4      # Секунды (ок. 30 км ошибки дальности)
    clock_drift_init: float = 1e-7     # Сек/сек (ок. 30 м/с ошибки доплера)
    clock_bias_noise: float = 1e-9     # Шум дрейфа часов
    # Спутники
    num_satellites: int = 36            # Количество видимых спутников
    satellite_radius: float = 26600000.0 # Радиус орбиты GPS (м)
    raw_pr_sigma: float = 2.0          # СКО шума псевдодальности (м)
    raw_doppler_sigma: float = 0.1     # СКО шума доплера (м/с)

ZERO_A = np.array([0.0, 0.0, 0.0])

# Сценарий движения
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

# Более сложный сценарий для демонстрации преимуществ ИНС/ESKF
stages_scenario_hard = [
    Stage(6.0,  np.array([ 1.5,  0.0,  0.0]), ZERO_A,                 "Интенсивный разгон"),
    Stage(10.0, ZERO_A,                             ZERO_A,             "Быстрый крейсерский полет"),

    Stage(4.0,  np.array([ 0.2,  0.0,  0.0]), np.array([0.0, 0.02, 0.10]), "Вход в левый разворот с набором"),
    Stage(6.0,  np.array([ 0.0,  0.0,  0.2]), ZERO_A,                        "Левый разворот, удержание набора"),
    Stage(4.0,  np.array([-0.2,  0.0,  0.0]), np.array([0.0,-0.02,-0.10]), "Выход из левого разворота"),

    Stage(5.0,  ZERO_A,                             ZERO_A,             "Короткий прямолинейный участок"),

    Stage(4.0,  np.array([ 0.2,  0.0,  0.0]), np.array([0.0,-0.02,-0.12]), "Вход в правый разворот со снижением"),
    Stage(6.0,  np.array([ 0.0,  0.0, -0.2]), ZERO_A,                        "Правый разворот, удержание снижения"),
    Stage(4.0,  np.array([-0.2,  0.0,  0.0]), np.array([0.0, 0.02, 0.12]), "Выход из правого разворота"),

    Stage(6.0,  np.array([ 0.5,  0.4,  0.0]), ZERO_A,                 "Маневр с боковой составляющей"),
    Stage(6.0,  np.array([-1.5, -0.2,  0.1]), ZERO_A,                 "Энергичное торможение"),
    Stage(6.0,  ZERO_A,                             ZERO_A,             "Стабилизированный полет"),
]
