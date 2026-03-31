
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
    dt_gnss: float = 1    # Шаг ГНСС (1 Гц)
    
    # Генераторы случайных чисел (для воспроизводимости)
    seed_gnss: int = 42
    seed_imu: int = 42
    
    # Опорная точка (нейтральная, экватор/нулевой меридиан для простоты)
    ref_lat: float = np.deg2rad(0)    # [рад]
    ref_lon: float = np.deg2rad(0)    # [рад]
    ref_alt: float = 0    # [м]
    
    # Физика и геометрия
    g: float = 9.81                 # Ускорение свободного падения [м/с^2]
    gnss_pos_sigma: float = 3.0     # СКО шума ГНСС [м]
    lever_arm: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))    # Вынос ИМУ -> Антенна ГНСС в ССК [м]

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
    raw_pr_sigma: float = 5.0          # СКО шума псевдодальности (м)
    raw_doppler_sigma: float = 3.0    # СКО шума доплера (м/с)

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

stages_scenario_extreme = [
    Stage(12.0, np.array([ 1.8,  0.0,  0.0]), ZERO_A, "Жёсткий разгон"),
    Stage(18.0, ZERO_A, ZERO_A, "Быстрый прямой участок"),

    Stage(3.0,  ZERO_A, np.array([0.0, 0.0,  0.20]), "Раскрутка влево"),
    Stage(10.0, np.array([ 0.2,  0.5,  0.15]), ZERO_A, "Левый вираж с набором"),
    Stage(3.0,  ZERO_A, np.array([0.0, 0.0, -0.20]), "Торможение вращения"),

    Stage(4.0,  np.array([-0.8,  0.2,  0.0]), ZERO_A, "Резкая перекладка"),
    Stage(3.0,  ZERO_A, np.array([0.0, 0.0, -0.22]), "Раскрутка вправо"),
    Stage(10.0, np.array([ 0.1, -0.6, -0.15]), ZERO_A, "Правый вираж со снижением"),
    Stage(3.0,  ZERO_A, np.array([0.0, 0.0,  0.22]), "Торможение вращения"),

    Stage(8.0,  np.array([ 0.7,  0.0,  0.0]), ZERO_A, "Повторный разгон"),
    Stage(6.0,  ZERO_A, np.array([0.0, 0.06, 0.0]), "Короткий тангаж вверх"),
    Stage(6.0,  np.array([ 0.0,  0.0,  0.35]), ZERO_A, "Набор высоты"),
    Stage(6.0,  ZERO_A, np.array([0.0,-0.06, 0.0]), "Возврат тангажа"),

    Stage(2.5,  ZERO_A, np.array([0.0, 0.0,  0.28]), "Зиг 1: раскрутка влево"),
    Stage(5.0,  np.array([ 0.3,  0.8,  0.0]), ZERO_A, "Зиг 1: удержание"),
    Stage(2.5,  ZERO_A, np.array([0.0, 0.0, -0.56]), "Зиг 2: переброс вправо"),
    Stage(5.0,  np.array([ 0.3, -0.8,  0.0]), ZERO_A, "Зиг 2: удержание"),
    Stage(2.5,  ZERO_A, np.array([0.0, 0.0,  0.56]), "Зиг 3: переброс влево"),
    Stage(5.0,  np.array([ 0.3,  0.8,  0.0]), ZERO_A, "Зиг 3: удержание"),
    Stage(2.5,  ZERO_A, np.array([0.0, 0.0, -0.28]), "Выход из слалома"),

    Stage(10.0, ZERO_A, ZERO_A, "Короткая стабилизация"),

    Stage(4.0,  ZERO_A, np.array([ 0.10, 0.00,  0.18]), "Комбинированный вход"),
    Stage(12.0, np.array([ 0.2,  0.3, -0.25]), ZERO_A, "Спираль вниз"),
    Stage(4.0,  ZERO_A, np.array([-0.10, 0.00, -0.18]), "Выход из спирали"),

    Stage(8.0,  np.array([-1.6,  0.0,  0.2]), ZERO_A, "Интенсивное торможение"),
    Stage(10.0, ZERO_A, ZERO_A, "Почти прямолинейный участок"),

    Stage(3.0,  ZERO_A, np.array([0.0, 0.0,  0.25]), "Финальный левый вход"),
    Stage(8.0,  np.array([ 0.0,  0.7,  0.0]), ZERO_A, "Быстрый вираж"),
    Stage(3.0,  ZERO_A, np.array([0.0, 0.0, -0.25]), "Финальный выход"),

    Stage(12.0, np.array([-1.0, -0.2, -0.2]), ZERO_A, "Длинное торможение"),
    Stage(15.0, ZERO_A, ZERO_A, "Финишный участок"),
]

stages_scenario_city = [
    Stage(6.0,  np.array([ 1.5,  0.0,  0.0]), ZERO_A, "Старт со светофора"),
    Stage(10.0, ZERO_A, ZERO_A, "Ровное движение по прямой"),
    Stage(2.0,  np.array([-4.4,  0.0,  0.0]), ZERO_A, "Резкое торможение"),
    Stage(2.0,  ZERO_A, ZERO_A, "Короткая остановка"),

    Stage(5.0,  np.array([ 1.6,  0.0,  0.0]), ZERO_A, "Повторный разгон"),
    Stage(1.0,  ZERO_A, np.array([0.0, 0.0,  1.57]), "Резкий вход в правый поворот"),
    # Stage(4.0,  ZERO_A, ZERO_A, "Доворот направо"),
    Stage(1.0,  ZERO_A, np.array([0.0, 0.0, -1.57]), "Гашение вращения"),
    Stage(8.0,  ZERO_A, ZERO_A, "Движение по новой улице"),

    Stage(1.0,  np.array([ 1.0,  7.0,  0.0]), ZERO_A, "Перестроение влево"),
    Stage(1.0,  np.array([ 0.0,  -7.0,  0.0]), ZERO_A, "Перестроение влево"),
    Stage(2.0,  ZERO_A, ZERO_A, "Удержание полосы"),
    Stage(1.0,  np.array([ 0.0,  -7.0,  0.0]), ZERO_A, "Возврат вправо"),
    Stage(1.0,  np.array([ -1.0,  7.0,  0.0]), ZERO_A, "Возврат вправо"),
    Stage(5.0,  ZERO_A, ZERO_A, "Стабилизация после манёвра"),

    # Stage(3.0,  np.array([ 1.0,  0.0,  0.0]), ZERO_A, "Обгон: ускорение"),
    # Stage(1.5,  np.array([ 0.2,  2.5,  0.0]), ZERO_A, "Выход влево на обгон"),
    # Stage(4.0,  ZERO_A, ZERO_A, "Параллельное движение"),
    # Stage(1.5,  np.array([ 0.2, -2.5,  0.0]), ZERO_A, "Возврат в полосу"),
    # Stage(3.0,  np.array([-1.0,  0.0,  0.0]), ZERO_A, "Сброс скорости после обгона"),

    Stage(6.0,  ZERO_A, ZERO_A, "Длинный прямой участок"),
    Stage(4.0,  np.array([-1.95,  0.0,  0.0]), ZERO_A, "Подъезд к перекрёстку"),
    Stage(2.0,  ZERO_A, ZERO_A, "Остановка на перекрёстке"),

    Stage(4.0,  np.array([ 1.0,  0.0,  0.0]), ZERO_A, "Начало разворота"),
    Stage(1.2,  ZERO_A, np.array([0.0, 0.0,  2.2]), "Раскрутка влево"),
    # Stage(5.0,  np.array([ 0.2,  0.8,  0.0]), ZERO_A, "Первая половина разворота"),
    # Stage(5.0,  np.array([ 0.2,  0.8,  0.0]), ZERO_A, "Вторая половина разворота"),
    Stage(1.2,  ZERO_A, np.array([0.0, 0.0, -2.2]), "Выход из разворота"),

    Stage(8.0,  ZERO_A, ZERO_A, "Движение в обратном направлении"),
    Stage(2.0,  ZERO_A, np.array([0.0, 0.0, -0.40]), "Резкий вход в левый поворот"),
    # Stage(4.0,  ZERO_A, ZERO_A, "Поворот налево"),
    Stage(2.0,  ZERO_A, np.array([0.0, 0.0,  0.40]), "Гашение вращения"),
    Stage(6.0,  ZERO_A, ZERO_A, "Выезд на длинную улицу"),

    Stage(1.5,  np.array([ 0.5, -5.0,  0.0]), ZERO_A, "Быстрый объезд препятствия вправо"),
    Stage(1.5,  np.array([ 0.0,  5.0,  0.0]), ZERO_A, "Быстрый объезд препятствия вправо"),
    Stage(1.5,  np.array([ 0.0,  5.0,  0.0]), ZERO_A, "Возврат в траекторию"),
    Stage(1.5,  np.array([ 0.5, -5.0,  0.0]), ZERO_A, "Возврат в траекторию"),
    Stage(8.0,  ZERO_A, ZERO_A, "Стабильное движение"),

    Stage(3.0,  np.array([-1.82,  0.0,  0.0]), ZERO_A, "Финальное торможение"),
    Stage(6.0,  ZERO_A, ZERO_A, "Финишная остановка"),
]
