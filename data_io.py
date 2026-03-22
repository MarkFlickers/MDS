import pandas as pd
import numpy as np
import json
from dataclasses import asdict
from math import degrees
from csv_2_SignalSim.csv_to_signalsim_traj import main as csv_2_SignalSim

def save_trajectories(df_imu_clean: pd.DataFrame, df_imu_noisy: pd.DataFrame, df_gnss_clean: pd.DataFrame, df_gnss_noisy: pd.DataFrame, df_gnss_raw: pd.DataFrame):
    df_imu_clean.to_csv("output/trajectory_imu_ideal.csv", index=False)    # Cохраняем идеал для отладки
    df_imu_noisy.to_csv("output/trajectory_imu_noisy.csv", index=False)          # Тут реалистичные измерения с шумом
    df_gnss_clean.to_csv("output/trajectory_gnss_true.csv", index=False)   
    df_gnss_meas = df_gnss_noisy[['t', 'E', 'N', 'U', 'lat', 'lon', 'alt', 'X_ecef', 'Y_ecef', 'Z_ecef']]   # Для фильтра Калмана ЛР1 оставляем только обязательные столбцы (позицию)
    df_gnss_meas.to_csv("output/gnss_measurements.csv", index=False)
    df_gnss_raw.to_csv("output/gnss_raw_observables.csv", index=False)
    df_for_nmea = df_gnss_clean[['lat', 'lon']].apply(np.vectorize(degrees))
    df_for_nmea.to_csv("output/gnss_for_nmea.csv", index=False)
    df_for_csv_to_sim = df_gnss_clean[['lat', 'lon']].apply(np.vectorize(degrees)).assign(**df_gnss_clean[['alt']])
    df_for_csv_to_sim.to_csv("output/CSV_to_SignalSim_track.csv", index=False, header=False)
    csv_2_SignalSim(
                ["output/CSV_to_SignalSim_track.csv",
                "output/SignalSimInput.json",
                "--template", "CSV_to_SignalSim_template.json",
                "--init-course-unit", "rad",
                "--turn-angle-unit", "degree",
                "--angleunit-field", "degree"])

def save_metadata(config, filename="output/trajectory_metadata.json"):
    meta = asdict(config)
    # Преобразуем numpy массивы в списки для JSON
    for key, value in meta.items():
        if hasattr(value, "tolist"):
            meta[key] = value.tolist()
            
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)