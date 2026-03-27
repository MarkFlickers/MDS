import pandas as pd
import numpy as np
import json
from dataclasses import asdict
from math import degrees
from csv_2_SignalSim.csv_to_signalsim_traj import main as csv_2_SignalSim
from pathlib import Path
import subprocess
from coord_conversion import ecef_to_enu


def save_trajectories(df_imu_clean: pd.DataFrame, df_imu_noisy: pd.DataFrame, df_gnss_clean: pd.DataFrame, df_gnss_noisy: pd.DataFrame, df_gnss_raw: pd.DataFrame):
    Path("output").mkdir(parents=True, exist_ok=True)
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
                "--init-course-unit", "degree",
                "--turn-angle-unit", "degree",
                "--angleunit-field", "degree"])
    
    exe_path = Path("SignalSim/bin/JsonObsGen.exe").resolve()
    json_path = Path("output/SignalSimInput.json").resolve()

    subprocess.run(
        [str(exe_path), str(json_path)],
        cwd=str(Path("SignalSim").resolve()),
        check=True
    )

def save_metadata(config, filename="output/trajectory_metadata.json"):
    meta = asdict(config)
    # Преобразуем numpy массивы в списки для JSON
    for key, value in meta.items():
        if hasattr(value, "tolist"):
            meta[key] = value.tolist()
            
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

def parse_rtklib_pos(filepath: str, ref_lat: float, ref_lon: float, ref_alt: float) -> pd.DataFrame:
    rows = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            parts = line.split()
            if len(parts) < 15:
                continue

            date_str = parts[0]
            time_str = parts[1]

            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            q = int(parts[5])
            ns = int(parts[6])
            sdx = float(parts[7])
            sdy = float(parts[8])
            sdz = float(parts[9])
            sdxy = float(parts[10])
            sdyz = float(parts[11])
            sdzx = float(parts[12])
            age = float(parts[13])
            ratio = float(parts[14])

            ts = pd.to_datetime(f"{date_str} {time_str}", format="%Y/%m/%d %H:%M:%S.%f")
            rows.append({
                'datetime': ts,
                'X': x,
                'Y': y,
                'Z': z,
                'Q': q,
                'ns': ns,
                'sdx': sdx,
                'sdy': sdy,
                'sdz': sdz,
                'sdxy': sdxy,
                'sdyz': sdyz,
                'sdzx': sdzx,
                'age': age,
                'ratio': ratio,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    t0 = df['datetime'].iloc[0]
    df['t'] = (df['datetime'] - t0).dt.total_seconds()

    enu = [
        ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)
        for x, y, z in zip(df['X'], df['Y'], df['Z'])
    ]
    df['E'], df['N'], df['U'] = zip(*enu)

    return df