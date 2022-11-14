import pathlib

import pandas as pd
import numpy as np


def generate_input(
    igrow_width,
    domain_height,
    drain_width,
    drain_bottom,
    drain_level,
    recharge,
    kxx,
    kzz,
    kxz,
    c_drain,
    c_seepage,
    c_base,
    H_base,
    cell_width,
    model_bottom="closed",
    label=None,
):
    column_dtype = {
        "cross section form": str,
        "model bottom": str,
        "riparian connection": str,
        "c(drainage-seepage) values": str,
        "catchment form": str,
        "iGrOw width (L/2)": np.float64,
        "domain height": np.float64,
        "drain width (B/2)": np.float64,
        "drain bottom": np.float64,
        "top talud angle": np.float64,
        "trap talud 1 over": np.float64,
        "catchment curvature": np.float64,
        "drain level": np.float64,
        "recharge *1000": np.float64,
        "kxx": np.float64,
        "kzz": np.float64,
        "kxz": np.float64,
        "c drain": np.float64,
        "c seepage": np.float64,
        "c base": np.float64,
        "H base": np.float64,
        "CELL width": np.float64,
        "# hor under drain": int,
        "# hor right of drain": int,
        "# verleft": int,
        "# verright": int,
        "# dens factor": np.float64,
    }

    n = igrow_width.size
    cross_section_form = np.full(n, "trapezoidal")
    model_bottom = np.full(n, model_bottom)
    riparian_connection = np.full(n, "impermeable")
    c_interpolation = np.full(n, "non-interpolated")
    catchment_form = np.full(n, "parallel")

    top_talud_angle = np.full(n, np.nan)
    trap_talud_1_over = np.full(n, 0.0)
    catchment_curvature = np.full(n, 0.0)

    n_hor_under_drain = np.full(n, 20)
    n_hor_right_of_drain = np.full(n, 30)
    verleft = np.full(n, 15)
    verright = np.full(n, 8)
    dens_factor = np.full(n, 0.4)

    columns = list(column_dtype.keys())
    args = [
        cross_section_form,
        model_bottom,
        riparian_connection,
        c_interpolation,
        catchment_form,
        igrow_width,
        domain_height,
        drain_width,
        drain_bottom,
        top_talud_angle,
        trap_talud_1_over,
        catchment_curvature,
        drain_level,
        recharge,
        kxx,
        kzz,
        kxz,
        c_drain,
        c_seepage,
        c_base,
        H_base,
        cell_width,
        n_hor_under_drain,
        n_hor_right_of_drain,
        verleft,
        verright,
        dens_factor,
    ]
    parameters = np.column_stack(args)
    df = pd.DataFrame(data=parameters, columns=columns)
    df["comment"] = label
    return df


def rand_within(n: int, low: float, high: float) -> np.ndarray:
    delta = high - low
    return np.random.rand(n) * delta + low


def update_case(case, key, value):
    new = case.copy()
    if key not in new:
        raise KeyError(f"{key} not found in existing iGrOw case")
    new[key] = value
    return new


def write_igrow_input(df: pd.DataFrame, path: str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep="NaN")
    return


def read_igrow_output(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).iloc[1:]
    df["id"] = df["id"].astype(np.int64) - 1
    df = df.set_index("id")
    return df
