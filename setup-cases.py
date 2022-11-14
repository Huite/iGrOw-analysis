# %%
from pathlib import Path

import pandas as pd
import numpy as np
import igrow_utils
import postprocess

np.random.seed(12345)

# %%

df = postprocess.extract_columns(
    igrow_utils.read_igrow_output(f"output/singlelayer-output.csv")
)
df = postprocess.assign_analytical_resistance(df)

results = postprocess.finite_volume_heads(df, celldrain_columns=["c_entry", "c_iGrOw"])

# %%


# %%


def generate_clay_parameters(n):
    kxx = igrow_utils.rand_within(n, 0.5, 1.0)
    vertical_anisotropy = igrow_utils.rand_within(n, 0.1, 0.5)
    kzz = kxx * vertical_anisotropy
    c0 = igrow_utils.rand_within(n, 1.0, 2.5)
    width = np.full(n, 100.0) * 0.5
    drain_width = igrow_utils.rand_within(n, 1.0, 3.0) * 0.5

    parameters = dict(
        igrow_width=width,
        domain_height=np.full(n, 3.0),
        drain_width=drain_width,
        drain_bottom=np.full(n, 1.5),
        drain_level=np.full(n, 2.0),
        recharge=np.full(n, 1.0),
        kxx=kxx,
        kzz=kzz,
        kxz=np.full(n, 0.0),
        c_drain=c0,
        c_seepage=c0,
        c_base=np.full(n, np.nan),
        H_base=np.full(n, np.nan),
        cell_width=width,
    )
    return parameters


def generate_peat_parameters(n):
    kxx = igrow_utils.rand_within(n, 0.5, 1.0)
    vertical_anisotropy = igrow_utils.rand_within(n, 0.1, 0.5)
    kzz = kxx * vertical_anisotropy
    c0 = igrow_utils.rand_within(n, 1.0, 2.5)
    domain_height = np.full(n, 1.6)
    drain_level = domain_height - 0.6
    drain_bottom = drain_level - 0.5
    width = np.full(n, 30.0) * 0.5
    drain_width = igrow_utils.rand_within(n, 1.0, 5.0) * 0.5

    parameters = dict(
        igrow_width=width,
        domain_height=domain_height,
        drain_width=drain_width,
        drain_bottom=drain_bottom,
        drain_level=drain_level,
        recharge=np.full(n, 1.0),
        kxx=kxx,
        kzz=kzz,
        kxz=np.full(n, 0.0),
        c_drain=c0,
        c_seepage=c0,
        c_base=np.full(n, np.nan),
        H_base=np.full(n, np.nan),
        cell_width=width,
    )
    return parameters


def generate_sand_parameters(n):
    kxx = igrow_utils.rand_within(n, 0.4, 3.0)
    vertical_anisotropy = igrow_utils.rand_within(n, 0.1, 0.5)
    kzz = kxx * vertical_anisotropy
    c0 = igrow_utils.rand_within(n, 1.0, 2.5)
    domain_height = igrow_utils.rand_within(n, 2.0, 15.0)
    drain_level = domain_height - 1.0
    drain_bottom = drain_level - 0.3
    width = np.full(n, 250.0) * 0.5
    drain_width = igrow_utils.rand_within(n, 1.0, 3.0) * 0.5

    parameters = dict(
        igrow_width=width,
        domain_height=domain_height,
        drain_width=drain_width,
        drain_bottom=drain_bottom,
        drain_level=drain_level,
        recharge=np.full(n, 1.0),
        kxx=kxx,
        kzz=kzz,
        kxz=np.full(n, 0.0),
        c_drain=c0,
        c_seepage=c0,
        c_base=np.full(n, np.nan),
        H_base=np.full(n, np.nan),
        cell_width=width,
    )
    return parameters


def generate_brook_parameters(n):
    kxx = igrow_utils.rand_within(n, 0.4, 3.0)
    vertical_anisotropy = igrow_utils.rand_within(n, 0.1, 0.5)
    kzz = kxx * vertical_anisotropy
    c0 = np.full(n, 0.1)
    domain_height = igrow_utils.rand_within(n, 5.0, 15.0)
    drain_level = domain_height - 1.0
    drain_bottom = drain_level - 2.0
    width = np.full(n, 1000.0) * 0.5
    drain_width = igrow_utils.rand_within(n, 10.0, 20.0) * 0.5

    parameters = dict(
        igrow_width=width,
        domain_height=domain_height,
        drain_width=drain_width,
        drain_bottom=drain_bottom,
        drain_level=drain_level,
        recharge=np.full(n, 1.0),
        kxx=kxx,
        kzz=kzz,
        kxz=np.full(n, 0.0),
        c_drain=c0,
        c_seepage=c0,
        c_base=np.full(n, np.nan),
        H_base=np.full(n, np.nan),
        cell_width=width,
    )
    return parameters


# %%


def clay_width25(base_parameters):
    n = base_parameters["igrow_width"].size
    parameters = igrow_utils.update_case(
        base_parameters, "cell_width", np.full(n, 12.5)
    )
    parameters = igrow_utils.update_case(parameters, "igrow_width", np.full(n, 62.5))
    return parameters


def peat_width25(base_parameters):
    n = base_parameters["igrow_width"].size
    parameters = igrow_utils.update_case(
        base_parameters, "cell_width", np.full(n, 12.5)
    )
    parameters = igrow_utils.update_case(parameters, "igrow_width", np.full(n, 12.5))
    return parameters


def sand_width25(base_parameters):
    n = base_parameters["igrow_width"].size
    parameters = igrow_utils.update_case(
        base_parameters, "cell_width", np.full(n, 12.5)
    )
    parameters = igrow_utils.update_case(parameters, "igrow_width", np.full(n, 112.5))
    return parameters


def brook_width25(base_parameters):
    n = base_parameters["igrow_width"].size
    parameters = igrow_utils.update_case(
        base_parameters, "cell_width", np.full(n, 12.5)
    )
    parameters = igrow_utils.update_case(parameters, "igrow_width", np.full(n, 512.5))
    return parameters


# %%


def multilayer(base_parameters):
    n = base_parameters["igrow_width"].size
    c_base = igrow_utils.rand_within(n, 10.0, 1000.0)
    H_base = base_parameters["drain_level"] + 0.1
    parameters = igrow_utils.update_case(base_parameters, "c_base", c_base)
    parameters = igrow_utils.update_case(parameters, "H_base", H_base)
    parameters["model_bottom"] = "open"
    return parameters


# %%

n = 2
cases = {
    "clay": generate_clay_parameters(n),
    "peat": generate_peat_parameters(n),
    "sand": generate_sand_parameters(n),
    "brook": generate_brook_parameters(n),
}
fine_cases = {
    "clay": clay_width25(cases["clay"]),
    "peat": peat_width25(cases["peat"]),
    "sand": sand_width25(cases["sand"]),
    "brook": brook_width25(cases["brook"]),
}
cases_multilayer = {k: multilayer(v) for k, v in cases.items()}
cases_fine_multilayer = {k: multilayer(v) for k, v in fine_cases.items()}

# %%
prefixes = (
    "singlelayer",
    "singlelayer-25m",
    "multilayer",
    "multilayer-25m",
)
# %%
all_cases = (
    cases,
    fine_cases,
    cases_multilayer,
    cases_fine_multilayer,
)
# %%
for prefix, case_collection in zip(prefixes, all_cases):
    dfs = []
    for name, case in case_collection.items():
        df = igrow_utils.generate_input(**case, label=f"{prefix}-{name}")
        dfs.append(df)

    df = pd.concat(dfs)
    path = Path("input") / f"{prefix}.csv"
    igrow_utils.write_igrow_input(df, path)


# A manual step is required. For every case:
#
# * Open iGrOw3.x.x
# * Upload one of the created CSV files.
# * Wait for the computation to finish.
# * Download table to CSV; use the case name, and append "-output.csv".
#
# Run "analyze-cases.py" next.
# %%
