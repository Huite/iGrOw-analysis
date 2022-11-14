import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import analytical


def labda_B(kh, D, c0, c1):
    # Use c1 if available, if NaN, use c0 only.
    labda_B_c1 = np.sqrt((kh * D * c1 * c0) / (c1 + c0))
    labda_B_c0 = np.sqrt(kh * D * c0)
    return labda_B_c1.combine_first(labda_B_c0)


def extract_columns(results: pd.DataFrame) -> pd.DataFrame:
    extraction = pd.DataFrame()
    width = 2.0 * results["CELL width"]

    extraction["B"] = 2.0 * results["drain width (B/2)"]
    extraction["L"] = width - extraction["B"]
    extraction["kh"] = results["kxx"]
    extraction["kv"] = results["kzz"]
    extraction["D"] = results["domain height"]
    extraction["c0"] = results["c drain"]
    extraction["c1"] = results["c base"]
    lab_B = labda_B(
        kh=extraction["kh"],
        D=extraction["D"],
        c0=extraction["c0"],
        c1=extraction["c1"],
    )
    extraction["B_eff"] = analytical.effective_perimeter(extraction["B"], lab_B)
    extraction["c_iGrOw"] = results["CELL c_fleak"]
    extraction["label"] = results["comment"]

    # For head computation
    extraction["cell_width"] = results["CELL width"]
    extraction["drain_level"] = results["drain level"]
    extraction["h_base"] = results["H base"]
    extraction["recharge"] = results["recharge *1000"] / 1000.0
    extraction["iGrOw_width"] = results["iGrOw width (L/2)"]
    extraction["iGrOw_wet_width"] = results["Ow Wet Width"]
    extraction["iGrOw_h_mean"] = results["CELL H_mean"]
    extraction["iGrOw_h_max"] = results["H max"]
    extraction["iGrOw_recharge_total"] = results["Q tot recharge"]
    extraction["iGrOw_drain_total"] = results["Q tot drain"]
    extraction["iGrOw_base_total"] = results["Q tot base"]

    return extraction


def assign_analytical_resistance(results):
    results["c_modflow"] = analytical.c_modflow(
        L=results["L"],
        B=results["B"],
        c0=results["c0"],
    )
    results["c_horizontal"] = analytical.c_horizontal_multilayer(
        L=results["L"],
        D=results["D"],
        kh=results["kh"],
        c1=results["c1"].fillna(1.0e6),
    )
    results["c_vertical"] = analytical.c_vertical(
        D=results["D"],
        kv=results["kv"],
    )
    results["c_radial"] = analytical.c_radial(
        L=results["L"],
        B=results["B"],
        kh=results["kh"],
        kv=results["kv"],
        D=results["D"],
    )
    results["c_radial_B_eff"] = analytical.c_radial(
        L=results["L"],
        B=results["B_eff"],
        kh=results["kh"],
        kv=results["kv"],
        D=results["D"],
    )
    results["c_entry"] = analytical.c_entry(
        L=results["L"],
        B=results["B"],
        c0=results["c0"],
    )
    results["c_entry_B_eff"] = analytical.c_entry(
        L=results["L"],
        B=results["B"],
        c0=results["c0"],
    )
    results["c_ernst"] = analytical.c_ernst_multilayer(
        L=results["L"],
        B=results["B"],
        kh=results["kh"],
        kv=results["kv"],
        D=results["D"],
        c0=results["c0"],
        c1=results["c1"].fillna(1.0e6),
    )
    results["c_ernst-c_vertical"] = analytical.c_ernst_multilayer_no_vertical(
        L=results["L"],
        B=results["B"],
        kh=results["kh"],
        kv=results["kv"],
        D=results["D"],
        c0=results["c0"],
        c1=results["c1"].fillna(1.0e6),
    )
    results["c_ernst_B_eff"] = analytical.c_ernst_multilayer(
        L=results["L"],
        B=results["B_eff"],
        kh=results["kh"],
        kv=results["kv"],
        D=results["D"],
        c0=results["c0"],
        c1=results["c1"].fillna(1.0e6),
    )
    results["c_de_lange_1997"] = analytical.c_de_lange_1997(
        L=results["L"],
        B=results["B"],
        kh=results["kh"],
        kv=results["kv"],
        D=results["D"],
        c0=results["c0"],
        c1=results["c1"].fillna(1.0e6),
    )
    results["c_de_lange_2022"] = analytical.c_de_lange_2022(
        L=results["L"],
        B=results["B"],
        kh=results["kh"],
        kv=results["kv"],
        D=results["D"],
        c0=results["c0"],
        c1=results["c1"].fillna(1.0e6),
    )
    return results


def mae(x, y):
    """
    Mean absolute error.
    """
    error = np.abs(y - x)
    return np.mean(error)


def R2(x, y):
    residual_sum_of_squares = np.sum((y - x) ** 2)
    total_sum_of_squares = np.sum((y - np.mean(x)) ** 2)
    return 1.0 - (residual_sum_of_squares / total_sum_of_squares)


def c_scatter_plots(results, path, title, columns0, columns1):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ncol = max(len(columns0), len(columns1))
    fig, axes = plt.subplots(2, ncol, figsize=(5 * ncol, 5 * 2))
    fig.suptitle(title)

    for i, columns in enumerate([columns0, columns1]):
        for j, column in enumerate(columns):
            ax = axes[i, j]
            results.plot.scatter(x="c_iGrOw", y=column, ax=ax, alpha=0.2)
            column_mae = mae(results["c_iGrOw"], results[column])
            column_r2 = R2(results["c_iGrOw"], results[column])
            ax.set_title(f"MAE:{column_mae:.0f} \n" f"R2: {column_r2:.3f}")

    a = results["c_iGrOw"].min()
    b = results["c_iGrOw"].max()
    all_c = results[columns0 + columns1].values
    y0 = all_c.min()
    y1 = all_c.max()
    for ax in axes.ravel():
        ax.axline((a, a), (b, b), color="r")
        ax.set_ylim((y0, y1))
        ax.grid()

    fig.tight_layout()
    fig.savefig(f"{path}", dpi=300, bbox_inches="tight")
    return


def is_divisor(a, b):
    """
    Test whether b is a whole divisor of a, taking finite precision into
    account.
    """
    remainder = a % b
    return np.allclose(remainder, 0.0) or np.allclose(remainder, b)


def compute_ncell(domain_width, cell_width):
    """
    Compute how many cells are required for the domain, given the desired cell
    width.
    """
    whole_cells = domain_width - cell_width
    if not is_divisor(whole_cells, cell_width):
        raise ValueError(
            "cell width is not a whole divisor of domain width for all cases."
        )
    dx = 2.0 * cell_width
    # Mirror around center, and add river cell.
    ncell = (whole_cells / dx * 2 + 1).astype(int)
    return ncell


def finite_volume_model(
    domain_width,
    domain_height,
    drain_level,
    recharge,
    kxx,
    c_celldrain,
    cell_width,
    c_base=None,
    h_base=None,
    wet_width=None,
):
    """
    This function implements a single layer finite volume model (like MODFLOW).
    For efficiency, it solves all cases in one go. However, this means the
    cases must have the same number of cells. Domain_width and cell_width are
    required to result in the same number of cells; it will error otherwise.
    The head of every cell is returned for every case.

    If this is not the case, split the cases beforehand into categories with an
    equal number of cells.

    Note that while iGrOw computes a symmetric profile, this is inconvenient
    for this finite volume model. Instead, the domain is mirrored so that the
    drain is found exactly in the middle.

    Parameters
    ----------
    domain_width: np.ndarray of floats with shape (N,)
    domain_height: np.ndarray of floats with shape (N,)
    drain_level: np.ndarray of floats with shape (N,)
    recharge: np.ndarray of floats with shape (N,)
    kxx: np.ndarray of floats with shape (N,)
    c_celldrain: np.ndarray of floats with shape (N,)
    cell_width: np.ndarray of floats with shape (N,)
    c_base: optional, np.ndarray of floats with shape (N,)
    h_base: optional, np.ndarray of floats with shape (N,)

    Returns
    -------
    head: np.ndarray with shape (N, ncell)
    """
    ncell = compute_ncell(domain_width, cell_width)
    unique = np.unique(ncell)
    if unique.size > 1:
        raise ValueError(
            "Can only simulate multiple flow situations at once with the same "
            "number of cells."
        )
    ncell = unique[0]

    if ncell % 2 == 0:
        raise ValueError(
            "ncell should always be uneven: adjust domain width or cell width."
        )
    n = domain_width.size
    dx = cell_width * 2.0

    river_cell = int(ncell / 2)
    conductance = kxx * domain_height / dx

    # Setup numerical solution
    A = np.zeros((n, ncell, ncell))
    # formulate recharge
    rch = np.multiply.outer(-recharge * dx, np.ones(ncell))
    if wet_width is not None:
        fraction = 1.0 - (wet_width / cell_width)
        if not (fraction > 0.0).all():
            raise ValueError("fraction wet must exceed 0.0")
        if not (fraction < 1.0).all():
            raise ValueError("fraction wet exceeds 1.0")
        rch[:, river_cell] *= fraction

    b = rch.copy()

    # Only set cell to cell flow if there are 3 or more cells.
    # Two cells aren't allowed, should be caught by check above (as well as any
    # other even number.)
    if ncell == 1:
        i = [0]
        j = [0]
    else:
        # Set corner diagonal values
        A[:, 0, 0] = -conductance
        A[:, -1, -1] = -conductance
        # Set inner diagonal values
        i, j = np.diag_indices(ncell)
        inner_i = i[1:-1]
        inner_j = i[1:-1]
        A[:, inner_i, inner_j] = np.repeat(-2 * conductance, ncell - 2).reshape((n, -1))
        # Set off-diagonal values
        above_i = i[:-1]
        above_j = j[1:]
        below_i = i[1:]
        below_j = j[:-1]
        A[:, above_i, above_j] = np.repeat(conductance, ncell - 1).reshape((n, -1))
        A[:, below_i, below_j] = np.repeat(conductance, ncell - 1).reshape((n, -1))

    # Formulate river boundary
    drain_conductance = dx / c_celldrain
    b[:, river_cell] -= drain_conductance * drain_level
    A[:, river_cell, river_cell] -= drain_conductance

    # Formulate lower aquifer boundary condition
    if c_base is not None and h_base is not None:
        no_base = ~(np.isfinite(c_base) & np.isfinite(h_base))
        base_cond = dx / c_base
        base_hcof = np.multiply.outer(-base_cond, np.ones(ncell))
        base_b = base_hcof * h_base[:, np.newaxis]
        base_hcof[no_base] = 0.0
        base_b[no_base] = 0.0
        A[:, i, j] += base_hcof.reshape((n, -1))
        b[:, :] += base_b.reshape((n, -1))

    # Solve the system of equations; linalg.inv can deal with "stacked" matrices.
    head = np.linalg.solve(A, b)
    return head, river_cell


def finite_volume_heads(results, celldrain_columns):
    """
    Compute a finite volume model for every cell drain value, and append
    the columns of the mean and maximum heads.

    Parameters
    ----------
    results: pd.DataFrame
    celldrain_columns: list of strings
        Which columns to use as a celldrain resistance.

    Returns
    -------
    results: pd.DataFrame
        Results with a column with a computed mean head attached for every
        celldrain column.
    """

    def fv_model(results):
        for column in celldrain_columns:
            head, river_cell = finite_volume_model(
                domain_width=results["iGrOw_width"].values,
                cell_width=results["cell_width"].values,
                kxx=results["kh"].values,
                domain_height=results["D"].values,
                recharge=results["recharge"].values,
                c_celldrain=results[column].values,
                drain_level=results["drain_level"].values,
                c_base=results["c1"].values,
                h_base=results["h_base"].values,
                wet_width=results["iGrOw_wet_width"].values,
            )
            # Express all levels relative to drain level.
            results[f"center_head_{column}"] = (
                head[:, river_cell] - results["drain_level"]
            )
            results[f"mean_head_{column}"] = head.mean(axis=1) - results["drain_level"]
            results[f"max_head_{column}"] = head.max(axis=1) - results["drain_level"]
            results[f"drain_budget_{column}"] = (
                results[f"center_head_{column}"]
                / results[column]
                * results["cell_width"]
            )
        return results

    results = results.copy()
    # Compute the number of required finite volume cells.
    domain_width = results["iGrOw_width"].values
    cell_width = results["cell_width"].values
    results["ncell"] = compute_ncell(domain_width, cell_width)
    results["center_head_iGrOw"] = results["iGrOw_h_mean"] - results["drain_level"]
    results["max_head_iGrOw"] = results["iGrOw_h_max"] - results["drain_level"]
    return results.groupby("ncell", group_keys=False).apply(fv_model)
