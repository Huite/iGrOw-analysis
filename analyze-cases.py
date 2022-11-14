# %%
import postprocess
import igrow_utils

# %%

prefixes = (
    "singlelayer",
    "singlelayer-25m",
    "multilayer",
    "multilayer-25m",
)

results = {}
for prefix in prefixes:
    df = postprocess.extract_columns(
        igrow_utils.read_igrow_output(f"output/{prefix}-output.csv")
    )
    results[prefix] = postprocess.assign_analytical_resistance(df)

# %%
# Make some plots.

columns0 = [
    "c_modflow",
    "c_ernst",
    "c_ernst_B_eff",
    "c_de_lange_1997",
    "c_de_lange_2022",
    "c_ernst-c_vertical",
]
columns1 = [
    "c_horizontal",
    "c_vertical",
    "c_radial",
    "c_radial_B_eff",
    "c_entry",
    "c_entry_B_eff",
]

for prefix, all_df in results.items():
    for label, df in all_df.groupby("label"):
        postprocess.c_scatter_plots(
            df,
            title=f"{label}",
            path=f"images/{label}.png",
            columns0=columns0,
            columns1=columns1,
        )

# %%
