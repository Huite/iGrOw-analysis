import igrow_utils
import postprocess

# %%

df = postprocess.extract_columns(
    igrow_utils.read_igrow_output(f"output/singlelayer-output.csv")
)
df = postprocess.assign_analytical_resistance(df)

results = postprocess.finite_volume_heads(df, celldrain_columns=["c_entry", "c_iGrOw"])

# %%
# Columns0 end up in the first row of the plot
# Columns1 end up in the second row of the plot

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

postprocess.c_scatter_plots(
    df,
    title="singlelayer",
    path="images/singlayer.png",
    columns0=columns0,
    columns1=columns1,
)

# %%
