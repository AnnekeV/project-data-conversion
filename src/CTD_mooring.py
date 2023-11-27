# %% FUNCTIONS

from datetime import datetime, timedelta
from pathlib import Path
import ctd
from matplotlib import dates
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pathlib
import os
from os import listdir

plt.style.use("ggplot")
pd.options.plotting.backend = "matplotlib"

# %% define paths

path_parent = Path.cwd().parent.parent
figpath = path_parent.joinpath(Path("Figures"))
fname = "/Users/annek/Library/CloudStorage/OneDrive-SharedLibraries-NIOZ/PhD Anneke Vries - General/Data/Moorings/20190612_SBE_GF10/20180828_SBE2989_5m.cnv"

fpath = str(path_parent) + "/Data/Moorings/20190612_SBE_GF10/"
fpath_GF13 = str(path_parent.joinpath("Data", "Moorings", "20190802_SBE_GF13")) + "/"
fpath_GF10 = fpath


# define paths
plt.style.use("ggplot")
pd.options.plotting.backend = "matplotlib"
try:
    path_parent = pathlib.Path(__file__).parent.parent.resolve()
except NameError:
    path_parent = Path.cwd().parent.resolve()
path_data = path_parent.joinpath("data", "raw", "Moorings", "20190612_SBE_GF10")

fileNames = [
    f for f in listdir(path_data) if (f.endswith(".cnv"))
]
print(fileNames)

# %% FUNCTIONS


def open_cnv(fname, remove_5m=True):
    """Open cnv file and export dataframe down and metadata and cast"""
    cast = ctd.from_cnv(fname)  #
    cast = rename_variables(cast, remove_5m)
    down, up = cast.split()
    metadata = cast._metadata
    return down, metadata, cast


def rename_variables(df, remove_5m):
    """renames the variables as in cnv to shorter and equal names  AND REMOVES OUTLIERS"""
    df = df.rename(
        columns={
            "potemp090C": "temp",
            "sal00": "sal",
            "density00": "dens",
            "timeJV2": "time",
            "timeJ": "time",
            "sigma-t00": "sigma-dens",
            "sigma-È00": "sigma-dens",
            "sigma-�00": "sigma-dens",
        }
    )
    df = df.reset_index()
    df = remove_above_zero(df)
    if remove_5m:
        df = remove_outliers_5m(df)
        df = remove_outliers_5m(df)  # twice for the best result

    return df


def remove_above_zero(cast):
    """remove all vales lower than 0.1 dbar"""
    rows_before = len(cast.index)
    start = (cast["Pressure [dbar]"] > 0.15).idxmax()
    cast = cast.iloc[start:]
    cast = cast[cast["Pressure [dbar]"] > 0.15].reset_index()
    print(f"Nr. rows below water surface (dbar<0.15):{rows_before-len(cast.index)}")
    return cast


def remove_outliers_5m(cast, nr_per_hour=6):
    """Removes outliers based on temp and sal by finding a x times std over 4 weeks
    nr_per_hour  = nr obs per hour, default every 10 min, is 6 times per hour,
    with built in safety so you don't do it for other depth than 5 m"""
    if cast["Pressure [dbar]"].mean() < 10:
        df5 = cast[["temp", "sal"]]
        window = nr_per_hour * 24 * 28  # 4 weeks
        max_z = 3  # max std dev
        z_scores_alt = (
            df5 - df5.rolling(window, center=True, min_periods=window // 2).mean()
        ) / df5.rolling(window, center=True, min_periods=window // 2).std()
        abs_z_scores = z_scores_alt.abs()
        filtered_entries = (abs_z_scores < max_z).all(axis=1)
        new_df = df5[filtered_entries]
        print(
            f"{len(df5) - len(new_df)} values removed, {(len(df5) - len(new_df))/len(new_df):.3f} %"
        )
        return cast[filtered_entries]
    else:
        return cast


def remove_outliers(cast, nr_per_hour=6):
    """Removes outliers based on temp and sal by finding a x times std over 4 weeks
    nr_per_hour  = nr obs per hour, default every 10 min, is 6 times per hour"""
    df5 = cast[["temp", "sal"]]
    window = nr_per_hour * 24 * 28  # 4 weeks
    max_z = 3  # max std dev
    z_scores_alt = (
        df5 - df5.rolling(window, center=True, min_periods=window // 2).mean()
    ) / df5.rolling(window, center=True, min_periods=window // 2).std()
    abs_z_scores = z_scores_alt.abs()
    filtered_entries = (abs_z_scores < max_z).all(axis=1)
    new_df = df5[filtered_entries]
    print(
        f"{len(df5) - len(new_df)} values removed, {(len(df5) - len(new_df))/len(new_df):.3f} %"
    )
    return cast[filtered_entries]


def time_to_date(fname, cast, start_time="Jan 1 2018"):
    """Changes nr of Julian days to datetime object
    Insert dataframe with at least one column 'time'  and get adjusted dataframe back
    """

    dt_start_time = datetime.strptime(start_time, "%b %d %Y")
    cast["timedelta"] = cast.time.apply(lambda x: timedelta(x))
    cast["date"] = cast["timedelta"] + dt_start_time


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# %%

labels = {
    "temp": "Potential temperature [C]",
    "sal": "Salinity [PSU]",
    "month": "Month",
    "depth": "Depth [m]",
    "2018.0": "",
    "2019.0": "",
}

# %%

fname = "/Users/annek/Library/CloudStorage/OneDrive-SharedLibraries-NIOZ/PhD Anneke Vries - General/Data/Moorings/20190612_SBE_GF10/20190612_SBE5968_540m.cnv"
# fname = fpath_GF13+"20190802_SBE16523_20m.cnv"


start_time = "Jan 1 2018"


fig, axes = plt.subplots(1, 3, sharex=True, figsize=[20, 10])

down, metadata, cast = open_cnv(fname, remove_5m=False)
time_to_date(fname, cast, start_time)


y = "date"
cast.plot.scatter(y="temp", x=y, ax=axes[1], color="red")
cast.plot.scatter(y="sal", x=y, ax=axes[2], color="red")
cast.plot.scatter(y="dens", x=y, ax=axes[0], color="red")


# cast = remove_outliers_5m(cast)
# cast = remove_outliers_5m(cast)

cast.plot.scatter(y="temp", x=y, ax=axes[1], title="Temperature")
cast.plot.scatter(y="sal", x=y, ax=axes[2], title="Salinity")
cast.plot.scatter(y="dens", x=y, ax=axes[0], title="Density")
cast.plot.scatter(y="Pressure [dbar]", x=y, title="Pressure")


# cast = cast.reset_index(d)
plt.show()

cast.plot(x="date", y="Pressure [dbar]")

# %%
# Diffusion for mooring at 540 m
from sklearn.linear_model import LinearRegression

pd.options.plotting.backend = "matplotlib"


down, metadata, cast_540 = open_cnv(f"{fpath_GF10}20190612_SBE5968_540m.cnv", False)
cast_diffusion = cast_540.set_index("time")[:470].reset_index()
X = cast_diffusion["time"].values.reshape(
    -1, 1
)  # values converts it into a numpy array
Y = cast_diffusion["dens"].values.reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column
Z = cast_diffusion["Pressure [dbar]"].values.reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column


def lin_reg(X, Y):
    """Linear regression for Array X and Y
    Creates plot also"""
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color="blue")

    r_sq = linear_regressor.score(X, Y)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {linear_regressor.intercept_}")
    print(f"slope: {linear_regressor.coef_}")
    return np.squeeze(linear_regressor.coef_)


slope = lin_reg(X, Y)
print(f"slope: {np.squeeze(slope)/(24*3600)} kg/m3/s")

plt.xlabel("Days since 1-1-2018")
plt.ylabel("Density [kg/m3 ]")
plt.title("Diffusion at 540 m")
plt.show()

sc = plt.scatter(Z, Y - 1000, c=X, s=1)
cax = plt.colorbar(sc)
cax.set_label("Time [days since 1-1-2018]")
plt.xlabel("Pressure [dbar]")
plt.ylabel("Density [kg/m3]")
plt.title("Mooring showing vertical diffusion at depth")
plt.tight_layout()
plt.savefig(f"{figpath}/Vertical_diffusion_from_microcat.png")
plt.show()

# %%
pd.options.plotting.backend = "plotly"

cast_540["tidal_amp"] = cast_540["Pressure [dbar]"] - cast_540["Pressure [dbar]"].mean()
cast_540["spring_neap"] = (
    cast_540["tidal_amp"].rolling(center=True, window=6 * 25).max()
)
fig = cast_540.set_index("date", drop=True)["spring_neap"].plot()
fig.update_yaxes(title_text="Tidal amplitude [dbar]")
# remove legend
fig.update_layout(showlegend=False)
# save as html
fig.write_html(f"{figpath}/Tidal_amplitude.html")

# %% Grid
# Grid
pd.options.plotting.backend = "matplotlib"


variables = ["temp", "sal", "sigma-dens", "Pressure [dbar]"]
mooring = "GF13"


if mooring == "GF10":
    # GF10 microcat files
    files = [
        "20190612_SBE5968_540m.cnv",
        "20190612_SBE5969_330m.cnv",
        "20190612_SBE5970_150m.cnv",
        "20190612_SBE5971_60m.cnv",
        "20190821_SBE4247_5m.cnv",
        "20180828_SBE2989_5m.cnv",
    ]
    files = files[::-1]
    fpath = fpath_GF10
elif mooring == "GF13":
    # GF13 microcat files
    files = [
        "20190802_SBE16523_20m.cnv",
        "20190802_SBE15390_60m.cnv",
        "20190802_SBE15399_120m.cnv",
    ]
    fpath = fpath_GF13
else:
    files = ""

nr_moorings = len(files)
fig, axes = plt.subplots(nr_moorings, len(variables), sharex=True, figsize=[20, 10])
for i in range(nr_moorings):
    cast, metadata, cast = open_cnv(fpath + files[i])
    if files[i] == "20180828_SBE2989_5m.cnv":
        start_time = "Jan 1 2017"
    else:
        start_time = "Jan 1 2018"
    time_to_date(fpath + files[i], cast, start_time)
    for j in range(len(variables)):
        title = files[i].split("_")[2].split(".")[0] + " " + variables[j]
        ax = axes[i, j]
        cast = cast.set_index("date")
        cast = cast.rolling(6 * 50).mean()
        cast = cast.reset_index()
        cast.plot.line(
            x="date",
            y=variables[j],
            ax=ax,
            # title = title,
            legend=True,
            label=title,
            # s=.5
        )
        # remove_outliers(cast).plot.scatter(x='date',y=variables[j], ax=ax )
        ax.xaxis.set_major_formatter(
            dates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        # ax.set_ylim([0,10])
# plt.xlim(datetime(2018, 10, 1), datetime(2018, 12, 1))
plt.tight_layout()

# plt.savefig(f"{figpath}/grid_50_hr_rolling_mean_{mooring}.png")


# %% Plot difference compared to mean pressure
pd.options.plotting.backend = "matplotlib"

ax = plt.figure()
for i in range(nr_moorings):
    cast, metadata, cast = open_cnv(fpath + files[i])
    time_to_date(fpath + files[i], cast, start_time)
    ax = (
        cast.groupby(cast.date.dt.date)
        .mean()["Pressure [dbar]"]
        .apply(lambda x: x - cast["Pressure [dbar]"].mean())
        .iloc[:-1]
        .plot(label=files[i])
    )
ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
plt.legend()
plt.ylabel("Pressure [dbar]")
plt.title("Relative pressure difference per day compared to mean")
plt.show()


ax2 = plt.figure()
for i in range(nr_moorings):
    cast, metadata, cast = open_cnv(fpath + files[i])
    time_to_date(fpath + files[i], cast, start_time)
    ax2 = (
        cast["Pressure [dbar]"]
        .apply(lambda x: x - cast["Pressure [dbar]"].mean())
        .iloc[:-1]
        .plot(label=files[i].split("_")[2].split(".")[0])
    )
    max_tide = (
        cast["Pressure [dbar]"]
        .apply(lambda x: x - cast["Pressure [dbar]"].mean())
        .abs()
        .max()
    )
    print(f"Max pressure is {max_tide} m")

ax2.xaxis.set_major_formatter(dates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
plt.legend()
plt.ylabel("Pressure [dbar]")
plt.title("Relative pressure difference compared to mean")
plt.show()

# %% Plot T/S


fig, ax = plt.subplots(1, 1, sharex=True, figsize=[10, 10])
for i in range(nr_moorings):
    title = files[i].split("_")[2].split(".")[0] + " "
    cast, metadata, cast = open_cnv(fpath + files[i])
    time_to_date(fpath + files[i], cast)
    # cast.groupby(cast.date.dt.date).mean().plot(x='sal',y='temp', ax=ax,
    #             legend=True,
    #             label = "avg date " + title,
    # )
    cast.rolling(6 * 24 * 7).mean().plot(
        x="sal",
        y="temp",
        ax=ax,
        legend=True,
        label=title,
    )
plt.ylabel("Temperature")
plt.xlabel("Salinity")
plt.title("Weekly rolling average")
plt.tight_layout()


# %%   Group all files together
# Open monthly data

all_moorings = pd.DataFrame()

for i in range(nr_moorings):
    depth = files[i].split("_")[2].split("m.")[0]
    cast, metadata, cast = open_cnv(fpath + files[i])
    if files[i] == "20180828_SBE2989_5m.cnv":
        start_time = "Jan 1 2017"
    else:
        start_time = "Jan 1 2018"
    time_to_date(fpath + files[i], cast, start_time)
    cast["depth"] = depth
    all_moorings = pd.concat([all_moorings, cast]).reset_index(drop=True)
    print(i)


# all_moorings.to_csv(f"{path_parent.joinpath('Processing')}/intermediate_files/mooring_gf13.csv")


df_monthly = pd.read_csv(
    Path.cwd().parent.joinpath("intermediate_files", "monthly_18_19_gf10.csv"),
    index_col=0,
    parse_dates=True,
)
all_moorings["id"] = all_moorings["depth"]
df_combi = pd.concat([df_monthly, all_moorings])


# %% 3D plot
# fig = px.line_3d(df_combi,
#     x="temp", y="date",
#     z="Pressure [dbar]", color ="id")


# fig.update_layout(scene = dict(
#                     xaxis_title='Potential Temperature (C)'),
#                     width=700,
#                     margin=dict(r=20, b=10, l=10, t=10))

# # fig.write_html(f"{figpath}/3d.html")
# fig.show()

# #%% 2D plot
# # fig = px.line_3d(df_combi,
# #     x="temp", y="date",
# #     z="Pressure [dbar]", color ="id")


# # fig.update_layout(scene = dict(
# #                     xaxis_title='Potential Temperature (C)'),
# #                     width=700,
# #                     margin=dict(r=20, b=10, l=10, t=10))

# # # fig.write_html(f"{figpath}/3d.html")
# # fig.show()

# Z = pd.DataFrame(data = df_combi)
# Z.Type[(Z['Type'] != 'CTD')]
# Z = Z.set_index('date')


# fig = go.Figure(data =
#     go.Contour(
#         z=df_combi['temp'].rolling(window=50 #pd.Timedelta(hours=50)
#         , center=True, min_periods=12 ).mean(),
#         x=df_combi['date'],
#         y=df_combi['Pressure [dbar]'],
#         # line_smoothing=0,
#         # connectgaps=True,
#         # colorscale='PRGn',
#         # colorbar=dict(
#         #     title='Velocity (m/s)',
#         #     titleside ='right'),
#         # contours=dict(
#         #     coloring='fill',
#         #     showlines=False,
#         #     start=-0.15,
#         #     end=0.15,
#         #     size=.05,
#         #     ),
#         ),
#     layout=go.Layout(
#         # title=go.layout.Title(text=f"Across fjord velocity + north {hours} hrs",
#         # x=0.5),
#         yaxis= dict(autorange="reversed")
#     )    )

# fig.show()

# %%

from turtle import color

d = 60
depths = all_moorings.id.unique()
vari = ["temp", "sal", "sigma-dens"]
if nr_moorings == 6:
    nr_moorings -= 1
fig, axes = plt.subplots(nr_moorings, 3, sharex=True, figsize=[40, 30])

for i in range(3):
    for j in range(nr_moorings):
        d = int(depths[j])
        var = vari[i]
        ax = axes[j, i]
        month_mask = (df_monthly["Pressure [dbar]"] > d - 1) & (
            df_monthly["Pressure [dbar]"] < d + 1
        )
        all_moorings[all_moorings.id == str(d)].plot.scatter(
            x="date", y=var, ax=ax, color="red", s=0.5, label="mooring"
        )
        df_monthly[month_mask].plot(
            x="date", y=var, ax=ax, color="darkred", label="monthly"
        )
        df_monthly[month_mask].plot.scatter(
            x="date", y=var, ax=ax, color="black", label=None
        )
        ax.set_title(f"{var} {d}m")
plt.tight_layout()
fig.show()


# %% SCATTER 3D

# fig = px.scatter_3d(df_combi, x="temp", y="date", z="Pressure [dbar]", color ="id",size_max=0.1)
# fig.update_yaxes(autorange="reversed")
# # tight layout
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# fig.update_traces(marker=dict(size=3))
# fig.show()

# %%
df_monthly[
    df_monthly["Pressure [dbar]"]
    == min(df_monthly["Pressure [dbar]"], key=lambda x: abs(x - d))
]


# %% Frequency analysis ================================================

from scipy import signal


var_freq = "temp"

x = all_moorings[all_moorings.depth == "60"][var_freq]
fs = 6  # per hour
f, Pxx_den = signal.periodogram(x, fs)

mask = Pxx_den > 1


plt.semilogy(f, Pxx_den)
plt.xlabel("frequency [1/hr]")
plt.ylabel("PSD [V**2/Hz]")
plt.show()
plt.plot(x)
plt.show()

plt.semilogy(1 / f[1:], Pxx_den[1:])
plt.xlabel("Period [hr]")
plt.ylabel("PSD [V**2/Hz]")
plt.xlim([0, 50])
plt.show()

plt.bar(1 / f[mask], Pxx_den[mask])
plt.xlabel("Period [hr]")
plt.ylabel("PSD [V**2/Hz]")
plt.xlim([0, 50])
plt.show()

mask = Pxx_den[1:] > 1

df = pd.DataFrame(data={"f": f[1:], "P": 1 / f[1:], "Pxx_den": Pxx_den[1:]})
fig = px.scatter(df[mask], x="P", y="Pxx_den", log_y=True)
fig.update_xaxes(range=[0, 50])
fig.show()


maski = df[mask][
    df[mask].rolling(3, center=True).max().Pxx_den == df[mask].Pxx_den
].index

fig = go.Figure()
df.P = df.P / 24
fig.add_trace(
    go.Scatter(
        x=df.P,
        y=df.Pxx_den,
        fill="tozeroy",
        mode="none",
    )
)
dfi = df.loc[maski]
fig.add_trace(
    go.Scatter(
        x=dfi["P"],
        y=dfi["Pxx_den"],
        text=dfi["P"].round(1),
        textposition="top right",
        # mode ='markers+text',
        mode="markers",
    )
)
fig.update_xaxes(range=[0, 100])
# fig.update_traces(marker_size=1000)
fig.update_yaxes(type="log", range=[-6, 6])
fig.update_layout(
    title=var_freq,
    showlegend=False,
    xaxis_title="Period [d]",
    yaxis_title="Power density",
)

# fig.write_image(f"{figpath}/temp/spectral_analysis_{var_freq}.png")
fig.show()


# %% TS scatter ================================================
pd.options.plotting.backend = "plotly"


all_moorings["year"] = all_moorings.date.apply(lambda x: int(x.year))
all_moorings["day"] = all_moorings.date.apply(lambda x: x.day)
all_moorings["month"] = all_moorings.date.apply(
    lambda x: x.strftime("%B") + " "
) + all_moorings["year"].astype(str)


n_colors = len(all_moorings["month"].unique())
colors = px.colors.sample_colorscale(
    "plasma", [n / (n_colors - 1) for n in range(n_colors)]
)


# .sort_values([ "date"]).groupby(['depth','month', 'day'],as_index=False, sort=False).mean()
# mask = all_moorings["depth"]=='60'
# mask = :
fig = (
    all_moorings.sort_values(["date"])
    .groupby(["depth", "month", "day"], as_index=False, sort=False)
    .mean()
    .plot.scatter(
        x="sal",
        y="temp",
        # ax=ax,
        color="month",
        symbol="year",
        # legend=True,
        # label =  title,
        labels=labels,
        facet_col="depth",
        facet_col_wrap=3,
        color_discrete_sequence=colors,
        # px.colors.cyclical.Phase,
        width=1400,
        height=800,
    )
)

# fig.update_traces(marker=dict(size=2))
fig.update_yaxes(matches=None, showticklabels=True)
fig.update_xaxes(matches=None, showticklabels=True)
fig.update_layout(margin=dict(r=20, b=10, l=10, t=20))
# fig.write_image(f"{figpath}/temp/T-S_per_month_averaged.svg")
# fig.write_html(f"{figpath}/temp/T-S_per_month_averaged.html")
fig.show()


# %%
df_TS = (
    all_moorings.sort_values(["date"])
    .groupby(["depth", "month", "day"], as_index=False, sort=False)
    .mean()
)


# %% ==================================
# Combination of mooring and monthly
# Interpolation of nan values in between
#  ==================================


# %%   Group all files together
# Open monthly data

resample_interval = "1H"
start_time = "Jan 1 2018"

all_moorings_hourly = pd.DataFrame()
nr_moorings = 6

for i in range(nr_moorings):
    depth = files[i].split("_")[2].split("m.")[0]
    cast, metadata, cast = open_cnv(fpath + files[i])
    if files[i] == "20180828_SBE2989_5m.cnv":
        cast.time = cast.time - 365
    time_to_date(fpath + files[i], cast, start_time)
    cast = (
        cast.set_index("date")
        .resample(resample_interval)
        .mean()
        .reset_index()
        .round({"Pressure [dbar]": 2})
    )
    cast["depth"] = cast["Pressure [dbar]"].mean()
    all_moorings_hourly = pd.concat([all_moorings_hourly, cast]).reset_index(drop=True)


all_moorings_hourly.to_csv(
    f"{path_parent}/Processing/intermediate_files/mooring_gf10_{resample_interval}hourly.csv"
)


df18 = pd.read_csv(f"{path_parent}/Processing/intermediate_files/monthly_2018_gf10.csv")
df19 = pd.read_csv(f"{path_parent}/Processing/intermediate_files/monthly_2019_gf10.csv")
df19["date"] = df19.time.apply(lambda x: timedelta(days=x)) + datetime(2019, 1, 1)
df19["time"] = (
    df19.time.apply(lambda x: timedelta(days=x)) + timedelta(days=365)
) / np.timedelta64(1, "D")
df18.date = df18.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"), "broadcast")
df18["date"] = df18.time.apply(lambda x: timedelta(days=x)) + datetime(2018, 1, 1)

df_monthly = pd.concat([df18, df19]).reset_index(drop=True)
df_monthly["id"] = df_monthly["Name"].copy()


all_moorings_hourly["Type"] = "Mooring"
all_moorings_hourly = all_moorings_hourly[all_moorings_hourly["Pressure [dbar]"] > 0.1]
all_moorings_hourly["id"] = all_moorings_hourly["depth"]
# all_moorings_hourly["Pressure [dbar]"] = all_moorings_hourly["depth"].astype("int")


df_combi_hr = pd.concat([df_monthly, all_moorings_hourly])

df_combi_hr = df_combi_hr.dropna(subset=["Pressure [dbar]", "time"], how="any")

df_mi = df_combi_hr.set_index(["Pressure [dbar]", "time"]).drop_duplicates(keep="first")
# da_sal = df_mi["sal"].to_xarray()

# ## Intermediate step to look at the data
# df_piv = df_combi_hr.pivot(columns='Pressure [dbar]',index='time', values='sal')
# df_piv.iloc[:2000,:].to_csv("intermediate_files/multi_index_sal_test.csv")

# %%  preparations
from scipy.interpolate import griddata
import cmocean

var = "dens"


if var == "sal":
    levels_f = np.concatenate(
        (
            [10, 15, 20, 25, 30, 31],
            np.arange(32, 33.5, 0.2),
            np.arange(33.5, 33.6, 0.05),
        )
    )
    cmap = cmocean.cm.haline
    variable = "Salinity [PSU]"

elif var == "temp":
    levels_f = np.concatenate(
        (np.arange(-1.5, 0.5, 0.5), np.arange(0.5, 2.5, 0.25), np.arange(2.5, 5.5, 0.5))
    )
    cmap = cmocean.cm.thermal
    variable = "Potential temperature [C]"


else:
    levels_f = np.concatenate((np.arange(1005, 1025, 5), np.arange(1025, 1030, 1)))
    levels_f -= 1000
    cmap = cmocean.cm.dense
    var = "dens"
    variable = "Density"

# cmap = matplotlib.cm.get_cmap('gist_rainbow')

levels_f = np.sort(np.concatenate([levels_f, (levels_f[1:] + levels_f[:-1]) / 2]))
levels_f = np.sort(np.concatenate([levels_f, (levels_f[1:] + levels_f[:-1]) / 2]))

levels_l = levels_f[::4]
norm = matplotlib.colors.BoundaryNorm(levels_f, cmap.N, extend="both")


df_short = (
    df_mi.reset_index()
    .set_index("date")["2018-01-01":"2020-01-01"]
    .reset_index()
    .set_index(["Pressure [dbar]", "time"])
)
df_short.dens = df_short.dens - 1000
# df_short = df_mi

x_dates = df_short.date

long_px = df_short[var].reset_index().time
long_py = df_short[var].reset_index()["Pressure [dbar]"]
y_depths = df_short.reset_index().depth.fillna(
    long_py
)  # data series with all the depths with mounting depth for
long_p_para = df_short[var].reset_index()[var].values  # parameter


npts = 200
pts_rand = np.random.choice(len(long_px), npts)
px, py, p_para = long_px[pts_rand], long_py[pts_rand], long_p_para[pts_rand]

# px,py, p_para = long_px, long_py, long_p_para
plt.clf()
plt.scatter(px, py, c=p_para, cmap=cmap, norm=norm)
plt.colorbar()
plt.ylim([550, 0])
plt.show()

# points = np.transpose(np.squeeze([px.values, py.values]))
points = (px, py)
X, Y = np.meshgrid(px, py)
Ti = griddata(points, p_para, (X, Y), "linear")

# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[20,16])
# # Plot the model function and the randomly selected sample points
# ax[0,0].scatter(px,py, c=p_para, cmap=cmap, norm=norm,s=150)
# ax[0,0].set_title('Sample points ')
# ax[0,0].set_ylim([550,0])


# # Interpolate using three different methods and plot
# for i, method in enumerate(('nearest', 'linear', 'cubic')):
#     Ti = griddata((px, py), p_para, (X, Y), method=method)
#     r, c = (i+1) // 2, (i+1) % 2
#     ax[r,c].contourf(X, Y, Ti, cmap=cmap, norm= norm, levels=levels_f,)
#     ax[r,c].set_title("method = '{}'".format(method))
#     ax[r,c].set_ylim([500,0])

# plt.tight_layout()
# fig.savefig(f"{figpath}/temp/interpolated_example.png")


# %%  interpolating the actual data

X, Y = np.meshgrid(
    np.arange(round(min(long_px)), max(long_px), 1), np.arange(2, 553, 1)
)
Z = griddata((long_px.values, long_py.values), long_p_para, (X, Y), method="linear")
X = X * timedelta(days=1) + datetime(2018, 1, 1)  # make from decimal days datetime

# removing part in order not to over represent
# %% exporting

df_interpolated = pd.DataFrame(index=X[0, :], columns=Y[:, 0], data=Z.T)
df_interpolated.to_csv(
    path_parent.joinpath(
        "Processing",
        "intermediate_files",
        "interpolated_mooring_profile_2018_2019",
        f"{var}.csv",
    )
)


# %% plotting it on a contour plot

divider = 0


fig, axes = plt.subplots(1, 1, figsize=[20, 8], sharex=True, constrained_layout=True)
ax2 = axes
# ax1=axes[0]
# ax2=axes[1]
# fig1, ax1 = plt.subplots(1,1, figsize=[20,10], sharex =True , constrained_layout=True)


contourf = ax2.contourf(X, Y, Z, levels=levels_f, norm=norm, cmap=cmap, extend="both")
# contourf = ax1.contourf(X,Y,Z,levels=levels_f, norm=norm, cmap=cmap, extend='both')
contourl = ax2.contour(X, Y, Z, colors="k", levels=levels_l, linewidths=0.2, font=5)
# contourl1 = ax1.contour(X,Y,Z,colors= 'k', levels=levels_l, linewidths= 0.2, font=5)
ax2.scatter(x_dates.dt.floor("D"), y_depths, c="white", s=0.1, alpha=0.5)
# ax1.scatter(x_dates.dt.floor('D'), y_depths, c="white", s=.1,alpha =.5)

ax2.clabel(contourl, colors="k")
# ax1.clabel(contourl1, colors = 'k')
# ax1.set_ylim([0,divider])
ax2.set_ylim([divider, max(py)])
# ax2.set_xlim([datetime(2018,5,29), datetime(2019,6,12)])
# ax2.set_ylim([divider,100])

ax2.set_xlim([datetime(2018, 1, 1), datetime(2020, 1, 1)])
ax2.xaxis.set_minor_locator(mdates.MonthLocator())

# ax1.invert_yaxis()
ax2.invert_yaxis()

ax2.set_ylabel("Pressure [dbar]")
fig.subplots_adjust(right=0.8, wspace=0.02, hspace=0.01)
cb_ax = fig.add_axes([0.83, 0.2, 0.02, 0.6])
fig.colorbar(contourf, cax=cb_ax, shrink=0.8, label=variable, ticks=levels_f[::4])
fig.savefig(f"{figpath}/temp/interpolated_{var}_one_plot_{cmap.name}.png")

# %%

# %%

# %% just the data points plotted


fig, axes = plt.subplots(1, 1, figsize=[20, 10], sharex=True, constrained_layout=True)
ax2 = axes
# ax1=axes[0]
# ax2=axes[1]

# contourf = ax2.contourf(X,Y,Z,levels=levels_f, norm=norm, cmap=cmap, extend='both')
# contourf = ax1.contourf(X,Y,Z,levels=levels_f, norm=norm, cmap=cmap, extend='both')
# contourl = ax2.contour(X,Y,Z,colors= 'k', levels=levels_l,linewidths= 0.2,font=5)
# contourl = ax1.contour(X,Y,Z,colors= 'k', levels=levels_l, linewidths= 0.2, font=5)
sc = ax2.scatter(x_dates, long_py.values, c=long_p_para, norm=norm, cmap=cmap, s=20)
# ax1.scatter(x_dates, long_py.values, c=long_p_para,  norm=norm, cmap=cmap,s=20)

# ax2.clabel(contourl,  colors = 'k')
# ax1.clabel(contourl, colors = 'k')
# ax1.set_ylim([0,divider])
ax2.set_ylim([divider, max(py)])
ax2.set_xlim(datetime(2018, 6, 1), datetime(2018, 7, 1))
# ax1.invert_yaxis()
ax2.invert_yaxis()

ax2.set_ylabel("Pressure [dbar]")
fig.subplots_adjust(right=0.8, wspace=0.02, hspace=0.01)
cb_ax = fig.add_axes([0.83, 0.2, 0.02, 0.6])
fig.colorbar(sc, cax=cb_ax, shrink=0.8, label=variable, ticks=levels_f[::2])
# fig.savefig(f"{figpath}/temp/interpolated_{var}_long_only_scatter_only_monthly.png")

# %% interpolating new with wjb method


def dfprofile_to_array(df, direction):
    if direction == "H":
        return [
            df.date.apply(lambda x: x.to_julian_date()).values,
            df[var].values,
            np.mean(df["Pressure [dbar]"].values),
        ]
    elif direction == "V":
        return [
            df["Pressure [dbar]"].values,
            df[var].values,
            np.mean(df.date.apply(lambda x: x.to_julian_date()).values),
        ]
    else:
        print("error")


print(var)
maskV1 = (
    (df_combi_hr.Type == "CTD")
    & (df_combi_hr.date > datetime(2018, 6, 1))
    & (df_combi_hr.date < datetime(2018, 6, 15))
    & (df_combi_hr["Pressure [dbar]"] < 60)
    & (df_combi_hr["Pressure [dbar]"] > 5)
)
maskV2 = mask = (
    (df_combi_hr.Type == "CTD")
    & (df_combi_hr.date > datetime(2018, 6, 15))
    & (df_combi_hr.date < datetime(2018, 7, 1))
    & (df_combi_hr["Pressure [dbar]"] < 60)
    & (df_combi_hr["Pressure [dbar]"] > 5)
)
V1 = df_combi_hr[maskV1]
V2 = df_combi_hr[maskV2]
# choose mask horizontal mooring 1 as between dates vertical moorings and depth lower than 10
maskH1 = (
    (df_combi_hr.Type == "Mooring")
    & (df_combi_hr.date > V1.date.max())
    & (df_combi_hr.date < V2.date.min())
    & (df_combi_hr.depth < 10)
)
maskH2 = (
    (df_combi_hr.Type == "Mooring")
    & (df_combi_hr.date > V1.date.max())
    & (df_combi_hr.date < V2.date.min())
    & (df_combi_hr.depth > 10)
    & (df_combi_hr.depth < 100)
)
H1 = df_combi_hr[maskH1]
H2 = df_combi_hr[maskH2]


V1profile = dfprofile_to_array(V1, "V")
V2profile = dfprofile_to_array(V2, "V")
H1profile = dfprofile_to_array(H1, "H")
H2profile = dfprofile_to_array(H2, "H")
# continue here!!!


TargetWJB = [
    np.arange(6, 60, 3),
    np.arange(
        V1.date.max().to_julian_date(), V2.date.min().to_julian_date(), 1.0 / 24.0
    ),
]

# from WJB_profile_interpolate_v2 import WJB_profile_interpolate

Q, [QV1B, QV1T, QV1], [QHD1, QHD2, QHD] = WJB_profile_interpolate(
    V1profile, V2profile, H2profile, H1profile, TargetWJB, "LinearV"
)
#  WJB_profile_interpolate(Vprofile1, Vprofile2, HserieUp, HserieDown, Target, Method)

# %%


scat = plt.scatter(
    np.repeat(V1profile[2], len(V1profile[0])), V1profile[0], c=V1profile[1]
)
plt.scatter(np.repeat(V2profile[2], len(V2profile[0])), V2profile[0], c=V2profile[1])
plt.scatter(H1profile[0], np.repeat(H1profile[2], len(H1profile[0])), c=H1profile[1])
plt.scatter(H2profile[0], np.repeat(H2profile[2], len(H2profile[0])), c=H2profile[1])
plt.colorbar()

# %%

# %%
