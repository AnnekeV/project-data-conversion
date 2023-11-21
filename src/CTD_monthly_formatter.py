# -*- coding: utf-8 -*-
"""
This files analyses and combines monthly CTD files

"""
# %%
from pathlib import Path
from re import X

import cmocean
import ctd
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from os import listdir
import os.path
import seaborn as sns
from geopy import Point


plt.style.use("ggplot")
pd.options.plotting.backend = "matplotlib"
path_parent = r"/Users/annek/Library/CloudStorage/OneDrive-SharedLibraries-NIOZ/PhD Anneke Vries - General"
path_parent = Path.cwd().parent.parent
path_intermediate_files = Path.cwd().parent.joinpath("intermediate_files")
figpath = os.path.join(path_parent, "Figures")
# %% Importing station data and combining it to a bigger datase
from basic_station_data import stat_loc, dist_lat_lon, find_distance_from_fjordmouth

all_years = pd.DataFrame()

for year in ["2018", "2019"]:
    path_data = os.path.join(path_parent, "Data", "CTD", year)

    # % manually importing station information
    station_info = path_parent.joinpath(Path("Data", "CTD", year, f"{year}.txt"))
    widths = [8, 5, 6, 5, 9, 5, 7, 12, 12, 7, 8, 1000]  # nr of characters
    stat = pd.read_fwf(
        rf"{station_info}",
        header=1,
        encoding="latin1",
        on_bad_lines="skip",
        widths=widths,
        skipfooter=4,
    )
    stat["Latitude"] = stat["Latitude"].apply(
        lambda x: float(x.split(" ")[0]) + float(x.split(" ")[1]) / 60
    )
    stat["Longitude"] = stat["Longitude"].apply(
        lambda x: -1 * (float(x.split(" ")[0]) + float(x.split(" ")[1]) / 60)
    )
    stat["Coordinates"] = stat.apply(
        lambda row: Point(latitude=row["Latitude"], longitude=row["Longitude"]), axis=1
    )
    stat["Date"] = pd.to_datetime(stat["YYYYMMDD"], format="%Y%m%d")
    stat["Name"] = stat["St.No."].copy()
    stat = stat.set_index(stat["St.No."])
    stat["Distance"] = stat.merge(
        stat_loc.rename(columns={"Station": "St."}), on="St.", how="left"
    )["LengthMouth"].values
    nan_dist = stat[stat["Distance"].isna()].index
    for i in nan_dist:
        p = ((stat.loc[i, "Latitude"]), (stat.loc[i, "Longitude"]))
        dist = find_distance_from_fjordmouth(p)
        if dist == 999:
            print(i)
        else:
            stat.loc[i, "Distance"] = dist

    stat[["YYYYMMDD", "St."]].value_counts().reset_index().pivot(
        index="YYYYMMDD", columns="St."
    ).to_csv(f"{path_data}/overview_per_date.csv")

    # Import profile every file as a ctd

    all_thisyear = pd.DataFrame()
    counter = 0
    plot_cast = False
    fileNames = [
        f for f in listdir(path_data) if os.path.isfile(os.path.join(path_data, f))
    ]

    for fname in Path(path_data).rglob("*.cnv"):
        counter += 1
        print(counter)
        cast = ctd.from_cnv(fname)  #
        down, up = cast.split()
        down = down.rename(
            columns={
                "potemp090C": "Potential temperature [°C]",
                "sal00": "Salinity [PSU]",
                "density00": "Density [kg/m3]",
            }
        )
        up = up.rename(
            columns={
                "potemp090C": "Potential temperature [°C]",
                "sal00": "Salinity [PSU]",
                "density00": "Density [kg/m3]",
            }
        )
        metadata = cast._metadata

        if not metadata["name"] in stat.index:
            print(f"\n Station {metadata['name']} is not in the overview file !!! \n")
            continue
        this_stat = stat.loc[metadata["name"]]
        meta = [
            "Latitude",
            "Longitude",
            "St.",
            "Date",
            "Type",
            "Name",
            "YYYYMMDD",
            "Distance",
        ]
        for key in meta:
            down[key] = this_stat[key]

        if plot_cast == True:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            down.reset_index().plot(
                x="Potential temperature [°C]",
                y="Pressure [dbar]",
                ax=ax1,
                legend=False,
            ).invert_yaxis()
            down.reset_index().plot(
                x="Salinity [PSU]", y="Pressure [dbar]", ax=ax2, legend=False
            )
            down.reset_index().plot(
                x="Density [kg/m3]", y="Pressure [dbar]", ax=ax3, legend=False
            )
            fig.suptitle(
                f"{metadata['name']}, {this_stat['St.']}, {this_stat['Date']}, {this_stat['Type']}"
            )
            ax1.set_ylabel("Pressure [dbar]")
            fig.show()

        down = down.reset_index()
        all_thisyear = pd.concat([all_thisyear, down], axis="index")

    # ==================
    #  selecting GF10
    # ==================
    print(all_thisyear.columns)

    GF10 = (
        all_thisyear[all_thisyear["St."] == "GF10"]
        .sort_values(["Name", "Pressure [dbar]"])
        .reset_index(drop=True)
    )

    StGF10 = stat[stat["St."] == "GF10"]["Name"].values
    for i in range(len(StGF10)):
        down = all_thisyear[all_thisyear["Name"] == StGF10[i]]

    # Single year
    df10 = all_thisyear[all_thisyear["Name"].isin(StGF10)]
    column_rename = {
        "potemp090C": "temp",
        "sal00": "sal",
        "density00": "dens",
        "timeJV2": "time",
        "timeJ": "time",
        "sigma-t00": "sigma-dens",
        "sigma-È00": "sigma-dens",
        "sigma-�00": "sigma-dens",
        "Date": "date",
        "Potential temperature [°C]": "temp",
        "Salinity [PSU]": "sal",
        "Density [kg/m3]": "dens",
    }
    df10.rename(columns=column_rename).reset_index().to_csv(
        f"{path_intermediate_files}/monthly_{year}_gf10.csv"
    )

    # ==================
    #  All stations all years
    # ==================

    all_years = pd.concat([all_years, all_thisyear]).reset_index(drop=True)

all_years.to_csv(
    f"{path_intermediate_files}/monthly_all_years_all_stations.csv", index=None
)

#  Combine 2018 and 2019
df18 = pd.read_csv(f"{path_intermediate_files}/monthly_2018_gf10.csv")
df19 = pd.read_csv(f"{path_intermediate_files}/monthly_2019_gf10.csv")
df_monthly = pd.concat([df18, df19]).reset_index(drop=True)


df_monthly["id"] = df_monthly["Name"].copy()
df_monthly.date = df_monthly.date.apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d"), "broadcast"
)
df_monthly.to_csv(f"{path_intermediate_files}/monthly_18_19_gf10.csv")


# %%

fnames = []
for fname in Path(os.path.join(path_parent, "Data", "CTD")).rglob("*.cnv"):
    fnames.append(str(fname))


def open_ctd(stat_name):
    fname = [i for i in fnames if stat_name in i][0]
    cast = ctd.from_cnv(fname)  #
    down, up = cast.split()
    down = down.rename(
        columns={
            "potemp090C": "Potential temperature [°C]",
            "sal00": "Salinity [PSU]",
            "density00": "Density [kg/m3]",
        }
    )
    up = up.rename(
        columns={
            "potemp090C": "Potential temperature [°C]",
            "sal00": "Salinity [PSU]",
            "density00": "Density [kg/m3]",
        }
    )
    metadata = cast._metadata

    if not metadata["name"] in stat.index:
        print(f"\n Station {metadata['name']} is not in the overview file !!! \n")
    this_stat = stat.loc[metadata["name"]]
    meta = [
        "Latitude",
        "Longitude",
        "St.",
        "Date",
        "Type",
        "Name",
        "YYYYMMDD",
        "Distance",
    ]
    for key in meta:
        down[key] = this_stat[key]
    return down, up, cast


down, up, cast = open_ctd("HS190123")

pd.options.plotting.backend = "plotly"
fig = go.Figure()
for col in down.columns:
    fig = down.plot(y=cast.index, x=col)
    fig.update_yaxes(autorange="reversed")
    fig.show()

# %%
plt.style.use("ggplot")
pd.options.plotting.backend = "matplotlib"


# %% depth vs profile

dSet = all_thisyear.set_index(["Date", "Pressure [dbar]", "Longitude"]).to_xarray()


# for day in dSet['Date']:
#     fig = dSet.sel(Date=day)['Potential temperature [°C]'].plot(yincrease=False, levels=np.arange(-8,10,1), cmap=plt.get_cmap('viridis'))
#     plt.show()

g_simple = dSet["Potential temperature [°C]"].plot.contourf(
    x="Longitude",
    y="Pressure [dbar]",
    col="Date",
    col_wrap=4,
    yincrease=False,
    cmap=plt.get_cmap("inferno"),
    levels=np.arange(-2, 4, 0.2),
)

g_simple = dSet["Salinity [PSU]"].plot(
    x="Longitude",
    y="Pressure [dbar]",
    col="Date",
    col_wrap=4,
    yincrease=False,
    cmap=plt.get_cmap("viridis"),
    levels=np.arange(25, 35, 0.5),
)


# %%

no_stat = (all_thisyear["St."].isna()) & (all_thisyear["Type"] == "CTD")
# dSt = all_thisyear[~no_stat].set_index(['Date','Pressure [dbar]', 'St.']).to_xarray()
# g_simple = dSt['Salinity [PSU]'].plot(x="Longitude", y="Pressure [dbar]", col="Date", col_wrap=4,
#                                                    yincrease=False, cmap=plt.get_cmap('viridis'),levels=np.arange(25,35,.5))


# %% time vs profile


Dates = np.sort(all_thisyear.Date.unique())
april = all_thisyear[all_thisyear.Date == Dates[0]].sort_values(
    ["Pressure [dbar]", "Longitude"]
)
df = april.pivot(index="Pressure [dbar]", columns="St.", values="Salinity [PSU]")
g = sns.heatmap(df)
g.set_yticks(np.arange(0, 630, 100))
g.set_yticklabels(np.arange(0, 630, 100))
plt.show()
print("seperate figure printed")

var = "Potential temperature [°C]"

fig, axes = plt.subplots(4, 4, figsize=(15, 10), sharey=True)
fig.suptitle(var)
# cbar_ax = fig.add_axes([.91, .3, .03, .4])

all_thisyear = all_thisyear.sort_values(["Pressure [dbar]", "Longitude"])
ax_count = 0
for i in range(len(Dates)):
    print(Dates[i])
    try:
        ax = axes[int(np.floor(ax_count / 4)), ax_count % 4]
        month = all_thisyear[all_thisyear.Date == Dates[i]]
        df = month.pivot(index="Pressure [dbar]", columns="St.", values=var)
        g = sns.heatmap(df, ax=ax, vmin=-2, vmax=4)
        g.set_yticks(np.arange(0, 630, 100))
        g.set_yticklabels(np.arange(0, 630, 100))
        g.set_ylabel("")
        g.set_xlabel("")
        g.set_title(pd.to_datetime(str(Dates[i])).strftime("%d-%m-%Y"))
        ax_count += 1
    except:
        print("didn't work")

fig.tight_layout(rect=[0, 0, 0.9, 1])
# plt.savefig(f"{path_parent.joinpath(Path('Figures'))}/{var}.png")

# cb_ax = fig.add_axes([1.1, 0.1, 0.02, 0.8])
# cbar = fig.colorbar(g, cax=cb_ax)


# %% Select GF10
pd.options.plotting.backend = "plotly"
variable = "sbox0Mm/Kg"


fig = px.line(GF10, x=variable, y="Pressure [dbar]", facet_col="Name", facet_col_wrap=5)
fig.update_yaxes(autorange="reversed")
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()
# %% SALINITY
pd.options.plotting.backend = "matplotlib"

StGF10 = stat[stat["St."] == "GF10"]["Name"].values
fig, axes = plt.subplots(3, 5, sharey=True, figsize=[15, 25], sharex=True)
for i, ax in enumerate(axes.flat):
    if i == len(StGF10):
        break
    down = all_thisyear[all_thisyear["Name"] == StGF10[i]]
    if i == 0:
        down.reset_index().plot(
            x="Salinity [PSU]",
            y="Pressure [dbar]",
            ax=ax,
            legend=False,
            title=down["YYYYMMDD"][0],
        ).invert_yaxis()
        ax.set_ylabel("Pressure [dbar]")

    else:
        down.reset_index().plot(
            x="Salinity [PSU]",
            y="Pressure [dbar]",
            ax=ax,
            legend=False,
            title=down["YYYYMMDD"][0],
        )
    if i % 5 == 0:
        ax.set_ylabel("Pressure [dbar]")

# fig.suptitle('Salinity in GF10')
plt.tight_layout()
# plt.savefig(f"{path_parent.joinpath(Path('Figures'))}/GF10_salinity.png")

# %% POTENTIAL TEMPERATURE
fig, axes = plt.subplots(3, 5, sharey=True, figsize=[20, 20], sharex=True)
for i, ax in enumerate(axes.flat):
    if i == len(StGF10):
        break
    down = all_thisyear[all_thisyear["Name"] == StGF10[i]]
    if i == 0:
        down.reset_index().plot(
            x="Potential temperature [°C]",
            y="Pressure [dbar]",
            ax=ax,
            legend=False,
            title=down["YYYYMMDD"][0],
        ).invert_yaxis()
        ax.set_ylabel("Pressure [dbar]")

    else:
        down.reset_index().plot(
            x="Potential temperature [°C]",
            y="Pressure [dbar]",
            ax=ax,
            legend=False,
            title=down["YYYYMMDD"][0],
        )
    if i % 5 == 0:
        ax.set_ylabel("Pressure [dbar]")
    ax.set_xlim([-1.5, 3])
# fig.suptitle('Salinity in GF10')
plt.tight_layout()
# plt.savefig(f"{path_parent.joinpath(Path('Figures'))}/GF10_pot_temperature.png")

# %% T P
fig, axes = plt.subplots(3, 5, sharey=True, figsize=[25, 20], sharex=True)
for i, ax in enumerate(axes.flat):
    if i == len(StGF10):
        break
    down = all_thisyear[all_thisyear["Name"] == StGF10[i]]
    down.reset_index().plot(
        y="Potential temperature [°C]",
        x="Salinity [PSU]",
        ax=ax,
        legend=False,
        title=down["YYYYMMDD"][0],
    )
    if i % 5 == 0:
        ax.set_ylabel("Potential temperature [°C]")
# fig.suptitle('Salinity in GF10')
plt.tight_layout()
# plt.savefig(f"{path_parent.joinpath(Path('Figures'))}/GF10_temp_sal.png")

# %%
fig, ax = plt.subplots(1, 1, figsize=[15, 15])
for i in range(len(StGF10)):
    down = all_thisyear[all_thisyear["Name"] == StGF10[i]]
    down.reset_index().plot(
        y="Potential temperature [°C]",
        x="Salinity [PSU]",
        ax=ax,
        label=down["YYYYMMDD"][0],
    )
ax.set_xlim([30.5, 34])
# fig.suptitle('Salinity in GF10')
plt.tight_layout()
# plt.savefig(f"{figpath}/GF10_temp_sal_all.png")
plt.show()
# %% SURFER PLOT
# SURFER PLOT


variable = "Salinity [PSU]"
# variable = 'Potential temperature [°C]'
# variable = 'Density [kg/m3]'

if variable == "Salinity [PSU]":
    levels_f = np.concatenate(
        (
            [10, 15, 20, 25, 30, 31],
            np.arange(32, 33.5, 0.2),
            np.arange(33.5, 33.6, 0.05),
        )
    )
    cmap = cmocean.cm.haline
    var = "sal"

elif variable == "Potential temperature [°C]":
    levels_f = np.concatenate(
        (np.arange(-1.5, 0.5, 0.5), np.arange(0.5, 2.5, 0.25), np.arange(2.5, 5.5, 0.5))
    )
    cmap = cmocean.cm.thermal
    var = "temp"

else:
    levels_f = np.concatenate((np.arange(1005, 1025, 1.5), np.arange(1025, 1030, 0.5)))
    cmap = cmocean.cm.dense
    var = "dens"

levels_l = levels_f[::2]

divider = 30

# # Single year
# df10 = all_thisyear[all_thisyear['Name'].isin(StGF10)]
# df10piv = df10.pivot_table(values=variable, index="Pressure [dbar]", columns="YYYYMMDD")
# Y =df10piv.index.values
# X = df10piv.columns.values
# X = pd.to_datetime(df10piv.columns, format='%Y%m%d')
# Xpiv, Ypiv = np.meshgrid(df10piv.index.values, df10piv.columns.values)
# Z=df10piv.values

# Mooring year
df_monthly = pd.read_csv(
    f"{path_intermediate_files}/monthly_18_19_gf10.csv",
    index_col=0,
    parse_dates=["date"],
)
df_gf10 = df_monthly[(df_monthly["St."] == "GF10") & (df_monthly["Type"] == "CTD")]
df10piv = df_gf10.drop_duplicates(["date", "Pressure [dbar]"]).pivot(
    "date", "Pressure [dbar]", var
)["2018-05-29":"2019-06-12"]
X = df10piv.index.values
Y = df10piv.columns.values
Xpiv, Ypiv = np.meshgrid(df10piv.index.values, df10piv.columns.values)
Z = df10piv.values.T

for i in range(11):
    print(df10piv.stack().quantile(i / 10))
    plt.scatter(i, df10piv.stack().quantile(i / 10))
plt.show()

norm = matplotlib.colors.BoundaryNorm(levels_f, cmap.N, extend="both")

fig, axes = plt.subplots(2, 1, figsize=[12, 3], sharex=True, constrained_layout=True)
ax1 = axes[0]
ax2 = axes[1]

contourf = ax2.contourf(X, Y, Z, levels=levels_f, norm=norm, cmap=cmap, extend="both")
contourf = ax1.contourf(X, Y, Z, levels=levels_f, norm=norm, cmap=cmap, extend="both")
contourl = ax2.contour(X, Y, Z, colors="k", levels=levels_l, linewidths=0.2, font=5)
contourl = ax1.contour(X, Y, Z, colors="k", levels=levels_l, linewidths=0.2, font=5)
ax2.scatter(X, np.ones(len(X)) * max(Y) - 1, color="black", marker="^")
ax2.clabel(contourl, colors="k")
ax1.clabel(contourl, colors="k")
ax1.set_ylim([0, divider])
ax2.set_ylim([divider, max(Y)])
ax1.invert_yaxis()
ax2.invert_yaxis()

ax2.set_ylabel("Pressure [dbar]")
fig.subplots_adjust(right=0.8, wspace=0.02, hspace=0.01)
cb_ax = fig.add_axes([0.83, 0.2, 0.02, 0.6])
fig.colorbar(contourf, cax=cb_ax, shrink=0.8, label=variable, ticks=levels_f[::2])
ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

fig.savefig(f"{figpath}/temp/GF10_surfer_mooring_{variable.split()[0]}_poster.svg")
plt.show()

# %% SURFER PLOT ONLY one


variable = "Potential temperature [°C]"


df_monthly = pd.read_csv(
    f"{path_parent}/Processing/intermediate_files/monthly_18_19_gf10.csv",
    index_col=0,
    parse_dates=["date"],
)
df_gf10 = df_monthly[(df_monthly["St."] == "GF10") & (df_monthly["Type"] == "CTD")]
df10piv = df_gf10.drop_duplicates(["date", "Pressure [dbar]"]).pivot(
    "date", "Pressure [dbar]", "temp"
)["2018-05-29":"2019-06-12"]

X = df10piv.index.values
Y = df10piv.columns.values
Xpiv, Ypiv = np.meshgrid(df10piv.index.values, df10piv.columns.values)
Z = df10piv.values.T

cmap = cmocean.cm.thermal
# levels_f = np.concatenate(([5,10,15,20,25,30,31],np.arange(32,33.6,.2)))
# levels_f = np.concatenate((np.arange(5,32,2.5),np.arange(32,33.6,.1)))

norm = matplotlib.colors.BoundaryNorm(levels_f, cmap.N, extend="both")
# levels_l = levels_f[5::2]
# levels_f = np.concatenate((np.arange(5,32,5),np.arange(32,33.6,.2)))

fig, (ax2) = plt.subplots(1, 1, figsize=[11, 4], sharex=True)

contourf = ax2.contourf(X, Y, Z, levels=levels_f, norm=norm, cmap=cmap)
contourl = ax2.contour(X, Y, Z, colors="k", levels=levels_l, linewidths=0.5)

ax2.scatter(X, np.ones(len(X)) / 2)
ax2.clabel(contourl, colors="k", fmt="%2.1f", fontsize=8)
ax2.invert_yaxis()
ax2.set_ylabel("Depth")
plt.ylim([500, 0])
fig.colorbar(
    contourf,
    orientation="vertical",
    label=variable,
    ticks=levels_f[::2],
)

ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


# fig.suptitle(variable)
fig.tight_layout()
# fig.savefig(f"{figpath}/GF10_monthly_temperature_poster.svg")
plt.show()
# %%
pd.options.plotting.backend = "plotly"

df_both = pd.read_csv(
    f"{path_parent}/Processing/intermediate_files/monthly_18_19_gf10.csv", index_col=0
)
names = df_both.Name.unique()

bin = 20  # m
df_both["Pressure_grouped"] = df_both["Pressure [dbar]"].apply(
    lambda x: x - np.mod(x, bin)
)
bins = [0, 20, 60, 150, 350, 600]
labels = [f"{i} - {j}" for i, j in zip(bins[:-1], bins[1:])]
df_both["Layers"] = pd.cut(df_both["Pressure [dbar]"], bins=bins, labels=labels)


sal_max = 34.7  # PSU
df_both["rel_sal"] = (sal_max - df_both.sal) / sal_max
df_both.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"), "broadcast")

df_fresh = (
    df_both.groupby(["date", "Layers"])
    .agg({"rel_sal": "sum", "sal": "mean", "temp": "mean"})
    .reset_index()
)

pd.options.plotting.backend = "plotly"

df_fresh_profile = (
    df_both.groupby(["date", "Pressure_grouped"])
    .agg({"rel_sal": "mean", "sal": "mean", "temp": "mean"})
    .reset_index()
)
df_fresh_profile.rel_sal *= bin

fig = df_fresh_profile.plot.bar(
    x="rel_sal",
    y="Pressure_grouped",
    facet_col="date",
    facet_col_wrap=9,
    orientation="h",
    title="Freshwater content",
    width=1200,
    height=500,
)
fig.update_yaxes(autorange="reversed")
fig.write_image(f"{figpath}/Freshwater content per date.png")
fig.show()

mean_var = "sal"

fig = df_fresh.plot.line(
    x="date",
    y=mean_var,
    # facet_col="Pressure_grouped", facet_col_wrap=1,
    color="Layers",
    # title="Freshwater content [m]",
    markers=True,
    width=1000,
    height=400,
)
fig.update_yaxes(matches=None)
fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", ticklabelmode="period")
# fig.write_image(f"{figpath}/Freshwater content per layer.png")
# fig.write_image(f"{figpath}/Mean {mean_var} per layer.png")

fig.show()


# %% GRID

var = "dens"
df_sorted = df_both.drop_duplicates(["date", "Pressure [dbar]"]).sort_values(
    ["date", "Pressure [dbar]"]
)
fig = df_sorted.plot.line(
    y="Pressure [dbar]",
    x=var,
    facet_col="date",
    facet_col_wrap=9,
    facet_row_spacing=0.03,
    facet_col_spacing=0.02,
    height=1000,
    width=1500,
    # color = 'sal'
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
if var == "temp":
    fig.update_xaxes(range=[-1, 5], dtick=2)
fig.update_yaxes(range=[550, 0])
fig.write_image(f"{figpath}/All profiles {var}.png")

# if var=='temp':fig.update_xaxes(range=[-2,10], dtick=2)
# else:fig.update_xaxes( range = [33  ,33.7], dtick=0.1)

# fig.update_yaxes(range=[20,0])
# fig.write_image(f"{figpath}/temp/All profiles {var} zoom.png")

# fig.update_yaxes(range=[600,0])

# fig.write_html(f"{figpath}/temp/All profiles.html")


fig.show()

df_sorted.to_csv("intermediate/df_sorted.csv")


# %% GRID SCATTER
# GRID SCATTER
# manually change color scal and range color
def find_accompanying_value_colorbar(cmin, cmax, percentage):
    """find the accompanying value for a custom discontinuous colorbar
    cmin = minimum in range color
    cmax = maximum in range color
    percentage is value in custom scale between (0,1)
    returns value
    """
    return (cmax - cmin) * percentage + cmin


def find_accompanying_percentage_colorbar(cmin, cmax, goal_value):
    """find the accompanying percentage for an aimed value in a custom discontinuous colorbar
    cmin = minimum in range color
    cmax = maximum in range color
    goal_value is value in variable
    returns percentage (0,1)
    """
    return (goal_value - cmin) / (cmax - cmin)


custom_scale = [
    (0.0, "gold"),
    (0.94, "lemonchiffon"),
    (0.97, "mediumseagreen"),
    (0.989, "navy"),
    (1, "black"),
]
var = "temp"
other_var = "sal"
df_sorted = df_both.sort_values(["date", "Pressure [dbar]"])
fig = df_sorted.plot.scatter(
    y="Pressure [dbar]",
    x=var,
    facet_col="date",
    facet_col_wrap=9,
    facet_row_spacing=0.03,
    facet_col_spacing=0.02,
    height=1000,
    width=1500,
    color=other_var,
    # color_continuous_scale=px.colors.sequential.haline,
    color_continuous_scale=custom_scale,
    range_color=[10, 33.7],
)


fig.update_layout(
    coloraxis_colorbar=dict(
        title="Salinity [PSU]",
        tickvals=[10, 15, 20, 25, 30, 31, 32, 32.5, 33, 33.44, 33.7],
    )
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
if var == "temp":
    fig.update_xaxes(range=[-1, 5], dtick=2)
fig.update_yaxes(range=[600, 0])
# fig.write_image(f"{figpath}/All profiles {var} scatter.png")
fig.show()

# %% d rho /d z
from sklearn.linear_model import LinearRegression


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


var = "dens"
slope_dz = []
date = []
for i in df_sorted.date.unique():
    diff_monthly = (
        df_sorted[df_sorted.date == i]
        .set_index("Pressure [dbar]")[530:550]
        .reset_index()
    )
    diff_monthly.date = pd.to_datetime(df_sorted["date"])
    X = diff_monthly["Pressure [dbar]"].values.reshape(
        -1, 1
    )  # values converts it into a numpy array
    Y = diff_monthly["dens"].values.reshape(
        -1, 1
    )  # -1 means that calculate the dimension of rows, but have 1 column
    if not diff_monthly["dens"].isnull().values.all():
        slope = lin_reg(X, Y)
        print(f"slope: {slope:.3e} kg/m3/m")
        plt.xlabel("Depth [m]")
        plt.ylabel("Density [kg/m3 ]")
        plt.title("Diffusion at depth (d rho/dz)")
        plt.show()
        slope_dz.append(slope)
        date.append(pd.to_datetime(i))

plt.plot(date, slope_dz)
plt.ylabel("Coefficient [kg/m3/m]")
plt.title("Coefficient density change with depth drho/dz 350-deepest point")

# %%
# Making a pivot plot


df_monthly = pd.read_csv(
    f"{path_intermediate_files}/monthly_18_19_gf10.csv",
    index_col=0,
    parse_dates=["date"],
)
df_gf10 = df_monthly[(df_monthly["St."] == "GF10") & (df_monthly["Type"] == "CTD")]
Z_1819 = df_gf10.drop_duplicates(["Pressure [dbar]", "date"], "first").pivot(
    index="Pressure [dbar]", columns="date", values="sal"
)

# %% CONTOURPLOT IN PLOTLY ===============

variable = "Salinity [PSU]"
fig, (ax2) = plt.subplots(1, 1, figsize=[8, 4], sharex=True)

contourf = ax2.contourf(
    Z_1819.columns, Z_1819.index, Z_1819.values, levels=levels_f, norm=norm, cmap=cmap
)
# contourl = ax2.contour(X,Y,Z,colors= 'k', levels=levels_l)
ax2.clabel(contourl, colors="k")  # , fmt = '%2.1f', fontsize=12)
ax2.invert_yaxis()
ax2.set_ylabel("Pressure [dbar]")
fig.colorbar(contourf, orientation="vertical")
fig.suptitle(variable)
fig.tight_layout()
# fig.savefig(f"{figpath}/GF10_time_{variable.split()[0]}_p.png")
plt.show()
# %%

# %%
