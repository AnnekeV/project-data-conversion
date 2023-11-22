# -*- coding: utf-8 -*-
"""
This files analyses and combines monthly CTD files

"""
# %%
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
import os.path
import ctd
from geopy import Point
from basic_station_data import stat_loc, find_distance_from_fjordmouth
import functions as functions
import importlib
importlib.reload(functions)

# define paths
plt.style.use("ggplot")
pd.options.plotting.backend = "matplotlib"
path_parent = Path.cwd().parent
path_intermediate_files = Path.cwd().parent.joinpath("data", "temp")
path_intermediate_files_netcdf = path_intermediate_files.joinpath("netcdf")
figpath = os.path.join(path_parent, "Figures")

# %% if it's not there yet, create the intermediate folder
if not os.path.exists(path_intermediate_files_netcdf):
    os.makedirs(path_intermediate_files_netcdf)

# %% Importing station data and combining it to a bigger dataset


def split_header(header):
    """
    Split header into a dictionary, based on the = sign
    For example:
    start_time = Mar 29 2018 14:00:00

    Parameters
    ----------
    header : string
        header or config of the CTD file, as a string

    Returns
    -------
    data_dict : dictionary
        dictionary with the header information
    """
    lines = header.split("\n")

    data_dict = {}
    for line in lines:
        sublines = line.split(",")
        for subline in sublines:
            if "=" in subline:
                key, value = subline.split("=")
                key = key.strip() 
                value = (
                    value.strip()
                )  # Remove leading/trailing whitespaces from the value
                if "* " in key:
                    key = key.split("* ")[1]
                if "# " in key:
                    key = key.split("# ")[1]

                try:
                    value = float(value)
                except ValueError:
                    pass
                data_dict[key] = value
    return data_dict


all_years = pd.DataFrame()

for year in ["2018", "2019"]:
    path_data = os.path.join(path_parent, "data", "raw", "CTD", year)

    # % manually importing station information
    station_info = path_parent.joinpath(Path("data", "raw", "CTD", year, f"{year}.txt"))
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
            print(f"\n Station {metadata['name']} is not in the overview file !!")
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
        metadata["header"] = split_header(metadata["header"])
        metadata["config"] = split_header(metadata["config"])

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

# df_monthly.to_csv(f"{path_intermediate_files}/monthly_18_19_gf10.csv")
# %%
# =================================
# Assign attributes and convert to xarray
# =================================
ds_single = down.set_index("Pressure [dbar]").to_xarray()
# set attributes
time_coverage_start = datetime.strptime(
    metadata["config"]["start_time"].split("[")[0].strip(), "%b %d %Y %H:%M:%S"
).strftime("%Y-%m-%d %H:%M:%S")

# assign attributes to ds_single
ds_single.attrs["time_coverage_start"] = time_coverage_start

