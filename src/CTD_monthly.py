# -*- coding: utf-8 -*-
"""
This files analyses and combines monthly CTD files an converts them to netcdf files
https://adc.met.no/node/4

"""
# %%
from datetime import datetime, timedelta
from pathlib import Path
import pathlib
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
path_parent = pathlib.Path(__file__).parent.parent.resolve()
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
    stat["YYYYMMDD UTC"] = pd.to_datetime(
        stat["YYYYMMDD"].astype(str) + " " + stat["UTC"].astype(str),
        format="%Y%m%d %H%M",
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
                "potemp090C": "Potential temperature",
                "sal00": "Salinity",
                "density00": "Density",
            }
        )

        metadata = cast._metadata

        if not metadata["name"] in stat.index:
            print(f"\n Station {metadata['name']} is not in the overview file !!")
            continue
        this_stat = stat.loc[metadata["name"]]
        down["timeJyear"] = pd.to_timedelta(
            down.timeJ - 1, unit="days"
        ) + pd.to_datetime(
            f"{(this_stat['YYYYMMDD']).astype('str')[:4]}-01-01 00:00:00"
        )  # substract 1 because than it matches with independent time records
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
    #  All stations all years
    # ==================

    all_years = pd.concat([all_years, all_thisyear]).reset_index(drop=True)

all_years.to_csv(
    f"{path_intermediate_files}/monthly_all_years_all_stations.csv", index=None
)

#  Combine 2018 and 2019
# df18 = pd.read_csv(f"{path_intermediate_files}/monthly_2018_gf10.csv")
# df19 = pd.read_csv(f"{path_intermediate_files}/monthly_2019_gf10.csv")
# df_monthly = pd.concat([df18, df19]).reset_index(drop=True)


# df_monthly["id"] = df_monthly["Name"].copy()

# df_monthly.to_csv(f"{path_intermediate_files}/monthly_18_19_gf10.csv")
# %%
# =================================
# Assign attributes and convert to xarray, according to ADC.met.no Arctic data centre
# =================================
ds_single = down.set_index("Pressure [dbar]").to_xarray()
ds_single = ds_single[["Potential temperature", "Salinity", "Density"]]

# even more generic
user_name = "Anneke Vries"

# generic data  attributes
data_type = "CTD"
featureType = "profile"
instrument = "Sea-Bird SBE19plus"
keyword_vocabulary = "GCMD Science Keywords"
keywords = "EARTH SCIENCE>OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE, EARTH SCIENCE>OCEANS>OCEAN PRESSURE>WATER PRESSURE,EARTH SCIENCE>OCEANS>SALINITY/DENSITY>SALINITY"
Conventions = "ACDD-1.3"
datenow_utc =  f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}"
history = f"{datenow_utc} converted to netcdf with xarray by {user_name}"
processing_level = "binning and manual inspection"
date_created = datenow_utc
creator_type = "person"
creator_institution = "NIOZ Royal Netherlands Institute for Sea Research"
creator_name = user_name
creator_email = "anneke.vries@nioz.nl"
creator_url = "https://orcid.org/0000-0001-9970-1189"
project = "UU-NIOZ Greenland fjords as gateways between the ice sheet and the ocean"
platform = "ship"
license =  "https://creativecommons.org/licenses/by/4.0/"
iso_topic_category = "oceans"

# profile specific attributes
start_time = datetime.strptime(
    metadata["config"]["start_time"].split("[")[0].strip(), "%b %d %Y %H:%M:%S"
)  # when the instrument started recording
time_coverage_start = down["timeJyear"].iloc[0]  # first measurement in dataframe
time_coverage_end = down["timeJyear"].iloc[-1]
time_recorded = this_stat["YYYYMMDD UTC"]  # as recorded in seperate sheet
if abs(time_coverage_start - time_recorded) > timedelta(hours=1):
    raise ValueError(
        f"Time coverage start {time_coverage_start} and time recorded {time_recorded} are more than 1 hour apart"
    )
time_coverage_start = time_coverage_start.strftime("%Y-%m-%dT%H:%MZ")
time_coverage_end = time_coverage_end.strftime("%Y-%m-%dT%H:%MZ")



GCRC_station_number = this_stat["Name"]
latitude = this_stat["Latitude"]
longitude = this_stat["Longitude"]
geospatial_lat_min = latitude
geospatial_lat_max = latitude
geospatial_lon_min = longitude
geospatial_lon_max = longitude
source = f"{data_type} #{GCRC_station_number}"



title = f"{data_type} {featureType} {this_stat['Name']} on {this_stat['Date']}, in Nuup Kangerlua, Greenland"
summary = f"The file contains potential temperature, practical salinity and depth measurements binned into 1 db bins. The raw data was measured at {latitude:.3f}N, {longitude:.3f}E, on {this_stat['Date']} UTC , with a {instrument}. The data was collected by the Greenland Climate Research Center (GCRC)"

attributes_to_be_included = ["user_name", "data_type", "featureType", "instrument", "keywords", "Conventions", "history", "processing_level", "date_created", "creator_type", "creator_institution", "creator_name", "creator_email", "creator_url", "project", "platform", "license", "iso_topic_category", "time_coverage_start", "time_coverage_end", "GCRC_station_number", "latitude", "longitude", "geospatial_lat_min", "geospatial_lat_max", "geospatial_lon_min", "geospatial_lon_max", "source", "title", "summary"]

for att in attributes_to_be_included:
    ds_single.attrs[att] = eval(att)

ds_single
ds_single.to_netcdf(f"{path_intermediate_files_netcdf}/test2.nc")


# %%
# combine column YYYYMMDD and UTC IN stat
