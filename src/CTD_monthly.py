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
import os.path
import ctd
from geopy import Point
from basic_station_data import stat_loc, find_distance_from_fjordmouth
import functions as functions
import importlib
import math
from mpl_toolkits.basemap import Basemap
from os import listdir
import xarray as xr


importlib.reload(functions)

# define paths
plt.style.use("ggplot")
pd.options.plotting.backend = "matplotlib"
try:
    path_parent = pathlib.Path(__file__).parent.parent.resolve()
except NameError:
    path_parent = Path.cwd().parent.resolve()
path_intermediate_files = path_parent.joinpath("data", "temp")
path_intermediate_files_netcdf = path_intermediate_files.joinpath("netcdf")
print(path_intermediate_files_netcdf)
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


def extract_station_info(path_parent, year):
    path_data = path_parent.joinpath(Path("data", "raw", "CTD", year))

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

    return path_data, stat


def extract_CTD(fname, stat):
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
        print(
            f"\n Station {metadata['name']} is not in the overview file, will be skipped"
        )
        return None, None, None
    this_stat = stat.loc[metadata["name"]]
    down["timeJyear"] = pd.to_timedelta(down.timeJ - 1, unit="days") + pd.to_datetime(
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
    return down, metadata, this_stat


# %%
# =================================
# Assign attributes and convert to xarray, according to ADC.met.no Arctic data centre
# =================================


def make_xarray_with_attributes(down, metadata, this_stat):
    """
    Make xarray with attributes, according to ADC.met.no Arctic data centre
    Parameters
    ----------
    down : dataframe
        dataframe with the downcast data
    metadata : dictionary
        metadata of the cast
    this_stat : dataframe
        dataframe with the station information
    Returns
    -------
    ds_single : xarray of CTD with attributes
    """

    ds_single = down.set_index("Pressure [dbar]").to_xarray()
    ds_single = ds_single[["Potential temperature", "Salinity", "Density"]]
    #  set units, standard_name, long_name
    ds_single["Potential temperature"].attrs["units"] = "degree)C"
    ds_single["Potential temperature"].attrs[
        "standard_name"
    ] = "sea_water_potential_temperature"
    ds_single["Potential temperature"].attrs[
        "long_name"
    ] = "Potential temperature of sea water"
    ds_single["Salinity"].attrs["units"] = "psu"
    ds_single["Salinity"].attrs["standard_name"] = "sea_water_practical_salinity"
    ds_single["Salinity"].attrs["long_name"] = "Salinity"
    ds_single["Density"].attrs["units"] = "kg/m3"
    ds_single["Density"].attrs["standard_name"] = "sea_water_density"
    ds_single["Density"].attrs["long_name"] = "Density of sea water"

    # even more generic
    user_name = "Anneke Vries"

    # generic data  attributes
    data_type = "CTD"
    featureType = "profile"
    instrument = "Sea-Bird SBE19plus"
    keyword_vocabulary = "GCMD Science Keywords"
    keywords = "EARTH SCIENCE>OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE, EARTH SCIENCE>OCEANS>OCEAN PRESSURE>WATER PRESSURE,EARTH SCIENCE>OCEANS>SALINITY/DENSITY>SALINITY"
    Conventions = "ACDD-1.3"
    datenow_utc = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}"
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
    license = "https://creativecommons.org/licenses/by/4.0/"
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
    GCRC_standard_station = this_stat["St."]

    title = f"{data_type} {featureType} {this_stat['Name']} on {this_stat['Date']}, in Nuup Kangerlua, Greenland"
    summary = f"The file contains potential temperature, practical salinity and depth measurements binned into 1 db bins. The raw data was measured at {latitude:.3f}N, {longitude:.3f}E, on {this_stat['Date']} UTC , with a {instrument}. The data was collected by the Greenland Climate Research Center (GCRC)"

    attributes_to_be_included = [
        "user_name",
        "data_type",
        "featureType",
        "keywords",
        "Conventions",
        "history",
        "processing_level",
        "date_created",
        "creator_type",
        "creator_institution",
        "creator_name",
        "creator_email",
        "creator_url",
        "project",
        "platform",
        "license",
        "iso_topic_category",
        "time_coverage_start",
        "time_coverage_end",
        "GCRC_station_number",
        "latitude",
        "longitude",
        "geospatial_lat_min",
        "geospatial_lat_max",
        "geospatial_lon_min",
        "geospatial_lon_max",
        "source",
        "title",
        "summary",
        "GCRC_standard_station",
    ]

    for att in attributes_to_be_included:
        ds_single.attrs[att] = eval(att)
    return ds_single


# %%
def plot_variables(data):
    """
    Plot each variable in the dataset as a subplot
    Parameters
    ----------
    data : xarray
        xarray with the data
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(data.variables) - 1, figsize=(12, 4))

    # Iterate over variables and plot each one
    for i, (var_name, var_data) in enumerate(data.variables.items()):
        y_dimension = var_data.dims[0]
        if var_name == y_dimension:
            continue
        data[var_name].plot(ax=axs[i], y=y_dimension)
        axs[i].set_title(var_name)
        # if y_dimension contains "Pr"
        if "Pr" in y_dimension:
            axs[i].invert_yaxis()
        # set suptitle
        fig.suptitle(
            f"CTD {data.attrs['GCRC_station_number']} on {data.attrs['time_coverage_start']}, St. {data.attrs['GCRC_standard_station']}"
        )
    return fig


# %%


# %%

# make an easy map with a scatter data["geospatial_lat_min"], data["geospatial_lon_min"] and annotate with  data["GCRC_station_number"]


def plot_coordinates(latitudes, longitudes, annotations, size_km=50):
    # Calculate the bounding box for the given size in kilometers
    lat_center, lon_center = (max(latitudes) + min(latitudes)) / 2, (
        max(longitudes) + min(longitudes)
    ) / 2
    lat_degrees_per_km = 1 / 111.32  # Approximate degrees per kilometer for latitude
    lon_degrees_per_km = (
        500 / 6.3e3 * math.cos(math.radians(lat_center))
    )  # Approximate degrees per kilometer for longitude

    lat_span = size_km * lat_degrees_per_km
    lon_span = size_km * lon_degrees_per_km

    lat_min, lat_max = lat_center - lat_span / 2, lat_center + lat_span / 2
    lon_min, lon_max = lon_center - lon_span / 2, lon_center + lon_span / 2

    # Set up the basemap
    fig, ax = plt.subplots(figsize=(8, 8))
    if size_km < 100:
        resolution = "h"
    else:
        resolution = "i"
    m = Basemap(
        projection="merc",
        llcrnrlat=lat_min,
        urcrnrlat=lat_max,
        llcrnrlon=lon_min,
        urcrnrlon=lon_max,
        resolution=resolution,
    )

    # Draw coastlines
    m.drawcoastlines()

    # Convert latitude and longitude to x and y coordinates
    x, y = m(longitudes, latitudes)

    # Scatter plot
    m.scatter(x, y, marker="o", color="red", s=100)

    # Annotate points with a box containing the station number
    for i, txt in enumerate(annotations):
        ax.annotate(txt, (x[i], y[i]), fontsize=8, ha="right")

    # Show the plot
    return fig


def plot_coordinates_CTD(data):
    """
    Plot the coordinates of the CTD
    Parameters
    ----------
    data : xarray
        xarray with the data
    """
    fig = plot_coordinates(
        latitudes=[
            data.attrs["geospatial_lat_min"],
            data.attrs["geospatial_lat_max"],
        ],
        longitudes=[
            data.attrs["geospatial_lon_min"],
            data.attrs["geospatial_lon_max"],
        ],
        annotations=[data.attrs["GCRC_station_number"]],
        size_km=99,
    )
    return fig


# %%
#  save

# %%


# %%


def save_dsCTD_to_netcdf(ds_single_CTD, path_intermediate_files_netcdf):
    """
    Save xarray to netcdf file
    Parameters
    ----------
    ds_single_CTD : xarray
        xarray with the data
    path_intermediate_files_netcdf : string
        path to the intermediate files
    """
    print(path_intermediate_files_netcdf)

    # Save to netcdf
    ds_single_CTD.to_netcdf(
        f"{path_intermediate_files_netcdf}/CTD_{ds_single_CTD.attrs['GCRC_station_number']}_{ds_single_CTD.attrs['time_coverage_start'].split('T')[0]}_{ds_single_CTD.attrs['geospatial_lat_min']:.2f}N_{ds_single_CTD.attrs['geospatial_lon_min']:.2f}E.nc"
    )


# save_dsCTD_to_netcdf(ds_single_CTD, path_intermediate_files_netcdf)


class InteractivePlot:
    def __init__(self, dataset, function_figure=plot_variables):
        self.figMap = plot_coordinates_CTD(dataset)
        self.fig = function_figure(dataset)

        # Connect key press events to the corresponding method
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        # Show the plot
        plt.show()

    def on_key_press(self, event):
        # Check if the pressed key is 'y' or 'n'
        if event.key == "y":
            print("Yes")
            plt.close(self.fig)
            plt.close(self.figMap)
            save_dsCTD_to_netcdf(ds_single_CTD, path_intermediate_files_netcdf)
        elif event.key == "n":
            print("No")
            plt.close(self.fig)
            plt.close(self.figMap)
        elif event.key == "escape":
            plt.close(self.fig)
            plt.close(self.figMap)
        else:
            print("Invalid key")


# remove last line
def remove_last_line(dfCTD):
    dfCTD = dfCTD.iloc[:-1]
    return dfCTD


def is_string_in_filenames(directory, search_string):
    # List all files in the directory
    files = os.listdir(directory)

    # Check if the search string is in any filename
    for file in files:
        if search_string in file:
            return True

    return False


if __name__ == "__main__":
    list_of_all_stations = []
    for year in ["2018", "2019"]:
        path_data, stat = extract_station_info(path_parent, year)
        # Import profile every file as a ctd
        counter = 0
        fileNames = [
            f for f in listdir(path_data) if os.path.isfile(os.path.join(path_data, f))
        ]

        for fname in Path(path_data).rglob("*.cnv"):
            counter += 1
            down, metadata, this_stat = extract_CTD(fname, stat)
            if down is None:
                continue
            print(f"{this_stat['Name']}")

            if this_stat["Name"] in ["GF18090"]:
                down = remove_last_line(down)
            ds_single_CTD = make_xarray_with_attributes(down, metadata, this_stat)
            if is_string_in_filenames(
                path_intermediate_files_netcdf, this_stat["Name"]
            ):
                print(f"{this_stat['Name']} already converted to netcdf")
            else:
                save_dsCTD_to_netcdf(ds_single_CTD, path_intermediate_files_netcdf)

            # check of str is in list

            # interactive_plot = InteractivePlot(ds_single_CTD)
            list_of_all_stations.append(
                ds_single_CTD.assign_coords(
                    time=ds_single_CTD.attrs["time_coverage_start"],
                    latitude=ds_single_CTD.attrs["geospatial_lat_min"],
                    longitude=ds_single_CTD.attrs["geospatial_lon_min"],
                    station=ds_single_CTD.attrs["GCRC_station_number"],
                )
            )

    combined = xr.concat(list_of_all_stations, dim="station")
    combined.to_netcdf(f"{path_intermediate_files_netcdf}/CTD_all_stations.nc")


# %%
