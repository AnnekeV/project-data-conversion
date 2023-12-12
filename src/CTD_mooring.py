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
import functions_convert as func
import importlib
import xarray as xr
from CTD_monthly import extract_station_info




importlib.reload(func)


plt.style.use("fast")
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
_, dfStationOverview = extract_station_info(path_parent=path_parent, year="2018")
dfMooringOverview18 =  dfStationOverview[dfStationOverview.Type == "MOR"]
_, dfStationOverview = extract_station_info(path_parent=path_parent, year="2019")
dfMooringOverview19 =  dfStationOverview[dfStationOverview.Type == "MOR"]
dfMooringOverview = pd.concat([dfMooringOverview18, dfMooringOverview19])

def open_cnv(fname, remove_5m=True):
    """Open cnv file and export dataframe down and metadata and cast"""
    def read_first_line(fname):
        with open(fname, 'r', encoding='ISO-8859-1') as f:
            first_line = f.readline()
        return first_line

    cast = ctd.from_cnv(fname)  #
    metadata = cast._metadata
    metadata["config_original"] = metadata["config"]
    metadata["header_original"] = metadata["header"]
    metadata["header"] = func.split_header(metadata["header"])
    metadata["config"] = func.split_header(metadata["config"])
    metadata["StatNumber"] = metadata["name"].split("SBE")[1][:4]
    metadata["Instrument"] = read_first_line(fname).strip("*").split("Data")[0].strip()
    
    for stat in dfMooringOverview.index:
        if metadata["StatNumber"] in dfMooringOverview.loc[stat, "Comments"]:
            for col in dfMooringOverview.columns:
                metadata[col] = dfMooringOverview.loc[stat, col]

    cast = rename_variables(cast)
    return cast, metadata


def rename_variables(df):
    """renames the variables as in cnv to shorter and equal names  AND REMOVES OUTLIERS"""
    df = df.rename(
        columns={
            "t090": "temp_insitu",
            "tv290C": "temp_insitu",
            "potemp090C": "temp_pot",
            "c0S/m": "cond",
            "cond0S/m": "cond",
            "sal00": "sal_prac",
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

    return df


def remove_above_zero(cast):
    """remove all vales lower than 0.1 dbar"""
    cast = cast.reset_index()
    cast.loc[cast["Pressure [dbar]"] < 0.15, "flag"] = True
    print(f"Nr. rows above water surface (dbar<0.15): {cast.flag.sum()}")
    # cast = cast.set_index("Pressure [dbar]")
    return cast




def time_to_date(fname, cast, start_time="Jan 1 2018"):
    """Changes nr of Julian days to datetime object
    Insert dataframe with at least one column 'time'  and get adjusted dataframe back
    """

    dt_start_time = datetime.strptime(start_time, "%b %d %Y")
    cast["timedelta"] = cast.time.apply(lambda x: timedelta(x))
    cast["date"] = cast["timedelta"] + dt_start_time
    cast["date"] = cast["date"].dt.round("1min")


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# %%

# define global atttributes
def define_global_attributes(metadata=None):
    datenow_utc = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    user_name = "Anneke Vries"
    if metadata is None:
        metadata = {
            "Latitude": 64.61238333333333,
            "Longitude": -50.954233333333335
        }
    global_attributes = {
        "data_type": "CTD",
        "featureType": "timeseries",
        "keyword_vocabulary": "GCMD Science Keywords",
        "keywords": "EARTH SCIENCE>OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE, EARTH SCIENCE>OCEANS>OCEAN PRESSURE>WATER PRESSURE,EARTH SCIENCE>OCEANS>SALINITY/DENSITY>SALINITY",
        "Conventions": "ACDD-1.3",
        "datenow_utc": f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "history": f"{datenow_utc} converted to netcdf with xarray by {user_name}",
        "processing_level": "manual inspection, removing outliers (>3x std), remove measurements close to water surface (<0.15 dbar)",
        "date_created": datenow_utc,
        "creator_type": "person",
        "creator_institution": "NIOZ Royal Netherlands Institute for Sea Research",
        "creator_name": user_name,
        "creator_email": "anneke.vries@nioz.nl",
        "creator_url": "https://orcid.org/0000-0001-9970-1189",
        "project": "NIOZ-UU Greenland fjords as gateways between the ice sheet and the ocean",
        "platform": "mooring",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "iso_topic_category": "oceans",
        "geospatial_lat_min": metadata["Latitude"],
        "geospatial_lat_max": metadata["Latitude"],
        "geospatial_lon_min": metadata["Longitude"],
        "geospatial_lon_max": metadata["Longitude"],
        "source": "Mooring with CTD at GF10",
        "GCRC_standard_station": "GF10",
        "Bottom depth": "560 m",
    }
    return global_attributes


# only select files that start with 2019
files = [f for f in fileNames if f.startswith("2019")]
files = fileNames
nr_moorings = len(files)
all_moorings = pd.DataFrame()

dsAllMoorings = xr.Dataset()

def check_if_var_not_nan(df, var):
    """Checks if variable contains only NaN values, if so, raises error
    Parameters
    ----------
    df : dataframe
        dataframe with variable
    var : string
    """
    if df[var].isnull().values.all():
        print(f"Variable {var} contains only NaN values")
        raise ValueError(f"Variable {var} for {df.depth[0]} m contains only NaN values")


for i in range(nr_moorings):
    depth = files[i].split("_")[2].split("m.")[0]
    if depth == "5":
        continue
    cast, metadata = open_cnv(f"{path_data}/{files[i]}")
    # parse dates
    StartTimeInstrument = datetime.strptime(metadata["config"]["start_time"].split(" [")[0],  "%b %d %Y %H:%M:%S")
    StartTimeJulianDays = datetime.strptime(f"{StartTimeInstrument.year}-01-01", "%Y-%m-%d")

    time_to_date(f"{path_data}/{files[i]}", cast, StartTimeJulianDays.strftime("%b %d %Y"))

    cast["depth"] = depth
    # rename Pressure [dbar] to pressure
    cast = cast.rename(columns={"Pressure [dbar]": "pressure", "time": "days_julian"})
    for var in ["temp_pot", "sal_prac", "dens", "temp_insitu", "cond"]:
        check_if_var_not_nan(cast, var)


    all_moorings = pd.concat([all_moorings, cast]).reset_index(drop=True)

    cast["id"] = cast["depth"].astype("int")
    cast = cast.set_index(["id", "date"])
    dsSingleMooring = cast.drop(columns=["index", "timedelta"]).to_xarray()

    print(f"\nMean pressure: {cast.pressure.mean():.0f} dbar, std: {cast.pressure.std():.2f} dbar")
    print(f"Mean time interval: {cast.days_julian.diff().mean()*24*60*60:.0f} s")
    print(f"Instrument used is {metadata['Instrument']}")
    start_time_depth_str, end_time_depth_str = StartTimeInstrument.strftime("%Y-%m-%dT%H:%MZ"),cast.reset_index().date.max().strftime("%Y-%m-%dT%H:%MZ")
    print(f"Measured between {start_time_depth_str} and {end_time_depth_str}\n")

    #set attributes    
    dsSingleMooring["temp_pot"].attrs = {"long_name": "Potential temperature", "units": "degree C", "short_name": r"$\theta$ [$^o$C]"}
    dsSingleMooring["temp_insitu"].attrs = {"long_name": "In situ temperature as measured ", "units": "degree C", "short_name": r"$T$ [$^o$C]"}
    dsSingleMooring["sal_prac"].attrs = {"long_name": "Practical salinity", "units": "PSU", "short_name": r"$S$ [PSU]"}
    dsSingleMooring["dens"].attrs = {"long_name": "Density", "units": "kg/m3", "short_name": r"$\rho$ [kg m$^{-3}$]"}
    dsSingleMooring["cond"].attrs = {"long_name": "Conductivity as measured", "units": "S/m", "short_name": r"cond [S m$^{-1}$]"}
    dsSingleMooring["depth"].attrs = {"long_name": "Planned depth of instrument", "units": "m", "short_name": r"z [m]"}
    dsSingleMooring["days_julian"].attrs = {"long_name": "Time in julian days since start of specific year", "units": f"days since {StartTimeJulianDays}", "short_name": r"t [days]"}
    dsSingleMooring["date"].attrs = {"long_name": "Date in datetime format, rounded to 1 min",}
    dsSingleMooring["pressure"].attrs = {"long_name": "Pressure", "units": "dbar", "short_name": r"p [dbar]"}
    dsSingleMooring["flag"].attrs = {"long_name": f"Flagged measurements, 1 if measurement is classified as outlier, because of being too close to the ocean surface (<0.15 dBar)"}
    
    # More attributes for single mooring
    global_attributes = define_global_attributes(metadata)
    dsSingleMooring = dsSingleMooring.assign_attrs({
        "instrument": metadata['Instrument'],
        "geospatial_vertical_min": cast.pressure.max(),
        "geospatial_vertical_max": cast.pressure.min(),
        "time_coverage_start" : start_time_depth_str,
        "time_coverage_end" : end_time_depth_str,
        "title" : f"{global_attributes['data_type']} {global_attributes['featureType']} at Station GF10 on between{start_time_depth_str} and {end_time_depth_str}, for an approximate depth of {depth} m in Nuup Kangerlua, Greenland",         
        "summary": f"The file contains potential temperature, practical salinity and depth measurements every 10 minutes at depth {depth} m  at station GF10. The raw data was measured at {metadata['Latitude']:.3f}N, {metadata['Longitude']:.3f}E, with a Sea-Bird SBE37SM. The data was collected by the Greenland Climate Research Center (GCRC)",
    })
    dsSingleMooring.attrs = {**dsSingleMooring.attrs, **global_attributes}    
    ncName = f"Mooring_CTD_GCRC_GF10_{depth}m_{StartTimeInstrument.strftime('%Y-%m-%d')}_{cast.reset_index().date.max().strftime('%Y-%m-%d')}_{metadata['Latitude']:.2f}N_{metadata['Longitude']:.2f}E.nc"
    dsSingleMooring.to_netcdf(f"{path_parent.joinpath('data', 'temp', 'netcdf', ncName)}")
    

    dsAllMoorings = xr.merge([dsAllMoorings, dsSingleMooring])
    dsAllMoorings.attrs = {**dsAllMoorings.attrs, **dsSingleMooring.attrs}

latest_date = pd.to_datetime(dsAllMoorings.date).max().strftime("%Y-%m-%dT%H:%MZ")
earliest_date = pd.to_datetime(dsAllMoorings.date).min().strftime("%Y-%m-%dT%H:%MZ")

attributes_combined_moorings = {
    "time_coverage_end" : latest_date,
    "time_coverage_start" : earliest_date,
    "title": f"{global_attributes['data_type']} {global_attributes['featureType']} at Station GF10 on between {earliest_date} and {latest_date}, for depths {dsAllMoorings.id.to_numpy()} m in Nuup Kangerlua, Greenland",
    "summary": f"The file contains potential temperature, practical salinity and depth measurements every 10 minutes at depths {dsAllMoorings.id.to_numpy()} m  at station GF10. The raw data was measured at {metadata['Latitude']:.3f}N, {metadata['Longitude']:.3f}E, with a Sea-Bird SBE37. The data was collected by the Greenland Climate Research Center (GCRC). ID is planned mooring depth of specific instrument. Flagged for outliers (bad ='True')",
    }

dsAllMoorings= dsAllMoorings.assign_attrs(global_attributes)
dsAllMoorings= dsAllMoorings.assign_attrs(attributes_combined_moorings)


depth_string = f"{dsAllMoorings.id.to_numpy()}".replace(" ", "_").replace("[", "").replace("]","")

ncName = f"Mooring_CTD_GCRC_GF10{depth_string}m_{earliest_date[:10]}_{latest_date[:10]}_{metadata['Latitude']:.2f}N_{metadata['Longitude']:.2f}E.nc"
dsAllMoorings.to_netcdf(f"{path_parent.joinpath('data', 'temp', 'netcdf', ncName)}")


print("Done")
print(attributes_combined_moorings["summary"])
print(attributes_combined_moorings["title"])

plotting = True

plt.style.use('seaborn-ticks')
if plotting is True:
    for var in ["temp_pot", "sal_prac", "dens", "temp_insitu", "cond"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        dsAllMoorings[var].rename({'id': 'Mounting\ndepth[m]'}).plot.scatter(x='date', s=1, zorder=1, color = 'k', alpha = 0.5)
        # plot rolling mean
        dsAllMoorings[var].rename({'id': 'Mounting\ndepth[m]'}).rolling(date=int(25*6), center=True).mean().plot.line(x='date', zorder=2)
        plt.xlabel("")
        plt.ylabel(dsAllMoorings[var].short_name)
        plt.title(dsAllMoorings[var].long_name)
        plt.xlim(dsAllMoorings.date.min(), dsAllMoorings.date.max())
        plt.show()




# %%
