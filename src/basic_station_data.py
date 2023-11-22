# %%
import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic as GD
from geopy.distance import great_circle as GRC


# %%
class Error(Exception):
    """"""

    "Base class for other exceptions"
    ""
    pass


class ValueTooLargeError(Error):
    """Raised when the input value is too large"""

    pass


# %%
path_parent = Path.cwd().parent
path_stations = path_parent.joinpath("data", "raw", "Stations GHF.csv")
stat_loc = pd.read_csv(path_stations)


def dist_lat_lon(pointA, pointB):
    """pointA,B as tuples, returns in km
    (latitude, longitude)"""
    return GD(pointA, pointB).km


def find_closest_stations(pointA):
    """find two closest stations in Nuup Kangerdlua
    point A is tuple (latitude, longitude)
    returns dictionaries: closest station, and second closest station with keys "distance" in km and "name"
    """
    closest_st = {"name": "", "distance": 999}
    second_closest = {"name": "", "distance": 999}
    for i in range(len(stat_loc.index)):
        dist_st = dist_lat_lon(
            (stat_loc.loc[i, "Latitude"], stat_loc.loc[i, "Longitude"]), pointA
        )  # distance to station
        if dist_st < closest_st["distance"]:
            second_closest["distance"] = closest_st["distance"]
            second_closest["name"] = closest_st["name"]

            closest_st["distance"] = dist_st
            closest_st["name"] = stat_loc.loc[i, "Station"]

        elif dist_st < second_closest["distance"]:
            second_closest["distance"] = dist_st
            second_closest["name"] = stat_loc.loc[i, "Station"]
    return closest_st, second_closest


def find_distance_from_fjordmouth(pointA):
    closest, second = find_closest_stations(pointA)
    station_index = stat_loc.set_index("Station")
    dist_between = (
        station_index.loc[closest["name"], "Distance"]
        - station_index.loc[second["name"], "Distance"]
    )  # distance between stations
    if closest["distance"] < 1.0:
        dist_from_mouth = station_index.loc[closest["name"], "Distance"]

    elif (closest["distance"] > abs(dist_between) + 1) or (closest["distance"] > 15):
        dist_from_mouth = 999
        print(
            f"Something is wrong with {pointA}, distance to closest station is {closest['distance']:.1f} km"
        )
    else:
        if (
            dist_between < 0
        ):  # if the closest station is closer to the mouth of the fjord
            dist_from_mouth = (
                station_index.loc[closest["name"], "Distance"] + closest["distance"]
            )
        else:  # if the second closest is closer to the mouth of the fjord
            dist_from_mouth = (
                station_index.loc[closest["name"], "Distance"] - closest["distance"]
            )
    return dist_from_mouth


if __name__ == "__main__":
    find_closest_stations((64.4205, -50.5577166667))
    find_distance_from_fjordmouth((64.5005, -50.5577166667))


# %%
