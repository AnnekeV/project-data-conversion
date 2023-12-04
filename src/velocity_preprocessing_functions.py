#%%
import scipy.io as sio
import numpy as np
import datetime
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import math as m
from pytides2.tide import Tide
import gsw
from sklearn.decomposition import PCA


def rotate_velocities_with_PCA(ds_velo):
    """
        rotate velocities using PCA, takes in a dataset with SerEmmpersec and SerNmmpersec\
        returns a dataset with rotated velocities
        !!!! Make sure axis point in the direction you want
        """
    X_array = np.squeeze(
        [
            ds_velo.SerEmmpersec.to_numpy().flatten(),
            ds_velo.SerNmmpersec.to_numpy().flatten(),
        ]
    ).transpose()
    arg_no_nans = ~np.isnan(X_array).any(axis=1)
    # remove nans from X
    X = X_array[arg_no_nans]
    pca = PCA(n_components=2).fit(X)
    components = pca.components_
    if components[0][0] < 0:
        # make sure main axis points in the easterly direction
        components[0][0] = components[0][0] * -1
        components[0][1] = components[0][1] * -1
    if components[1][1] > 0:
        # make sure second axis points in the southerly direction
        components[1][0] = components[1][0] * -1
        components[1][1] = components[1][1] * -1

    X_trans = np.dot(X, components.T)
    # add nans back in
    X_pca = np.full(np.shape(X_array), np.nan)
    X_pca[arg_no_nans] = X_trans

    # make a 2d histogram of the data
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    bins = np.arange(-500, 500, 10)
    axes[0].hist2d(X[:, 0], X[:, 1], bins=bins, cmap="Blues")
    axes[1].hist2d(X_pca[:, 1], X_pca[:, 0], bins=bins, cmap="Blues")
    # use same aspect ratio for both plots
    axes[0].set_aspect("equal", "box")
    axes[1].set_aspect("equal", "box")
    axes[0].set(xlabel="E", ylabel="N")
    axes[1].set(xlabel="PCA2, Across", ylabel="PCA1, Along")
    for i, (comp, var) in enumerate(zip(components, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        axes[0].plot(
            [0, comp[0] / 10],
            [0, comp[1] / 10],
            label=f"Component {i}",
            linewidth=1,
            color=f"C{i + 2}",
        )
    axes[0].legend(loc="lower left")
    axes[0].set_title("Original Data")
    axes[1].set_title("PCA Transformed Data")

    print(f"Components {components}")
    print(f"Explained Variance {pca.explained_variance_}")
    print(
        f"Degrees of components {m.degrees(np.arctan2(components[0,1], components[0,0])):.0f}, {m.degrees(np.arctan2(components[1,1], components[1,0])):.0f}"
    )
    # print mean of X_pca and of X
    print(f"Mean of X_pca {np.round(np.nanmean(X_pca, axis=0),1)}")
    print(f"Mean of X {np.round(np.mean(X, axis=0),1)}")

    # calculate magnitude (X[:,0]**2 + X[:,1]**2)**0.5 and print
    print(f"Mean of Magnitude of X {np.mean((X[:,0]**2 + X[:,1]**2)**0.5):.2f} mm/s")
    print(
        f"Mean of Magnitude of X_pca {np.nanmean((X_pca[:,0]**2 + X_pca[:,1]**2)**0.5):.2f} mm/s"
    )

    speed_along = X_pca[:, 0]
    speed_across = X_pca[:, 1]

    return speed_along, speed_across, pca, components


def open_and_proces_mat_adcp(fname, main_angle=None):
    """
    Open and process mat file from ADCP, based on GF10 2018-2019, for both 75 and 300 kHz
    fname = path to mat file (str)
    If other ADCP is used, check if the nbins to be removed
    Returns: ds_velo, constituent, df_const, my_tide_along, my_tide_across, my_prediction_along, my_prediction_across, mean_velocity_water_column_along, mean_velocity_water_column_across, ds_velo_no_outlier_removal
    """
    mat_file = sio.loadmat(fname)
    fkeys = []
    for i in mat_file.keys():
        fkeys.append(i)
    ns = len(np.squeeze(mat_file["SerDay"]))  # number of samples
    nb = len(np.squeeze(mat_file["SerBins"]))  # number of bins
    print("Number of samples: ", ns)
    print("Number of bins: ", nb)
    print(f"Bin size: {int(mat_file['RDIBinSize'])} m")

    # time vector
    time_array = []
    for ii in np.arange(ns, dtype="int"):
        time_array.append(
            datetime(
                year=int(mat_file["SerYear"][ii] + 2000),
                month=mat_file["SerMon"][ii][0],
                day=mat_file["SerDay"][ii][0],
                hour=mat_file["SerHour"][ii][0],
                minute=mat_file["SerMin"][ii][0],
                second=mat_file["SerSec"][ii][0],
            )
        )
    n_obs_day = 86400 / (time_array[1] - time_array[0]).seconds
    print(f"Number of observations per day: {n_obs_day}")
    print("first ping: ", time_array[0])
    print("last ping: ", time_array[-1])
    print(f"sampling interval: {np.mean(np.diff(time_array)).seconds/60:.2f} minutes")

    # construct vertical axis
    mdepth = (
        gsw.z_from_p(
            p=np.mean(mat_file["AnDepthmm"][500:-500].squeeze()) * 1e-3, lat=64
        )
        * -1
    ).round(
        0
    )  # mean of pressure sensor record
    # alternatively, you can use water depth (if known) and mooring sketch
    print(f"Mounting depth: {int(mdepth)}")
    zax = (
        mdepth
        - mat_file["RDIBin1Mid"].squeeze()
        - np.concatenate(
            ([0], np.cumsum(mat_file["RDIBinSize"] * np.ones(nb - 1)))
        ).squeeze()
    )  # z = water depth

    # make xarray from dictionary
    coordinates = ["time", "zax"]
    data = [
        "SerEmmpersec",
        "SerNmmpersec",
        "SerVmmpersec",
        "SerErmmpersec",
        "SerMagmmpersec",
        "SerDir10thDeg",
        "SerPG4",
        "test",
    ]

    data_time = [
        "AnP100thDeg",
        "AnR100thDeg",
        "AnH100thDeg",
        "AnT100thDeg",
        "AnDepthmm",
    ]

    # check if every value in data is in fkeys, otherwise drop it
    for i in data:
        if i not in fkeys:
            data.remove(i)
    for j in data_time:
        if j not in fkeys:
            data_time.remove(j)

    # make xarray dataset with coords and dims time and zax
    ds_velo = xr.Dataset(
        coords={"time": np.array(time_array), "z": zax}, attrs={"units": "mm/s",}
    )
    for i in range(len(data)):
        da = xr.DataArray(
            mat_file[data[i]],
            coords={"time": np.array(time_array), "z": zax},
            attrs={"_FillValue": -32768, "units": "mm/s",},
            name=data[i],
        )
        ds_velo = xr.merge([ds_velo, da],)

    ds_pitch = xr.Dataset(
        coords={"time": np.array(time_array)}
    )  # all variables that only have time as a coordinate
    for i in range(len(data_time)):
        da = xr.DataArray(
            mat_file[data_time[i]].squeeze(),
            coords={"time": np.array(time_array)},
            name=data_time[i],
        )
        ds_pitch = xr.merge([ds_pitch, da],)

    # find all AnDepthmm that are lower then 1000, drop from dataset
    press_threshold = (max(zax) - 20) * 1000
    ds_velo = ds_velo.where(ds_pitch.AnDepthmm > press_threshold, drop=True)
    ds_pitch = ds_pitch.where(ds_pitch.AnDepthmm > press_threshold, drop=True)

    # drop first and last time step
    ds_velo = ds_velo.isel(time=slice(1, -1))
    ds_pitch = ds_pitch.isel(time=slice(1, -1))

    # replace fill values with nan
    ds_velo = ds_velo.where(ds_velo != -32768, np.nan)

    # remove all values where zax < 0
    ds_velo = ds_velo.where(ds_velo.z > 0, drop=True)
    print(f"Number of bins after removing above water surface: {nb}")

    # drop highest bin, depending on which adcp remove x number of bins
    if mdepth > 500:
        # According to Kiki 2 , according to John 10 percent
        n_bad_bins = 2
    else:
        # According to Kiki 1 , according to John 10 percent
        n_bad_bins = 2
    ds_velo = ds_velo.isel(z=slice(0, -n_bad_bins))

    print(
        f"Removing approx to 10 percent, or upper bins based on inspections: {n_bad_bins} bins"
    )

    # update number of bins and time
    nb = len(ds_velo.z)
    ns = len(ds_velo.time)
    zax = ds_velo.z
    time = ds_velo.time

    print("Final number of bins: ", nb)
    print(f"Shallowest bin:  {float(np.min(zax[0])):.1f} m")

    ds_velo_no_outlier_removal = ds_velo.copy(deep=True)

    # find and remove outliers: where velocity is more than 3 standard deviations from the rolling median
    std_dev = ds_velo.SerEmmpersec.std(dim="time") * 3
    ds_velo = ds_velo.where(
        np.abs(
            ds_velo.SerMagmmpersec
            - ds_velo.SerMagmmpersec.rolling(
                time=int(n_obs_day / 24 * 25),
                center=True,
                min_periods=int(0.7 * n_obs_day),
            ).median()
        )
        < std_dev,
        other=np.nan,
    )

    # flagging orignal data
    ds_velo_no_outlier_removal["flag"] = 0
    # make all values in flag 0 where ds_velo.Magmmpersec is nan
    ds_velo_no_outlier_removal["flag"] = ds_velo_no_outlier_removal.flag.where(
        ~np.isnan(ds_velo_no_outlier_removal.SerMagmmpersec), 1
    )

    print("\n Rotating velocities")

    if main_angle:
        print(f"Using fixed angle {main_angle} to rotate velocities")
        # calculate along and across axis and rotte
        vector_along = np.expand_dims(
            [1 * np.cos(m.radians(main_angle)), 1 * np.sin(m.radians(main_angle))],
            axis=1,
        )
        vector_across = np.expand_dims(
            [
                1 * np.cos(m.radians(main_angle + 90)),
                1 * np.sin(m.radians(main_angle + 90)),
            ],
            axis=1,
        )

        v_ori = np.squeeze(
            [
                ds_velo.SerEmmpersec.to_numpy().flatten(),
                ds_velo.SerNmmpersec.to_numpy().flatten(),
            ]
        ).transpose()
        speed_along = v_ori.dot(vector_along)
        speed_across = v_ori.dot(vector_across)
    else:
        print("Using PCA to rotate velocities")
        speed_along, speed_across, pca, _ = rotate_velocities_with_PCA(ds_velo)
    print("\n")

    ds_velo["Alongmmpersec"] = xr.DataArray(
        speed_along.transpose().reshape(ns, len(zax)),
        dims=["time", "z"],
        attrs={"units": "mm/s"},
    )
    ds_velo["Acrossmmpersec"] = xr.DataArray(
        speed_across.transpose().reshape(ns, len(zax)),
        dims=["time", "z"],
        attrs={"units": "mm/s"},
    )

    # calculate tides and remove

    print("\n Calculating tides and removing them")

    def calculate_tidal_prediction(data_array):
        # first remove mean per layer than average over time
        mean_velocity_water_column = (data_array - data_array.mean(dim="time")).mean(
            dim="z"
        )
        mean_velocity_water_column -= mean_velocity_water_column.mean()

        # For a quicker decomposition, we'll only use hourly readings rather than 6-minutely readings.
        v = mean_velocity_water_column.to_numpy().astype("float")
        # fill nan with 0
        v[np.isnan(v)] = 0
        t = mean_velocity_water_column.time

        t_hours = []
        for i in range(len(t)):
            t_hours.append(np_to_datetime(t[i]))

        ##Fit the tidal data to the harmonic model using Pytides
        my_tide = Tide.decompose(
            v, t_hours, n_period=10
        )  # increase n_period to filter out long tidal periods
        ##Predict the tides using the Pytides model.
        my_prediction = my_tide.at(t_hours)

        # print stand deviation of orignal data and prediction
        print(f"Std of original data:  {np.std(v):.2f}")
        print(f"Std of prediction: {np.std(my_prediction):.2f}")

        return my_prediction, mean_velocity_water_column, my_tide

    print("Along")
    (
        my_prediction_along,
        mean_velocity_water_column_along,
        my_tide_along,
    ) = calculate_tidal_prediction(ds_velo.Alongmmpersec)
    print("Across")
    (
        my_prediction_across,
        mean_velocity_water_column_across,
        my_tide_across,
    ) = calculate_tidal_prediction(ds_velo.Acrossmmpersec)

    constituent = [c.name for c in my_tide_along.model["constituent"]]

    df_const = (
        pd.DataFrame(my_tide_across.model, index=constituent)
        .drop("constituent", axis=1)
        .sort_values(by="amplitude", ascending=False)
        .rename(
            columns={"amplitude": "Velocity amplitude [mm/s]", "phase": "Phase [deg]"}
        )
    )
    df_const["%"] = (
        df_const["Velocity amplitude [mm/s]"]
        / df_const["Velocity amplitude [mm/s]"].sum()
        * 100
    ).astype(int)
    df_const.head(15).round(2).style.background_gradient(cmap="Reds", subset=["%"])

    # calculate variance from tide and from original data
    variance_tide = np.var(my_prediction_along)
    variance_original = np.var(mean_velocity_water_column_along)
    print(f"Variance of original data: {variance_original:.2f}")
    print(f"Variance of prediction: {variance_tide:.2f}")
    print(
        f"Variance of prediction as percentage of original data: {variance_tide/variance_original*100:.2f}%"
    )

    ds_velo["Along_res"] = ds_velo["Alongmmpersec"] - xr.DataArray(
        my_prediction_along, dims=["time"]
    )
    ds_velo["Across_res"] = ds_velo["Acrossmmpersec"] - xr.DataArray(
        my_prediction_across, dims=["time"]
    )
    ds_velo.Along_res.attrs = {"units": "mm/s"}
    ds_velo.Across_res.attrs = {"units": "mm/s"}

    return (
        ds_velo,
        constituent,
        df_const,
        my_tide_along,
        my_tide_across,
        my_prediction_along,
        my_prediction_across,
        mean_velocity_water_column_along,
        mean_velocity_water_column_across,
        ds_velo_no_outlier_removal,
    )


if __name__ == "__main__":
    (
        ds_velo,
        constituent,
        df_const,
        my_tide_along,
        my_tide_across,
        my_prediction_along,
        my_prediction_across,
        mean_velocity_water_column_along,
        mean_velocity_water_column_across,
        ds_velo_no_outlier_removal,
    ) = open_and_proces_mat_adcp(f300)


# %%


f300 = "/Users/annek/Library/CloudStorage/OneDrive-SharedLibraries-NIOZ/PhD Anneke Vries - General/Data/Moorings/20190612_ADCP_GF10/ADCP300.mat"
f75 = "/Users/annek/Library/CloudStorage/OneDrive-SharedLibraries-NIOZ/PhD Anneke Vries - General/Data/Moorings/20190612_ADCP_GF10/GF18_/GF18_/ADCP75.mat"

(
    ds_velo_300,
    _,
    df_cons300,
    my_tide_along,
    _,
    _,
    _,
    _,
    _,
    ds_velo_no_outlier_removal_300,
) = open_and_proces_mat_adcp(f300)
(
    ds_velo_75,
    _,
    df_cons75,
    _,
    _,
    _,
    _,
    _,
    _,
    ds_velo_no_outlier_removal_75,
) = open_and_proces_mat_adcp(f75)
print("Total amplitude tide : ", df_cons75["Velocity amplitude [mm/s]"].sum().round(2))

# %%
