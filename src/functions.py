# %%
import ctd
import pandas as pd
import warnings


def dataframe_CTD_from_cnv(fname):
    """
    DataFrame constructor to open Seabird CTD CNV-ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> cast = ctd.from_cnv(data_path.joinpath('CTD_big.cnv.bz2'))
    >>> downcast, upcast = cast.split()
    >>> ax = downcast['t090C'].plot_cast()

    """
    f = ctd.read._read_file(fname)
    metadata = ctd.read._parse_seabird(f.readlines(), ftype="cnv")

    f.seek(0)
    df = pd.read_fwf(
        f,
        header=None,
        index_col=None,
        names=metadata["names"],
        skiprows=metadata["skiprows"],
        delim_whitespace=True,
        widths=[11] * len(metadata["names"]),
    )
    f.close()

    prkeys = ["prM ", "prE", "prDM", "pr50M", "pr50M1", "prSM", "prdM", "pr", "depSM"]
    prkey = [key for key in prkeys if key in df.columns]
    if len(prkey) != 1:
        # if prkey contains "depSM"
        if "pr" in prkey:
            prkey = "pr"
        elif "prdM" in prkey:
            prkey = "prdM"
        else:
            raise ValueError(f"Expected one pressure/depth column, got {prkey}.")
    df.set_index(prkey, drop=True, inplace=True)
    df.index.name = "Pressure [dbar]"
    if prkey == "depSM":
        lat = metadata.get("lat", None)
        if lat is not None:
            df.index = gsw.p_from_z(
                df.index,
                lat,
                geo_strf_dyn_height=0,
                sea_surface_geopotential=0,
            )
        else:
            warnings.war(
                f"Missing latitude information. Cannot compute pressure! Your index is {prkey}, "
                "please compute pressure manually with `gsw.p_from_z` and overwrite your index.",
            )
            df.index.name = prkey

    name = ctd.read._basename(fname)[1]

    dtypes = {"bpos": int, "pumps": bool, "flag": bool}
    for column in df.columns:
        if column in dtypes:
            df[column] = df[column].astype(dtypes[column])
        else:
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                ctd.read.warnings.warn("Could not convert %s to float." % column)

    metadata["name"] = str(name)
    setattr(df, "_metadata", metadata)
    return df


# %%


def extended_metadata(lines, ftype):
    """Parse searbird formats."""
    # Initialize variables.
    metadata = {}
    header, config, names = [], [], []
    for k, line in enumerate(lines):
        line = line.strip()

        # Only cnv has columns names, for bottle files we will use the variable row.
        if ftype == "cnv":
            if "# name" in line:
                name, unit = line.split("=")[1].split(":")
                name, unit = list(map(_normalize_names, (name, unit)))
                names.append(name)

        # Seabird headers starts with *.
        if line.startswith("*"):
            header.append(line)

        # Seabird configuration starts with #.
        if line.startswith("#"):
            config.append(line)

        # NMEA position and time.
        if "NMEA Latitude" in line:
            hemisphere = line[-1]
            lat = line.strip(hemisphere).split("=")[1].strip()
            lat = np.float_(lat.split())
            if hemisphere == "S":
                lat = -(lat[0] + lat[1] / 60.0)
            elif hemisphere == "N":
                lat = lat[0] + lat[1] / 60.0
            else:
                raise ValueError("Latitude not recognized.")
        if "NMEA Longitude" in line:
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split("=")[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == "W":
                lon = -(lon[0] + lon[1] / 60.0)
            elif hemisphere == "E":
                lon = lon[0] + lon[1] / 60.0
            else:
                raise ValueError("Latitude not recognized.")
        if "NMEA UTC (Time)" in line:
            time = line.split("=")[-1].strip()
            # Should use some fuzzy datetime parser to make this more robust.
            time = datetime.strptime(time, "%b %d %Y %H:%M:%S")

        # cnv file header ends with *END* while
        if ftype == "cnv":
            if line == "*END*":
                skiprows = k + 1
                break
        else:  # btl.
            # There is no *END* like in a .cnv file, skip two after header info.
            if not (line.startswith("*") | line.startswith("#")):
                # Fix commonly occurring problem when Sbeox.* exists in the file
                # the name is concatenated to previous parameter
                # example:
                #   CStarAt0Sbeox0Mm/Kg to CStarAt0 Sbeox0Mm/Kg (really two different params)
                line = re.sub(r"(\S)Sbeox", "\\1 Sbeox", line)

                names = line.split()
                skiprows = k + 2
                break
    if ftype == "btl":
        # Capture stat names column.
        names.append("Statistic")
    metadata.update(
        {
            "header": "\n".join(header),
            "config": "\n".join(config),
            "names": _remane_duplicate_columns(names),
            "skiprows": skiprows,
            "time": time,
            "lon": lon,
            "lat": lat,
        },
    )
    return metadata
