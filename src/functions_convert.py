# %%
import ctd
import pandas as pd
import warnings

def split_subline_with_multiple_equal_signs(subline):
    """
    Split subline into a dictionary, based on multile = sign

    Parameters
    ----------
    subline : string
        subline of the header or config of the CTD file, as a string

    Returns
    -------
    data_dict : dictionary
        dictionary with the subline information
    """
    data_dict = {}
    subsublines = subline.split(" ")
    for subsubline in subsublines:
        if "=" in subsubline:
            key, value = subsubline.split("=")
            # Remove leading/trailing whitespaces from the value
            key = key.strip().strip("<")
            value = value.strip().strip(">")
            try:
                value = float(value)
            except ValueError:
                pass
            data_dict[key] = value
        else: 
            continue
    return data_dict

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
                try: 
                    key, value = subline.split("=")
                except ValueError:
                    split_subline_with_multiple_equal_signs(subline)
                key = key.strip().strip("<")
                value = (
                    value.strip().strip("/>")
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

# %%
