import matplotlib.pyplot as plt
import CTD_monthly
import importlib
from pathlib import Path

# Reload module
importlib.reload(CTD_monthly)
import os
from os import listdir
import CTD_monthly
import importlib
from pathlib import Path

import matplotlib.pyplot as plt

# Reload module
importlib.reload(CTD_monthly)


class InteractivePlot:
    def __init__(self, dataset, function_figure=CTD_monthly.plot_variables):
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
            CTD_monthly.save_dsCTD_to_netcdf(
                ds_single_CTD, CTD_monthly.path_intermediate_files_netcdf
            )

        elif event.key == "n":
            print("No")

            plt.close(self.fig)

        elif event.key == "escape":
            plt.close(self.fig)
        else:
            print("Invalid key")


if __name__ == "__main__":
    for year in ["2018", "2019"]:
        path_data, stat = CTD_monthly.extract_station_info(
            CTD_monthly.path_parent, year
        )

        # Import profile every file as a ctd

        counter = 0
        fileNames = [
            f for f in listdir(path_data) if os.path.isfile(os.path.join(path_data, f))
        ]

        for fname in Path(path_data).rglob("*.cnv"):
            counter += 1
            down, metadata, this_stat = CTD_monthly.extract_CTD(fname, stat)
            if len(down) < 1:
                continue

            ds_single_CTD = CTD_monthly.make_xarray_with_attributes(
                down, metadata, this_stat
            )

            interactive_plot = InteractivePlot(ds_single_CTD)
            if counter >= 1:
                break
