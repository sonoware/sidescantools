from pathlib import Path
import argparse
import yaml
import numpy as np
from sidescan_file import SidescanFile
from sidescan_preproc import SidescanPreprocessor
from georef_thread import Georeferencer
from aux_functions import convert_to_dB, hist_equalization
from timeit import default_timer as timer
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# TODO: delete
ACTIVE_GEOREF = False


class SidescanToolsMain:
    filepath: Path
    cfg: dict

    def __init__(self, file_path, cfg_path):
        self.filepath = Path(file_path)
        f = open(
            cfg_path,
            "r",
        )
        loaded_dict = yaml.safe_load(f)
        f.close()
        self.cfg = loaded_dict

        # TODO: checkt that these files exist
        # TODO: write cfg parser to make working with cfg parameters more convenient

    def process(self):
        # TODO: get this from meta info and build functionalities to estimate bottom line without needing user input
        # TODO: Structuring and so on...

        # Load file
        start_timer_processing = timer()
        sidescan_file = SidescanFile(filepath=self.filepath)

        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=self.cfg["Slant chunk size"],
        )

        # --- Processing
        print(f"Pie slice filtering {self.filepath}")
        preproc.apply_pie_slice_filter()
        print(f"Slant range correcting {self.filepath}")
        preproc.slant_range_correction(
            nadir_angle=self.cfg["Slant nadir angle"],
            use_intern_altitude=True,
        )
        # TODO: Decide whether to use EGN or BAC via CFG
        # TODO: Therefore one could implement and incremental EGN Table building and corection
        print(f"Apply BAC to {self.filepath}")
        preproc.apply_beam_pattern_correction()
        preproc.apply_energy_normalization()
        preproc.egn_corrected_mat = np.hstack(
            (
                np.fliplr(preproc.sonar_data_proc[0]),
                preproc.sonar_data_proc[1],
            )
        )

        ping_len = int(np.shape(preproc.egn_corrected_mat)[1] / 2)
        proc_data_0 = preproc.egn_corrected_mat[:, 0:ping_len]
        proc_data_0 = np.nan_to_num(
            proc_data_0
        )  # remove nans from excluding far/nadir unknown values
        proc_data_1 = preproc.egn_corrected_mat[:, ping_len:]
        proc_data_1 = np.nan_to_num(proc_data_1)
        # Convert data to dB
        proc_data_out_0 = convert_to_dB(proc_data_0)
        proc_data_out_1 = convert_to_dB(proc_data_1)
        # Apply CLAHE
        proc_data_out_0 = hist_equalization(proc_data_out_0)
        proc_data_out_1 = hist_equalization(proc_data_out_1)

        end_timer_processing = timer()

        # --- Georeferencing
        if ACTIVE_GEOREF:
            start_timer_georef = timer()
            # TODO: get settings from CFG
            georeferencer = Georeferencer(
                filepath=self.filepath,
                channel=0,
                output_folder=self.cfg["Georef dir"],
                proc_data=proc_data_out_0,
                vertical_beam_angle=60,
            )
            georeferencer.process()
            georeferencer = Georeferencer(
                filepath=self.filepath,
                channel=1,
                output_folder=self.cfg["Georef dir"],
                proc_data=proc_data_out_1,
                vertical_beam_angle=60,
            )
            georeferencer.process()

            print(f"Cleaning ...")
            for file in os.listdir(self.cfg["Georef dir"]):
                file_path = os.path.join(self.cfg["Georef dir"], file)
                if (
                    str(file_path).endswith(".png")
                    or str(file_path).endswith(".txt")
                    or str(file_path).endswith("tmp.tif")
                    or str(file_path).endswith(".points")
                    or str(file_path).endswith(".xml")
                    or str(file_path).endswith("tmp.csv")
                ):
                    try:
                        os.remove(file_path)
                    except FileNotFoundError:
                        print(f"File Not Found: {file_path}")

            print("Cleanup done")

            end_timer_georef = timer()

        # Also write processed data as png for comparison
        img_data = np.hstack((proc_data_out_0, proc_data_out_1))
        img_data /= np.max(np.abs(img_data))
        img_data *= 255
        Georeferencer.write_img(
            self.filepath.parent / (self.filepath.stem + "_processed.png"),
            img_data.astype(np.uint8),
        )

        # Plotting to find errors and test bottom line strategy
        btm_line_mat_port = np.zeros_like(sidescan_file.data[0])
        btm_line_mat_star = np.zeros_like(sidescan_file.data[0])
        for line_idx in range(len(preproc.portside_bottom_dist.astype(int))):
            btm_line_mat_port[
                line_idx, preproc.portside_bottom_dist.astype(int)[line_idx]
            ] = 1
            btm_line_mat_star[
                line_idx, preproc.starboard_bottom_dist.astype(int)[line_idx]
            ] = 1
        overlay_cmap = ListedColormap(["none", "red"])
        plt.figure()
        plt.imshow(20 * np.log10(sidescan_file.data[0]))
        plt.imshow(btm_line_mat_port, cmap=overlay_cmap)

        plt.figure()
        plt.imshow(20 * np.log10(sidescan_file.data[1]))
        plt.imshow(btm_line_mat_star, cmap=overlay_cmap)

        plt.figure()
        plt.imshow(img_data)
        plt.show(block=True)

        print(f"Processing took {end_timer_processing - start_timer_processing} s")
        print(f"Georef took {end_timer_georef - start_timer_georef} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to process sidescan sonar data")
    parser.add_argument("filepath", metavar="FILE", help="Path to xtf/jsf file")
    parser.add_argument("cfg", metavar="FILE", help="Path to cfg")

    args = parser.parse_args()
    print("args:", args)

    sidescantools = SidescanToolsMain(args.filepath, args.cfg)
    sidescantools.process()

# T:\\projekte\\intern\\geisternetze\\seekuh2024_kiel\\990F\\StarfishLog_20240822_131816.xtf U:\\git\\ghostnetdetector\\sidescan_out_kiel\\project_info.yml
# T:\\projekte\\intern\\geisternetze\\seekuh2024_kolding\\sonar\\2024-08-08a\\StarfishLog_20240808_115824.xtf U:\\git\\ghostnetdetector\\sidescan_out_2\\project_info.yml
# D:\\sidescan_greinert\\2025-03-17_08-35-38_0.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\sidescan_greinert\\2025-03-17_08-30-44_0.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\sidescan_greinert\\2025-03-17_08-25-31_0.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\downloads\\MGDS_Download_1\\CentralAmerica_NicLakes\\niclakes.sidescan.26-MAY-2006-03.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\downloads\\MGDS_Download_1\\NBP0505\\NBP050501C.XTF D:\\sidescan_greinert\\project_info.yml
