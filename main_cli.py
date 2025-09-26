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
    cfg_path: Path
    sidescan_files_path: list
    cfg: dict

    def __init__(self, file_path, cfg_path):
        self.filepath = Path(file_path)
        self.cfg_path = Path(cfg_path)
        # Check input sidescan file path
        if self.filepath.is_dir():
            self.sidescan_files_path = []
            for f_path in list(self.filepath.glob("*.xtf")):
                self.sidescan_files_path.append(f_path)
            for f_path in list(self.filepath.glob("*.xtf")):
                self.sidescan_files_path.append(f_path)
        elif self.filepath.is_file():
            self.sidescan_files_path = [self.filepath]
        else:
            raise ValueError(
                f"Argument path: {self.filepath} is not a valid directory or file."
            )

        # Read CFG
        if self.cfg_path.is_file() and self.cfg_path.suffix == ".yml":
            f = open(
                cfg_path,
                "r",
            )
            loaded_dict = yaml.safe_load(f)
            f.close()
            self.cfg = loaded_dict
        else:
            raise ValueError(
                f"CFG can't be found or is no valid yml file: {self.cfg_path}"
            )

        # summarize found files which will be processed
        print(
            f"SidescanTools found the following sonar files:{self.sidescan_files_path}"
        )
        print(
            f"Processed PNGs will be written to: {self.sidescan_files_path[0].parent}"
        )
        print(f"Georeferencing will be working in: {self.cfg["Georef dir"]}")

    def process(self):

        # Process files
        for sidescan_path in self.sidescan_files_path:
            start_timer_processing = timer()
            sidescan_file = SidescanFile(filepath=sidescan_path)

            preproc = SidescanPreprocessor(
                sidescan_file=sidescan_file,
                chunk_size=self.cfg["Slant chunk size"],
            )

            # --- Processing
            if self.cfg["Acitve pie slice filter"]:
                print(f"Pie slice filtering {sidescan_path}")
                preproc.apply_pie_slice_filter()
            print(f"Slant range correcting {sidescan_path}")
            preproc.slant_range_correction(
                nadir_angle=self.cfg["Slant nadir angle"],
                use_intern_altitude=True,
            )
            # TODO: "Nacharbeiten" der Flughöhe
            if self.cfg["Slant gain norm strategy"] == 0:
                print(f"Apply BAC to {sidescan_path}")
                preproc.apply_beam_pattern_correction()
                preproc.apply_energy_normalization()
            elif self.cfg["Slant gain norm strategy"] == 1:
                print(f"Apply EGN to {sidescan_path}")
                preproc.do_EGN_correction(self.cfg["EGN table path"])
            else:
                raise NotImplementedError(
                    f"Gain normalization strategy {self.cfg["Slant gain norm strategy"]} not implemented. Valid options are:\n 0: BAC\n 1: EGN"
                )
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
            if self.cfg["Active convert dB"]:
                proc_data_out_0 = convert_to_dB(proc_data_0)
                proc_data_out_1 = convert_to_dB(proc_data_1)
            # Apply CLAHE
            if self.cfg["Active hist equal"]:
                proc_data_out_0 = hist_equalization(proc_data_out_0)
                proc_data_out_1 = hist_equalization(proc_data_out_1)

            end_timer_processing = timer()

            # --- Georeferencing
            if ACTIVE_GEOREF:
                start_timer_georef = timer()
                # TODO: get settings from CFG
                georeferencer = Georeferencer(
                    filepath=sidescan_path,
                    channel=0,
                    output_folder=self.cfg["Georef dir"],
                    proc_data=proc_data_out_0,
                    vertical_beam_angle=self.cfg["Slant vertical beam angle"],
                )
                georeferencer.process()
                georeferencer = Georeferencer(
                    filepath=sidescan_path,
                    channel=1,
                    output_folder=self.cfg["Georef dir"],
                    proc_data=proc_data_out_1,
                    vertical_beam_angle=self.cfg["Slant vertical beam angle"],
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
            proc_data_out_0 /= np.max(np.abs(proc_data_out_0))
            proc_data_out_0 *= 255
            proc_data_out_1 /= np.max(np.abs(proc_data_out_1))
            proc_data_out_1 *= 255
            img_data = np.hstack((proc_data_out_0, proc_data_out_1))
            Georeferencer.write_img(
                sidescan_path.parent / (sidescan_path.stem + "_processed.png"),
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

            print(f"Processing took {end_timer_processing - start_timer_processing} s")
            if ACTIVE_GEOREF:
                print(f"Georef took {end_timer_georef - start_timer_georef} s")

        plt.show(block=True)

    def gen_egn_table(self):
        raise NotImplementedError(
            "Generation of EGN tables is currently not implemented. Please use the UI variant."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to process sidescan sonar data")
    parser.add_argument("filepath", metavar="FILE", help="Path to xtf/jsf file")
    parser.add_argument("cfg", metavar="FILE", help="Path to cfg")
    parser.add_argument("-g", "--gen_egn", action="store_true")

    args = parser.parse_args()
    print("args:", args)

    sidescantools = SidescanToolsMain(args.filepath, args.cfg)
    if args.gen_egn:
        sidescantools.gen_egn_table()
    else:
        sidescantools.process()

# T:\\projekte\\intern\\geisternetze\\seekuh2024_kiel\\990F\\StarfishLog_20240822_131816.xtf U:\\git\\ghostnetdetector\\sidescan_out_kiel\\project_info.yml
# T:\\projekte\\intern\\geisternetze\\seekuh2024_kolding\\sonar\\2024-08-08a\\StarfishLog_20240808_115824.xtf U:\\git\\ghostnetdetector\\sidescan_out_2\\project_info.yml
# D:\\sidescan_greinert\\2025-03-17_08-35-38_0.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\sidescan_greinert\\2025-03-17_08-30-44_0.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\sidescan_greinert\\2025-03-17_08-25-31_0.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\downloads\\MGDS_Download_1\\CentralAmerica_NicLakes\\niclakes.sidescan.26-MAY-2006-03.xtf D:\\sidescan_greinert\\project_info.yml
# D:\\downloads\\MGDS_Download_1\\NBP0505\\NBP050501C.XTF D:\\sidescan_greinert\\project_info.yml
