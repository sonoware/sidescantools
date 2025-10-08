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

PLOT = False
if PLOT:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
from egn_table_build import generate_egn_info, generate_egn_table_from_infos
from datetime import datetime


class SidescanToolsMain:
    filepath: Path
    cfg_path: Path
    sidescan_files_path: list
    cfg: dict
    active_georef: bool

    def __init__(self, file_path, cfg_path, no_georef):
        self.filepath = Path(file_path)
        self.cfg_path = Path(cfg_path)
        self.active_georef = not no_georef
        # Check input sidescan file path
        if self.filepath.is_dir():
            self.sidescan_files_path = []
            for f_path in list(self.filepath.glob("*.xtf")):
                self.sidescan_files_path.append(f_path)
            for f_path in list(self.filepath.glob("*.jsf")):
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
            f"SidescanTools found the following sonar files: {self.sidescan_files_path}"
        )
        print(
            f"Processed PNGs will be written to: {self.sidescan_files_path[0].parent}"
        )
        if self.active_georef:
            print(f"Georeferencing will be working in: {self.cfg["Georef dir"]}")
        else:
            print("Georeferencing disabled")

    def process(self):

        # Process files
        for sidescan_path in self.sidescan_files_path:
            start_timer_processing = timer()
            sidescan_file = SidescanFile(filepath=sidescan_path)

            preproc = SidescanPreprocessor(
                sidescan_file=sidescan_file,
                chunk_size=self.cfg["Slant chunk size"],
            )

            # Refine altitude
            # Therefore apply pie slice filter beforehand to remove artifacts
            if self.cfg["Active pie slice filter"]:
                print(f"Pie slice filtering {sidescan_path}")
                preproc.apply_pie_slice_filter()
            search_range = self.cfg["Bottom line refinement search range"]
            active_depth_refine = self.cfg["Active bottom line refinement"]
            active_single_altitude_offset = self.cfg[
                "Active btm refinement shift by offset"
            ]
            active_btm_smoothing = self.cfg["Active bottom line smoothing"]
            additional_inset = self.cfg["Additional bottom line inset"]
            use_intern_altitude = True
            if active_depth_refine:
                preproc.refine_detected_bottom_line(
                    search_range,
                    active_single_altitude_offset=active_single_altitude_offset,
                    active_bottom_smoothing=active_btm_smoothing,
                    additional_inset=additional_inset,
                )
                use_intern_altitude = False

            # --- Processing
            print(f"Slant range correcting {sidescan_path}")
            preproc.slant_range_correction(
                nadir_angle=self.cfg["Slant nadir angle"],
                use_intern_altitude=use_intern_altitude,
            )

            if self.cfg["Slant gain norm strategy"] == 0:
                print(f"Apply BAC to {sidescan_path}")
                preproc.apply_beam_pattern_correction(
                    angle_num=self.cfg["BAC resolution"]
                )
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
            if self.active_georef:
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
            if active_depth_refine:
                btm_line_mat_port_intern = np.zeros_like(sidescan_file.data[0])
                btm_line_mat_star_intern = np.zeros_like(sidescan_file.data[0])
            for line_idx in range(len(preproc.portside_bottom_dist.astype(int))):
                btm_line_mat_port[
                    line_idx, preproc.portside_bottom_dist.astype(int)[line_idx]
                ] = 1
                btm_line_mat_star[
                    line_idx, preproc.starboard_bottom_dist.astype(int)[line_idx]
                ] = 1
                if active_depth_refine:
                    btm_line_mat_port_intern[
                        line_idx, preproc.intern_altitude_port.astype(int)[line_idx]
                    ] = 1
                    btm_line_mat_star_intern[
                        line_idx, preproc.intern_altitude_star.astype(int)[line_idx]
                    ] = 1

            print(f"Processing took {end_timer_processing - start_timer_processing} s")
            if self.active_georef:
                print(f"Georef took {end_timer_georef - start_timer_georef} s")

            if PLOT:
                overlay_cmap = ListedColormap(["none", "red"])
                overlay_cmap_2 = ListedColormap(["none", "green"])

                plt.figure()
                plt.imshow(20 * np.log10(sidescan_file.data[0]))
                plt.imshow(btm_line_mat_port, cmap=overlay_cmap)
                if active_depth_refine:
                    plt.imshow(btm_line_mat_port_intern, cmap=overlay_cmap_2)

                plt.figure()
                plt.imshow(20 * np.log10(sidescan_file.data[1]))
                plt.imshow(btm_line_mat_star, cmap=overlay_cmap)
                if active_depth_refine:
                    plt.imshow(btm_line_mat_star_intern, cmap=overlay_cmap_2)

                if self.cfg["Active pie slice filter"]:
                    plt.figure()
                    plt.imshow(20 * np.log10(preproc.dat_pie_slice_copy[0]))
                    plt.imshow(btm_line_mat_port, cmap=overlay_cmap)
                    if active_depth_refine:
                        plt.imshow(btm_line_mat_port_intern, cmap=overlay_cmap_2)

                    plt.figure()
                    plt.imshow(20 * np.log10(preproc.dat_pie_slice_copy[1]))
                    plt.imshow(btm_line_mat_star, cmap=overlay_cmap)
                    if active_depth_refine:
                        plt.imshow(btm_line_mat_star_intern, cmap=overlay_cmap_2)

                plt.figure()
                plt.imshow(img_data)

                plt.show(block=True)

    def gen_egn_table(self):

        egn_infos = []
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        egn_table_path = Path(self.sidescan_files_path[0]).parent / (
            "egn_table_" + time_str + ".npz"
        )
        for sidescan_path in self.sidescan_files_path:
            sonar_file_path = Path(sidescan_path)
            bottom_path = sonar_file_path.parent / (
                sonar_file_path.stem + "_bottom_info.npz"
            )
            out_path = sonar_file_path.parent / (sonar_file_path.stem + "_egn_info.npz")
            active_btm_detection_downsampling = False
            if self.cfg["Btm downsampling"] > 1:
                active_btm_detection_downsampling = True
            generate_egn_info(
                filename=sidescan_path,
                bottom_file=bottom_path,
                out_path=out_path,
                chunk_size=self.cfg["Slant chunk size"],
                nadir_angle=self.cfg["Slant nadir angle"],
                active_intern_depth=self.cfg["Slant use intern depth"],
                active_bottom_detection_downsampling=active_btm_detection_downsampling,
                egn_table_parameters=self.cfg["EGN table resolution parameters"],
            )
            egn_infos.append(out_path)

        # generate final EGN Table
        generate_egn_table_from_infos(egn_infos, egn_table_path)

        # clean up
        for egn_info in egn_infos:
            os.remove(egn_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to process sidescan sonar data")
    parser.add_argument(
        "filepath",
        metavar="FILE",
        help="Path to xtf/jsf file or dir containing multiple files",
    )
    parser.add_argument("cfg", metavar="FILE", help="Path to cfg")
    parser.add_argument("-g", "--gen_egn", action="store_true")
    parser.add_argument("-n", "--no_georef", action="store_true")

    args = parser.parse_args()
    print("args:", args)

    sidescantools = SidescanToolsMain(args.filepath, args.cfg, args.no_georef)
    if args.gen_egn:
        sidescantools.gen_egn_table()
    else:
        sidescantools.process()
