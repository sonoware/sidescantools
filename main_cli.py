from pathlib import Path
import argparse
import yaml
import numpy as np
from sidescan_file import SidescanFile
from sidescan_preproc import SidescanPreprocessor
from georef_thread import Georeferencer
from custom_widgets import (
    convert_to_dB,
    hist_equalization,
)  # TODO: move these function somewhere else to remove Qt import here
from timeit import default_timer as timer
import os


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

        # TODO: This is only a helper to laod bottom data - needs to be rewritten with automatic bottom detection from sonar file information
        bottom_file = self.filepath.parent / (self.filepath.stem + "_bottom_info.npz")
        bottom_info = np.load(bottom_file)
        # Check if downsampling was applied
        try:
            downsampling_factor = bottom_info["downsampling_factor"]
        except:
            downsampling_factor = 1

        portside_bottom_dist = bottom_info["bottom_info_port"].flatten()[
            : sidescan_file.num_ping
        ]
        starboard_bottom_dist = bottom_info["bottom_info_star"].flatten()[
            : sidescan_file.num_ping
        ]

        if downsampling_factor != 1:
            # rescale bottom info
            portside_bottom_dist = portside_bottom_dist * downsampling_factor
            starboard_bottom_dist = starboard_bottom_dist * downsampling_factor
            downsampling_factor = 1

        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=self.cfg["Slant chunk size"],
            downsampling_factor=downsampling_factor,
        )
        # flip order for xtf files to contain backwards compability
        if self.filepath.suffix.casefold() == ".xtf":
            portside_bottom_dist = np.flip(portside_bottom_dist)
            starboard_bottom_dist = np.flip(starboard_bottom_dist)

        preproc.portside_bottom_dist = portside_bottom_dist
        preproc.starboard_bottom_dist = starboard_bottom_dist
        preproc.napari_portside_bottom = np.zeros(
            (preproc.num_chunk, preproc.chunk_size), dtype=int
        )
        preproc.napari_starboard_bottom = np.zeros(
            (preproc.num_chunk, preproc.chunk_size), dtype=int
        )
        for chunk_idx in range(preproc.num_chunk):
            port_chunk = portside_bottom_dist[
                chunk_idx * preproc.chunk_size : (chunk_idx + 1) * preproc.chunk_size
            ]
            preproc.napari_portside_bottom[chunk_idx, : len(port_chunk)] = port_chunk
            star_chunk = starboard_bottom_dist[
                chunk_idx * preproc.chunk_size : (chunk_idx + 1) * preproc.chunk_size
            ]
            preproc.napari_starboard_bottom[chunk_idx, : len(star_chunk)] = star_chunk

        # --- Processing
        print(f"Pie slice filtering {self.filepath}")
        preproc.apply_pie_slice_filter()
        print(f"Slant range correcting {self.filepath}")
        preproc.slant_range_correction(
            active_interpolation=True,
            nadir_angle=self.cfg["Slant nadir angle"],
            save_to=None,
            active_mult_slant_range_resampling=True,
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
                or str(file_path).endswith(".vrt")
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
            self.cfg["Georef dir"] + "\\" + self.filepath.stem + "_processed.png",
            img_data.astype(np.uint8),
        )

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
