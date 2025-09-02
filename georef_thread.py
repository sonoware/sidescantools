import argparse
from pathlib import Path
import os, copy
import numpy as np
import utm
import math
from sidescan_file import SidescanFile
import subprocess
import itertools
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pyproj import CRS, datadir
from scipy.signal import savgol_filter


class Georeferencer:
    filepath: str | os.PathLike
    sidescan_file: SidescanFile
    channel: int
    active_utm: bool
    active_poly: bool
    proc_data: np.array
    output_folder: str | os.PathLike
    active_proc_data: bool
    GCP_SPLIT: list
    POINTS_SPLIT: list
    LALO_OUTER: list
    PING: np.ndarray
    chunk_indices: np.array
    vertical_beam_angle: int
    epsg_code: str
    warp_options: dict = {
        "Polynomial 1 (recommended)": "SRC_METHOD=GCP_POLYNOMIAL, ORDER=1",
        "Homography (experimental)": "SRC_METHOD=GCP_HOMOGRAPHY",
    }
    resolution_options: dict = {
        "Same": "same",
        "Highest": "highest",
        "Lowest": "lowest",
        "Average": "average",
        "Common": "common",
    }
    resampling_options: dict = {
        "Near": "near",
        "Bilinear": "bilinear",
        "Cubic": "cubicspline",
        "Lanczos": "lanczos",
        "Average": "average",
        "RMS": "rms",
        "Mode": "mode",
        "Maximum": "max",
        "Minimum": "min",
        "Median": "med",
        "1. Quartile": "q1",
        "3. Quartile": "q3",
        "Weighted Sum": "sum",
    }
    LOLA_plt: np.ndarray
    HEAD_plt: np.ndarray
    LOLA_plt_ori: np.ndarray
    HEAD_plt_ori: np.ndarray

    def __init__(
        self,
        filepath: str | os.PathLike,
        channel: int = 0,
        active_utm: bool = True,
        active_poly: bool = True,
        proc_data=None,
        output_folder: str | os.PathLike = "./georef_out",
        vertical_beam_angle: int = 60,
        warp_algorithm: str = "SRC_METHOD=GCP_POLYNOMIAL, ORDER=1",
        resolution_mode: str = "average",
        resampling_method: str = "near",
    ):
        self.filepath = Path(filepath)
        self.sidescan_file = SidescanFile(self.filepath)
        self.channel = channel
        self.active_utm = active_utm
        self.active_poly = active_poly
        self.output_folder = Path(output_folder)
        self.vertical_beam_angle = vertical_beam_angle
        self.warp_algorithm = warp_algorithm
        self.resolution_mode = resolution_mode
        self.resampling_method = resampling_method
        self.active_proc_data = False
        self.GCP_SPLIT = []
        self.POINTS_SPLIT = []
        self.LALO_OUTER = []
        #self.PING_NO = np.empty_like(proc_data)
        self.LOLA_plt = np.empty_like(proc_data)
        self.HEAD_plt = np.empty_like(proc_data)
        self.LOLA_plt_ori = np.empty_like(proc_data)
        self.HEAD_plt_ori = np.empty_like(proc_data)
        if proc_data is not None:
            self.proc_data = proc_data
            self.active_proc_data = False
        self.setup_output_folder()

    def setup_output_folder(self):
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=False, exist_ok=True)
            if not self.output_folder.exists():
                print(
                    f"Error setting up output folder. Path might be invalid: {self.output_folder}"
                )
                raise FileNotFoundError

    def prep_data(self):
        # Extract metadata for each ping in sonar channel
        self.PING = self.sidescan_file.packet_no
        LON_ori = self.sidescan_file.longitude
        LAT_ori = self.sidescan_file.latitude
        HEAD_ori = self.sidescan_file.sensor_heading
        SLANT_RANGE = self.sidescan_file.slant_range[self.channel]
        GROUND_RANGE = []
        swath_len = len(self.PING)
        if self.active_proc_data:
            swath_width = len(self.proc_data[0])
        else:
            swath_width = len(self.sidescan_file.data[self.channel][0])

        self.PING = np.ndarray.flatten(np.array(self.PING))
        LON_ori = np.ndarray.flatten(np.array(LON_ori))
        LAT_ori = np.ndarray.flatten(np.array(LAT_ori))
        HEAD_ori = np.ndarray.flatten(np.array(HEAD_ori))
        SLANT_RANGE = np.ndarray.flatten(np.array(SLANT_RANGE))

        ground_range = [
            math.cos(self.vertical_beam_angle) * slant_range * (-1)
            for slant_range in SLANT_RANGE
        ]
        GROUND_RANGE.append(ground_range)
        GROUND_RANGE = np.ndarray.flatten(np.array(GROUND_RANGE))

        MASK = LON_ori != 0

        LON_ori = LON_ori[MASK]
        LAT_ori = LAT_ori[MASK]
        HEAD_ori = HEAD_ori[MASK]

        GROUND_RANGE = GROUND_RANGE[MASK]
        SLANT_RANGE = SLANT_RANGE[MASK]
        self.PING = self.PING[MASK]
        self.PING = self.PING
        print(f"shape ping original: {np.shape(self.PING)}")

        # Smooth Coordinates and Heading
        LAT = savgol_filter(LAT_ori, 100, 2)
        LON = savgol_filter(LON_ori, 100, 2)

        # convert heading angle to vectors to avoid jumps when crossing 0/360° degree angle
        HEAD_ori_rad = np.deg2rad(HEAD_ori)
        x_head = np.cos(HEAD_ori_rad)
        y_head = np.sin(HEAD_ori_rad)
    
        x_head_savgol = savgol_filter(x_head, 100, 2)
        y_head_savgol = savgol_filter(y_head, 100, 2)

        # convert back to angles -> degree and map from -180/180 -> 0/360
        HEAD = np.arctan2(y_head_savgol, x_head_savgol)
        HEAD = np.rad2deg(HEAD) % 360

        # filter high turn radii to exclude from data
        turn_rad = np.gradient(HEAD, self.PING, edge_order=1)
        #turn_rad = np.gradient(x_head_savgol, y_head_savgol, edge_order=1)

        # plot ping vs. turnrad
        from matplotlib import pyplot as plt
        plt.plot(self.PING, turn_rad)
        plt.show()

        mov_win =+ 0
        for i, rad in enumerate(turn_rad):
            if rad >= 0.025 or rad <= -0.025:
                turn_rad[i-mov_win:i+mov_win] = np.nan

        plt.plot(self.PING, turn_rad)
        #plt.plot(self.PING_NO, HEAD)
        plt.show()

        TURN_MASK = [ False if np.isnan(rad) else True for rad in turn_rad ]

        # Apply turn radius mask to cut turns
        LON = LON[TURN_MASK]
        LAT = LAT[TURN_MASK]
        HEAD = HEAD[TURN_MASK]  
        self.PING = self.PING[TURN_MASK]

        # Create arrays for heading and coords for plotting in GUI
        x = range(len(HEAD))
        self.HEAD_plt = np.column_stack((x, HEAD))
        self.HEAD_plt_ori = np.column_stack((x, HEAD_ori))
        self.LOLA_plt = np.column_stack((LON, LAT))
        self.LOLA_plt_ori = np.column_stack((LON_ori, LAT_ori))

        # Convert to UTM to calculate outer swath coordinates for both channels
        UTM = np.full_like(LAT_ori, np.nan)
        UTM = UTM.tolist()
        for idx, (la, lo) in enumerate(zip(LAT, LON)):
            try:
                UTM[idx] = utm.from_latlon(la, lo)
            except:
                ValueError("Values or lon and/or lat must not be 0")

        if UTM:
            NORTH = [utm_coord[0] for utm_coord in UTM]
            EAST = [utm_coord[1] for utm_coord in UTM]
            UTM_ZONE = [utm_coord[2] for utm_coord in UTM]
            UTM_LET = [utm_coord[3] for utm_coord in UTM]
            crs = CRS.from_dict({"proj": "utm", "zone": UTM_ZONE[0], "south": False})
            epsg = crs.to_authority()
            self.epsg_code = f"{epsg[0]}:{epsg[1]}"

        if self.channel == 0:
            EAST_OUTER = np.array(
                [
                    ground_range * math.sin(np.deg2rad(head)) + east
                    for ground_range, head, east in zip(GROUND_RANGE, HEAD, EAST)
                ]
            )
            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(np.deg2rad(head)) * -1) + north
                    for ground_range, head, north in zip(GROUND_RANGE, HEAD, NORTH)
                ]
            )
            self.LALO_OUTER = [
                utm.to_latlon(north_ch1, east_ch1, utm_zone, utm_let)
                for (north_ch1, east_ch1, utm_zone, utm_let) in zip(
                    NORTH_OUTER, EAST_OUTER, UTM_ZONE, UTM_LET
                )
            ]
            LA_OUTER, LO_OUTER = map(np.array, zip(*self.LALO_OUTER))

        elif self.channel == 1:
            EAST_OUTER = np.array(
                [
                    (ground_range * math.sin(np.deg2rad(head)) * -1) + east
                    for ground_range, head, east in zip(GROUND_RANGE, HEAD, EAST)
                ]
            )
            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(np.deg2rad(head))) + north
                    for ground_range, head, north in zip(GROUND_RANGE, HEAD, NORTH)
                ]
            )
            self.LALO_OUTER = [
                utm.to_latlon(north_ch2, east_ch2, utm_zone, utm_let)
                for (north_ch2, east_ch2, utm_zone, utm_let) in zip(
                    NORTH_OUTER, EAST_OUTER, UTM_ZONE, UTM_LET
                )
            ]
            LA_OUTER, LO_OUTER = map(np.array, zip(*self.LALO_OUTER))

        chunksize = 5
        self.chunk_indices = int(swath_len / chunksize)
        print(f"Fixed chunk size: {chunksize} pings.")

        # UTM
        if self.active_utm:
            lo_split_ce = np.array_split(NORTH, self.chunk_indices, axis=0)
            la_split_ce = np.array_split(EAST, self.chunk_indices, axis=0)
            lo_split_e = np.array_split(NORTH_OUTER, self.chunk_indices, axis=0)
            la_split_e = np.array_split(EAST_OUTER, self.chunk_indices, axis=0)

        else:
            lo_split_ce = np.array_split(LON, self.chunk_indices, axis=0)
            la_split_ce = np.array_split(LAT, self.chunk_indices, axis=0)
            lo_split_e = np.array_split(LO_OUTER, self.chunk_indices, axis=0)
            la_split_e = np.array_split(LA_OUTER, self.chunk_indices, axis=0)

        """
        Calculate edge coordinates for first and last coordinates in chunks:
        - Convert to utm, add ground range, convert back to lon/lat
        """
        for chunk_num, (lo_chunk_ce, la_chunk_ce, lo_chunk_e, la_chunk_e) in enumerate(
            zip(lo_split_ce, la_split_ce, lo_split_e, la_split_e)
        ):

            """
            Define corner coordinates for chunks and set gcps:
            - ul, ur: upper left/right --> nadir coordinates
            - ll, lr: lower left/right --> edge coordinates, calculated with ground range & heading
            - image coordinates: im_x_right = length of chunk
            """

            lo_ce_ul = lo_chunk_ce[0]
            lo_ce_ur = lo_chunk_ce[-1]
            la_ce_ul = la_chunk_ce[0]
            la_ce_ur = la_chunk_ce[-1]
            lo_e_ll = lo_chunk_e[0]
            lo_e_lr = lo_chunk_e[-1]
            la_e_ll = la_chunk_e[0]
            la_e_lr = la_chunk_e[-1]

            # add/substract small values to ensure overlap on outer edges (move inwards/outwards at nadir/outer edge)

            im_x_left_nad = 1  # 1
            im_x_right_nad = np.shape(lo_chunk_ce)[0] - 1
            im_x_left_outer = 1  # -1
            im_x_right_outer = np.shape(lo_chunk_ce)[0] - 1  # -1

            im_y_outer = swath_width
            im_y_nad = 0

            gcp = np.array(
                (
                    (im_x_left_nad, im_y_nad, lo_ce_ul, la_ce_ul),
                    (im_x_left_outer, im_y_outer, lo_e_ll, la_e_ll),
                    (im_x_right_nad, im_y_nad, lo_ce_ur, la_ce_ur),
                    (im_x_right_outer, im_y_outer, lo_e_lr, la_e_lr),
                )
            )

            points = np.array(
                (
                    (lo_ce_ul, la_ce_ul, im_x_left_nad, im_y_nad),
                    (lo_e_ll, la_e_ll, im_x_left_outer, im_y_outer * (-1)),
                    (lo_ce_ur, la_ce_ur, im_x_right_nad, im_y_nad),
                    (lo_e_lr, la_e_lr, im_x_right_outer, im_y_outer * (-1)),
                )
            )

            self.GCP_SPLIT.append(gcp)
            self.POINTS_SPLIT.append(points)

    def channel_stack(self):
        """- Work on raw or processed data, depending on `self.active_proc_data`
        - Stack channel so that the largest axis is horizontal
        - Norm data to max 255 for pic generation
        """

        # check whether processed data is present
        if self.active_proc_data:
            ch_stack = self.proc_data
        else:
            ch_stack = self.sidescan_file.data[self.channel]

        # Extract metadata for each ping in sonar channel
        PING = self.sidescan_file.packet_no
        swath_len = len(PING)
        swath_width = len(ch_stack[0])
        print(f"swath_len: {swath_len}, swath_width: {swath_width}")
        # print(f"ch_stack.shape[0], ch_stack.shape[1]: {ch_stack.shape[0], ch_stack.shape[1]}")

        # Transpose (always!) so that the largest axis is horizontal
        ch_stack = ch_stack.T

        ch_stack = np.array(ch_stack, dtype=float)

        # Hack for alter transparency
        ch_stack /= np.max(np.abs(ch_stack)) / 254
        ch_stack = np.clip(ch_stack, 1, 255)

        # Flip array ---> Note: different for .jsf and .xtf!
        # print(f"ch_stack shape after transposing: {np.shape(ch_stack)}")
        ch_stack = np.flip(ch_stack, axis=0)

        return ch_stack.astype(np.uint8)

    @staticmethod
    def write_img(im_path, data, alpha=None):
        # flip data to show first ping at bottom
        data = np.flipud(data)
        image_to_write = Image.fromarray(data)
        if alpha is not None:
            alpha = Image.fromarray(alpha)
            image_to_write.putalpha(alpha)
        png_info = PngInfo()
        png_info.add_text("Info", "Generated by SidescanTools")
        image_to_write.save(im_path, pnginfo=png_info)
        image_to_write.save(im_path)

    @staticmethod
    def run_command(command):
        """
        Starts a subprocess to run shell commands.
        """
        cur_env = copy.copy(os.environ)
        cur_env["PROJ_LIB"] = datadir.get_data_dir()
        cur_env["PROJ_DATA"] = datadir.get_data_dir()
        result = subprocess.run(command, capture_output=True, text=True, env=cur_env)
        if result.returncode != 0:
            print(f"Error occurred: {result.stderr}")

    def georeference(self, ch_stack, otiff, progress_signal=None):
        """
        array_split: Creates [chunk_size]-ping chunks per channel and extracts corner coordinates for chunks from GCP list. \
        Assigns extracted corner coordinates as GCPs (gdal_translate) and projects them (gdal_warp).
            find indices where lon/lat change (they are the same for multiple consequitive pings) 
            and add a '1' to obtain correct index (diff-array is one index shorter than original) \
            and another '1' to move one coordinate up, else it would be still the same coordinate

        gdal GCP format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], [elevation], \
            [image column index(x)], [image row index (y)]

        X:L; Y:U                   X:R; Y:U
        LO,LA(0):CE                LO,LA(chunk):CE
                      [       ]
                      [       ]
                      [       ]
                      [       ]
        LO,LA(0):E                LO,LA(chunk):E
        X:L; Y:L                  X:R; Y:L

        new in gdal version 3.11: homography algorithm for warping. Important note: Does not work when gcps are not in right order, i.e. if for example \
        lower left and lower right image coorainte are switched. this can sometimes happrns when there is no vessel movement or vessel turns etc. \
        Right now, these chunks are ignored (the data look crappy anyway). 
        """
        ch_split = np.array_split(ch_stack, self.chunk_indices, axis=1)

        for chunk_num, (ch_chunk, gcp_chunk, points_chunk) in enumerate(
            zip(ch_split, self.GCP_SPLIT, self.POINTS_SPLIT)
        ):
            if chunk_num < len(ch_split) - 1:

                im_path = otiff.with_stem(f"{otiff.stem}_{chunk_num}_tmp").with_suffix(
                    ".png"
                )
                chunk_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_chunk_tmp"
                ).with_suffix(".tif")
                warp_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_georef_chunk_tmp"
                ).with_suffix(".tif")

                # Flip image chunks according to side
                if self.channel == 0:
                    ch_chunk_flip = np.flip(ch_chunk, 1)
                elif self.channel == 1:
                    ch_chunk_flip = np.flip(ch_chunk, 0)

                alpha = np.ones(np.shape(ch_chunk_flip), dtype=np.uint8) * 255
                alpha[ch_chunk_flip == 0] = 0
                data_stack = np.stack(
                    (ch_chunk_flip, ch_chunk_flip, ch_chunk_flip), axis=-1
                )
                image_to_write = Image.fromarray(data_stack)
                alpha = Image.fromarray(alpha)
                image_to_write.putalpha(alpha)
                png_info = PngInfo()
                png_info.add_text("Info", "Generated by SidescanTools")
                image_to_write.save(im_path, pnginfo=png_info)

                try:
                    coords = [
                        (im_x, im_y, lo, la) for (im_x, im_y, lo, la) in gcp_chunk
                    ]
                    im_x, im_y, lo, la = zip(*coords)

                except Exception as e:
                    print(f"gcp chunk: {np.shape(gcp_chunk)}")

                gdal_translate = ["gdal_translate", "-of", "GTiff"]

                gdal_warp_utm = [
                    "gdal",
                    "raster",
                    "reproject",
                    "-r",
                    self.resampling_method,
                    "--to",
                    self.warp_algorithm,
                    "--co",
                    "COMPRESS=DEFLATE",
                    "--overwrite",
                    "-d",
                    self.epsg_code,
                    "-i",
                    str(chunk_path),
                    "-o",
                    str(warp_path),
                ]

                gdal_warp_wgs84 = [
                    "gdal",
                    "raster",
                    "reproject",
                    "-r",
                    self.resampling_method,
                    "--to",
                    self.warp_algorithm,
                    "--co",
                    "COMPRESS=DEFLATE",
                    "--overwrite",
                    "-d",
                    "EPSG:4326",
                    "-i",
                    str(chunk_path),
                    "-o",
                    str(warp_path),
                ]

                for i in range(len(gcp_chunk)):
                    gdal_translate.extend(
                        ["-gcp", str(im_x[i]), str(im_y[i]), str(lo[i]), str(la[i])]
                    )

                gdal_translate.extend([str(im_path), str(chunk_path)])

                # gdal 3.11 syntax
                if self.active_utm:
                    gdal_warp = gdal_warp_utm
                else:
                    gdal_warp = gdal_warp_wgs84

                if progress_signal is not None:
                    progress_signal.emit(1000 / (len(ch_stack[1])) * 0.005)

                self.run_command(gdal_translate)
                self.run_command(gdal_warp)

            elif chunk_num == len(ch_split) - 1:
                pass

    def mosaic(self, mosaic_tiff, txt_path, progress_signal=None):
        """
        Merges tif chunks created in the georef function.
        Args.:
        - path: Path to geotiffs
        - creates temporary txt file that lists tif chunks
        - mosaic_tiff: mosaicked output tif

        """
        # create list from warped tifs and merge
        TIF_ch0 = []
        TIF_ch1 = []

        txt_path_ch0 = os.path.join(self.output_folder, "chunks_tif_ch0.txt")
        txt_path_ch1 = os.path.join(self.output_folder, "chunks_tif_ch1.txt")

        for root, dirs, files in os.walk(self.output_folder):
            for name in files:
                if (
                    name.endswith("_georef_chunk_tmp.tif")
                    and "ch0" in name
                    and not name.startswith("._")
                ):
                    TIF_ch0.append(os.path.join(root, name))

        for root, dirs, files in os.walk(self.output_folder):
            for name in files:
                if (
                    name.endswith("_georef_chunk_tmp.tif")
                    and "ch1" in name
                    and not name.startswith("._")
                ):
                    TIF_ch1.append(os.path.join(root, name))

        if progress_signal is not None:
            progress_signal.emit(0.2)

        np.savetxt(txt_path_ch0, TIF_ch0, fmt="%s")
        np.savetxt(txt_path_ch1, TIF_ch1, fmt="%s")

        # delete merged file if it already exists
        if mosaic_tiff.exists():
            mosaic_tiff.unlink()

        # gdal 3.11 syntax
        gdal_mosaic = [
            "gdal",
            "raster",
            "mosaic",
            "-i",
            f"@{txt_path}",
            "-o",
            str(mosaic_tiff),
            "--src-nodata",
            "0",
            "--resolution",
            self.resolution_mode,
            "--co",
            "COMPRESS=DEFLATE",
            "--co",
            "TILED=YES",
        ]

        self.run_command(gdal_mosaic)

    def process(self, progress_signal=None):
        file_name = self.filepath.stem
        tif_path = self.output_folder / f"{file_name}_ch{self.channel}.tif"
        mosaic_tif_path = self.output_folder / f"{file_name}_ch{self.channel}_stack.tif"
        txt_path = self.output_folder / f"chunks_tif_ch{self.channel}.txt"
        nav_ch = self.output_folder / f"Navigation_{file_name}_ch{self.channel}.csv"

        self.prep_data()
        ch_stack = self.channel_stack()

        try:
            print(
                f"Processing chunks in channel {self.channel} with warp method {self.warp_algorithm}..."
            )

            self.georeference(
                ch_stack=ch_stack, otiff=tif_path, progress_signal=progress_signal
            )

            # save Navigation to .csv
            print(f"Saving navinfo to {nav_ch}")
            #nav = np.column_stack((self.PING_NO, self.LALO_OUTER, self.LOLA_plt, self.HEAD_plt[:, 1]))
            #np.savetxt(
            #    nav_ch,
            #    nav,
            #    fmt="%s",
            #    delimiter=";",
            #    header="Ping No; Outer Latitude; Outer Longitude; Nadir Longitude; Nadir Latitude; Heading",
            #)

            nav = np.column_stack((self.PING, self.LOLA_plt, self.HEAD_plt[:, 1]))
            np.savetxt(
                nav_ch,
                nav,
                fmt="%s",
                delimiter=";",
                header="Ping No; Nadir Longitude; Nadir Latitude; Heading",
            )
        except Exception as e:
            print(f"An error occurred during georeferencing: {str(e)}")

        try:
            print(
                f"Mosaicking channel {self.channel} with resolution mode {self.resolution_mode}..."
            )
            self.mosaic(mosaic_tif_path, txt_path, progress_signal=progress_signal)

        except Exception as e:
            print(f"An error occurred during mosaicking: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Tool to process sidescan sonar data")
    parser.add_argument("xtf", metavar="FILE", help="Path to xtf/jsf file")
    parser.add_argument(
        "channel",
        type=int,
        default=0,
        help="Channel number (can be 0 or 1, default: 0)",
    )
    parser.add_argument(
        "--dynamic_chunking",
        type=bool,
        default=False,
        help="Implements chunking based on GPS information density; only use for bad/scarce GPS data",
    )
    parser.add_argument(
        "--UTM",
        type=bool,
        default=True,
        help="Uses UTM projection rather than WGS84. Default is UTM",
    )

    parser.add_argument(
        "--poly",
        type=bool,
        default=True,
        help="Uses polynomial order 1 transformation (affine) instead of homographic for georeferencing. Default is homographic.",
    )

    args = parser.parse_args()
    print("args:", args)

    georeferencer = Georeferencer(
        args.xtf, args.channel, args.dynamic_chunking, args.UTM, args.poly
    )
    georeferencer.process()


if __name__ == "__main__":
    main()
