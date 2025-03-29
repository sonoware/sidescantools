import argparse
from osgeo import gdal  # Leads to Segmentation fault when exiting main GUI -.-
from pathlib import Path, PurePath
import os
import numpy as np
import utm
import math
from .sidescan_file import SidescanFile
import subprocess
import itertools
from PIL import Image
from PIL.PngImagePlugin import PngInfo


# TODO: Doc/Type hints
class SidescanGeoreferencer:
    filepath: Path
    sidescan_file: SidescanFile
    channel: int
    dynamic_chunking: bool
    output_folder: Path
    proc_data: np.array
    active_proc_data: bool
    GCP_SPLIT: list
    POINTS_SPLIT: list
    chunk_indices: np.array
    vertical_beam_angle: int

    def __init__(
        self,
        filepath: str | os.PathLike,
        channel: int = 0,
        dynamic_chunking: bool = False,
        proc_data=None,
        output_folder: str | os.PathLike = "./georef_out",
        vertical_beam_angle: int = 60,
    ):
        self.filepath = Path(filepath)
        self.sidescan_file = SidescanFile(self.filepath)
        self.channel = channel
        self.dynamic_chunking = dynamic_chunking
        self.output_folder = Path(output_folder)
        self.vertical_beam_angle = vertical_beam_angle

        self.active_proc_data = False
        self.GCP_SPLIT = []
        self.POINTS_SPLIT = []
        if proc_data is not None:
            self.proc_data = proc_data
            self.active_proc_data = True
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
        PING = self.sidescan_file.packet_no
        LON = self.sidescan_file.longitude
        LAT = self.sidescan_file.latitude
        HEAD = np.radians(self.sidescan_file.sensor_heading)
        SLANT_RANGE = self.sidescan_file.slant_range[self.channel]
        GROUND_RANGE = []
        swath_len = len(PING)
        print(f"swath_len: {swath_len}")

        PING = np.ndarray.flatten(np.array(PING))
        LON = np.ndarray.flatten(np.array(LON))
        LAT = np.ndarray.flatten(np.array(LAT))
        HEAD = np.ndarray.flatten(np.array(HEAD))
        SLANT_RANGE = np.ndarray.flatten(np.array(SLANT_RANGE))

        ground_range = [
            math.cos(self.vertical_beam_angle) * slant_range * (-1)
            for slant_range in SLANT_RANGE
        ]
        GROUND_RANGE.append(ground_range)
        GROUND_RANGE = np.ndarray.flatten(np.array(GROUND_RANGE))

        MASK = LON != 0

        LON = LON[MASK]
        LAT = LAT[MASK]
        HEAD = HEAD[MASK]
        GROUND_RANGE = GROUND_RANGE[MASK]
        SLANT_RANGE = SLANT_RANGE[MASK]
        PING = PING[MASK]

        UTM = []
        for la, lo in zip(LAT, LON):
            try:
                UTM.append((utm.from_latlon(la, lo)))
            except:
                ValueError("Values or lon and/or lat must not be 0")

            if UTM:
                NORTH = [utm_coord[0] for utm_coord in UTM]
                EAST = [utm_coord[1] for utm_coord in UTM]
                UTM_ZONE = [utm_coord[2] for utm_coord in UTM]
                UTM_LET = [utm_coord[3] for utm_coord in UTM]

        if self.channel == 0:
            EAST = np.array(
                [
                    ground_range * math.sin(head) + east
                    for ground_range, head, east in zip(GROUND_RANGE, HEAD, EAST)
                ]
            )
            NORTH = np.array(
                [
                    (ground_range * math.cos(head) * -1) + north
                    for ground_range, head, north in zip(GROUND_RANGE, HEAD, NORTH)
                ]
            )
            LALO_EDGE = [
                utm.to_latlon(north_ch1, east_ch1, utm_zone, utm_let)
                for (north_ch1, east_ch1, utm_zone, utm_let) in zip(
                    NORTH, EAST, UTM_ZONE, UTM_LET
                )
            ]
            LA_EDGE, LO_EDGE = map(np.array, zip(*LALO_EDGE))

        elif self.channel == 1:
            EAST = np.array(
                [
                    (ground_range * math.sin(head) * -1) + east
                    for ground_range, head, east in zip(GROUND_RANGE, HEAD, EAST)
                ]
            )
            NORTH = np.array(
                [
                    (ground_range * math.cos(head)) + north
                    for ground_range, head, north in zip(GROUND_RANGE, HEAD, NORTH)
                ]
            )
            LALO_EDGE = [
                utm.to_latlon(north_ch2, east_ch2, utm_zone, utm_let)
                for (north_ch2, east_ch2, utm_zone, utm_let) in zip(
                    NORTH, EAST, UTM_ZONE, UTM_LET
                )
            ]
            LA_EDGE, LO_EDGE = map(np.array, zip(*LALO_EDGE))

        if self.dynamic_chunking:
            print("Dynamic chunking active.")
            self.chunk_indices = np.where(np.diff(LAT) != 0)[0] + 2

        elif not self.dynamic_chunking:
            chunksize = 5
            self.chunk_indices = int(swath_len / chunksize)
            print(f"Fixed chunk size: {chunksize} pings.")

        lo_split_ce = np.array_split(LON, self.chunk_indices, axis=0)
        la_split_ce = np.array_split(LAT, self.chunk_indices, axis=0)
        lo_split_e = np.array_split(LO_EDGE, self.chunk_indices, axis=0)
        la_split_e = np.array_split(LA_EDGE, self.chunk_indices, axis=0)

        """
        Calculate edge coordinates for first and last coordinates in chunks:
        - Convert to utm, add ground range, convert back to lon/lat
        """
        for chunk_num, (lo_chunk_ce, la_chunk_ce, lo_chunk_e, la_chunk_e) in enumerate(
            zip(lo_split_ce, la_split_ce, lo_split_e, la_split_e)
        ):

            if self.dynamic_chunking:

                # print(f'chunk_num, chunk_indices: {chunk_num, len(chunk_indices)}')
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

                im_x_right = np.shape(lo_chunk_ce)[0]
                im_x_left = 0
                im_y_up = 0
                im_y_low = 1000
                # TODO: IS 1000 samples/Swathwidth cottrect in every case? Or read it out and implement as argument?

                gcp = np.array(
                    (
                        (im_x_left, im_y_up, lo_ce_ul, la_ce_ul),
                        (im_x_left, im_y_low, lo_e_ll, la_e_ll),
                        (im_x_right, im_y_up, lo_ce_ur, la_ce_ur),
                        (im_x_right, im_y_low, lo_e_lr, la_e_lr),
                    )
                )
                self.GCP_SPLIT.append(gcp)

            elif not self.dynamic_chunking:

                im_x = np.array([x for x in range(len(lo_chunk_ce))])
                im_y_nadir = np.array([0 for y in range(len(lo_chunk_ce))])
                im_y_outer = np.array([-1000 for y in range(len(lo_chunk_ce))])

                # Create one DF for each channel and combine with '-gcp' for gdal command

                # gcp_cmd = np.array([ '-gcp' for gcp in range(chunksize) ])
                Nadir_GCP = np.column_stack(
                    (im_x, im_y_nadir, lo_chunk_ce, la_chunk_ce)
                ).astype(object)
                Outer_GCP = np.column_stack(
                    (im_x, im_y_outer, lo_chunk_e, la_chunk_e)
                ).astype(object)
                gcp = np.vstack([Nadir_GCP, Outer_GCP])

                self.GCP_SPLIT.append(gcp)


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

        # TODO: Insert gmt logic here, if applied in the future


        # Transpose so that the largest axis is horizontal
        ch_stack = ch_stack if ch_stack.shape[0] < ch_stack.shape[1] else ch_stack.T
        ch_stack = np.array(ch_stack, dtype=float)

        # Hack for alter transparency
        ch_stack /= np.max(np.abs(ch_stack)) / 254
        ch_stack = np.clip(ch_stack, 1, 255)

        # Flip array
        print(f"ch_stack shape: {np.shape(ch_stack)}")
        ch_stack = np.flip(ch_stack, axis=1)
        ch_stack = np.flip(ch_stack, axis=0)

        return ch_stack.astype(np.uint8)

    @staticmethod
    def run_command(command):
        """
        Starts a subprocess to run shell commands.

        """
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            pass
            # print(f"Command executed successfully: {' '.join(command)}")
        else:
            print(f"Error occurred: {result.stderr}")

    def georeference(self, ch_stack, otiff):
        """
        array_split: Creates [chunk_size]-ping chunks per channel and extracts corner coordinates for chunks from GCP list. \
        Assigns extracted corner coordinates as GCPs (gdal_translate) and projects them (gdal_warp).
        Dynamic chunking: chunk_indices = number of pings within one chunk;  \
            find indices where lon/lat change (they are the same for multiple consequitive pings) 
            and add a '1' to obtain correct index (diff-array is one index shorter than original) \
            and another '1' to move one coordinate up, else it would be still the same coordinate

        gdal GCP format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], [elevation], \
            [image column index(x)], [image row index (y)]

        # X:L; Y:U                   X:R; Y:U
        # LO,LA(0):CE                LO,LA(chunk):CE
        #               [       ]
        #               [       ]
        #               [       ]
        #               [       ]
        # LO,LA(0):E                LO,LA(chunk):E
        # X:L; Y:L                  X:R; Y:L

        """
        ch_split = np.array_split(ch_stack, self.chunk_indices, axis=1)

        for chunk_num, (ch_chunk, gcp_chunk) in enumerate(
            zip(ch_split, self.GCP_SPLIT)
        ):
            if chunk_num < len(ch_split) - 1:

                im_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_ch{self.channel}_tmp"
                ).with_suffix(".png")
                chunk_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_ch{self.channel}_chunk_tmp"
                ).with_suffix(".tif")
                warp_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_ch{self.channel}_WGS84_chunk_tmp"
                ).with_suffix(".tif")
                csv_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_ch{self.channel}_tmp"
                ).with_suffix(".csv")

                # optional: export points
                #points_path = otiff.with_stem(
                #    f"{otiff.stem}_{chunk_num}_ch{self.channel}_points_tmp"
                #).with_suffix(".points")

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

                    # optional: export points
                    # points = [(lo, la, im_x, im_y) for (lo, la, im_x, im_y) in points_chunk]
                    # np.savetxt(csv_path, points, fmt="%s", delimiter=",")

                except Exception as e:
                    print(f"gcp chunk: {np.shape(gcp_chunk)}")


                gdal_translate = ["gdal_translate", "-of", "GTiff"]

                if self.dynamic_chunking:
                    for i in range(4):
                        gdal_translate.extend(
                            ["-gcp", str(im_x[i]), str(im_y[i]), str(lo[i]), str(la[i])]
                        )
                    gdal_translate.extend([str(im_path), str(chunk_path)])
                    print(gdal_translate)

                    gdal_warp = [
                        "gdalwarp",
                        "-r",
                        "bilinear",
                        "-order",
                        "1",
                        "-co",
                        "COMPRESS=DEFLATE",
                        "-t_srs",
                        "EPSG:4326",
                        str(chunk_path),
                        str(warp_path),
                    ]

                elif not self.dynamic_chunking:
                    for i in range(len(gcp_chunk)):
                        gdal_translate.extend(
                            ["-gcp", str(im_x[i]), str(im_y[i]), str(lo[i]), str(la[i])]
                        )
                    gdal_translate.extend([str(im_path), str(chunk_path)])

                    gdal_warp = [
                        "gdalwarp",
                        "-r",
                        "bilinear",
                        "-tps",
                        "-co",
                        "COMPRESS=DEFLATE",
                        "-t_srs",
                        "EPSG:4326",
                        str(chunk_path),
                        str(warp_path),
                    ]

                    # optional: append .points header and fist and last center point

                    #first_line = '#CRS: GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984                  (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World                   Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],               ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT           ["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.        0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE             ["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
                    #second_line = "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"
                    #with open(csv_path, "r") as ori_f:
                    #    ori_txt = ori_f.read()
                    #    # print(ori_txt)
                    #with open(points_path, "w") as mod_f:
                    #    mod_f.write(
                    #        first_line + "\n" + second_line + "\n" + ori_txt
                    #    )

                self.run_command(gdal_translate)
                self.run_command(gdal_warp)

            elif chunk_num == len(ch_split) - 1:
                pass

    def mosaic(self, mosaic_tiff):
        """
        Merges tif chunks created in the georef function.
        Args.:
        - path: Path to geotiffs
        - creates temporary txt file that lists tif chunks
        - mosaic_tiff: mosaicked output tif

        """
        # create list from warped tifs and merge
        TIF = []
        if self.channel == 0:
            txt_path = os.path.join(self.output_folder, "ch1_tif.txt")
        elif self.channel == 1:
            txt_path = os.path.join(self.output_folder, "ch2_tif.txt")

        for root, dirs, files in os.walk(self.output_folder):
            for name in files:
                if name.endswith("_WGS84_chunk_tmp.tif") and not name.startswith("._"):
                    TIF.append(os.path.join(root, name))
                    np.savetxt(txt_path, TIF, fmt="%s")

        # delete merged file if it already exists
        if mosaic_tiff.exists():
            mosaic_tiff.unlink()

        gdal_merge = [
            "gdal_merge",
            "-o",
            str(mosaic_tiff),
            "-n",
            "0",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "TILED=YES",
            "--optfile",
            str(txt_path),
        ]

        self.run_command(gdal_merge)


    def process(self):
        file_name = self.filepath.stem
        tif_path = self.output_folder / f"{file_name}_ch{self.channel}.tif"
        mosaic_tif_path = self.output_folder / f"{file_name}_ch{self.channel}_stack.tif"
        csv_ch = self.output_folder / f"{file_name}_ch{self.channel}.csv"

        self.prep_data()
        ch_stack = self.channel_stack()

        try:
            print(f"Georeferencing channel {self.channel}...")

            self.georeference(ch_stack=ch_stack, otiff=tif_path)

            print(f"Mosaicking channel {self.channel}...")
            self.mosaic(mosaic_tif_path)

            # save GCPs to .csv
            print(f"Saving GCPs to {csv_ch}")
            GCP_SPLIT = list(itertools.chain.from_iterable(self.GCP_SPLIT))
            np.savetxt(csv_ch, GCP_SPLIT, fmt="%s", delimiter=";")

        except IndexError as i:
            print(f"Something with indexing went wrong... {str(i)}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        self.cleanup()

    def cleanup(self):
        print(f"Cleaning ...")
        for file in os.listdir(self.output_folder):
            file_path = self.output_folder / file
            if (
                str(file_path).endswith("_tmp.png")
                or str(file_path).endswith("_tmp.txt")
                or str(file_path).endswith("_tmp.csv")
                or str(file_path).endswith("_tmp.tif")
                or str(file_path).endswith(".xml")
            ):
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    print(f"File Not Found: {file_path}")
        print("Cleanup done")

    @staticmethod
    def write_img(im_path, data, alpha=None):
        image_to_write = Image.fromarray(data)
        if alpha is not None:
            alpha = Image.fromarray(alpha)
            image_to_write.putalpha(alpha)
        png_info = PngInfo()
        png_info.add_text("Info", "Generated by SidescanTools")
        image_to_write.save(im_path, pnginfo=png_info)
        image_to_write.save(im_path)

def main():
    parser = argparse.ArgumentParser(description="Tool to georeference sidescan data")
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
        help="Implements chunking based on GPS information density",
    )

    args = parser.parse_args()
    print("args:", args)

    georeferencer = SidescanGeoreferencer(args.xtf, args.channel, args.dynamic_chunking)
    georeferencer.process()


if __name__ == "__main__":
    main()
