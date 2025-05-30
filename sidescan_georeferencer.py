import argparse
from pathlib import Path, PurePath
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
import napari


# TODO: Doc/Type hints

class SidescanGeoreferencer:
    filepath: Path
    sidescan_file: SidescanFile
    channel: int
    dynamic_chunking: bool
    active_utm: bool
    active_poly: bool
    output_folder: Path
    proc_data: np.array
    active_proc_data: bool
    GCP_SPLIT: list
    POINTS_SPLIT: list
    chunk_indices: np.array
    vertical_beam_angle: int
    epsg_code: str
    resolution_mode: dict = {
        "Same": "same",
        "Highest": "highest", 
        "Lowest": "lowest", 
        "Average": "average", 
        "Common": "common"
        }
    warp_algorithm: dict = {
        "Polynomial 1": "SRC_METHOD=GCP_POLYNOMIAL, ORDER=1", 
        "Homography": "SRC_METHOD=GCP_HOMOGRAPHY"
    }

    def __init__(
        self,
        filepath: str | os.PathLike,
        channel: int = 0,
        dynamic_chunking: bool = False,
        active_utm: bool = True,
        active_poly: bool = True,
        proc_data = None,
        output_folder: str | os.PathLike = "./georef_out",
        vertical_beam_angle: int = 60,
        resolution_mode: dict = {
        "Same": "same",
        "Highest": "highest", 
        "Lowest": "lowest", 
        "Average": "average", 
        "Common": "common"
        },
        warp_algorithm: dict = {
        "Polynomial 1": "SRC_METHOD=GCP_POLYNOMIAL, ORDER=1", 
        "Homography": "SRC_METHOD=GCP_HOMOGRAPHY"
    }
    ):
        self.filepath = Path(filepath)
        self.sidescan_file = SidescanFile(self.filepath)
        self.channel = channel
        self.dynamic_chunking = dynamic_chunking
        self.active_utm = active_utm
        self.active_poly = active_poly
        self.output_folder = Path(output_folder)
        self.vertical_beam_angle = vertical_beam_angle
        self.resolution_mode = resolution_mode
        self.warp_algorithm = warp_algorithm
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
        LON_ori = self.sidescan_file.longitude
        LAT_ori = self.sidescan_file.latitude
        HEAD_ori = np.radians(self.sidescan_file.sensor_heading)
        SLANT_RANGE = self.sidescan_file.slant_range[self.channel]
        GROUND_RANGE = []
        swath_len = len(PING)
        swath_width = len(self.sidescan_file.data[self.channel][0])

        PING = np.ndarray.flatten(np.array(PING))
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
        PING = PING[MASK]

        HEAD = savgol_filter(HEAD_ori, 120, 1)
        x = range(len(HEAD))
        head_data = np.stack([x, HEAD], axis=0)
        #view_head = napari.Viewer()
        #view_head.add_image(head_data, name = 'Heading')
        #layer_head = view_head.layers['Heading']
        #layer_head.save('./Heading.png')
        #plt.title("Heading")
        #plt.plot(x, HEAD_ori, label='Original Heading')
        #plt.plot(x, HEAD, label='Smoothed Heading')
        #plt.legend()
        #plt.show()

        LAT = savgol_filter(LAT_ori, 120, 1)
        LON = savgol_filter(LON_ori, 120, 1)
        lola_data = np.stack([LON, LAT], axis = 0)
        #view_lola = napari.Viewer()
        #view_lola.add_image(lola_data, name = 'Navigation')
        #layer_lola = view_head.layers['Navigation']
        #layer_lola.save('./Navigation.png')
        #plt.title("Navigation")
        #plt.plot(LON_ori, LAT_ori, label='Original Navigation')
        #plt.plot(LON, LAT, label='Smoothed Navigation')
        #plt.legend()
        #plt.show()

        napari.run()

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
                crs = CRS.from_dict({'proj': 'utm', 'zone': UTM_ZONE[0], 'south': False})
                epsg = crs.to_authority()
                self.epsg_code = f'{epsg[0]}:{epsg[1]}'


        if self.channel == 0:
            EAST_OUTER = np.array(
                [
                    ground_range * math.sin(head) + east
                    for ground_range, head, east in zip(GROUND_RANGE, HEAD, EAST)
                ]
            )
            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(head) * -1) + north
                    for ground_range, head, north in zip(GROUND_RANGE, HEAD, NORTH)
                ]
            )
            LALO_OUTER = [
                utm.to_latlon(north_ch1, east_ch1, utm_zone, utm_let)
                for (north_ch1, east_ch1, utm_zone, utm_let) in zip(
                    NORTH_OUTER, EAST_OUTER, UTM_ZONE, UTM_LET
                )
            ]
            LA_OUTER, LO_OUTER = map(np.array, zip(*LALO_OUTER))

        elif self.channel == 1:
            EAST_OUTER = np.array(
                [
                    (ground_range * math.sin(head) * -1) + east
                    for ground_range, head, east in zip(GROUND_RANGE, HEAD, EAST)
                ]
            )
            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(head)) + north
                    for ground_range, head, north in zip(GROUND_RANGE, HEAD, NORTH)
                ]
            )
            LALO_OUTER = [
                utm.to_latlon(north_ch2, east_ch2, utm_zone, utm_let)
                for (north_ch2, east_ch2, utm_zone, utm_let) in zip(
                    NORTH_OUTER, EAST_OUTER, UTM_ZONE, UTM_LET
                )
            ]
            LA_OUTER, LO_OUTER = map(np.array, zip(*LALO_OUTER))

        if self.dynamic_chunking:
            print("Dynamic chunking active.")
            self.chunk_indices = np.where(np.diff(LAT_ori) != 0)[0] + 2

        elif not self.dynamic_chunking:
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
                # TODO: make curvature-dependent

                im_x_left_nad = 1   #1
                im_x_right_nad = np.shape(lo_chunk_ce)[0] -1
                im_x_left_outer = 1 #-1
                im_x_right_outer = np.shape(lo_chunk_ce)[0] -1  #-1
                im_y_nad = 0    #1
                im_y_outer = -swath_width #-swath_width

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
                        (lo_e_ll, la_e_ll, im_x_left_outer, im_y_outer*(-1)),
                        (lo_ce_ur, la_ce_ur, im_x_right_nad, im_y_nad),
                        (lo_e_lr, la_e_lr, im_x_right_outer, im_y_outer*(-1)),
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
            # convert to dB (just if script is run standalone)
            #ch_stack = 20 * np.log10(np.abs(ch_stack) + 0.1)

        # Extract metadata for each ping in sonar channel
        PING = self.sidescan_file.packet_no
        swath_len = len(PING)
        swath_width = len(self.sidescan_file.data[self.channel][0])
        print(f"swath_len: {swath_len}, swath_width: {swath_width}")


        # Transpose so that the largest axis is horizontal
        ch_stack = ch_stack if ch_stack.shape[0] < ch_stack.shape[1] else ch_stack.T
        if False:
            if swath_len >= swath_width:
                ch_stack = ch_stack
                    #ch_stack = ch_stack if ch_stack.shape[0] < ch_stack.shape[1] else ch_stack.T
            elif swath_len <= swath_width:
                ch_stack = ch_stack.T

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
        cur_env = copy.copy(os.environ)
        cur_env["PROJ_LIB"] = datadir.get_data_dir()
        cur_env["PROJ_DATA"] = datadir.get_data_dir()
        result = subprocess.run(command, capture_output=True, text=True, env=cur_env)
        if result.returncode == 0:
            pass
            #print(result.stdout)
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

        # new in gdal version 3.11: homography algorithm for warping. Important note: Does not work when gcps are not in right order, i.e. if for example \
        lower left and lower right image coorainte are switched. this can sometimes happrns when there is no vessel movement or vessel turns etc. \
        Right now, these chunks are ignored (the data look crappy anyway). 
        """
        ch_split = np.array_split(ch_stack, self.chunk_indices, axis=1)

        for chunk_num, (ch_chunk, gcp_chunk, points_chunk) in enumerate(
            zip(ch_split, self.GCP_SPLIT, self.POINTS_SPLIT)
        ):
            if chunk_num < len(ch_split) - 1:

                im_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_tmp"
                ).with_suffix(".png")
                chunk_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_chunk_tmp"
                ).with_suffix(".tif")
                warp_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_georef_chunk_tmp"
                ).with_suffix(".tif")
                csv_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_tmp"
                ).with_suffix(".csv")

                # optional: export points
                points_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_points_tmp"
                    ).with_suffix(".points")

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

                except Exception as e:
                    print(f"gcp chunk: {np.shape(gcp_chunk)}")

                # optional: export points
                try:
                    points = [(lo, la, im_x, im_y) for (lo, la, im_x, im_y) in points_chunk]
                    np.savetxt(csv_path, points, fmt="%s", delimiter=",")
                except Exception as e:
                    print(f'Exception: {e}')

                gdal_translate = ["gdal_translate", "-of", "GTiff"]

                gdal_warp_utm = [
                        "gdal",
                        "raster",
                        "reproject",
                        "-r",
                        "near",
                        "--to",
                        self.warp_algorithm,   
                        "--co",
                        "COMPRESS=DEFLATE",
                        "-d",
                        self.epsg_code,
                        "-i",
                        str(chunk_path),
                        "-o",
                        str(warp_path)
                    ]
                
                gdal_warp_wgs84 = [
                        "gdal",
                        "raster",
                        "reproject",
                        "-r",
                        "near",
                        "--to",
                        self.warp_algorithm,
                        "--co",
                        "COMPRESS=DEFLATE",
                        "-d",
                        "EPSG:4326",
                        "-i",
                        str(chunk_path),
                        "-o",
                        str(warp_path)
                    ]
                

                if self.dynamic_chunking:
                    for i in range(4):
                        gdal_translate.extend(
                            ["-gcp", str(im_x[i]), str(im_y[i]), str(lo[i]), str(la[i])]
                        )
                    gdal_translate.extend([str(im_path), str(chunk_path)])
    
    
                    # gdal < 3.11 syntax
                    if False:
                        if self.active_utm:
                             gdal_warp = [
                                 "gdalwarp",
                                 "-r",
                                 "near",
                                 "-order",
                                 "1",
                                 "-co",
                                 "COMPRESS=DEFLATE",
                                 "-t_srs",
                                 self.epsg_code,
                                 str(chunk_path),
                                 str(warp_path),
                             ]
                        else:
                             gdal_warp = [
                                 "gdalwarp",
                                 "-r",
                                 "near",
                                 "-order",
                                 "1",
                                 "-co",
                                 "COMPRESS=DEFLATE",
                                 "-t_srs",
                                 "EPSG:4326",
                                 str(chunk_path),
                                 str(warp_path),
                             ]
                    
                # gdal 3.11 syntax
                #gdal raster reproject -r near --to SRC_METHOD=GCP_HOMOGRAPHY --co COMPRESS=DEFLATE -d=EPSG:4326 -i 2025-03-17_08-30-44_0_ch0_0_chunk_tmp.tif -o 2025-03-17_08-30-44_0_ch0_0_chunk_tmp_WGS84.tif
                    if self.active_utm:
                        gdal_warp = gdal_warp_utm 
                        #if self.active_poly:
                        #    gdal_warp = gdal_warp_utm_poly
                    else:    
                        gdal_warp = gdal_warp_wgs84
                        #if self.active_poly:
                        #    gdal_warp = gdal_warp_wgs84_poly


                elif not self.dynamic_chunking:
                    for i in range(len(gcp_chunk)):
                        gdal_translate.extend(
                            ["-gcp", str(im_x[i]), str(im_y[i]), str(lo[i]), str(la[i])]
                        )

                    gdal_translate.extend([str(im_path), str(chunk_path)])


                    # gdal 3.11 syntax
                    if self.active_utm:
                        gdal_warp = gdal_warp_utm 
                        #if self.active_poly:
                        #    gdal_warp = gdal_warp_utm_poly
                    else:    
                        gdal_warp = gdal_warp_wgs84
                        #if self.active_poly:
                        #    gdal_warp = gdal_warp_wgs84_poly


                # optional: append .points header and fist and last center point

                try:
                    first_line = '#CRS: GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984(Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World                   Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],               ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT           ["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.        0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE             ["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
                    second_line='mapX,mapY,sourceX,sourceY,enable,dX,dY,residual'
                    with open(csv_path, 'r') as ori_f:
                        ori_txt = ori_f.read()
                        #print(ori_txt)
                    with open(points_path, 'w') as mod_f:
                        mod_f.write(first_line + '\n' + second_line + '\n' + ori_txt)
                except Exception as e:
                    print(f'Exception: {e}')

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
                if name.endswith("_georef_chunk_tmp.tif") and not name.startswith("._"):
                    TIF.append(os.path.join(root, name))
                    np.savetxt(txt_path, TIF, fmt="%s")

        # delete merged file if it already exists
        if mosaic_tiff.exists():
            mosaic_tiff.unlink()

    # gdal < 3.11 syntax - still working fine
        if False:
                gdal_mosaic = [
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
    
    # gdal 3.11 syntax
        if True:
            #resolution_mode = self.resolution_mode == self.resolution_mode[3]
            gdal_mosaic = [
                "gdal", "raster", "mosaic",
                "-i", f"@{txt_path}",
                "-o", str(mosaic_tiff),
                "--src-nodata", "0",
                "--resolution", self.resolution_mode,
                "--co", "COMPRESS=DEFLATE",
                "--co", "TILED=YES"
            ]


        self.run_command(gdal_mosaic)


    def process(self):
        file_name = self.filepath.stem
        tif_path = self.output_folder / f"{file_name}_ch{self.channel}.tif"
        mosaic_tif_path = self.output_folder / f"{file_name}_ch{self.channel}_stack.tif"
        csv_ch = self.output_folder / f"{file_name}_ch{self.channel}.csv"

        self.prep_data()
        ch_stack = self.channel_stack()

        try:
            print(f"Processing chunks in channel {self.channel} with warp method {self.warp_algorithm}...")

            self.georeference(ch_stack=ch_stack, otiff=tif_path)

            print(f"Mosaicking channel {self.channel} with resolution mode {self.resolution_mode}...")
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
                or str(file_path).endswith("_tmp.points")
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

    georeferencer = SidescanGeoreferencer(args.xtf, args.channel, args.dynamic_chunking, args.UTM, args.poly)
    georeferencer.process()


if __name__ == "__main__":
    main()
