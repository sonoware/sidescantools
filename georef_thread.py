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
from scipy import interpolate
from matplotlib import pyplot as plt




class Georeferencer:
    filepath: str | os.PathLike
    sidescan_file: SidescanFile
    channel: int
    active_utm: bool
    active_poly: bool
    active_export_navdata: bool
    proc_data: np.array
    output_folder: str | os.PathLike
    active_proc_data: bool
    GCP_SPLIT: list
    POINTS_SPLIT: list
    LALO_OUTER: list
    PING: list
    turn_rate: list
    cog_smooth: np.ndarray
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
        active_export_navdata: bool = True,
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
        self.active_export_navdata = active_export_navdata
        self.output_folder = Path(output_folder)
        self.vertical_beam_angle = vertical_beam_angle
        self.warp_algorithm = warp_algorithm
        self.resolution_mode = resolution_mode
        self.resampling_method = resampling_method
        self.active_proc_data = False
        self.GCP_SPLIT = []
        self.POINTS_SPLIT = []
        self.LALO_OUTER = []
        self.PING = []
        self.turn_rate = []
        self.cog_smooth = np.empty_like(proc_data)
        self.LOLA_plt = np.empty_like(proc_data)
        self.HEAD_plt = np.empty_like(proc_data)
        self.LOLA_plt_ori = np.empty_like(proc_data)
        self.HEAD_plt_ori = np.empty_like(proc_data)
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
            
    def moving_segments(self, track_array, window_size):
        """
        Moving window to isolate turns within window_size. 
        """
        for i in range(len(track_array - window_size +1)):
            yield track_array[i:i+window_size]

    def calc_bearing(self, lo, la):
        """
        Calculates bearing according to haversine. Suitable for large scale datasets, 
        less for datasets that e.g. are within one UTM zone.
        """
        # convert to radians
        lon_rad = np.deg2rad(lo)
        lat_rad = np.deg2rad(la)

        # Create Coordinate pairs
        lon1 = lon_rad[:-1]
        lat1 = lat_rad[:-1]
        lon2 = lon_rad[1:]
        lat2 = lat_rad[1:]
        # Differences longitudes
        dlon = lon2 - lon1
        # Calculate bearing based on haversine bearing on sphere
        bearing = np.arctan2(
            np.sin(dlon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )
        # warp from 0-360°
        cog = (np.degrees(bearing) + 360) % 360
        # insert same value at the beginning to avoid artificial jumps
        cog = np.insert(cog, 0, cog[0])
        if False:
            # --- Smooth with moving average window size 5---
            kernel = np.ones(30) / 30
            cog = np.convolve(cog, kernel, mode="same")

    def calculate_distance(self, lo, la):
        """
        Calculate distance between pings
        """
        import geopy.distance
        DIST = np.empty_like(lo)
        for i, (lo, la, dst) in enumerate(zip(lo, la, DIST)):
            c_a = (la ,lo)
            c_b = (la[i-1], lo[i-1])
            DIST[i] = geopy.distance.distance(c_a, c_b).meters
        # Set first value to avoid jumps
        DIST[0] = 0

    def calculate_cog(self, lo, la, ping_unique, ping_uniform):
        """
        Calculate Course over Ground (COG)/true heading based on difference between single coordinates. 
        Note that coordinates must be unique!
        Paras:
            - lo: Longitude or Easting, unique and smoothed (savgol filteres) if neccessary
            - la: Latitude or Northing, unique and smoothed (savgol filteres) if neccessary

        Returns:
            - Course over ground in degrees (-180° -> 180°)
        """
        cog = np.empty_like(lo)
        LON_DIFF = np.diff(lo, prepend=np.nan)
        LAT_DIFF = np.diff(la, prepend=np.nan)
        for i, (lo, la, cang) in enumerate(zip(LON_DIFF, LAT_DIFF, cog)):
            # Set first value same as second to avoid nan or false differences
            if i == 0:
                course_ang = np.arctan2(LAT_DIFF[1], LON_DIFF[1])
                cog[i] = course_ang
            else:
                course_ang = np.atan2(la,lo)
                cog[i] = course_ang
        cog[0] = cog[1]
        cog = np.unwrap(cog)
        cog = (np.rad2deg(cog))

        # Interpolate cog with univariate to get smooth curve; smoothing factor have been empirically defined
        cog_spl = interpolate.UnivariateSpline(ping_unique, cog, k=3, s=30)
        cog_intp = cog_spl(ping_uniform)
        self.cog_smooth = savgol_filter(cog_intp, 100, 3)

    def generate_gcps(self, chunksize, northing, easting, northing_outer, easting_outer, lon, lat, lon_outer, lat_outer):
        """
        Generates ground control points (GCPs) for each chunk of chunksize (usually something like 3-5 pings).
        Paras:
            - chunksize: number of pings that are processed together as one chunk
            - northing/easting/lon/lat: Nadir UTM/unprojected coordinates
            - northing_outer, easting_outer/lon_outer/lat_outer: UTM/unprojected coordinates calculated for outer swath
        
        Returns:
            - List of Ground control points:
            Define corner coordinates for chunks and set gcps:
                - ul, ur: upper left/right --> nadir coordinates
                - ll, lr: lower left/right --> edge coordinates, calculated with ground range & heading
                - image coordinates: im_x_right = length of chunk
        
        """
        swath_len = len(self.PING)
        if self.active_proc_data:
            swath_width = len(self.proc_data[0])
        else:
            swath_width = len(self.sidescan_file.data[self.channel][0])

        self.chunk_indices = int(swath_len / chunksize)
        print(f"Fixed chunk size: {chunksize} pings.")
        print("swath_len, chunk_indices: ", swath_len, self.chunk_indices)

        # UTM
        if self.active_utm:
            lo_split_ce = np.array_split(easting, self.chunk_indices, axis=0)
            la_split_ce = np.array_split(northing, self.chunk_indices, axis=0)
            lo_split_e = np.array_split(easting_outer, self.chunk_indices, axis=0)
            la_split_e = np.array_split(northing_outer, self.chunk_indices, axis=0)

        else:
            lo_split_ce = np.array_split(lon, self.chunk_indices, axis=0)
            la_split_ce = np.array_split(lat, self.chunk_indices, axis=0)
            lo_split_e = np.array_split(lon_outer, self.chunk_indices, axis=0)
            la_split_e = np.array_split(lat_outer, self.chunk_indices, axis=0)

        # Calculate edge coordinates for first and last coordinates in chunks:        
        for chunk_num, (lo_chunk_ce, la_chunk_ce, lo_chunk_e, la_chunk_e) in enumerate(
            zip(lo_split_ce, la_split_ce, lo_split_e, la_split_e)
        ):

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
        print("PING_UNIQUE: ", len(self.PING))

        ZERO_MASK = LON_ori != 0
        print(ZERO_MASK)
        print(f"shape ping original: {np.shape(self.PING)}")

        LON_ori = LON_ori[ZERO_MASK]
        LAT_ori = LAT_ori[ZERO_MASK]
        HEAD_ori = HEAD_ori[ZERO_MASK]

        GROUND_RANGE = GROUND_RANGE[ZERO_MASK]
        SLANT_RANGE = SLANT_RANGE[ZERO_MASK]
        self.PING = self.PING[ZERO_MASK]
        print(f"shape ping zero: {np.shape(self.PING)}")

        # Unwrap to avoid jumps when crossing 0/360° degree angle
        HEAD_ori_rad = np.deg2rad(HEAD_ori)
        head_unwrapped = np.unwrap(HEAD_ori_rad)
        head_unwrapped_savgol = savgol_filter(head_unwrapped, 100, 2)
        HEAD_savgol = (np.rad2deg(head_unwrapped_savgol)) % 360

        # Remove duplicate values
        UNIQUE_MASK = np.empty_like(LON_ori)
        i = 0
        for i, (lo,la, uni) in enumerate(zip(LON_ori, LAT_ori, UNIQUE_MASK)):
            if LON_ori[i] == LON_ori[i-1] and LAT_ori[i] == LAT_ori[i-1]:
                UNIQUE_MASK[i] = np.nan
            else:
                UNIQUE_MASK[i] = 0
        UNIQUE_MASK = [False if np.isnan(unique_val) else True for unique_val in UNIQUE_MASK]
        LON_unique = LON_ori[UNIQUE_MASK]
        LAT_unique = LAT_ori[UNIQUE_MASK]
        PING_UNIQUE = self.PING[UNIQUE_MASK]

        # make uniform ping sequence for smooth curvature
        PING_uniform = np.linspace(self.PING[0], self.PING[-1], len(self.PING))

        # B-Spline lon/lats and filter to obtain esqual-interval, unique coordinates for each ping
        lo_spl = interpolate.make_interp_spline(PING_UNIQUE, LON_unique, k=3, bc_type="not-a-knot")  
        la_spl = interpolate.make_interp_spline(PING_UNIQUE, LAT_unique, k=3, bc_type="not-a-knot") 
        # Evaluate spline at equally spaced pings and smooth again with savgol filter
        lo_intp = lo_spl(PING_uniform)
        la_intp = la_spl(PING_uniform)
        lo_intp = savgol_filter(lo_intp, 100, 2)
        la_intp = savgol_filter(la_intp, 100, 2)

        # Create arrays for heading and coords for plotting in GUI
        # TODO: plot cog rather than heading?
        x = range(len(HEAD_savgol))
        x_ori = range(len(HEAD_ori))
        self.HEAD_plt = np.column_stack((x, HEAD_savgol))
        self.HEAD_plt_ori = np.column_stack((x_ori, HEAD_ori))
        self.LOLA_plt = np.column_stack((lo_intp, la_intp))
        self.LOLA_plt_ori = np.column_stack((LON_ori, LAT_ori))

        # Convert to UTM to calculate outer swath coordinates for both channels
        UTM = np.full_like(LAT_unique, np.nan)
        UTM = UTM.tolist()
        for idx, (la, lo) in enumerate(zip(LAT_unique, LON_unique)):
            try:
                UTM[idx] = utm.from_latlon(la, lo)
            except:
                ValueError("Values or lon and/or lat must not be 0")

        if UTM:
            EAST = [utm_coord[0] for utm_coord in UTM]
            NORTH = [utm_coord[1] for utm_coord in UTM]
            NORTH = np.asarray(NORTH)
            EAST = np.asarray(EAST)
            UTM_ZONE = [utm_coord[2] for utm_coord in UTM]
            UTM_LET = [utm_coord[3] for utm_coord in UTM]
            crs = CRS.from_dict({"proj": "utm", "zone": UTM_ZONE[0], "south": False})
            epsg = crs.to_authority()
            self.epsg_code = f"{epsg[0]}:{epsg[1]}"


        # calculate cog from east/north
        self.calculate_cog(EAST, NORTH, PING_UNIQUE, PING_uniform)


        # interpolate easting northing to full swath length
        east_spl = interpolate.make_interp_spline(PING_UNIQUE, EAST, k=3, bc_type="not-a-knot")  
        north_spl = interpolate.make_interp_spline(PING_UNIQUE, NORTH, k=3, bc_type="not-a-knot") 
        east_intp = east_spl(PING_uniform)
        north_intp = north_spl(PING_uniform)

        # Resize UTM Zone and Letter arrays to fit interpolated array sizes
        UTM_ZONE = np.resize(UTM_ZONE, len(east_intp))
        UTM_LET = np.resize(UTM_LET, len(east_intp))

        if self.channel == 1:
            EAST_OUTER = np.array(
                [
                    ground_range * math.sin(np.deg2rad(head)) + east
                    for ground_range, head, east in zip(GROUND_RANGE, self.cog_smooth, east_intp)
                ]
            )
            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(np.deg2rad(head)) * -1) + north
                    for ground_range, head, north in zip(GROUND_RANGE, self.cog_smooth, north_intp)
                ]
            )

            east_out_intp_savgol = savgol_filter(EAST_OUTER, 300, 2)
            north_out_intp_savgol = savgol_filter(NORTH_OUTER, 300, 2)

            self.LALO_OUTER = [
                utm.to_latlon(east_ch1, north_ch1, utm_zone, utm_let)
                for (east_ch1, north_ch1, utm_zone, utm_let) in zip(
                    east_out_intp_savgol, north_out_intp_savgol, UTM_ZONE, UTM_LET
                )
            ]


        elif self.channel == 0:
            EAST_OUTER = np.array(
                [
                    (ground_range * math.sin(np.deg2rad(head)) * -1) + east
                    for ground_range, head, east in zip(GROUND_RANGE, self.cog_smooth, east_intp)
                ]
            )

            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(np.deg2rad(head))) + north
                    for ground_range, head, north in zip(GROUND_RANGE, self.cog_smooth, north_intp)
                ]
            )

            east_out_intp_savgol = savgol_filter(EAST_OUTER, 300, 2)
            north_out_intp_savgol = savgol_filter(NORTH_OUTER, 300, 2)

            self.LALO_OUTER = [
                utm.to_latlon(east_ch2, north_ch2, utm_zone, utm_let)
                for (east_ch2, north_ch2, utm_zone, utm_let) in zip(
                    east_out_intp_savgol, north_out_intp_savgol, UTM_ZONE, UTM_LET
                )
            ]

        la_out_intp_savgol, lo_out_intp_savgol = map(np.array, zip(*self.LALO_OUTER))

        # Generate lists of gcps for each chunk
        self.generate_gcps(chunksize=5, northing=north_intp, easting=east_intp, northing_outer=north_out_intp_savgol, easting_outer=east_out_intp_savgol,
                           lon=lo_intp, lat=la_intp, lon_outer=lo_out_intp_savgol, lat_outer=la_out_intp_savgol)

        # Export navigation data
        if self.active_export_navdata:
            print(f"Saving navinfo to file")
            nav = np.column_stack((self.PING, lo_intp, la_intp, lo_out_intp_savgol, la_out_intp_savgol, east_intp, north_intp, east_out_intp_savgol, north_out_intp_savgol, self.cog_smooth))
            nav_ch = self.output_folder / f"Navigation_{self.filepath.stem}_ch{self.channel}.csv"

            np.savetxt(
                nav_ch,
                nav,
                fmt="%s",
                delimiter=";",
                header="Ping No; Nadir Longitude; Nadir Latitude; Outer Lon; Outer Lat; Nadir Easting; Nadir Northing; Outer Easting; Outer Northing; CoG [°]",
            )

        # Logic for cutting turn rates. Ignore for now. 
        if False:
            # Remove spikes in course ang (e.g. when ping order is messed up in very small sections of ~5-10 pings)
            window_spike = 10
            start_range = int(np.abs(np.min(cog)) - np.abs(np.median(cog)))
            stop_range = int(np.abs(np.max(cog)) - np.abs(np.median(cog)))
            accepted_range = range(start_range, stop_range)
            print("accepted_range: ", accepted_range, np.max(cog), np.min(cog))
            for idx, window_spike in enumerate(self.moving_segments(cog, window_spike)):
                if np.abs(cog[idx]) - np.abs(np.median(window_spike)) > np.abs(np.median(cog)):
                    #print("idx, window_spike, np.median(window_spike),: ", idx, np.abs(self.COURSE_ANG[idx]), np.abs(np.median(window_spike)), "outliers")
                    cog[idx] = np.median(window_spike)


            # calculate turn rates in [°/ping] (usually of magnitudes 0.xx°/ping) inside moving window of size X pings. to get an idea of turn rates within a distance, 
            # the rates are summed up to make the windows comparable
            # TODO: make window size depending on between-ping distances so that segment are ~50 (high turn rate within 50m is bad), 
            # remove outliers from course_ang if order of pings is messed up  - take median as threshold maybe?

            #window_seg = int(np.abs(np.ceil(30/np.median(DIST))))
            #print(window_seg)
            window_seg = 100
            self.turn_rate = []
            TR = []
            for window_seg in self.moving_segments(cog, window_seg):
                tr = np.abs(np.diff(window_seg, prepend=window_seg[0]))
                # find max turn rate within each window
                tr_max = np.max(tr)
                self.turn_rate.append(tr_max)
                TR.append(tr)

            print("np.ceil(np.max(self.turn_rate) - np.abs(np.median(self.turn_rate))): ", np.ceil(np.max(self.turn_rate) - np.abs(np.median(self.turn_rate))))

            # Find segment indices of turn rates > threshold
            #TURN_MASK = [False if rate >= 10 else True for rate in self.turn_rate ]
            threshold = np.ceil(np.max(self.turn_rate) - np.abs(np.median(self.turn_rate))) + 5
            TURN_MASK = [np.nan if rate >= threshold else rate for rate in self.turn_rate ]
            TURN_IDX = np.where(np.isnan(TURN_MASK))
            #TURN_IDX = TURN_IDX[0]
            #print("len TURN_IDX: ", (TURN_IDX))

            # Apply turn radius mask to cut turns
            cog[TURN_IDX] = np.nan
            self.turn_rate = np.asarray(self.turn_rate)
            self.turn_rate[TURN_IDX] = np.nan
            self.PING[TURN_IDX] = np.nan

            # NAN where turn index > threshold (the other arrays into segments at TURN_IDX)
            lo_intp[TURN_IDX] = np.nan
            la_intp[TURN_IDX] = np.nan
            lo_out_intp_savgol[TURN_IDX] = np.nan
            la_out_intp_savgol[TURN_IDX] = np.nan
            NORTH[TURN_IDX] = np.nan
            EAST[TURN_IDX] = np.nan
            NORTH_OUTER[TURN_IDX] = np.nan
            EAST_OUTER[TURN_IDX] = np.nan


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
            # process only non-empty gcp chunks
            if chunk_num < len(ch_split) - 1:
                if not np.isnan(gcp_chunk).any():
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

                else:
                    print("Contains turns")
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

        self.prep_data()
        ch_stack = self.channel_stack()

        try:
            print(
                f"Processing chunks in channel {self.channel} with warp method {self.warp_algorithm}..."
            )

            self.georeference(ch_stack=ch_stack, otiff=tif_path, progress_signal=progress_signal)


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
    parser.add_argument(
        "--navdata",
        type=bool,
        default=True,
        help="If true, exports navigation data to csv"
    )

    args = parser.parse_args()
    print("args:", args)

    georeferencer = Georeferencer(
        args.xtf, args.channel, args.dynamic_chunking, args.UTM, args.poly, args.navdata
    )
    georeferencer.process()


if __name__ == "__main__":
    main()
