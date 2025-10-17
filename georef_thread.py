import argparse
from pathlib import Path
import os, copy
import numpy as np
import utm
import math
from sidescan_file import SidescanFile
from pyproj import CRS, datadir
from scipy.signal import savgol_filter
from scipy import interpolate
import numpy.ma as ma
import pygmt
import rioxarray as rio
from decimal import Decimal


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
    LALO_OUTER: list
    PING: list
    cog_smooth: np.ndarray
    vertical_beam_angle: int
    epsg_code: str
    LOLA_plt: np.ndarray
    HEAD_plt: np.ndarray
    LOLA_plt_ori: np.ndarray
    HEAD_plt_ori: np.ndarray

    # TODO: Adjust multithread module and main; 
    # add fields to enter user-defined resolution and search radius () or search radius = 2*radius;
    # Clean code (print stuff etc)
    # remove bool export nav

    def __init__(
        self,
        filepath: str | os.PathLike,
        channel: int = 0,
        active_utm: bool = True,
        active_poly: bool = True,
        active_export_navdata: bool = True,
        proc_data=None,
        nav = [],
        pix_size: float = 0.1,
        resolution: float = 1.0,
        search_radius: float = 2.0,
        output_folder: str | os.PathLike = "./georef_out",
        vertical_beam_angle: int = 60,
    ):
        self.filepath = Path(filepath)
        self.sidescan_file = SidescanFile(self.filepath)
        self.channel = channel
        self.active_utm = active_utm
        self.active_poly = active_poly
        self.active_export_navdata = active_export_navdata
        self.output_folder = Path(output_folder)
        self.vertical_beam_angle = vertical_beam_angle
        self.active_proc_data = False
        self.nav = nav
        self.pix_size = pix_size
        self.resolution = resolution
        self.search_radius = search_radius
        self.LALO_OUTER = []
        self.PING = []
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


    def get_pix_size(self, lo, la, res_factor):

        """
        Calculate distance between pings
        Calculate distance between coordinates in m 
        from a (randomly sampled) subset of coordinates array (else it takes very 
        long and the distances should be similiar throughout the interp. coords)
        Args: 
            - lo/la: arrays of interpolated(!) longitudes and latitudes
        """
        import geopy.distance

        start = int(len(lo)/2 - 100)
        stop = int(len(lo)/2 + 100)
        lo = lo[start:stop]
        la = la[start:stop]
        DIST = np.ones_like(lo)

        for i, (lon, lat, dst) in enumerate(zip(lo, la, DIST)):
            c_a = (lat, lon)
            c_b = (la[i - 1], lo[i - 1])
            DIST[i] = geopy.distance.distance(c_a, c_b).meters
        
        DIST[0] = np.nan
        # Round pixel resolution to 3 decimals and multiply by *factor* else too small
        self.pix_size = np.round(np.nanmedian(DIST), 3) * res_factor

        # Set first value to avoid jumps

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
                course_ang = np.atan2(la, lo)
                cog[i] = course_ang
        cog[0] = cog[1]
        cog = np.unwrap(cog)
        cog = np.rad2deg(cog)

        # Interpolate cog with univariate to get smooth curve; smoothing factor have been empirically defined
        cog_spl = interpolate.UnivariateSpline(ping_unique, cog, k=3, s=30)
        cog_intp = cog_spl(ping_uniform)
        self.cog_smooth = savgol_filter(cog_intp, 100, 3)

 
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

        ZERO_MASK = LON_ori != 0

        LON_ori = LON_ori[ZERO_MASK]
        LAT_ori = LAT_ori[ZERO_MASK]
        HEAD_ori = HEAD_ori[ZERO_MASK]

        GROUND_RANGE = GROUND_RANGE[ZERO_MASK]
        SLANT_RANGE = SLANT_RANGE[ZERO_MASK]
        self.PING = self.PING[ZERO_MASK]

        # Unwrap to avoid jumps when crossing 0/360° degree angle
        HEAD_ori_rad = np.deg2rad(HEAD_ori)
        head_unwrapped = np.unwrap(HEAD_ori_rad)
        head_unwrapped_savgol = savgol_filter(head_unwrapped, 100, 2)
        HEAD_savgol = (np.rad2deg(head_unwrapped_savgol)) % 360

        # Remove duplicate values
        UNIQUE_MASK = np.empty_like(LON_ori)
        i = 0
        for i, (lo, la, uni) in enumerate(zip(LON_ori, LAT_ori, UNIQUE_MASK)):
            if LON_ori[i] == LON_ori[i - 1] and LAT_ori[i] == LAT_ori[i - 1]:
                UNIQUE_MASK[i] = np.nan
            else:
                UNIQUE_MASK[i] = 0
        UNIQUE_MASK = [
            False if np.isnan(unique_val) else True for unique_val in UNIQUE_MASK
        ]
        LON_unique = LON_ori[UNIQUE_MASK]
        LAT_unique = LAT_ori[UNIQUE_MASK]
        PING_UNIQUE = self.PING[UNIQUE_MASK]

        # make uniform ping sequence for smooth curvature
        PING_uniform = np.linspace(self.PING[0], self.PING[-1], len(self.PING))

        # B-Spline lon/lats and filter to obtain esqual-interval, unique coordinates for each ping
        lo_spl = interpolate.make_interp_spline(
            PING_UNIQUE, LON_unique, k=3, bc_type="not-a-knot"
        )
        la_spl = interpolate.make_interp_spline(
            PING_UNIQUE, LAT_unique, k=3, bc_type="not-a-knot"
        )
        # Evaluate spline at equally spaced pings and smooth again with savgol filter
        lo_intp = lo_spl(PING_uniform)
        la_intp = la_spl(PING_uniform)
        lo_intp = savgol_filter(lo_intp, 100, 2)
        la_intp = savgol_filter(la_intp, 100, 2)

        # Create arrays for heading and coords for plotting in GUI
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
        east_spl = interpolate.make_interp_spline(
            PING_UNIQUE, EAST, k=3, bc_type="not-a-knot"
        )
        north_spl = interpolate.make_interp_spline(
            PING_UNIQUE, NORTH, k=3, bc_type="not-a-knot"
        )
        east_intp = east_spl(PING_uniform)
        north_intp = north_spl(PING_uniform)
        east_intp = savgol_filter(east_intp, 100, 2)
        north_intp = savgol_filter(north_intp, 100, 2)

        # Resize UTM Zone and Letter arrays to fit interpolated array sizes
        UTM_ZONE = np.resize(UTM_ZONE, len(east_intp))
        UTM_LET = np.resize(UTM_LET, len(east_intp))

        if self.channel == 1:
            EAST_OUTER = np.array(
                [
                    ground_range * math.sin(np.deg2rad(head)) + east
                    for ground_range, head, east in zip(
                        GROUND_RANGE, self.cog_smooth, east_intp
                    )
                ]
            )
            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(np.deg2rad(head)) * -1) + north
                    for ground_range, head, north in zip(
                        GROUND_RANGE, self.cog_smooth, north_intp
                    )
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
                    for ground_range, head, east in zip(
                        GROUND_RANGE, self.cog_smooth, east_intp
                    )
                ]
            )

            NORTH_OUTER = np.array(
                [
                    (ground_range * math.cos(np.deg2rad(head))) + north
                    for ground_range, head, north in zip(
                        GROUND_RANGE, self.cog_smooth, north_intp
                    )
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

        la_out_intp, lo_out_intp = map(np.array, zip(*self.LALO_OUTER))

        # linspace for coordinates along ping
        XX = []
        YY = []
        for x, y, x_out, y_out in zip(lo_intp, la_intp, lo_out_intp, la_out_intp):
            xx = np.linspace(x, x_out, swath_width)
            yy = np.linspace(y, y_out, swath_width)
            XX.append(xx)
            YY.append(yy)
        XX = np.ndarray.flatten(np.asarray(XX))
        YY = np.ndarray.flatten(np.asarray(YY))
        self.nav = np.column_stack((np.ndarray.flatten(XX),np.ndarray.flatten(YY)))

        # create empty grid with sample size as resolution (finest possible)

        #xx = np.linspace(np.min(lo_intp), np.max(lo_out_intp), swath_width)
        #yy = np.linspace(np.min(la_intp), np.max(la_out_intp), swath_width)
        #XX,YY = np.meshgrid(xx, yy)
        #print("np.shape(XX): ", np.shape(XX))
        #self.nav = np.column_stack((xx,yy))

    def channel_stack(self):
        """
        Work on raw or processed data, depending on `self.active_proc_data`
        - Stack channel so that the largest axis is horizontal
        - Norm data to max 255 for pic generation
        """

        # check whether processed data is present
        if self.active_proc_data:
            ch_stack = self.proc_data
        else:
            ch_stack = self.sidescan_file.data[self.channel]

        # Extract metadata for each ping in sonar channel, also longitude to mask invalid values
        PING = self.sidescan_file.packet_no
        lon = self.sidescan_file.longitude
        lon = np.ndarray.flatten(np.array(lon))
        mask_x = lon != 0
        mask_y = np.ones_like(ch_stack[0])

        # 'expand' to match ch_stack shape
        mask = mask_x[:, np.newaxis] * mask_y
        mask = mask.astype(bool)

        # Extract valid pings (same like ZERO mask for coordinates)
        ch_stack = ch_stack[np.all(mask, axis=1)]
        swath_len = len(PING)
        swath_width = len(ch_stack[0])
        print(f"swath_len: {swath_len}, swath_width: {swath_width}")

        # Transpose (always!) so that the largest axis is horizontal
        #ch_stack = ch_stack.T

        ch_stack = np.array(ch_stack, dtype=float)

        # Hack for alter transparency
        ch_stack /= np.max(np.abs(ch_stack)) / 254
        ch_stack = np.clip(ch_stack, 1, 255)

        # Flip array ---> Note: different for .jsf and .xtf!
        #ch_stack = np.flip(ch_stack, axis=0)
        ch_stack = np.flip(ch_stack, axis=1)
        ch_stack_flat = np.ndarray.flatten(ch_stack)

        return ch_stack.astype(np.uint8), ch_stack_flat.astype(np.uint8)

    def process(self):
        self.prep_data()
        chan_stack, chan_stack_flat = self.channel_stack()

        # Determine pixel size based on minimum distance between coordinates
        self.get_pix_size(self.nav[:,0], self.nav[:,1], res_factor=1)
        print("self.pix_res: ", self.pix_size)

        # Export navigation data
        if self.active_export_navdata:
            xyz = np.column_stack((self.nav, chan_stack_flat))
            nav_ch = (self.output_folder/f"Navigation_{self.filepath.stem}_ch{self.channel}.csv")
            print(f"Saving navinfo to {nav_ch}")
            outfile_median = (self.output_folder/f"outmedian_{self.filepath.stem}_ch{self.channel}.xyz")
            outfile_tiff = (self.output_folder/f"{self.filepath.stem}_{self.pix_size}m_ch{self.channel}_{self.epsg_code}.tif")
            
            np.savetxt(
                nav_ch,
                xyz,
                fmt="%s", 
                delimiter=";",
                #header="Nadir Longitude; Nadir Latitude; BS",
            )

        # Use pygmt to get region from xyz data, run blockmedian to reduce data size and then nearneighbor to produce interpolated grid.
        # output from nearneighbor is of type xarray so it can directly be used with rioxarray to assign CRS and save to geotiff - without gdal!
        # TODO: spatial indices, tiling/compressing? Reprojecting to utm, make radii user definable
        
        # Use coording precision as spacing of region parameter

        crd = Decimal(xyz[0,0])
        dgts = len(str(crd).split(".")[1]) - 2
        prec = 1/(10**dgts)
        region = pygmt.info(self.nav, per_column=True, spacing=(prec,prec))
        spacing = f"{self.pix_size}e"
        search_radius = f"{self.pix_size * 4}e"
        print("crd, prec, spacing, search_radius: ", crd, prec, spacing, search_radius)

        pygmt.blockmedian(
            data=xyz, 
            outfile=outfile_median, 
            output_type="file",
            coltypes="fg", 
            spacing=spacing, 
            region=region, 
            #verbose=True, 
            binary="o3d", 
            ) 
        data = pygmt.nearneighbor(
            data=outfile_median, 
            #outgrid=outfile_xyz, 
            coltypes="fg", 
            region=region, 
            #verbose=True, 
            binary="i3d", 
            spacing=spacing, 
            search_radius=search_radius 
            )

        # Clip data to range between 0 - 256
        data = data.clip(min=0.0, max=256.0)
        
        # Project to WGS84
        data_pr = data.rio.write_crs("EPSG:4326", inplace=True)

        # Reproject to utm if applied and save to geotiff
        if self.active_utm:
            self.epsg_code = self.epsg_code
            data_rpr = data_pr.rio.reproject(self.epsg_code, inplace=True)
            data_rpr.rio.to_raster(outfile_tiff)

        # Save as geotiff (set epsg_code for filename)
        else:
            self.epsg_code = "EPSG:4326"
            data_pr.rio.to_raster(outfile_tiff)

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
        help="If true, exports navigation data to csv",
    )

    args = parser.parse_args()
    print("args:", args)

    georeferencer = Georeferencer(
        args.xtf, args.channel, args.dynamic_chunking, args.UTM, args.poly, args.navdata
    )
    georeferencer.process()


if __name__ == "__main__":
    main()
