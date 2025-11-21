import argparse
from pathlib import Path
import os
import numpy as np
import utm
import math
from sidescan_file import SidescanFile
from pyproj import CRS
from scipy.signal import savgol_filter
from scipy import interpolate
import pygmt
from decimal import Decimal
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class Georeferencer:
    filepath: str | os.PathLike
    sidescan_file: SidescanFile
    channel: int
    active_utm: bool
    active_export_navdata: bool
    active_blockmedian: bool
    proc_data: np.array
    output_folder: str | os.PathLike
    active_proc_data: bool
    LALO_OUTER: list
    PING: list
    cog_smooth: np.ndarray
    vertical_beam_angle: int
    epsg_code: str
    pix_size: float
    resolution: float
    search_radius: float
    LOLA_plt: np.ndarray
    HEAD_plt: np.ndarray
    LOLA_plt_ori: np.ndarray
    HEAD_plt_ori: np.ndarray
    cable_out: float

    def __init__(
        self,
        filepath: str | os.PathLike,
        channel: int = 0,
        active_utm: bool = True,
        active_export_navdata: bool = False,
        active_blockmedian: bool = True,
        proc_data=None,
        nav=[],
        output_folder: str | os.PathLike = "./georef_out",
        vertical_beam_angle: int = 60,
        pix_size: float = 0.0,
        resolution: float = 0.0,
        search_radius: float = 0.0,
        cable_out: float = 0.0,
    ):
        self.filepath = Path(filepath)
        self.sidescan_file = SidescanFile(self.filepath)
        self.channel = channel
        self.active_utm = active_utm
        self.active_export_navdata = active_export_navdata
        self.active_blockmedian = active_blockmedian
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
        self.cable_out = cable_out
        if proc_data is not None:
            self.proc_data = proc_data
            self.active_proc_data = True
        self.setup_output_folder()
        self.PING = self.sidescan_file.packet_no

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
        from a middle subset of coordinates array (else it takes very
        long and the distances should be similar throughout the interp. coords)


        Parameters
        -----------
        lo: np.ndarray
            Array of interpolated(!) longitudes
        la: np.ndarray
            Array of interpolated(!) latitudes
        """
        import geopy.distance

        # define subset if array length is larger than 300 pings.
        if len(lo) > 300:
            start = int(len(lo) / 2 - 100)
            stop = int(len(lo) / 2 + 100)
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

        Parameters
        ----------
        lo: np.ndarray
            Longitude or Easting, unique and smoothed (savgol filteres) if neccessary
        la: np.ndarray
            Latitude or Northing, unique and smoothed (savgol filteres) if neccessary
        ping_unique:  np.ndarray
            Array of unique pings (without duplicates) to build spline
        ping_uniform:  np.ndarray
            Ping array for original length with monotonous ping numbers to evaluate spline
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
        PING_UNIQUE = [ping_no for ping_no in range(len(LON_unique))]

        # create uniform ping sequence for smooth curvature with original number of pings as length and last entry of unique ping for
        # maximum ping number, else b-spline will extrapolate which messes up coordinates
        PING_uniform = np.linspace(0, len(PING_UNIQUE) - 1, len(self.PING))

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

        # add offset https://apps.dtic.mil/sti/pdfs/AD1005010.pdf
        # 
        if self.cable_out:
            layback = math.sin(np.deg2rad(45)) * self.cable_out
            NORTH_LAY = []
            EAST_LAY = []
            LALO_LAY = []
            for east, north, utm_zone, letter, head in zip(EAST, NORTH, UTM_ZONE, UTM_LET, self.cog_smooth):
                east_lay = east - layback*math.sin(np.deg2rad(head))
                north_lay = north - layback*math.cos(np.deg2rad(head))
                lalo_lay = utm.to_latlon(east_lay, north_lay, utm_zone, letter)
                EAST_LAY.append(east_lay)
                NORTH_LAY.append(north_lay)
                LALO_LAY.append(lalo_lay)

            NORTH = NORTH_LAY
            EAST = EAST_LAY
            LAT_unique, LON_unique = map(np.array, zip(*LALO_LAY))
            print(layback, self.cable_out)


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


        # Create arrays for heading and coords for plotting in GUI
        x = range(len(self.cog_smooth))
        x_ori = range(len(HEAD_ori))
        self.HEAD_plt = np.column_stack((x, self.cog_smooth))
        self.HEAD_plt_ori = np.column_stack((x_ori, HEAD_ori))
        self.LOLA_plt = np.column_stack((lo_intp, la_intp))
        self.LOLA_plt_ori = np.column_stack((LON_ori, LAT_ori))

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
        self.nav = np.column_stack((np.ndarray.flatten(XX), np.ndarray.flatten(YY)))

    def channel_stack(self):
        """
        Work on raw or processed data, depending on `self.active_proc_data`
        - Norm data to max 255 for pic generation
        """

        # check whether processed data is present
        if self.active_proc_data:
            ch_stack = self.proc_data
        else:
            ch_stack = self.sidescan_file.data[self.channel]

        # Extract metadata for each ping in sonar channel, also longitude to mask invalid values
        lon = self.sidescan_file.longitude
        lon = np.ndarray.flatten(np.array(lon))
        mask_x = lon != 0
        mask_y = np.ones_like(ch_stack[0])

        # 'expand' to match ch_stack shape
        mask = mask_x[:, np.newaxis] * mask_y
        mask = mask.astype(bool)

        # Extract valid pings (same like ZERO mask for coordinates)
        ch_stack = ch_stack[np.all(mask, axis=1)]
        swath_len = len(lon)
        swath_width = len(ch_stack[0])
        print(f"swath_len: {swath_len}, swath_width: {swath_width}")

        ch_stack = np.array(ch_stack, dtype=float)

        # Hack for alter transparency
        ch_stack /= np.max(np.abs(ch_stack)) / 254
        ch_stack = np.clip(ch_stack, 1, 255)

        # Flip array ---> Note: different for .jsf and .xtf!
        if self.channel == 0:
            ch_stack = np.flip(ch_stack, axis=1)

        ch_stack_flat = np.ndarray.flatten(ch_stack)

        return ch_stack_flat.astype(np.uint8)

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

    def georeference(self, bs_data, progress_signal=None):
        """
        Method to georeference point cloud data.
        Uses pygmt to get region from xyz data by using coordinate precision as spacing.
        Runs blockmedian to reduce data size and nearneighbor on blockmedian output
        to produce final interpolated grid.
        Output from nearneighbor is of type xarray so it can directly
        be used with rioxarray to assign CRS and save to geotiff.

        Parameters
        ----------
        bs_data: np.ndarray
            1D array of backscatter data (can be amplitudes or greyscales)
        """

        # Determine pixel size based on minimum distance between coordinates
        self.get_pix_size(self.nav[:, 0], self.nav[:, 1], res_factor=1)

        resolution = f"{self.resolution}e"
        search_radius = f"{self.search_radius}e"

        # Define output file names
        # Convert resolution to mm to avoid "." in file name
        out_median = (
            self.output_folder / f"outmedian_{self.filepath.stem}_ch{self.channel}.xyz"
        )
        if self.resolution < 1.0:
            res_name = str(int(self.resolution * 100)) + "mm"
        else:
            res_name = str(int(self.resolution)) + "m"
        if self.active_utm:
            epsg_name = str(self.epsg_code).replace(":", "")
            out_tiff = (
                self.output_folder
                / f"{self.filepath.stem}_{res_name}_ch{self.channel}_{epsg_name}.tif"
            )
        else:
            out_tiff = (
                self.output_folder
                / f"{self.filepath.stem}_{res_name}_ch{self.channel}_EPSG4326.tif"
            )

        xybs = np.column_stack((self.nav, bs_data))
        crd = Decimal(xybs[0, 0])
        dgts = len(str(crd).split(".")[1]) - 2
        prec = 1 / (10**dgts)
        region = pygmt.info(self.nav, per_column=True, spacing=(prec, prec))

        if self.active_utm:
            print(
                f"Georeferencing with resolution {str(resolution).strip('e')}m and {str(self.search_radius).strip('e')}m in {self.epsg_code}."
            )
        else:
            print(
                f"Georeferencing with resolution {str(resolution).strip('e')}m and {str(self.search_radius).strip('e')}m in WGS84/EPSG:4326."
            )

        if self.active_blockmedian:
            print("Applying GMT Blockmedian...")
            pygmt.blockmedian(
                data=xybs,
                outfile=out_median,
                output_type="file",
                coltypes="fg",
                spacing=resolution,
                region=region,
                binary="o3d",
            )

            if progress_signal is not None:
                progress_signal.emit((1000 / len(bs_data)) * 0.05)

            print("Applying GMT Nearneighbour alg...")
            data_nn = pygmt.nearneighbor(
                data=out_median,
                coltypes="fg",
                region=region,
                binary="i3d",
                spacing=resolution,
                search_radius=search_radius,
            )
        else:
            print("Blockmedian off, using nearneighbor only")
            data_nn = pygmt.nearneighbor(
                data=xybs,
                coltypes="fg",
                region=region,
                binary="i3d",
                spacing=resolution,
                search_radius=search_radius,
            )

        # Clip data to range between 0 - 256
        data_clp = data_nn.clip(min=0.0, max=255.0)

        # Reproject to utm if applied and save to geotiff
        if self.active_utm:
            print(f"Saving to: {out_tiff}")
            data_pr = data_clp.rio.write_crs("EPSG:4326", inplace=True)
            data_rpr = data_pr.rio.reproject(self.epsg_code)
            data_rpr.rio.to_raster(out_tiff, compress="deflate", tiled=True)

        # Save as geotiff (set epsg_code for filename)
        else:
            print(f"Saving to: {out_tiff}")
            self.epsg_code = "EPSG:4326"
            data_pr = data_clp.rio.write_crs(self.epsg_code, inplace=True)
            data_pr.rio.to_raster(out_tiff, compress="deflate", tiled=True)

        if progress_signal is not None:
            progress_signal.emit(0.5)

    def process(self, progress_signal=None):
        # Check if enough data are present, otherwise quit
        if len(self.PING) > 300:
            self.prep_data()
            chan_stack_flat = self.channel_stack()

            try:
                self.georeference(
                    bs_data=chan_stack_flat, progress_signal=progress_signal
                )
            except Exception as e:
                print(str(e))

            # Export navigation data
            if self.active_export_navdata:
                xyz = np.column_stack((self.nav, chan_stack_flat))
                nav_ch = (
                    self.output_folder
                    / f"Navigation_{self.filepath.stem}_ch{self.channel}.csv"
                )
                print(f"Saving navinfo to {nav_ch}")

                np.savetxt(
                    nav_ch,
                    xyz,
                    fmt="%s",
                    delimiter=";",
                    header="Nadir Longitude; Nadir Latitude; BS",
                )
        else:
            print("Only {len(self.PING)} pings present. Quitting.")


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
        "--resolution",
        type=float,
        default=0.1,
        help="Output raster resolution",
    )
    parser.add_argument(
        "--search_radius",
        type=float,
        default=0.2,
        help="Search Radius for output raster creation. Usually 2 * resolution.",
    )
    parser.add_argument(
        "--UTM",
        type=bool,
        default=True,
        help="Uses UTM projection rather than WGS84. Default is UTM",
    )
    parser.add_argument(
        "--navdata",
        type=bool,
        default=False,
        help="If true, exports navigation data to csv",
    )
    parser.add_argument(
        "--blockmedian",
        type=bool,
        default=True,
        help="If True, uses blockmedian before nearneighbour alg. for gridding to reduce noise and data size. Default True.",
    )

    args = parser.parse_args()
    print("args:", args)

    georeferencer = Georeferencer(
        filepath=args.xtf,
        channel=args.channel,
        active_utm=args.UTM,
        active_export_navdata=args.navdata,
        active_blockmedian=args.blockmedian,
        resolution=args.resolution,
        search_radius=args.search_radius,
    )
    georeferencer.process()


if __name__ == "__main__":
    main()
