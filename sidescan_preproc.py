import numpy as np
import copy
import scipy.signal as scisig
from sidescan_file import SidescanFile
from skimage.morphology.misc import remove_small_holes, remove_small_objects
import skimage
from skimage import feature
from pathlib import Path
import geopy.distance as geo_dist
from aux_functions import convert_to_dB, hist_equalization

PLOT = True
if PLOT:
    import matplotlib.pyplot as plt


class SidescanPreprocessor:
    """Main class to apply preprocessing functionalities to sidescan sonar data:
    - Init by loading a SidescanFile with desired parameters
    - Bottom line detection by threshold and display in napari
    - Slant range correction
    - Gain correction by EGN table
    """

    num_ch: int
    sidescan_file: SidescanFile
    doFilt: bool
    chunk_size: int
    bottom_strategy_choices = [
        "Each side individually",
        "Combine both sides",
        "Only use portside",
        "Only use starboard",
    ]
    """["Each side individually", "Combine both sides", "Only use portside", "Only use starboard"]"""
    downsampling_factor: int
    _napari_active_click_pos: bool

    def __init__(
        self,
        sidescan_file: SidescanFile,
        chunk_size=500,
        num_ch=2,
        downsampling_factor=1,
    ):
        """Main class to apply preprocessing functionalities (reading SidescanFiles,
        bottom line detection using napari, slant range and EGN correction)

        Parameters
        ----------
        sidescan_file: SidescanFile
            Reference to SidescanFile class that holds a loaded sidescan data file (XTF/JSF)
        chunk_size: int
            Number of pings per single chunk
        num_ch: int
            Number of channels - defaults to 2. This should be the usual case.
            Other cases are only partly integrated right now and it is not clear
            if there is a use case for this. This param will probably be deleted in the future.
        downsampling_factor: int
            Factor used for decimation of ping signals
        """
        self.sidescan_file = sidescan_file
        self.sonar_data_proc = copy.deepcopy(self.sidescan_file.data)
        self.sonar_data_proc = np.array(self.sonar_data_proc).astype(float)

        self.chunk_size = chunk_size
        self.num_chunk = int(np.ceil(self.sonar_data_proc.shape[1] / self.chunk_size))
        self.ping_len = self.sidescan_file.ping_len
        self.num_ch = num_ch
        self.downsampling_factor = downsampling_factor

        # store old minimal but positive value that might be needed later if filter introduce negative values
        self.pre_dec_least_val = np.min(
            self.sonar_data_proc[np.where(self.sonar_data_proc > 0)]
        )
        if downsampling_factor != 1:
            pre_dec_min = np.min(self.sonar_data_proc)
            self.sonar_data_proc = scisig.decimate(
                self.sonar_data_proc, downsampling_factor, axis=2
            )
            # Decimating filter might introduce negative values, avoid these
            self.sonar_data_proc = np.clip(
                self.sonar_data_proc, a_min=pre_dec_min, a_max=None
            )
            self.ping_len = int(np.ceil(self.ping_len / self.downsampling_factor))

        # initialiazation of itnern variables
        self._napari_active_click_pos = False

        ## Print spatial information estimation
        start_idx = 0
        start_coord = (
            sidescan_file.longitude[start_idx],
            sidescan_file.latitude[start_idx],
        )
        while sidescan_file.longitude[start_idx] == 0:
            if start_idx >= (len(sidescan_file.longitude) - 1):
                break
            start_idx += 1
            start_coord = (
                sidescan_file.longitude[start_idx],
                sidescan_file.latitude[start_idx],
            )
        end_coord = (
            sidescan_file.longitude[-1],
            sidescan_file.latitude[-1],
        )
        print("------------------------------------------------------------")
        print("--- Estimated spatial information by SidescanPreprocessor:")
        if np.size(self.sidescan_file.slant_range) > 1:
            print(
                f"Resolution in ping direction: {self.sidescan_file.slant_range[0, 0]/self.ping_len} m"
            )
        else:
            print(
                f"Resolution in ping direction: {self.sidescan_file.slant_range/self.ping_len} m"
            )
        print("(Estimated from slant range of first ping)")
        try:
            print(
                f"Resolution in tow/heading direction: {geo_dist.geodesic(end_coord, start_coord).m / self.sidescan_file.num_ping} m"
            )
        except:
            print("Geo Error")
        print("(Estimated from start and end GPS position)")
        print("------------------------------------------------------------")

    def detect_bottom_line_t(
        self,
        threshold_bin=0.75,
        combine_both_sides=False,
        plotting=False,
    ):

        # normalize each ping individually
        portside = self.sonar_data_proc[0]
        indv_max_portside = np.max(portside, 1)
        portside = portside / indv_max_portside[:, None]
        starboard = self.sonar_data_proc[1]
        indv_max_starboard = np.max(starboard, 1)
        starboard = starboard / indv_max_starboard[:, None]

        # edge detection
        initial_guess = True
        while initial_guess:

            portside_edges, starboard_edges = self.detect_edges(
                portside, starboard, threshold_bin, combine_both_sides, plotting
            )

            # convert most likely edge to bottom distance
            self.portside_bottom_dist = self.edges_to_bottom_dist(
                portside_edges, threshold_bin, data_is_port_side=True
            )
            self.starboard_bottom_dist = self.edges_to_bottom_dist(
                starboard_edges, threshold_bin, data_is_port_side=True
            )
            if plotting:
                usr_in = input(
                    f"Thresh is {threshold_bin}, enter new thresh or nothing to keep current threshold: "
                )
                if usr_in != "":
                    try:
                        threshold_bin = float(usr_in)
                    except:
                        print(f"Couldn't interpret input: {usr_in}")
                        finding_thresh = True
                        while finding_thresh:
                            usr_in = input("Enter new threshold:")
                            try:
                                threshold_bin = float(usr_in)
                                finding_thresh = False
                            except:
                                print(f"Couldn't interpret input: {usr_in}")
                else:
                    initial_guess = False
            else:
                initial_guess = False

    def init_napari_bottom_detect(
        self,
        default_threshold,
        active_dB=False,
        active_hist_equal=False,
        depth_info=None,
    ):

        # if depth data is present, this is set here
        self.napari_depth_info = depth_info
        # normalize each ping individually
        portside = np.array(self.sonar_data_proc[0], dtype=float)
        starboard = np.array(self.sonar_data_proc[1], dtype=float)
        if active_dB:
            portside = convert_to_dB(portside)
            starboard = convert_to_dB(starboard)
        if np.min(portside) < 0:
            portside = portside - np.min(portside)
        if np.min(starboard) < 0:
            starboard = starboard - np.min(starboard)
        indv_max_portside = np.max(portside, 1)
        portside = portside / indv_max_portside[:, None]
        indv_max_starboard = np.max(starboard, 1)
        starboard = starboard / indv_max_starboard[:, None]

        if active_hist_equal:
            portside = hist_equalization(portside)
            starboard = hist_equalization(starboard)

        # do initial bottom line detection for start values
        self.detect_bottom_line_t(
            threshold_bin=default_threshold,
            combine_both_sides=True,
            plotting=False,
        )

        ## prepare data to chunks
        # build list containing each chunk
        self.num_chunk = int(np.ceil(len(self.portside_bottom_dist) / self.chunk_size))
        self.napari_portside_chunk = np.zeros(
            (self.num_chunk, self.chunk_size, self.ping_len)
        )
        self.napari_starboard_chunk = np.zeros(
            (self.num_chunk, self.chunk_size, self.ping_len)
        )
        self.napari_fullmat = np.zeros(
            (self.num_chunk, self.chunk_size, int(2 * self.ping_len))
        )
        self.napari_fullmat_bin = np.zeros(
            (self.num_chunk, self.chunk_size, int(2 * self.ping_len))
        )
        self.edges_mat = np.zeros(
            (self.num_chunk, self.chunk_size, int(2 * self.ping_len)), dtype=bool
        )
        self.napari_portside_bottom = np.zeros(
            (self.num_chunk, self.chunk_size), dtype=int
        )
        self.napari_starboard_bottom = np.zeros(
            (self.num_chunk, self.chunk_size), dtype=int
        )
        for chunk_idx in range(self.num_chunk):
            if chunk_idx < self.num_chunk - 1:
                num_ping = self.chunk_size
            else:
                num_ping = np.shape(
                    portside[
                        chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                    ]
                )[0]
            self.napari_portside_chunk[chunk_idx, :num_ping] = portside[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ]
            self.napari_starboard_chunk[chunk_idx, :num_ping] = starboard[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ]
            self.napari_fullmat[chunk_idx, :num_ping] = np.hstack(
                (
                    portside[
                        chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                    ],
                    starboard[
                        chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                    ],
                )
            )
            self.napari_portside_bottom[chunk_idx, :num_ping] = (
                self.portside_bottom_dist[
                    chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                ]
            )
            self.napari_starboard_bottom[chunk_idx, :num_ping] = (
                self.starboard_bottom_dist[
                    chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                ]
            )

            self.build_bottom_line_map()

    def build_bottom_line_map(self):

        ## build map for plotting
        map_shape = (self.num_chunk, self.chunk_size, self.ping_len)
        star_map = np.zeros(map_shape)
        port_map = np.zeros(map_shape)

        for chunk_idx in range(self.num_chunk):
            starboard_dep_chunk = self.starboard_bottom_dist[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ]
            portside_dep_chunk = self.portside_bottom_dist[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ]
            cur_chunk_len = len(portside_dep_chunk)
            for crnr in range(cur_chunk_len):

                # make wider if not at border
                if (
                    1 < starboard_dep_chunk[crnr] < map_shape[2] - 2
                    and 1 < portside_dep_chunk[crnr] < map_shape[2] - 2
                ):
                    star_map[
                        chunk_idx,
                        crnr,
                        starboard_dep_chunk[crnr] - 1 : starboard_dep_chunk[crnr] + 2,
                    ] = 1
                    port_map[
                        chunk_idx,
                        crnr,
                        portside_dep_chunk[crnr] - 1 : portside_dep_chunk[crnr] + 2,
                    ] = 1
                else:
                    star_map[
                        chunk_idx,
                        crnr,
                        starboard_dep_chunk[crnr] : starboard_dep_chunk[crnr] + 1,
                    ] = 1
                    port_map[
                        chunk_idx,
                        crnr,
                        portside_dep_chunk[crnr] : portside_dep_chunk[crnr] + 1,
                    ] = 1

        # chunk_data = np.hstack((portside_chunk, starboard_chunk))
        self.bottom_map = np.dstack((port_map, star_map))

    def detect_bottom_napari(
        self,
        chunk_idx,
        threshold_bin=0.6,
        bottom_strategy_choice="",
        add_line_width=0,
    ):
        """Do bottom line detection update for a single chunk of given idx"""

        # Apply custom contrast limits

        # edge detection
        combine_both_sides = bottom_strategy_choice == self.bottom_strategy_choices[1]
        portside_edges_chunk, starboard_edges_chunk = self.detect_edges(
            self.napari_portside_chunk[chunk_idx],
            self.napari_starboard_chunk[chunk_idx],
            threshold_bin,
            combine_both_sides,
            False,
            chunk_idx=chunk_idx,
        )
        # convert most likely edge to bottom distance
        self.napari_portside_bottom[chunk_idx] = self.edges_to_bottom_dist(
            portside_edges_chunk, threshold_bin, data_is_port_side=True
        )
        self.napari_starboard_bottom[chunk_idx] = self.edges_to_bottom_dist(
            starboard_edges_chunk, threshold_bin, data_is_port_side=False
        )

        if bottom_strategy_choice == self.bottom_strategy_choices[2]:
            self.napari_starboard_bottom[chunk_idx] = (
                self.ping_len - self.napari_portside_bottom[chunk_idx]
            )
        elif bottom_strategy_choice == self.bottom_strategy_choices[3]:
            self.napari_portside_bottom[chunk_idx] = (
                self.ping_len - self.napari_starboard_bottom[chunk_idx]
            )

        self.update_bottom_map_napari(chunk_idx, add_line_width=add_line_width)

    def set_depth_from_info(self, offset: int):
        for chunk_idx in range(self.num_chunk):
            depth_chunk = (
                self.napari_depth_info[
                    chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                ]
                + offset
            )
            self.napari_portside_bottom[chunk_idx, : len(depth_chunk)] = (
                self.ping_len - depth_chunk
            )
            self.napari_starboard_bottom[chunk_idx, : len(depth_chunk)] = depth_chunk
            self.update_bottom_map_napari(chunk_idx, add_line_width=1)

    def update_bottom_map_napari(self, chunk_idx, add_line_width=0):
        # update bottom map
        chunk_shape = (self.chunk_size, self.ping_len)
        star_map = np.zeros(chunk_shape)
        port_map = np.zeros(chunk_shape)
        for crnr in range(self.chunk_size):

            # make wider if not at border
            if (
                1 + add_line_width
                < self.napari_starboard_bottom[chunk_idx, crnr]
                < chunk_shape[1] - 2 - add_line_width
                and 1 + add_line_width
                < self.napari_portside_bottom[chunk_idx, crnr]
                < chunk_shape[1] - 2 - add_line_width
            ):
                star_map[
                    crnr,
                    self.napari_starboard_bottom[chunk_idx, crnr]
                    - add_line_width : self.napari_starboard_bottom[chunk_idx, crnr]
                    + 1
                    + add_line_width,
                ] = 1
                port_map[
                    crnr,
                    self.napari_portside_bottom[chunk_idx, crnr]
                    - add_line_width : self.napari_portside_bottom[chunk_idx, crnr]
                    + 1
                    + add_line_width,
                ] = 1
            else:
                star_map[
                    crnr,
                    self.napari_starboard_bottom[
                        chunk_idx, crnr
                    ] : self.napari_starboard_bottom[chunk_idx, crnr]
                    + 1,
                ] = 1
                port_map[
                    crnr,
                    self.napari_portside_bottom[
                        chunk_idx, crnr
                    ] : self.napari_portside_bottom[chunk_idx, crnr]
                    + 1,
                ] = 1

        self.bottom_map[chunk_idx] = np.hstack((port_map, star_map))

    def update_bottom_detect_plot_napari(
        self,
        image_layer,
        map_layer,
        portside_chunk,
        starboard_chunk,
        portside_dep_chunk,
        starboard_dep_chunk,
    ):
        map_shape = np.shape(portside_chunk)
        star_map = np.zeros(map_shape)
        port_map = np.zeros(map_shape)
        for crnr in range(map_shape[0]):

            # make wider if not at border
            if (
                1 < starboard_dep_chunk[crnr] < map_shape[0] - 2
                and 1 < portside_dep_chunk[crnr] < map_shape[0] - 2
            ):
                star_map[
                    crnr, starboard_dep_chunk[crnr] - 1 : starboard_dep_chunk[crnr] + 2
                ] = 1
                port_map[
                    crnr, portside_dep_chunk[crnr] - 1 : portside_dep_chunk[crnr] + 2
                ] = 1
            else:
                star_map[
                    crnr, starboard_dep_chunk[crnr] : starboard_dep_chunk[crnr] + 1
                ] = 1
                port_map[
                    crnr, portside_dep_chunk[crnr] : portside_dep_chunk[crnr] + 1
                ] = 1

        chunk_data = np.hstack((portside_chunk, starboard_chunk))
        cur_map = np.hstack((port_map, star_map))

        image_layer.data = chunk_data
        map_layer.data = cur_map

    # wrapper function for edge detection
    def detect_edges(
        self,
        portside,
        starboard,
        threshold_bin,
        combine_both_sides,
        plotting,
        chunk_idx=-1,
    ):
        # make binary and invert
        portside_bin = portside > threshold_bin
        portside_bin = portside_bin - 1
        portside_bin = np.abs(portside_bin)
        portside_bin = np.array(portside_bin, dtype=bool)
        starboard_bin = starboard > threshold_bin
        starboard_bin = starboard_bin - 1
        starboard_bin = np.abs(starboard_bin)
        starboard_bin = np.array(starboard_bin, dtype=bool)

        # remove holes
        remove_small_holes(portside_bin)
        remove_small_objects(portside_bin)
        remove_small_holes(starboard_bin)
        remove_small_objects(starboard_bin)

        # smoothing
        portside_bin = skimage.filters.gaussian(portside_bin, sigma=1)
        starboard_bin = skimage.filters.gaussian(starboard_bin, sigma=1)
        portside_bin = np.array(np.round(portside_bin), dtype=bool)
        starboard_bin = np.array(np.round(starboard_bin), dtype=bool)

        # combine both sides with AND if wanted
        if combine_both_sides:
            combined_bin = np.logical_not(
                np.logical_and(
                    np.logical_not(portside_bin),
                    np.logical_not(np.fliplr(starboard_bin)),
                )
            )  # no nand functionality? :(

        # edge detection
        if combine_both_sides:
            portside_edges = feature.canny(combined_bin, sigma=3)
            starboard_edges = feature.canny(np.fliplr(combined_bin), sigma=3)
        else:
            portside_edges = feature.canny(portside_bin, sigma=3)
            starboard_edges = feature.canny(starboard_bin, sigma=3)

        if chunk_idx != -1:
            self.napari_fullmat_bin[chunk_idx] = np.hstack(
                (portside_bin, starboard_bin)
            )
            port_edges = np.zeros_like(portside_bin)
            port_edges[portside_edges] = 1
            star_edges = np.zeros_like(starboard_bin)
            star_edges[starboard_edges] = 1
            self.edges_mat[chunk_idx] = np.array(
                np.hstack((port_edges, star_edges)), dtype=bool
            )

        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(portside)
        # axs[1].imshow(starboard)

        # print(f"Threshold is: {threshold_bin}")

        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(portside_bin)
        # axs[1].imshow(starboard_bin)

        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(portside_edges)
        # axs[1].imshow(starboard_edges)

        # plt.show(block=True)

        return (portside_edges, starboard_edges)

    # find depth TODO: is there a better way to do this with skimage?
    def edges_to_bottom_dist(
        self,
        edges,
        threshold_bin,
        data_is_port_side,
        click_pos=None,
        dist_at_ends=20,  # TODO: find a good val
    ):

        cand_start = 0
        # TODO: cand_start von außen einstellbar machen (per Mausclick) -> Loopen von da aus in beide Richtungen, um den den Candidaten zu verfolgen?
        idx = 0

        # if a position has been clicked, this shall be our start candidate
        if click_pos:
            pass
        else:
            # find first ping edge result, where only one candidate is present as initial guess
            ping_len = np.shape(edges)[1]
            while cand_start == 0 and idx < np.shape(edges)[0]:
                edge_list = (
                    np.where(edges[idx, dist_at_ends : -1 * dist_at_ends])[0]
                    + dist_at_ends
                )
                idx += 1
                if len(edge_list) == 1:
                    cand_start = edge_list[0]

            candidates = []
            # if no start idx is found, choose first or last idx, depending on where thresh is closer to
            if len(edge_list) == 0:
                if threshold_bin < 0.5:
                    if data_is_port_side:
                        cand_start = ping_len - 3
                    else:
                        cand_start = 2
                else:
                    if data_is_port_side:
                        cand_start = 2
                    else:
                        cand_start = ping_len - 3

        # iterate over all pings to determine bottom distance for each ping
        last_cand = cand_start
        for ping_idx in range(np.shape(edges)[0]):
            edge_list = (
                np.where(edges[ping_idx, dist_at_ends : -1 * dist_at_ends])[0]
                + dist_at_ends
            )
            # else choose closest candidate, otherwise if no edge is present, just keep last guess
            if len(edge_list > 0):
                closest_idx = np.argmin(abs(edge_list - last_cand))
                last_cand = edge_list[closest_idx]
            candidates.append(last_cand)

        return np.array(candidates, dtype=int)

    @staticmethod
    def get_sup_line_lin(sup_fact, width):
        if width >= 3:
            rad = int(np.round(width / 2) - 1)
            sup_fact_lin = np.linspace(0.0, sup_fact, rad)
            sup_fact_lin = np.hstack([sup_fact_lin, sup_fact, np.flip(sup_fact_lin)])
        elif width == 1:
            sup_fact_lin = np.array([0.0])
        else:
            sup_fact_lin = np.array([sup_fact])

        sup_fact_lin = 1 / (10 ** (sup_fact_lin / 20))
        return sup_fact_lin

    @staticmethod
    def build_pie_H(M, N, width_end=0.1, dist_to_mid=0.0, sup_fact=80, peak_pos=None):
        """
        Parameters
        ----------
        M: int
            Size of first image dimension
        N: int
            Size of second image dimension
        width_end: float
            Width of pie at end
        dist_to_mid: float
            Fractional distance to 0,0 where pie starts
        sup_fact: float
            Factor of attenuation in middle of pie filter
        peak_pos: None or np.array
            Position to rotate pie slice to go through"""

        H = np.ones((M, N))
        width_end = int(np.round(width_end * N))
        dist_to_mid = int(np.round(dist_to_mid * M / 2))
        pie_len = int(M / 2 - dist_to_mid)
        width_lin = np.linspace(width_end, 1, pie_len)
        width_lin = np.round(width_lin).astype(int)

        if peak_pos is not None:
            dist_fact = pie_len / peak_pos[0]
            max_shift = int(np.round(dist_fact * (peak_pos[1] - N / 2)))
            shift_vec = np.linspace(0, max_shift, pie_len).astype(int)
        else:
            shift_vec = np.linspace(0, 0, pie_len).astype(int)

        for l_idx in range(pie_len):
            cur_width = width_lin[l_idx]
            start_idx = int(N / 2 - np.round(cur_width / 2))
            sup_line = SidescanPreprocessor.get_sup_line_lin(sup_fact, cur_width)
            H[
                l_idx,
                start_idx
                + shift_vec[l_idx] : start_idx
                + len(sup_line)
                + shift_vec[l_idx],
            ] = sup_line

        H[int(M / 2) :] = np.flipud(H[: int(M / 2)])
        return H

    def apply_beam_pattern_correction(self):
        """Applies Beam Pattern Correction to the processing data"""

        num_ping = np.shape(self.sonar_data_proc[0])[0]
        angle_range = [-1 * np.pi / 2, np.pi / 2]
        angle_num = 360
        angle_stepsize = (angle_range[1] - angle_range[0]) / angle_num
        angle_sum = np.zeros(angle_num)
        angle_hits = np.zeros(angle_num)
        EPS = np.finfo(float).eps

        print("Estimating beam pattern...")
        son_dat = np.hstack(
            (np.fliplr(self.sonar_data_proc[0]), self.sonar_data_proc[1])
        )
        mean_depth = np.array(np.round(np.mean(self.dep_info, 0)), dtype=int)
        dd = mean_depth**2
        alpha_idx = np.zeros((num_ping, 2 * self.ping_len), dtype=int)
        for vector_idx in range(num_ping):
            r = np.sqrt(
                (
                    np.linspace(0, 2 * self.ping_len - 1, 2 * self.ping_len)
                    - self.ping_len
                )
                ** 2
                + dd[vector_idx]
            )
            alpha = np.sign(
                np.linspace(0, 2 * self.ping_len - 1, 2 * self.ping_len) - self.ping_len
            ) * np.arccos(mean_depth[vector_idx] / (r + EPS))
            alpha_idx[vector_idx] = np.array(
                np.round(alpha / angle_stepsize) + angle_num / 2, dtype=int
            )
            for ping_idx in range(self.ping_len):
                angle_sum[alpha_idx[vector_idx, ping_idx]] += son_dat[
                    vector_idx, ping_idx
                ]
                angle_hits[alpha_idx[vector_idx, ping_idx]] += 1
        print(f"Beam pattern done - correcting data.")

        # angle_sum /= angle_hits
        angle_sum = np.divide(
            angle_sum, angle_hits, out=np.zeros_like(angle_sum), where=angle_hits != 0
        )
        angle_sum[np.where(angle_hits == 0)] = np.nan
        angle_sum[np.where(angle_sum == 0)] = 1.0

        for vector_idx in range(num_ping):
            for ping_idx in range(self.ping_len):
                son_dat[vector_idx, ping_idx] /= angle_sum[
                    alpha_idx[vector_idx, ping_idx]
                ]

        self.sonar_data_proc[0] = np.fliplr(son_dat[:, : self.ping_len])
        self.sonar_data_proc[1] = son_dat[:, self.ping_len :]

    # Energy normalization
    def apply_energy_normalization(self):
        """Apply energy normalisation on each individual ping (is needed after BAC processing)"""
        for ch in range(self.num_ch):
            son_dat = self.sonar_data_proc[ch]
            num_ping = np.shape(son_dat)[0]
            num_norm = 40
            pow_vec = np.sum(son_dat[:40] ** 2, axis=1)
            for ping_idx in range(num_ping):
                if int(num_norm / 2) < ping_idx < num_ping - int(num_norm / 2):
                    pow_vec[:-1] = pow_vec[1:]
                    pow_vec[-1] = np.sum(son_dat[ping_idx + int(num_norm / 2)] ** 2)
                son_dat[ping_idx] /= np.sqrt(np.mean(pow_vec))
            self.sonar_data_proc[ch] = son_dat

    # Pie slice filter to remove noisy lines
    def apply_pie_slice_filter(self):
        """Apply pie slice filter to remove stripe noise in image"""
        if self.sidescan_file.num_ping < self.chunk_size:
            print("Not implemented for Nping < chunk size")
            return
        for ch in range(self.num_ch):
            son_dat = self.sonar_data_proc[ch]
            pre_filt_max = np.max(son_dat)
            pre_filt_min = np.min(son_dat[son_dat > 0])
            for chunk_idx in range(self.num_chunk):
                # avoid zero padding
                if chunk_idx == self.num_chunk - 1:
                    cur_chunk = son_dat[-1 * self.chunk_size :]
                else:
                    cur_chunk = son_dat[
                        chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                    ]

                # apply filter
                spec = np.fft.fft2(cur_chunk)
                spec_r = np.vstack(
                    [spec[int(self.chunk_size / 2) :], spec[: int(self.chunk_size / 2)]]
                )
                spec_r = np.hstack(
                    [
                        spec_r[:, int(self.ping_len / 2) :],
                        spec_r[:, : int(self.ping_len / 2)],
                    ]
                )

                # build rotated H
                spec_max_1 = np.max(20 * np.log10(np.abs(spec_r)), axis=1)
                search_lim = int(self.chunk_size / 2 * 0.9)
                peaks, _ = scisig.find_peaks(spec_max_1[:search_lim], prominence=10)
                # print(scisig.peak_prominences(spec_max_1[:search_lim], peaks))
                far_peak_pos = None
                if len(peaks) >= 2:
                    last_peaks = peaks[-2:]
                    peak_positions = np.zeros((2, 2))
                    for ii in range(2):
                        peak_positions[ii, 0] = last_peaks[ii]
                        peak_positions[ii, 1] = np.where(
                            20 * np.log10(np.abs(spec_r[last_peaks[ii]]))
                            == spec_max_1[last_peaks[ii]]
                        )[0]

                    # take pos which is furthest away from mid
                    far_peak_pos = peak_positions[
                        np.argmax(np.abs(peak_positions[:, 1] - self.ping_len / 2))
                    ]

                # Only apply filtering if atleast one peak is present
                if len(peaks) >= 1:
                    H = SidescanPreprocessor.build_pie_H(
                        self.chunk_size, self.ping_len, peak_pos=far_peak_pos
                    )
                    spec_filt_r = spec_r * H

                    spec_filt = np.vstack(
                        [
                            spec_filt_r[int(self.chunk_size / 2) :],
                            spec_filt_r[: int(self.chunk_size / 2)],
                        ]
                    )
                    spec_filt = np.hstack(
                        [
                            spec_filt[:, int(self.ping_len / 2) :],
                            spec_filt[:, : int(self.ping_len / 2)],
                        ]
                    )
                    chunk_filt = np.real(np.fft.ifft2(spec_filt))

                    if chunk_idx == self.num_chunk - 1:
                        son_dat[-1 * self.chunk_size :] = chunk_filt
                    else:
                        son_dat[
                            chunk_idx
                            * self.chunk_size : (chunk_idx + 1)
                            * self.chunk_size
                        ] = chunk_filt
                else:
                    print("No peak found, pie slice filter skipped")

            # filtering could introduce negative numbers and distortions
            # rescale the data here to old pre filtering range
            if not hasattr(self, "pre_dec_least_val"):
                self.pre_dec_least_val = pre_filt_min
            if np.min(son_dat) < 0:
                son_dat -= np.min(son_dat)
                son_dat += self.pre_dec_least_val
                son_dat /= np.max(son_dat)
                son_dat *= pre_filt_max
            self.sonar_data_proc[ch] = son_dat

        # TODO: Delete, this is curently used for later visualization
        self.dat_pie_slice_copy = copy.deepcopy(self.sonar_data_proc)

    @staticmethod
    def comp_D(u, v, M_2, N_2):
        return np.sqrt((u - M_2) ** 2 + (v - N_2) ** 2)

    def apply_sharpening_filter(self):
        """Use a homomorphic filter to emphasise higher frequencies for image sharpening"""
        print("Applying sharpening filter")
        if self.sidescan_file.num_ping < self.chunk_size:
            print("Not implemented for Nping < chunk size")
            return
        for ch in range(self.num_ch):
            son_dat = self.sonar_data_proc[ch]
            for chunk_idx in range(self.num_chunk):
                # avoid zero padding
                if chunk_idx == self.num_chunk - 1:
                    cur_chunk = son_dat[-1 * self.chunk_size :]
                else:
                    cur_chunk = son_dat[
                        chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                    ]

                if np.nanmin(cur_chunk) <= 0:
                    cur_chunk = np.clip(cur_chunk, a_min=1e-3, a_max=None)
                img_proc = np.log(cur_chunk + np.finfo(float).eps)
                img_spec = np.fft.fft2(img_proc)

                gamma_H = 3.0
                gamma_L = 0.3
                M_2 = self.chunk_size / 2
                N_2 = self.ping_len / 2
                const_c = 2
                D_0 = self.comp_D(0, 0, M_2, N_2)
                H = np.zeros((self.chunk_size, self.ping_len))
                for u in range(self.chunk_size):
                    for v in range(self.ping_len):
                        H[u, v] = (gamma_H - gamma_L) * (
                            1
                            - np.exp(
                                -1
                                * const_c
                                * ((self.comp_D(u, v, M_2, N_2) ** 2) / (D_0**2))
                            )
                        ) + gamma_L

                img_filtered = img_spec * H

                img_r = np.real(np.fft.ifft2(img_filtered))
                chunk_filt = np.exp(img_r)

                if chunk_idx == self.num_chunk - 1:
                    son_dat[-1 * self.chunk_size :] = chunk_filt
                else:
                    son_dat[
                        chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                    ] = chunk_filt

            self.sonar_data_proc[ch] = son_dat

    # Refine bottom detection
    def refine_detected_bottom_line(
        self,
        search_range,
        active_single_altitude_offset=False,  # TODO: CFG
        active_bottom_smoothing=True,  # TODO CFG
    ):
        # copy data
        son_data = np.copy(self.sonar_data_proc)

        # read altitude information
        raw_altitude = self.sidescan_file.sensor_primary_altitude
        if np.max(raw_altitude) == 0:
            raise ValueError("No depth information found in intern data.")
        raw_altitude = self.fill_zeros_with_last(raw_altitude)

        # read depth info and calc corresponding sample idx
        stepsize = self.sidescan_file.slant_range[0, :] / self.ping_len
        self.dep_info = [
            raw_altitude / stepsize,
            raw_altitude / stepsize,
        ]
        self.dep_info = np.clip(self.dep_info, a_min=1, a_max=self.ping_len - 1)
        self.portside_bottom_dist = self.ping_len - np.round(self.dep_info[0]).astype(
            int
        )
        self.intern_altitude_port = copy.copy(self.portside_bottom_dist)
        self.starboard_bottom_dist = np.round(self.dep_info[1]).astype(int)
        self.intern_altitude_star = copy.copy(self.starboard_bottom_dist)
        # TODO: add to CFG
        search_range_radius = int(np.round(search_range * self.ping_len / 2))
        additional_inset = 3

        # edge detection
        combine_both_sides = True  # this is currently mandatory
        for chunk_idx in range(self.num_chunk):

            # get current chunk of depth info and get snippet that ahs to be analyzed
            cur_depth_port = self.portside_bottom_dist[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ]
            cur_depth_star = self.starboard_bottom_dist[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ]
            cur_chunk = np.zeros((2, self.chunk_size, 2 * search_range_radius))
            out_len = len(
                self.portside_bottom_dist[
                    chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
                ]
            )
            for line_idx in range(self.chunk_size):
                if line_idx < len(cur_depth_port):
                    if line_idx < out_len:
                        tmp_chunk_0 = son_data[
                            0,
                            line_idx + (self.chunk_size * chunk_idx),
                            np.max(
                                (0, cur_depth_port[line_idx] - search_range_radius)
                            ) : cur_depth_port[line_idx]
                            + search_range_radius,
                        ]

                        tmp_chunk_1 = son_data[
                            1,
                            line_idx + (self.chunk_size * chunk_idx),
                            np.max(
                                (0, cur_depth_star[line_idx] - search_range_radius)
                            ) : cur_depth_star[line_idx]
                            + search_range_radius,
                        ]

                        chunk_len = len(tmp_chunk_0)
                        # check whether data needs to be padded because search radius would need samples with index <0 oder >PING_LEN
                        # therefore use zero padding to add "water" or padd ones for "ground"
                        if cur_depth_port[line_idx] - search_range_radius < 0:
                            # add ones for "negative" indices
                            cur_chunk[:, line_idx] = np.ones(
                                (2, 2 * search_range_radius)
                            ) * np.max(tmp_chunk_0)
                            cur_chunk[
                                0, line_idx, (2 * search_range_radius) - chunk_len :
                            ] = tmp_chunk_0
                            cur_chunk[1, line_idx, :chunk_len] = tmp_chunk_1

                        elif (
                            cur_depth_port[line_idx] + search_range_radius
                            > self.ping_len
                        ):
                            cur_chunk[0, line_idx, :chunk_len] = tmp_chunk_0
                            cur_chunk[
                                1, line_idx, (2 * search_range_radius) - chunk_len :
                            ] = tmp_chunk_1
                            # and if depth is low/chunk needs to be zero padded
                        else:
                            # default case
                            cur_chunk[0, line_idx] = tmp_chunk_0
                            cur_chunk[1, line_idx] = tmp_chunk_1

            # work on normalized data
            cur_chunk[0] /= np.max(np.abs(cur_chunk[0]))
            cur_chunk[1] /= np.max(np.abs(cur_chunk[1]))

            # Find best threshold for current chunk, currently median is used because roughly half water/land is expected
            threshold_bin = np.median(cur_chunk[:, :out_len])
            # TODO: delete visualization, just kept for current testing
            # plt.hist(cur_chunk.flatten(), bins=500)
            # plt.title(f"Histogram of current chunk, median={threshold_bin}")
            # plt.show(block=True)

            # clip threshold because it might be 0 for chunks with a lot of zero-padding
            threshold_bin = np.max((threshold_bin, 0.004))

            portside_edges_chunk, starboard_edges_chunk = self.detect_edges(
                cur_chunk[0],
                cur_chunk[1],
                threshold_bin,
                combine_both_sides,
                False,
            )
            # convert most likely edge to bottom distance
            # TODO test against bound/find potential crashes
            self.portside_bottom_dist[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ] = (
                self.edges_to_bottom_dist(
                    portside_edges_chunk,
                    threshold_bin,
                    data_is_port_side=True,
                    dist_at_ends=5,
                )[:out_len]
                - search_range_radius
                + cur_depth_port
                - additional_inset
            )
            self.starboard_bottom_dist[
                chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
            ] = (
                self.edges_to_bottom_dist(
                    starboard_edges_chunk,
                    threshold_bin,
                    data_is_port_side=False,
                    dist_at_ends=5,
                )[:out_len]
                - search_range_radius
                + cur_depth_star
                + additional_inset
            )

        # limit to [0, PING_LEN] if additional inset would lead to values out of bounds
        self.portside_bottom_dist = np.clip(
            self.portside_bottom_dist, a_min=1, a_max=self.ping_len - 1
        )
        self.starboard_bottom_dist = np.clip(
            self.starboard_bottom_dist, a_min=1, a_max=self.ping_len - 1
        )

        if active_single_altitude_offset:
            # find mean offset of intern altitude and detected bottom line
            altitude_offset = np.mean(
                [
                    self.intern_altitude_port - self.portside_bottom_dist,
                    -1 * (self.intern_altitude_star - self.starboard_bottom_dist),
                ]
            )
            altitude_offset = int(np.round(altitude_offset))
            # apply mean offset to resulting bottom lines
            self.starboard_bottom_dist = self.intern_altitude_star + altitude_offset
            self.portside_bottom_dist = self.intern_altitude_port - altitude_offset

        if active_bottom_smoothing:
            self.starboard_bottom_dist = scisig.savgol_filter(
                self.starboard_bottom_dist, 20, 3
            )
            self.portside_bottom_dist = scisig.savgol_filter(
                self.portside_bottom_dist, 20, 3
            )

    # Slant range correction
    def slant_range_correction(
        self,
        active_interpolation=True,
        save_to=None,
        nadir_angle=0,
        use_intern_altitude=False,
        active_mult_slant_range_resampling=False,
        progress_signal=None,
    ):
        """Correct slant range for current data. The current sidescan data is projected
        to the seafloor, assuming that the seafloor is flat and using the bottom line detection data.
        Therefore a new index for eachsample is calculated to determine its position on the ground range.
        This results in a new matrix ``self.slant_corrected_mat`` containing the ground range intensity
        values, which might contain gaps. These gaps are interpolated if active_interpolation is
        set to ``True``, otherwise the matrix contains NANs.

        Parameters
        ----------
        active_interpolation: bool
            If ``active`` gaps in slant range corrected signals are interpolated (highly recommended)
        save_to: str | os.PathLike | None
            If not ``None`` the resulting slant corrected matrix is saved as ``.npz`` to the provided path
        nadir_angle: int
            Angle below the sidescan sonar in degree which is invisible because of nadir (per side).
            Use 0 if it is not known.
        use_intern_depth: bool
            If ``True`` internal depth information is used. Otherwise the depth is estimated from the bottom detection data.
        active_mult_slant_range_resampling: bool
            If ``True`` pings with different slant ranges are resampled to the longest slant range.
            This results in a slant corrected matrix where all samples are equidistant.
        """

        # to make both channels have their wc left
        self.sonar_data_proc[0] = np.fliplr(self.sonar_data_proc[0])

        # check whether depth data is present
        if use_intern_altitude:
            # check all depth values and fill zeros with next non zero entry
            raw_altitude = self.sidescan_file.sensor_primary_altitude
            if np.max(raw_altitude) == 0:
                raise ValueError("No depth information found in intern data.")
            raw_altitude = self.fill_zeros_with_last(raw_altitude)

            # TODO: add smoothing/outlier detection here?
            # read depth info and calc corresponding sample idx
            stepsize = self.sidescan_file.slant_range[0, :] / self.ping_len
            self.dep_info = [
                raw_altitude / stepsize,
                raw_altitude / stepsize,
            ]  # is the same for both sides
            self.dep_info = np.clip(self.dep_info, a_min=1, a_max=self.ping_len - 1)
            self.portside_bottom_dist = self.ping_len - self.dep_info[0]
            self.starboard_bottom_dist = self.dep_info[1]
            print(
                "SidescanPreprocessor - Slant range correction: Using intern depth data."
            )
        elif nadir_angle != 0:
            print(
                f"SidescanPreprocessor - Slant range correction: No intern depth data used -> Using nadir angle {nadir_angle}° to estimate depth."
            )
            portside_dist = (self.ping_len - self.portside_bottom_dist) * np.sin(
                np.deg2rad(90 - nadir_angle)
            )
            starboard_dist = self.starboard_bottom_dist * np.sin(
                np.deg2rad(90 - nadir_angle)
            )
            self.dep_info = [portside_dist, starboard_dist]
        else:
            print(
                "SidescanPreprocessor - Slant range correction: No beam angle known -> Using raw slant values as depth info."
            )
            self.dep_info = [
                self.ping_len - self.portside_bottom_dist,
                self.starboard_bottom_dist,
            ]

        # check whether the slant range has been changed in the process
        # if yes -> decimate pings using resample to higher resolition by zero padding at the end
        portside_bottom_dist_conv = copy.copy(self.portside_bottom_dist)
        starboard_bottom_dist_conv = copy.copy(self.starboard_bottom_dist)
        slant_range = np.flip(self.sidescan_file.slant_range[0, :])
        if active_mult_slant_range_resampling:
            if np.min(slant_range) != np.max(slant_range):
                max_slant_range = np.max(slant_range)

                for ping_idx in range(self.sidescan_file.num_ping):
                    if slant_range[ping_idx] != max_slant_range:
                        num_res_sample = int(
                            np.round(
                                slant_range[ping_idx] / max_slant_range * self.ping_len
                            )
                        )

                        for ch in range(2):
                            pre_max = np.max(self.sonar_data_proc[ch][ping_idx])
                            sig_down = scisig.resample(
                                self.sonar_data_proc[ch][ping_idx], num_res_sample
                            )
                            sig_down = np.clip(sig_down, 0, pre_max)
                            self.sonar_data_proc[ch][
                                ping_idx, :num_res_sample
                            ] = sig_down
                            self.sonar_data_proc[ch][ping_idx, num_res_sample:] = 0
                            self.dep_info[ch][ping_idx] = int(
                                np.round(
                                    self.dep_info[ch][ping_idx]
                                    * slant_range[ping_idx]
                                    / max_slant_range
                                )
                            )
                        portside_bottom_dist_conv[ping_idx] = int(
                            np.round(
                                self.portside_bottom_dist[ping_idx]
                                * slant_range[ping_idx]
                                / max_slant_range
                            )
                        )
                        starboard_bottom_dist_conv[ping_idx] = int(
                            np.round(
                                self.starboard_bottom_dist[ping_idx]
                                * slant_range[ping_idx]
                                / max_slant_range
                            )
                        )

        for ch in range(2):
            slant_cor_mat = np.zeros(
                np.shape(self.sonar_data_proc[0]), dtype=np.float32
            )
            son_data = self.sonar_data_proc[ch]

            num_ping = np.shape(slant_cor_mat)[0]
            for ping_idx in range(num_ping):
                if ping_idx % 1000 == 0 and ping_idx != 0:
                    print(
                        f"\rSlant range correction progress: {ping_idx/num_ping*100:.2f}%"
                    )
                    if progress_signal is not None:
                        progress_signal.emit((1000 / num_ping) * 0.25)

                depth = int(self.dep_info[ch][ping_idx])  # in px
                dd = depth**2
                # vector to store reloacted pixels
                ping_dat = (
                    np.ones((slant_cor_mat.shape[1])).astype(np.float32)
                ) * np.nan

                for dep_idx in range(len(ping_dat)):
                    if dep_idx > depth:
                        if dep_idx**2 <= dd:
                            hor_idx = dep_idx
                        else:
                            hor_idx = int(round(np.sqrt(dep_idx**2 - dd), 0))
                        if hor_idx < len(ping_dat):
                            ping_dat[hor_idx] = son_data[ping_idx, dep_idx]

                # Process of relocating bed pixels will introduce across track gaps
                ## in the array so we will interpolate over gaps to fill them.
                if active_interpolation:
                    last_val = np.where(~np.isnan(ping_dat))[-1]
                    nans, x = np.isnan(ping_dat), lambda z: z.nonzero()[0]
                    if np.isnan(ping_dat).all():
                        ping_dat = np.zeros_like(ping_dat)
                    else:
                        ping_dat[nans] = np.interp(x(nans), x(~nans), ping_dat[~nans])
                    # TODO: probably no way to implement this with all other filtering steps?
                    # if len(last_val) > 0:
                    #     # remove all interpolated values after last known val
                    #     ping_dat[last_val[-1] + 1 :] = np.nan

                # remove remaining nadir
                # TODO: see above
                # if nadir_angle != 0:
                #     depth_on_ground = int(round(np.sqrt((depth + 1) ** 2 - dd), 0))
                #     ping_dat[:depth_on_ground] = np.nan

                if ch == 0:
                    ping_dat = np.flip(ping_dat)

                slant_cor_mat[ping_idx] = ping_dat

            self.sonar_data_proc[ch] = slant_cor_mat

        print("Slant range correction completed.")

        self.slant_corrected_mat = np.hstack(
            (self.sonar_data_proc[0], self.sonar_data_proc[1])
        )

        y_axis_m = self.gen_simple_y_axis()
        # revert flip
        self.sonar_data_proc[0] = np.fliplr(self.sonar_data_proc[0])

        # save intensity table/histogram for EGN
        if save_to is not None:
            save_to = Path(save_to)
            if save_to.suffix != ".npz":
                save_to = save_to.with_suffix(".npz")
            np.savez(
                save_to,
                slant_corr=self.slant_corrected_mat,
                depth_info=self.dep_info,
                y_axis_m=y_axis_m,
            )

    def gen_simple_y_axis(self):
        # quick and dirty y-Axis in m build
        # this might be useful later and is therefore kept for now
        stepsize = 1
        num_step = int(self.sidescan_file.num_ping / stepsize)
        y_axis_m = np.zeros(num_step)
        old_coord = 0
        new_coord = 0
        for ping_idx in range(num_step):
            old_coord = new_coord
            new_coord = (
                self.sidescan_file.longitude[ping_idx * stepsize],
                self.sidescan_file.latitude[ping_idx * stepsize],
            )
            if ping_idx > 0:
                if all(new_coord) and all(old_coord):
                    y_axis_m[ping_idx] = (
                        y_axis_m[ping_idx - 1]
                        + geo_dist.geodesic(old_coord, new_coord).m
                    )
                else:
                    y_axis_m[ping_idx] = y_axis_m[ping_idx - 1]
        # interpolate unknown points
        y_axis_m[y_axis_m == 0.0] = np.nan
        y_axis_m[0] = 0.0
        nans, x = np.isnan(y_axis_m), lambda z: z.nonzero()[0]
        y_axis_m[nans] = np.interp(x(nans), x(~nans), y_axis_m[~nans])

        return y_axis_m

    @staticmethod
    def fill_zeros_with_last(arr):
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        return arr[prev]

    def do_EGN_correction(self, egn_table_path, save_to=None):
        """

        Parameters
        ----------
        egn_table_path: str | os.PathLike
            Path to EGN table (``.npz``) file
        save_to: str | os.PathLike | None
            If not ``None`` the resulting slant corrected matrix is saved as ``.npz`` to the provided path
        """

        egn_info = np.load(egn_table_path)
        egn_table = egn_info["egn_table"]
        egn_hit_cnt = egn_info["egn_hit_cnt"]
        angle_range = egn_info["angle_range"]
        angle_num = egn_info["angle_num"]
        angle_stepsize = egn_info["angle_stepsize"]
        ping_len = egn_info["ping_len"]
        r_size = egn_info["r_size"]
        r_reduc_factor = egn_info["r_reduc_factor"]

        # do EGN
        self.egn_corrected_mat = np.zeros(
            np.shape(self.slant_corrected_mat)
        )  # TODO Rework to sono data proc
        num_ping = np.shape(self.slant_corrected_mat)[0]
        mean_depth = np.array(np.round(np.mean(self.dep_info, 0)), dtype=int)

        dd = mean_depth**2
        EPS = np.finfo(float).eps
        for vector_idx in range(num_ping):
            r = np.sqrt(
                (np.linspace(0, 2 * self.ping_len - 1, 2 * self.ping_len) - ping_len)
                ** 2
                + dd[vector_idx]
            )
            r_idx = np.array(np.round(r / r_reduc_factor), dtype=int)
            alpha = np.sign(
                np.linspace(0, 2 * self.ping_len - 1, 2 * self.ping_len) - ping_len
            ) * np.arccos(mean_depth[vector_idx] / (r + EPS))
            alpha_idx = np.array(
                np.round(alpha / angle_stepsize) + angle_num / 2, dtype=int
            )

            if vector_idx % 1000 == 0:
                print(f"EGN Progress: {vector_idx/num_ping:.2%}")
            for ping_idx in range(2 * ping_len):
                if (
                    0 <= r_idx[ping_idx] < r_size
                    and 0 <= alpha_idx[ping_idx] < angle_num
                ):
                    self.egn_corrected_mat[vector_idx, ping_idx] = (
                        self.slant_corrected_mat[vector_idx, ping_idx]
                        / (egn_table[r_idx[ping_idx], alpha_idx[ping_idx]] + EPS)
                    )

        if save_to is not None:
            np.savez(
                save_to,
                egn_table_path=egn_table_path,
                egn_corrected_mat=self.egn_corrected_mat,
            )
