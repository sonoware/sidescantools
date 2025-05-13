import pathlib
from sidescan_preproc import SidescanPreprocessor
import numpy as np
from sidescan_file import SidescanFile
import multiprocessing
from itertools import repeat


def calc_slant_correction_and_egn(
    sonar_file_path: str,
    bottom_file: str,
    out_path: str,
    chunk_size: int,
    nadir_angle: int,
    use_intern_depth: bool,
    use_bottom_detection_downsampling=False,
):
    """ Calculate slant range correction and EGN Table

    Parameters
    ----------
    sonar_file_path: str
        Path to sidescan file
    bottom_file: str
        Path to ``.npz`` file containing the bottom detection information
    out_path: str
        Path to output directory
    chunk_size: int
        Number of pings per single chunk
    nadir_angle: int
        Angle below the sidescan sonar in degree which is invisible because of nadir (per side). Use 0 if it is not known.
    use_intern_depth: bool
        If ``True`` internal depth information is used. Otherwise the depth is estimated from the bottom detection data.
    use_bottom_detection_downsampling: bool
        If ``True`` the data is downsampled by decimation using the same factor that has been used for the bottom detection.

    Returns
    -------
    Returns ``True`` on success.
    """

    print("---")
    print(f"Reading file: {sonar_file_path}")

    sidescan_file = SidescanFile(sonar_file_path)
    bottom_info = np.load(bottom_file)

    # TODO: Validate incoming sidescan files and all important information
    for ch in range(sidescan_file.num_ch):
        for idx in range(len(sidescan_file.slant_range[0])):
            if sidescan_file.slant_range[ch, idx] == 0:
                sidescan_file.slant_range[ch, idx] = sidescan_file.slant_range[ch, -1]

    # Check if downsampling was applied
    try:
        downsampling_factor = bottom_info["downsampling_factor"]
    except:
        downsampling_factor = 1

    portside_bottom_dist = bottom_info["bottom_info_port"].flatten()
    starboard_bottom_dist = bottom_info["bottom_info_star"].flatten()

    # If a bottom line distance value is 0, just use val from other side (solve that bug in bottom line detection)
    num_btm_line = len(portside_bottom_dist)
    for btm_idx in range(num_btm_line):
        if portside_bottom_dist[btm_idx] == 0:
            portside_bottom_dist[btm_idx] = starboard_bottom_dist[btm_idx]
        elif starboard_bottom_dist[btm_idx] == 0:
            starboard_bottom_dist[btm_idx] = portside_bottom_dist[btm_idx]

    if len(portside_bottom_dist) != len(starboard_bottom_dist):
        print(
            f"Reading bottom info {bottom_file}: detected bottom line lengths don't match!"
        )
        return False

    # check that data length and bottom detection length match
    if sidescan_file.num_ping != len(portside_bottom_dist):
        # if lengths don't match, bottom line might be padded to fill the last last chunk
        try:
            bottom_chunk_size = bottom_info["chunk_size"]
        except:
            bottom_chunk_size = chunk_size
        expected_full_chunk_size = int(
            np.ceil(sidescan_file.num_ping / bottom_chunk_size) * bottom_chunk_size
        )
        if len(portside_bottom_dist) != expected_full_chunk_size:
            print(
                f"Sizes of NUM ping ({sidescan_file.num_ping}) and bottom line info ({len(portside_bottom_dist)}) don't match!"
            )
            print(f"Sonar file: {sonar_file_path}")
            print(f"Bottom line detection: {bottom_file}")
            return False

    # Check whether data shall be downsampled using the bottom line detection factor
    if use_bottom_detection_downsampling:
        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=chunk_size,
            downsampling_factor=downsampling_factor,
        )
    else:
        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=chunk_size,
            downsampling_factor=1,
        )
        # rescale bottom info
        portside_bottom_dist = portside_bottom_dist * downsampling_factor
        starboard_bottom_dist = starboard_bottom_dist * downsampling_factor

    preproc.portside_bottom_dist = portside_bottom_dist
    preproc.starboard_bottom_dist = starboard_bottom_dist

    # slant range correction
    preproc.slant_range_correction(
        active_interpolation=True,
        nadir_angle=nadir_angle,
        use_intern_depth=use_intern_depth,
    )

    # compute egn info #TODO: check if this params shall be made adjustable
    angle_range = [-1 * np.pi / 2, np.pi / 2]
    angle_num = 360
    r_reduc_factor = 2
    r_size = int(preproc.ping_len * 1.1 / r_reduc_factor)
    angle_stepsize = (angle_range[1] - angle_range[0]) / angle_num
    egn_mat = np.zeros((r_size, angle_num))
    egn_hit_cnt = np.zeros((r_size, angle_num))

    # either use depth from annotation file or intern depth
    if use_intern_depth:
        stepsize = sidescan_file.slant_range[0, :] / preproc.ping_len
        mean_depth = sidescan_file.depth / stepsize  # is the same for both sides
    else:
        mean_depth = np.array(np.round(np.mean(preproc.dep_info, 0)), dtype=int)

    num_data = np.shape(preproc.slant_corrected_mat)[0]

    dd = mean_depth**2
    EPS = np.finfo(float).eps

    for vector_idx in range(num_data):
        if vector_idx % 1000 == 0:
            print(f"EGN Progress: {vector_idx/num_data:.2%}")
        r = np.sqrt(
            (
                np.linspace(0, 2 * preproc.ping_len - 1, 2 * preproc.ping_len)
                - preproc.ping_len
            )
            ** 2
            + dd[vector_idx]
        )
        r_idx = np.array(np.round(r / r_reduc_factor), dtype=int)
        alpha = np.sign(
            np.linspace(0, 2 * preproc.ping_len - 1, 2 * preproc.ping_len)
            - preproc.ping_len
        ) * np.arccos(mean_depth[vector_idx] / (r + EPS))
        alpha_idx = np.array(
            np.round(alpha / angle_stepsize) + angle_num / 2, dtype=int
        )
        for ping_idx in range(2 * preproc.ping_len):
            # TODO: check if neglection of 0 makes sense here
            if (
                0 <= r_idx[ping_idx] < r_size
                and 0 <= alpha_idx[ping_idx] < angle_num
                and preproc.slant_corrected_mat[vector_idx, ping_idx] != 0
            ):
                egn_mat[
                    r_idx[ping_idx], alpha_idx[ping_idx]
                ] += preproc.slant_corrected_mat[vector_idx, ping_idx]
                egn_hit_cnt[r_idx[ping_idx], alpha_idx[ping_idx]] += 1

            # else:
            #     print(f"r_idx: {r_idx} - alpha_idx: {alpha_idx}")
    np.savez(
        out_path / (sonar_file_path.stem + "_egn_info.npz"),
        egn_mat=egn_mat,
        egn_hit_cnt=egn_hit_cnt,
        angle_range=angle_range,
        angle_num=angle_num,
        angle_stepsize=angle_stepsize,
        ping_len=preproc.ping_len,
        r_size=r_size,
        r_reduc_factor=r_reduc_factor,
    )

    return True


def generate_slant_and_egn_files(
    sonar_files,
    out_path,
    nadir_angle,
    use_intern_depth,
    chunk_size,
    generate_final_egn_table,
    use_bottom_detection_downsampling,
    egn_table_name,
    active_multiprocessing,
    pool,
):
    """ Helper function to enable slant range correction and EGN table generation using multiprocessing"""
    res_sonar_path_list = []
    res_bottom_path_list = []
    res_egn_path_list = []
    out_path = pathlib.Path(out_path)

    for sonar_file_path in sonar_files:
        bottom_path = out_path / (sonar_file_path.stem + "_bottom_info.npz")
        if bottom_path.exists():
            res_sonar_path_list.append(sonar_file_path)
            res_bottom_path_list.append(bottom_path)

            if generate_final_egn_table:
                res_egn_path_list.append(
                    out_path / (sonar_file_path.stem + "_egn_info.npz")
                )

    # parallel execution of slant range correction and egn calculation
    active_generate_slant_and_egn_files = True # TODO: check whether there is a case when this is undesired
    if active_generate_slant_and_egn_files:
        if active_multiprocessing:

            res = pool.starmap(
                calc_slant_correction_and_egn,
                zip(
                    res_sonar_path_list,
                    res_bottom_path_list,
                    repeat(out_path),
                    repeat(chunk_size),
                    repeat(nadir_angle),
                    repeat(use_intern_depth),
                    repeat(use_bottom_detection_downsampling),
                ),
            )
            print(res)
        else:
            for sonar_file_path, bottom_file in zip(
                res_sonar_path_list, res_bottom_path_list
            ):
                calc_slant_correction_and_egn(
                    sonar_file_path,
                    bottom_file,
                    out_path,
                    chunk_size,
                    nadir_angle,
                    use_intern_depth,
                    use_bottom_detection_downsampling,
                )

    if generate_final_egn_table:
        do_init = True
        for egn_file in res_egn_path_list:
            egn_info = np.load(egn_file)

            if do_init:
                full_mat = egn_info["egn_mat"]
                full_hit_cnt = egn_info["egn_hit_cnt"]
                angle_range_init = egn_info["angle_range"]
                angle_num_init = egn_info["angle_num"]
                angle_stepsize_init = egn_info["angle_stepsize"]
                ping_len_init = egn_info["ping_len"]
                r_size_init = egn_info["r_size"]
                r_reduc_factor_init = egn_info["r_reduc_factor"]

                do_init = False

            else:
                if (
                    (angle_range_init == egn_info["angle_range"]).all()
                    and angle_num_init == egn_info["angle_num"]
                    and angle_stepsize_init == egn_info["angle_stepsize"]
                    and ping_len_init == egn_info["ping_len"]
                    and r_size_init == egn_info["r_size"]
                    and r_reduc_factor_init == egn_info["r_reduc_factor"]
                ):

                    full_mat += egn_info["egn_mat"]
                    full_hit_cnt += egn_info["egn_hit_cnt"]

                else:
                    print(f"EGN Parameter mismatch! Skipping file: {egn_file}")

        # build final egn table
        egn_table = full_mat / full_hit_cnt

        print("Saving " + str(out_path) + "/" + egn_table_name)
        np.savez(
            out_path / egn_table_name,
            egn_table=egn_table,
            egn_hit_cnt=full_hit_cnt,
            angle_range=angle_range_init,
            angle_num=angle_num_init,
            angle_stepsize=angle_stepsize_init,
            ping_len=ping_len_init,
            r_size=r_size_init,
            r_reduc_factor=r_reduc_factor_init,
            nadir_angle=nadir_angle,
        )


if __name__ == "__main__":
    filepath = ["add path to directry here"]
    filetype = "xtf"
    for path in filepath:
        sonar_files = list(pathlib.Path(path).rglob(f"*.{filetype}"))

    # nadir angle in degree, is needed if sonar data contains no/unwanted depth information
    nadir_angle = 0
    use_intern_depth = False  # use intern depth data for slant correction if present
    chunk_size = 1000

    generate_final_egn_table = True
    use_bottom_detection_downsampling = False
    out_path = ".\sidescan_out\\"
    egn_table_name = "EGN_table.npz"

    # multiprocessing
    active_multiprocessing = True
    num_worker = 4
    pool = multiprocessing.Pool(num_worker)
    generate_slant_and_egn_files(
        sonar_files,
        out_path,
        nadir_angle,
        use_intern_depth,
        chunk_size,
        generate_final_egn_table,
        use_bottom_detection_downsampling,
        egn_table_name,
        active_multiprocessing,
        pool,
    )
