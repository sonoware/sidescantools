from qtpy.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QProgressBar,
    QWidget,
    QPushButton,
    QMessageBox,
    QFileDialog,
)
import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
from sidescan_file import SidescanFile
from sidescan_preproc import SidescanPreprocessor
from georef_thread import Georeferencer
import numpy as np
import os
import pathlib
import pyqtgraph as pg
import pyqtgraph.exporters as exporters
from pathlib import Path
import os
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


class ImportThread(QtCore.QThread):
    status_signal = QtCore.Signal(str)
    progress_signal = QtCore.Signal(int)
    results_signal = QtCore.Signal(list)
    aborted_signal = QtCore.Signal(str)
    filenames = []

    def __init__(self, filenames: list, parent=None):
        super().__init__(parent)
        self.filenames = filenames

    def run(self):
        self.status_signal.emit("starting import")
        meta_list_html = []
        import_success = True
        err_str = ""
        for idx, filename in enumerate(self.filenames):
            self.progress_signal.emit(idx)
            try:
                sidescan_file = SidescanFile(filename)
            except Exception as err:
                err_str = f"Error while importing {filename}: \n {err}"
                print(f"Error while importing {filename}: \n {err}")
                import_success = False
                break

            meta_info = (
                f"<b>Date          :</b> " + str(sidescan_file.timestamp[0]) + "<br />"
            )
            meta_info += f"<b>Channels        :</b> {sidescan_file.num_ch}<br />"
            meta_info += f"<b>Number of pings :</b> {sidescan_file.num_ping}<br />"
            meta_info += f"<b>Samples per ping:</b> {sidescan_file.ping_len}<br />"
            meta_info += f"<b>Slant ranges    :</b> {np.min(sidescan_file.slant_range)} - {np.max(sidescan_file.slant_range)} m<br />"

            meta_list_html.append({filename: meta_info})
        if import_success:
            self.status_signal.emit("import finished")
            self.results_signal.emit(meta_list_html)
        else:
            self.status_signal.emit("import failed")
            self.aborted_signal.emit(err_str)


class FileImportManager(QWidget):
    results_ready = QtCore.Signal(list)
    aborted_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 50)
        self.pbar.setTextVisible(False)
        self.title_label = QLabel()
        self.title_label.setText("Importing Files ...")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label_font = QtGui.QFont()
        label_font.setBold(True)
        label_font.setPixelSize(20)
        self.title_label.setFont(label_font)
        self.box_layout = QVBoxLayout()
        self.box_layout.addWidget(self.title_label)
        self.box_layout.addWidget(self.pbar)
        self.setLayout(self.box_layout)
        self.setGeometry(300, 300, 550, 50)
        self.setWindowTitle("File Import Checking")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self.setStyleSheet(
            "background-color:black;border-color:darkgrey;border-style:solid;border-width:3px;"
        )
        self.show()

    def start_import(self, filenames: list) -> list:
        """Returns meta info for all files via results_ready signal or an error message via aborted_signal"""

        num_files = len(filenames)
        self.pbar.setRange(0, num_files)
        # starting import in its own thread
        self.import_thread = ImportThread(filenames, self)
        # connecting signals of thread with slots
        self.import_thread.status_signal.connect(lambda status: print(status))
        self.import_thread.progress_signal.connect(
            lambda progress: self.update_pbar(progress)
        )
        self.import_thread.results_signal.connect(
            lambda meta_info: self.send_results(meta_info)
        )
        self.import_thread.aborted_signal.connect(
            lambda msg_str: self.import_aborted(msg_str)
        )
        # start thread and return meta information as a list
        self.import_thread.start()

    def update_pbar(self, value: int):
        self.pbar.setValue(value)

    def send_results(self, results: list):
        self.results_ready.emit(results)
        self.deleteLater()

    def import_aborted(self, msg_str: str):
        self.aborted_signal.emit(msg_str)
        self.deleteLater()


class EGNTableProcessingWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(float)
    error_signal = QtCore.Signal(Exception)


class EGNTableProcessingWorker(QtCore.QRunnable):

    def __init__(
        self,
        filename: str,
        bottom_file: str,
        out_path: str,
        chunk_size: int,
        nadir_angle: int,
        active_intern_depth: bool,
        active_bottom_detection_downsampling: bool,
    ):
        """Init Worker to calculate slant range correction and EGN Table

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
        active_intern_depth: bool
            If ``True`` internal depth information is used. Otherwise the depth is estimated from the bottom detection data.
        active_bottom_detection_downsampling: bool
            If ``True`` the data is downsampled by decimation using the same factor that has been used for the bottom detection.
        """
        super().__init__()
        self.filename = filename
        self.bottom_file = bottom_file
        self.out_path = out_path
        self.chunk_size = chunk_size
        self.nadir_angle = nadir_angle
        self.active_intern_depth = active_intern_depth
        self.active_bottom_detection_downsampling = active_bottom_detection_downsampling
        self.signals = EGNTableProcessingWorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.calc_slant_correction_and_egn()
        except Exception as err:
            self.signals.error_signal.emit(err)
        finally:
            self.signals.finished.emit()

    def calc_slant_correction_and_egn(self):
        print("---")
        print(f"Reading file: {self.filename}")

        sidescan_file = SidescanFile(self.filename)
        bottom_info = np.load(self.bottom_file)

        # TODO: Validate incoming sidescan files and all important information
        for ch in range(sidescan_file.num_ch):
            for idx in range(len(sidescan_file.slant_range[0])):
                if sidescan_file.slant_range[ch, idx] == 0:
                    sidescan_file.slant_range[ch, idx] = sidescan_file.slant_range[
                        ch, -1
                    ]

        # Check if downsampling was applied
        try:
            downsampling_factor = bottom_info["downsampling_factor"]
        except:
            downsampling_factor = 1

        portside_bottom_dist = bottom_info["bottom_info_port"].flatten()[:sidescan_file.num_ping]
        starboard_bottom_dist = bottom_info["bottom_info_star"].flatten()[:sidescan_file.num_ping]
        # flip order for xtf files to contain backwards compability
        filepath = pathlib.Path(self.filename)
        if filepath.suffix.casefold() == ".xtf":
            portside_bottom_dist = np.flip(portside_bottom_dist)
            starboard_bottom_dist = np.flip(starboard_bottom_dist)

        # If a bottom line distance value is 0, just use val from other side (solve that bug in bottom line detection)
        num_btm_line = len(portside_bottom_dist)
        for btm_idx in range(num_btm_line):
            if portside_bottom_dist[btm_idx] == 0:
                portside_bottom_dist[btm_idx] = starboard_bottom_dist[btm_idx]
            elif starboard_bottom_dist[btm_idx] == 0:
                starboard_bottom_dist[btm_idx] = portside_bottom_dist[btm_idx]

        if len(portside_bottom_dist) != len(starboard_bottom_dist):
            print(
                f"Reading bottom info {self.bottom_file}: detected bottom line lengths don't match!"
            )
            return False

        # check that data length and bottom detection length match
        if sidescan_file.num_ping != len(portside_bottom_dist):
            # if lengths don't match, bottom line might be padded to fill the last last chunk
            try:
                bottom_chunk_size = bottom_info["chunk_size"]
            except:
                bottom_chunk_size = self.chunk_size
            expected_full_chunk_size = int(
                np.ceil(sidescan_file.num_ping / bottom_chunk_size) * bottom_chunk_size
            )
            if len(portside_bottom_dist) != expected_full_chunk_size:
                print(
                    f"Sizes of NUM ping ({sidescan_file.num_ping}) and bottom line info ({len(portside_bottom_dist)}) don't match!"
                )
                print(f"Sonar file: {self.filename}")
                print(f"Bottom line detection: {self.bottom_file}")
                return False

        # Check whether data shall be downsampled using the bottom line detection factor
        if self.active_bottom_detection_downsampling:
            preproc = SidescanPreprocessor(
                sidescan_file=sidescan_file,
                chunk_size=self.chunk_size,
                downsampling_factor=downsampling_factor,
            )
        else:
            preproc = SidescanPreprocessor(
                sidescan_file=sidescan_file,
                chunk_size=self.chunk_size,
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
            nadir_angle=self.nadir_angle,
            use_intern_depth=self.active_intern_depth,
            progress_signal=self.signals.progress,
        )
        # self.signals.progress.emit(0.5)

        # compute egn info #TODO: check if this params shall be made adjustable
        angle_range = [-1 * np.pi / 2, np.pi / 2]
        angle_num = 360
        r_reduc_factor = 2
        r_size = int(preproc.ping_len * 1.1 / r_reduc_factor)
        angle_stepsize = (angle_range[1] - angle_range[0]) / angle_num
        egn_mat = np.zeros((r_size, angle_num))
        egn_hit_cnt = np.zeros((r_size, angle_num))

        # either use depth from annotation file or intern depth
        if self.active_intern_depth:
            stepsize = sidescan_file.slant_range[0, :] / preproc.ping_len
            mean_depth = sidescan_file.depth / stepsize  # is the same for both sides
        else:
            mean_depth = np.array(np.round(np.mean(preproc.dep_info, 0)), dtype=int)

        num_data = np.shape(preproc.slant_corrected_mat)[0]

        dd = mean_depth**2
        EPS = np.finfo(float).eps

        for vector_idx in range(num_data):
            if vector_idx % 1000 == 0:
                self.signals.progress.emit(1000 / num_data * 0.5)
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
            self.out_path,
            egn_mat=egn_mat,
            egn_hit_cnt=egn_hit_cnt,
            angle_range=angle_range,
            angle_num=angle_num,
            angle_stepsize=angle_stepsize,
            ping_len=preproc.ping_len,
            r_size=r_size,
            r_reduc_factor=r_reduc_factor,
            nadir_angle=self.nadir_angle,
        )


class EGNTableBuilder(QWidget):
    table_finished = QtCore.Signal()
    aborted_signal = QtCore.Signal(str)
    pbar_val: float
    files_finished: int
    num_files: int

    def __init__(self, egn_table_path: os.PathLike):
        super().__init__()

        self.pbar_val = 0
        self.files_finished = 0
        self.egn_table_path = egn_table_path
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 50)
        self.pbar.setTextVisible(False)
        self.title_label = QLabel()
        self.title_label.setText("Generating EGN Table")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label_font = QtGui.QFont()
        label_font.setBold(True)
        label_font.setPixelSize(20)
        self.title_label.setFont(label_font)
        self.box_layout = QVBoxLayout()
        self.box_layout.addWidget(self.title_label)
        self.box_layout.addWidget(self.pbar)
        self.setLayout(self.box_layout)
        self.setGeometry(300, 300, 550, 50)
        self.setWindowTitle("EGN Table")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self.setStyleSheet(
            "background-color:black;border-color:darkgrey;border-style:solid;border-width:3px;"
        )

        self.threadpool = QtCore.QThreadPool()
        self.show()

    def build_egn_table(
        self,
        files: list,
        out_path: os.PathLike,
        nadir_angle: int,
        active_intern_depth: bool,
        chunk_size: int,
        active_downsampling: bool,
    ):

        # Build list of needed and to be written file names
        res_sonar_path_list = []
        res_bottom_path_list = []
        self.res_egn_path_list = []
        out_path = pathlib.Path(out_path)
        for sonar_file_path in files:
            bottom_path = out_path / (sonar_file_path.stem + "_bottom_info.npz")
            if bottom_path.exists():
                res_sonar_path_list.append(sonar_file_path)
                res_bottom_path_list.append(bottom_path)
                self.res_egn_path_list.append(
                    out_path / (sonar_file_path.stem + "_egn_info.npz")
                )

        self.num_files = len(files)
        self.pbar.setMaximum = 100
        for file_idx in range(self.num_files):
            new_worker = EGNTableProcessingWorker(
                res_sonar_path_list[file_idx],
                res_bottom_path_list[file_idx],
                self.res_egn_path_list[file_idx],
                chunk_size,
                nadir_angle,
                active_intern_depth,
                active_downsampling,
            )
            new_worker.signals.error_signal.connect(lambda err: self.build_aborted(err))
            new_worker.signals.progress.connect(
                lambda progress: self.update_pbar(progress)
            )
            new_worker.signals.finished.connect(self.files_finished_counter)
            self.threadpool.start(new_worker)

    def build_final_table(self):
        do_init = True
        for egn_file in self.res_egn_path_list:
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
                nadir_angle = egn_info["nadir_angle"]

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
        egn_table = np.divide(
            full_mat, full_hit_cnt, out=np.zeros_like(full_mat), where=full_hit_cnt != 0
        )
        egn_table[np.where(full_hit_cnt == 0)] = np.nan
        print("Saving " + str(self.egn_table_path))
        np.savez(
            self.egn_table_path,
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

    def update_pbar(self, progress: float):
        self.pbar_val += progress
        disp_var = self.pbar_val / self.num_files
        self.pbar.setValue(int(100 * disp_var))

    def files_finished_counter(self):
        self.files_finished += 1
        if self.files_finished == self.num_files:
            self.build_final_table()
            self.send_table_finished()

    def send_table_finished(self):
        self.table_finished.emit()
        self.deleteLater()

    def build_aborted(self, err: Exception):
        msg_str = str(err)
        self.aborted_signal.emit(msg_str)
        self.deleteLater()


class PreProcWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(list)
    progress = QtCore.Signal(float)
    error_signal = QtCore.Signal(Exception)


class PreProcWorker(QtCore.QRunnable):

    def __init__(
        self,
        filepath: str,
        bottom_file: str,
        work_dir: str,
        egn_table_path: str,
        chunk_size: int,
        nadir_angle: int,
        active_export_slant_corr_mat: bool,
        active_export_gain_corr_mat: bool,
        active_downsampling: bool,
        load_slant_data: bool,
        load_gain_data: bool,
        active_pie_slice_filter: bool,
        active_gain_norm: bool,
        active_egn: bool,
        active_bac: bool,
        active_sharpening_filter: bool,
    ):
        super().__init__()
        self.filepath = pathlib.Path(filepath)
        self.bottom_file = bottom_file
        self.work_dir = pathlib.Path(work_dir)
        self.egn_table_path = egn_table_path
        self.chunk_size = chunk_size
        self.nadir_angle = nadir_angle
        self.active_export_slant_corr_mat = active_export_slant_corr_mat
        self.active_export_gain_corr_mat = active_export_gain_corr_mat
        self.load_slant_data = load_slant_data
        self.load_gain_data = load_gain_data
        self.active_downsampling = active_downsampling
        self.signals = PreProcWorkerSignals()
        self.active_pie_slice_filter = active_pie_slice_filter
        self.active_gain_norm = active_gain_norm
        self.active_egn = active_egn
        self.active_bac = active_bac
        self.active_sharpening_filter = active_sharpening_filter

    @QtCore.Slot()
    def run(self):
        try:
            self.do_slant_corr_and_processing()
        except Exception as err:
            self.signals.error_signal.emit(err)

    def do_slant_corr_and_processing(self):
        sidescan_file = SidescanFile(filepath=self.filepath)
        bottom_info = np.load(self.bottom_file)
        # Check if downsampling was applied
        try:
            downsampling_factor = bottom_info["downsampling_factor"]
        except:
            downsampling_factor = 1
            
        portside_bottom_dist = bottom_info["bottom_info_port"].flatten()[:sidescan_file.num_ping]
        starboard_bottom_dist = bottom_info["bottom_info_star"].flatten()[:sidescan_file.num_ping]
        
        if not self.active_downsampling:
            if downsampling_factor != 1:
                # rescale bottom info
                portside_bottom_dist = portside_bottom_dist * downsampling_factor
                starboard_bottom_dist = starboard_bottom_dist * downsampling_factor
                downsampling_factor = 1

        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=self.chunk_size,
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
            port_chunk = portside_bottom_dist[chunk_idx * preproc.chunk_size:(chunk_idx+1) * preproc.chunk_size]
            preproc.napari_portside_bottom[chunk_idx, :len(port_chunk)] = port_chunk
            star_chunk = starboard_bottom_dist[chunk_idx * preproc.chunk_size:(chunk_idx+1) * preproc.chunk_size]
            preproc.napari_starboard_bottom[chunk_idx, :len(star_chunk)] = star_chunk

        # slant range correction and EGN data
        if self.active_export_slant_corr_mat:
            slant_data_path = self.work_dir / (
                self.filepath.stem + "_slant_corrected.npz"
            )
        else:
            slant_data_path = None

        self.signals.progress.emit(0.1)
        if self.load_slant_data:
            slant_data = np.load(slant_data_path)
            preproc.slant_corrected_mat = slant_data["slant_corr"]

        else:
            if self.active_pie_slice_filter:
                print(f"Pie slice filtering {self.filepath}")
                preproc.apply_pie_slice_filter()
            preproc.slant_range_correction(
                active_interpolation=True,
                nadir_angle=self.nadir_angle,
                save_to=slant_data_path,
                active_mult_slant_range_resampling=True,
            )
        self.signals.progress.emit(0.4)
        gain_corrected_path = self.work_dir / (
            self.filepath.stem + "_egn_corrected.npz"
        )
        if self.active_gain_norm:
            if self.active_egn:
                if not pathlib.Path(self.egn_table_path).exists():
                    raise FileNotFoundError(
                        f"The specified EGN table {self.egn_table_path} doesn't exist."
                    )

                if self.load_gain_data:
                    egn_data = np.load(gain_corrected_path)
                    preproc.egn_corrected_mat = egn_data["egn_corrected_mat"]
                else:
                    preproc.do_EGN_correction(
                        self.egn_table_path,
                        save_to=None,
                    )
                self.signals.progress.emit(0.4)
            elif self.active_bac:
                if self.load_gain_data:
                    gain_corrected_data = np.load(gain_corrected_path)
                    preproc.egn_corrected_mat = gain_corrected_data["egn_corrected_mat"]
                    self.signals.progress.emit(0.4)
                else:
                    preproc.apply_beam_pattern_correction()
                    self.signals.progress.emit(0.3)
                    preproc.apply_energy_normalization()
                    self.signals.progress.emit(0.1)
                    preproc.egn_corrected_mat = np.hstack(
                        (
                            np.fliplr(preproc.sonar_data_proc[0]),
                            preproc.sonar_data_proc[1],
                        )
                    )
        else:
            preproc.egn_corrected_mat = preproc.slant_corrected_mat
        if not self.load_gain_data:
            if self.active_sharpening_filter:
                preproc.apply_sharpening_filter()
        if self.active_export_gain_corr_mat:
            np.savez(
                gain_corrected_path,
                egn_table_path=self.egn_table_path,
                egn_corrected_mat=preproc.egn_corrected_mat,
            )
        self.signals.progress.emit(0.1)
        self.signals.finished.emit([sidescan_file, preproc])


class PreProcManager(QWidget):
    processing_finished = QtCore.Signal(list)
    new_res_present = QtCore.Signal(list)
    aborted_signal = QtCore.Signal(str)
    pbar_val: float
    files_finished: int
    num_files: int

    def __init__(self):
        super().__init__()

        self.pbar_val = 0
        self.files_finished = 0
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 50)
        self.pbar.setTextVisible(False)
        self.title_label = QLabel()
        self.title_label.setText("Processing Data")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label_font = QtGui.QFont()
        label_font.setBold(True)
        label_font.setPixelSize(20)
        self.title_label.setFont(label_font)
        self.box_layout = QVBoxLayout()
        self.box_layout.addWidget(self.title_label)
        self.box_layout.addWidget(self.pbar)
        self.setLayout(self.box_layout)
        self.setGeometry(300, 300, 550, 50)
        self.setWindowTitle("PreProcManager")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self.setStyleSheet(
            "background-color:black;border-color:darkgrey;border-style:solid;border-width:3px;"
        )

        self.threadpool = QtCore.QThreadPool()
        self.show()

    def proc_files(
        self,
        files: list,
        work_dir: os.PathLike,
        egn_table_path: str,
        chunk_size: int,
        nadir_angle: int,
        active_export_slant_corr_mat: bool,
        active_export_gain_corr_mat: bool,
        active_downsampling: bool,
        load_slant_data: bool,
        load_gain_data: bool,
        active_pie_slice_filter: bool,
        active_gain_norm: bool,
        active_egn: bool,
        active_bac: bool,
        active_sharpening_filter: bool,
    ):

        # change title to reflect actual processing
        if load_gain_data and load_slant_data:
            self.title_label.setText("Loading Data")

        # Build list of needed and to be written file names
        res_sonar_path_list = []
        res_bottom_path_list = []
        for sonar_file_path in files:
            sonar_file_path = pathlib.Path(sonar_file_path)
            bottom_path = work_dir / (sonar_file_path.stem + "_bottom_info.npz")
            if bottom_path.exists():
                res_sonar_path_list.append(sonar_file_path)
                res_bottom_path_list.append(bottom_path)

        self.num_files = len(res_sonar_path_list)
        if self.num_files > 0:
            self.pbar.setMaximum = 100
            for file_idx in range(self.num_files):
                new_worker = PreProcWorker(
                    filepath=res_sonar_path_list[file_idx],
                    bottom_file=res_bottom_path_list[file_idx],
                    work_dir=work_dir,
                    egn_table_path=egn_table_path,
                    chunk_size=chunk_size,
                    nadir_angle=nadir_angle,
                    active_export_slant_corr_mat=active_export_slant_corr_mat,
                    active_export_gain_corr_mat=active_export_gain_corr_mat,
                    active_downsampling=active_downsampling,
                    load_slant_data=load_slant_data,
                    load_gain_data=load_gain_data,
                    active_pie_slice_filter=active_pie_slice_filter,
                    active_gain_norm=active_gain_norm,
                    active_egn=active_egn,
                    active_bac=active_bac,
                    active_sharpening_filter=active_sharpening_filter,
                )
                new_worker.signals.error_signal.connect(lambda err: self.build_aborted(err))
                new_worker.signals.progress.connect(
                    lambda progress: self.update_pbar(progress)
                )
                new_worker.signals.finished.connect(
                    lambda res_list: self.files_finished_counter(res_list)
                )
                self.threadpool.start(new_worker)
        else:
            self.deleteLater()

    def update_pbar(self, progress: float):
        self.pbar_val += progress
        disp_var = self.pbar_val / self.num_files
        self.pbar.setValue(int(100 * disp_var))

    def files_finished_counter(self, res_list):
        self.files_finished += 1
        self.new_res_present.emit(res_list)
        if self.files_finished == self.num_files:
            self.processing_finished.emit(res_list)
            self.deleteLater()

    def build_aborted(self, err: Exception):
        msg_str = str(err)
        self.aborted_signal.emit(msg_str)
        self.deleteLater()

class NavPlotterSignals(QtCore.QObject):
    results_signal = QtCore.Signal(tuple)
    save_signal_head = QtCore.Signal(str,str)
    save_signal_nav = QtCore.Signal(str,str)    #filename, file_filter

class NavPlotter(QtCore.QThread):
    def __init__(self, filepath: pathlib.Path):
        super().__init__()
        self.filepath = filepath
        self.signals = NavPlotterSignals()
        self.thread = QtCore.QThread()


    #@QtCore.Slot()
    def run(self):
        print("NavPlotter started")
        get_nav = Georeferencer(filepath=self.filepath)
        get_nav.prep_data()
        print("Getting navigation data...")
        lola_data = get_nav.LOLA_plt
        head_data = get_nav.HEAD_plt
        lola_data_ori = get_nav.LOLA_plt_ori
        head_data_ori = get_nav.HEAD_plt_ori
        head_data = np.column_stack((head_data[:,0], head_data[:,1]*100))
        head_data_ori = np.column_stack((head_data_ori[:,0], head_data_ori[:,1]*100))
        self.signals.results_signal.emit((lola_data, lola_data_ori, head_data, head_data_ori))

class GeoreferencerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(float)
    error_signal = QtCore.Signal(Exception)


class GeoreferencerWorker(QtCore.QThread):
    finished = QtCore.Signal(str)
    progress = QtCore.Signal(float)
    error_signal = QtCore.Signal(Exception)
    filepath: Path
    sidescan_file: SidescanFile
    channel: int
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
    resolution_options: dict = {
        "Same": "same",
        "Highest": "highest", 
        "Lowest": "lowest", 
        "Average": "average", 
        "Common": "common"
        }
    warp_options: dict = {
        "Polynomial 1 (recommended)": "SRC_METHOD=GCP_POLYNOMIAL, ORDER=1", 
        "Homography (experimental)": "SRC_METHOD=GCP_HOMOGRAPHY"
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
        "Weighted Sum": "sum"
    }
    LOLA_plt: np.ndarray
    HEAD_plt: np.ndarray
    LOLA_plt_ori: np.ndarray
    HEAD_plt_ori: np.ndarray

    def __init__(
        self,
        filepath: str | os.PathLike,
        channel: int,
        active_utm: bool = True,
        active_poly: bool = True,
        proc_data = None,
        output_folder: str | os.PathLike = "./georef_out",
        vertical_beam_angle: int = 60,
        warp_algorithm: str = "SRC_METHOD=GCP_POLYNOMIAL, ORDER=1",
        resolution_mode: str = "average",
        resampling_method: str = "near"

    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.sidescan_file = SidescanFile(self.filepath)        
        self.channel = channel
        self.active_utm = active_utm
        self.active_poly = active_poly
        self.output_folder = pathlib.Path(output_folder)
        self.vertical_beam_angle = vertical_beam_angle
        self.resolution_mode = resolution_mode
        self.warp_algorithm = warp_algorithm
        self.resampling_method = resampling_method
        self.active_proc_data = False
        self.GCP_SPLIT = []
        self.POINTS_SPLIT = []
        self.LOLA_plt = np.empty_like(proc_data)
        self.HEAD_plt = np.empty_like(proc_data)
        self.LOLA_plt_ori = np.empty_like(proc_data)
        self.HEAD_plt_ori = np.empty_like(proc_data)
        if proc_data is not None:
            self.proc_data = proc_data
            self.active_proc_data = True
        self.setup_output_folder()
        self.signals = GeoreferencerSignals()

    @QtCore.Slot()
    def run(self):
        file_name = self.filepath.stem
        tif_path = self.output_folder / f"{file_name}_ch{self.channel}.tif"
        mosaic_tif_path = self.output_folder / f"{file_name}_stack.tif"

        self.prep_data()
        ch_stack = self.channel_stack()

        try:
            print(f"Processing chunks in channel {self.channel} with warp method {self.warp_algorithm}...")

            self.georeference(ch_stack=ch_stack, otiff=tif_path)

            print(f"Mosaicking channel {self.channel} with resolution mode {self.resolution_mode}...")
            self.mosaic(mosaic_tif_path)

        except Exception as e:
            self.signals.error_signal.emit(e)
        finally:
            self.signals.finished.emit()


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
        if self.active_proc_data:
            swath_width = len(self.proc_data[0])
        else:
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
        for idx, head in enumerate(HEAD_ori):
            if head < 0 or head > 3.6:
                HEAD_ori[idx] = HEAD_ori[idx-1]

        GROUND_RANGE = GROUND_RANGE[MASK]
        SLANT_RANGE = SLANT_RANGE[MASK]
        PING = PING[MASK]

        HEAD = savgol_filter(HEAD_ori, 120, 1)
        x = range(len(HEAD))
        self.HEAD_plt = np.column_stack((x, HEAD))
        self.HEAD_plt_ori = np.column_stack((x, HEAD_ori))

        LAT = savgol_filter(LAT_ori, 120, 1)
        LON = savgol_filter(LON_ori, 120, 1)
        self.LOLA_plt = np.column_stack((LON, LAT))
        self.LOLA_plt_ori = np.column_stack((LON_ori, LAT_ori))

        UTM = np.full_like(LAT_ori, np.nan)
        UTM = UTM.tolist()
        print(f"UTM: {type(UTM), len(UTM)}")
        for idx, (la, lo) in enumerate(zip(LAT, LON)):
            try:
                UTM[idx] = utm.from_latlon(la, lo)
            except:
                ValueError("Values or lon and/or lat must not be 0")
        print(f"UTM: {type(UTM), len(UTM)}")

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

        # Extract metadata for each ping in sonar channel
        PING = self.sidescan_file.packet_no
        swath_len = len(PING)
        swath_width = len(ch_stack[0])
        print(f"swath_len: {swath_len}, swath_width: {swath_width}")
        print(f"ch_stack.shape[0], ch_stack.shape[1]: {ch_stack.shape[0], ch_stack.shape[1]}")

        # Transpose (always!) so that the largest axis is horizontal
        ch_stack = ch_stack.T

        ch_stack = np.array(ch_stack, dtype=float)

        # Hack for alter transparency
        ch_stack /= np.max(np.abs(ch_stack)) / 254
        ch_stack = np.clip(ch_stack, 1, 255)

        # Flip array ---> Note: different for .jsf and .xtf!
        print(f"ch_stack shape after transposing: {np.shape(ch_stack)}")
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
            #print(result.stdout)
            # print(f"Command executed successfully: {' '.join(command)}")
            
    def georeference(self, ch_stack, otiff):
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

                im_path = otiff.with_stem(
                    f"{otiff.stem}_{chunk_num}_tmp"
                ).with_suffix(".png")
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
                        str(warp_path)
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
                        str(warp_path)
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
                if name.endswith("_georef_chunk_tmp.tif") and str(self.channel) in name and not name.startswith("._"):
                    TIF.append(os.path.join(root, name))
                    np.savetxt(txt_path, TIF, fmt="%s")

        # delete merged file if it already exists
        if mosaic_tiff.exists():
            mosaic_tiff.unlink()

    
    # gdal 3.11 syntax
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


class GeoreferencerManager(QWidget):

    results_ready = QtCore.Signal(list)
    aborted = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 50)
        self.pbar.setTextVisible(False)
        self.title_label = QLabel()
        self.title_label.setText("Georeferencing file...")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label_font = QtGui.QFont()
        label_font.setBold(True)
        label_font.setPixelSize(20)
        self.title_label.setFont(label_font)
        self.box_layout = QVBoxLayout()
        self.box_layout.addWidget(self.title_label)
        self.box_layout.addWidget(self.pbar)
        self.setLayout(self.box_layout)
        self.setGeometry(300, 300, 550, 50)
        self.setWindowTitle("File Georeferencing")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self.setStyleSheet(
            "background-color:black;border-color:darkgrey;border-style:solid;border-width:3px;"
        )
        self.show()

    def start_georef(self, 
                     filepath: str | os.PathLike, 
                     channel: int, 
                     active_utm: bool,
                     active_poly: bool,
                     output_folder: os.PathLike,
                     proc_data: np.array,
                     vertical_beam_angle: int,
                     resolution_mode: str,
                     warp_algorithm: str,
                     resampling_method: str                  
                     ):
        output_folder = pathlib.Path(output_folder)

        self.worker_ch0 = GeoreferencerWorker(            
                filepath,
                0,
                active_utm,
                active_poly,
                output_folder,
                proc_data,
                vertical_beam_angle,
                resolution_mode,
                warp_algorithm,
                resampling_method,
            )
        self.worker_ch0.start()

        self.worker_ch1 = GeoreferencerWorker(            
                filepath,
                1,
                active_utm,
                active_poly,
                output_folder,
                proc_data,
                vertical_beam_angle,
                resolution_mode,
                warp_algorithm,
                resampling_method,
            )
        self.worker_ch1.start()