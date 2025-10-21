from qtpy.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QProgressBar,
    QWidget,
    QMessageBox,
)
import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
from sidescan_file import SidescanFile
from sidescan_preproc import SidescanPreprocessor
from georef_thread import Georeferencer
import numpy as np
import os
import pathlib
from egn_table_build import generate_egn_info, generate_egn_table_from_infos


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
        egn_table_parameters: list,
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
        self.egn_table_parameters = egn_table_parameters

    @QtCore.Slot()
    def run(self):
        try:
            self.calc_slant_correction_and_egn()
        except Exception as err:
            self.signals.error_signal.emit(err)
        finally:
            self.signals.finished.emit()

    def calc_slant_correction_and_egn(self):
        generate_egn_info(
            filename=self.filename,
            bottom_file=self.bottom_file,
            out_path=self.out_path,
            chunk_size=self.chunk_size,
            nadir_angle=self.nadir_angle,
            active_intern_depth=self.active_intern_depth,
            active_bottom_detection_downsampling=self.active_bottom_detection_downsampling,
            progress_signal=self.signals.progress,
            egn_table_parameters=self.egn_table_parameters,
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
        egn_table_parameters: list,
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
                egn_table_parameters=egn_table_parameters,
            )
            new_worker.signals.error_signal.connect(lambda err: self.build_aborted(err))
            new_worker.signals.progress.connect(
                lambda progress: self.update_pbar(progress)
            )
            new_worker.signals.finished.connect(self.files_finished_counter)
            self.threadpool.start(new_worker)

    def build_final_table(self):
        generate_egn_table_from_infos(self.res_egn_path_list, self.egn_table_path)

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
        num_angle_bac: int,
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
        self.num_angle_bac = num_angle_bac

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

        portside_bottom_dist = bottom_info["bottom_info_port"].flatten()[
            : sidescan_file.num_ping
        ]
        starboard_bottom_dist = bottom_info["bottom_info_star"].flatten()[
            : sidescan_file.num_ping
        ]

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
            port_chunk = portside_bottom_dist[
                chunk_idx * preproc.chunk_size : (chunk_idx + 1) * preproc.chunk_size
            ]
            preproc.napari_portside_bottom[chunk_idx, : len(port_chunk)] = port_chunk
            star_chunk = starboard_bottom_dist[
                chunk_idx * preproc.chunk_size : (chunk_idx + 1) * preproc.chunk_size
            ]
            preproc.napari_starboard_bottom[chunk_idx, : len(star_chunk)] = star_chunk

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
                active_mult_slant_range_resampling=False,
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
                    preproc.apply_beam_pattern_correction(angle_num=self.num_angle_bac)
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
        num_angle_bac: int,
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
                    num_angle_bac=num_angle_bac,
                )
                new_worker.signals.error_signal.connect(
                    lambda err: self.build_aborted(err)
                )
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
    save_signal_head = QtCore.Signal(str, str)
    save_signal_nav = QtCore.Signal(str, str)  # filename, file_filter


class NavPlotter(QtCore.QThread):
    def __init__(self, filepath: pathlib.Path):
        super().__init__()
        self.filepath = filepath
        self.signals = NavPlotterSignals()
        self.thread = QtCore.QThread()

    # @QtCore.Slot()
    def run(self):
        print("NavPlotter started")
        get_nav = Georeferencer(filepath=self.filepath)
        get_nav.prep_data()
        print("Getting navigation data...")
        lola_data = get_nav.LOLA_plt
        head_data = get_nav.HEAD_plt
        lola_data_ori = get_nav.LOLA_plt_ori
        head_data_ori = get_nav.HEAD_plt_ori
        head_data = np.column_stack((head_data[:, 0], head_data[:, 1]))
        head_data_ori = np.column_stack((head_data_ori[:, 0], head_data_ori[:, 1]))
        self.signals.results_signal.emit(
            (lola_data, lola_data_ori, head_data, head_data_ori)
        )

class GeoreferencerThread(QtCore.QThread):
    finished = QtCore.Signal()
    progress_signal = QtCore.Signal(float)
    aborted_signal = QtCore.Signal(str)
    error_signal = QtCore.Signal(Exception)
    status_signal = QtCore.Signal(str)

    def __init__(
            self,  
            filepath: pathlib.Path, 
            active_utm: bool,
            active_export_navdata: bool,
            proc_data: list,
            output_folder: os.PathLike,
            vertical_beam_angle: int,
            resolution: float,
            search_radius: float,
            parent=None,
        ):
        super().__init__(parent)
        self.filepath = filepath
        self.active_utm = active_utm
        self.active_export_navdata = active_export_navdata
        self.proc_data = proc_data
        self.output_folder = output_folder
        self.vertical_beam_angle = vertical_beam_angle
        self.resolution = resolution
        self.search_radius = search_radius

    def georef(self):
        georef_success = True
        err_msg = f"Error while Georeferencing {self.filepath}"

        processor_0 = Georeferencer(
            filepath=self.filepath,
            channel=0,
            active_utm=self.active_utm,
            active_export_navdata=self.active_export_navdata,
            proc_data=None,
            output_folder=self.output_folder,
            vertical_beam_angle=self.vertical_beam_angle,
            resolution=self.resolution,
            search_radius=self.search_radius,
        )  # from georef.py
        processor_0.process(self.progress_signal)

        processor_1 = Georeferencer(
            filepath=self.filepath,
            channel=1,
            active_utm=self.active_utm,
            active_export_navdata=self.active_export_navdata,
            proc_data=None,
            output_folder=self.output_folder,
            vertical_beam_angle=self.vertical_beam_angle,
            resolution=self.resolution,
            search_radius=self.search_radius,
        )  # from georef.py
        processor_1.process(self.progress_signal)

        if georef_success:
            self.status_signal.emit("Georeferencing finished")
            self.finished.emit()
        else:
            self.error_signal.emit(str(err_msg))

    def run(self):
        try:
            self.georef()
        except Exception as e:
            self.error_signal.emit(e)
        finally:
            self.finished.emit()

class GeoreferencerManager(QWidget):
    processing_finished = QtCore.Signal(list)
    aborted = QtCore.Signal(str)
    pbar_val: float
    cleanup_cnt: int  # fix for now

    def __init__(self):
        super().__init__()
        self.pbar_val = 0
        self.cleanup_cnt = 0
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 50)
        self.pbar.setTextVisible(False)
        self.title_label = QLabel()
        self.title_label.setText("Georeferencing...")
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
                     filepath, 
                     active_utm, 
                     active_export_navdata, 
                     proc_data,
                     output_folder,
                     vertical_beam_angle,
                     resolution,
                     search_radius
                     ):

        self.georef_thread = GeoreferencerThread(
            filepath, 
            active_utm, 
            active_export_navdata,
            proc_data,
            output_folder,
            vertical_beam_angle,
            resolution,
            search_radius
            )
        self.georef_thread.progress_signal.connect(
            lambda progress: self.update_pbar(progress)
            )
        self.georef_thread.finished.connect(self.cleanup)
        self.georef_thread.finished.connect(self.georef_thread.deleteLater)
        self.georef_thread.aborted_signal.connect(            
            lambda msg_str: self.georef_aborted(msg_str)
        )
        self.georef_thread.start()
    
    def update_pbar(self, progress: float):
        self.pbar_val += progress  # for 2 channels
        disp_var = self.pbar_val
        self.pbar.setValue(int(100 * disp_var))

    def build_aborted(self, err: Exception):
        msg_str = str(err)
        print(msg_str)
        self.cleanup()

    def georef_aborted(self, msg_str: str):
        self.aborted_signal.emit(msg_str)
        self.deleteLater()
        self.cleanup()

    def cleanup(self):
        self.cleanup_cnt += 1
        if (
            self.cleanup_cnt > 1
        ):  # we need the signal 2 times for both channels to have finished
            print(f"Cleaning ...")
            for file in os.listdir(self.output_folder):
                file_path = os.path.join(self.output_folder, file)
                if (
                    str(file_path).startswith("outmedian")
                    or str(file_path).endswith(".xml")
                    or str(file_path).endswith(".xyz")
                ):
                    try:
                        os.remove(file_path)
                    except FileNotFoundError:
                        print(f"File Not Found: {file_path}")

            print("Cleanup done")
            self.deleteLater()
