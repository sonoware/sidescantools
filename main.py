import multiprocessing.dummy
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QLineEdit,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QStyle,
    QScrollArea,
    QTextEdit,
    QSizePolicy,
)
from qtpy.QtGui import QPixmap, QPalette, QColor, QShortcut, QKeySequence
import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
import sys, os, pathlib
from bottom_detection_napari_ui import run_napari_btm_line
from egn_building_batch import generate_slant_and_egn_files
from sidescan_georeferencer import SidescanGeoreferencer
import yaml
from math import log
import multiprocessing
import numpy as np
from sidescan_preproc import SidescanPreprocessor
from sidescan_file import SidescanFile
import napari
from napari.utils.colormaps import Colormap


class QHLine(QFrame):
    def __init__(self):
        """Helper class for a horizontal line"""
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class Labeled2Buttons(QHBoxLayout):
    label: QLabel
    button_1: QPushButton
    button_2: QPushButton

    def __init__(self, label_title: str, button_1: QPushButton, button_2: QPushButton):
        """Custom layout that stores 2 Buttons with an additional label in a HBox"""
        super().__init__()

        self.label = QLabel(label_title)
        self.label.setFixedWidth(160)  # for now for layout
        self.button_1 = button_1
        self.button_2 = button_2
        self.button_1.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding
        )
        self.button_2.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding
        )

        self.addWidget(self.label)
        self.addWidget(self.button_1)
        self.addWidget(self.button_2)


class LabeledLineEdit(QHBoxLayout):
    label: QLabel
    line_edit: QLineEdit

    def __init__(self, label_title: str, validator: QtGui.QValidator, start_val):
        """Custom layout for a LineEdit with an extra label in a HBox"""
        super().__init__()

        self.label = QLabel(label_title)
        self.line_edit = QLineEdit(str(start_val))
        if validator is not None:
            self.line_edit.setValidator(validator)
        self.addWidget(self.label)
        self.addWidget(self.line_edit)


class OverwriteWarnDialog(QDialog):
    def __init__(self, parent=None, filename: str = ""):
        """Warning dialog if a file is overwritten"""
        super().__init__(parent)

        self.setWindowTitle("Warning! Overwriting file")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QVBoxLayout()
        message = QLabel(
            f"Warning! The file {filename} exists. Do you want to overwrite it?"
        )
        layout.addWidget(message)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)


class FilePicker(QHBoxLayout):
    main_parent = None
    btn: QPushButton
    label: QLabel
    mode: QFileDialog.FileMode
    cur_dir: str
    dir_changed = QtCore.Signal()

    def __init__(
        self,
        btn_title: str,
        mode: QFileDialog.FileMode,
        start_val: str,
        filter=None,
        main_parent=None,
    ):
        """Helper class for a custom FilePicker with optional filter

        Parameters
        ----------
        btn_title: str
            Title of that button that triggers the file picker
        mode: QFileDialog.FileMode
            FileMode that is used for file picking
        start_val: str
            Inital ``str`` set as default file path
        filter: str
            Optional: Custom str that is interpreted by ``QFileDialog.setFilter()``
        main_parent:
            Optional: parent
        """

        super().__init__()

        self.main_parent = main_parent
        self.mode = mode
        self.filter = filter
        self.cur_dir = str(pathlib.Path(start_val).absolute())
        self.btn = QPushButton(btn_title)
        self.btn.setFixedWidth(150)
        self.label = QLabel(self.cur_dir)
        self.btn.clicked.connect(self.dir_selected)
        self.addWidget(self.btn)
        self.addWidget(self.label)

    def dir_selected(self):
        file_picker = QFileDialog()
        file_picker.setOption(QFileDialog.Option.ShowDirsOnly, True)
        file_picker.setFileMode(self.mode)
        file_picker.setDirectory(self.cur_dir)
        if self.filter != None:
            file_picker.setFilter(self.filter)
        if file_picker.exec_():
            filenames = file_picker.selectedFiles()
            self.cur_dir = filenames[0]
            self.label.setText(self.cur_dir)
        self.dir_changed.emit()
        if self.main_parent is not None:
            self.main_parent.update_right_view_size()

    def update_dir(self, dir: str):
        self.cur_dir = dir
        self.label.setText(dir)


class SidescanToolsMain(QWidget):
    file_dict = {
        "Path": [],
        "Bottom line": [],
        "Slant corrected": [],
        "EGN corrected": [],
        "File size": [],
    }
    settings_dict = {
        "Working dir": "./sidescan_out",
        "Georef dir": "./georef_out",
        "EGN table path": "./sidescan_out/EGN_table.npz",
        "Project filename": "project_info.yml",
        "EGN table name": "EGN_table.npz",
        "Btm chunk size": 1000,
        "Btm def thresh": 0.6,
        "Btm downsampling": 1,
        "Active convert dB": False,
        "Slant Vertical beam angle": 60,
        "Slant nadir angle": 0,
        "Slant use intern depth": False,
        "Slant chunk size": 1000,
        "Slant active use downsampling": True,
        "Slant active remove wc": True,
        "Slant active multiprocessing": True,
        "Slant num worker": 8,
        "Slant active export EGN data": True,
        "Slant active export slant data": True,
        "View reprocess file": False,
        "Georef active EGN": True,
        "Georef active dynamic chunking": False,
        "Georef UTM": True,
        "Path": [],
    }
    # TODO: is this a desired cmap or change that?
    sonar_dat_cmap = {
        "colors": [
            [0, 0, 0, 1],
            [0.43, 0.19, 0.17, 1],
            [0.65, 0.31, 0.15, 1],
            [0.9, 0.8, 0.25, 1],
            [0.95, 0.93, 0.33, 1],
        ],
        "name": "sonar_dat",
        "interpolation": "linear",
        "controls": [0, 0.2, 0.5, 0.9, 1],
    }

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SidescanTools")
        self.base_layout = QHBoxLayout()
        self.setLayout(self.base_layout)

        self.initGUI()

    def initGUI(self):

        # fonts
        title_font = QtGui.QFont()
        title_font.setBold(True)

        # File picker for JSF/XTF reading
        self.file_pick_btn = QPushButton("Add XTF/JSF file")
        self.file_pick_btn.clicked.connect(self.pick_new_files)

        self.file_table = QTableWidget()
        self.file_table.setAlternatingRowColors(True)

        self.file_table.setColumnCount(2)
        self.file_table.setColumnWidth(0, 600)
        self.file_table.setColumnWidth(1, 150)
        self.file_table.cellClicked.connect(self.always_select_row)
        self.file_table.cellClicked.connect(self.update_meta_info)

        ## Right side
        # Choose working directory and save/load project info
        if not pathlib.Path.exists(pathlib.Path(self.settings_dict["Working dir"])):
            os.mkdir(self.settings_dict["Working dir"])
        self.output_picker = FilePicker(
            "Working directory:",
            QFileDialog.FileMode.Directory,
            self.settings_dict["Working dir"],
            main_parent=self,
        )
        self.output_picker.dir_changed.connect(self.change_output_dir)
        self.project_save_button = QPushButton("Save Project Info")
        self.project_save_button.setToolTip(
            "Save project information as yaml in current working directory"
        )
        self.project_save_button.clicked.connect(self.save_project)
        self.project_load_button = QPushButton("Load Project Info")
        self.project_load_button.setToolTip(
            "Load project information from yaml in current working directory"
        )
        self.project_load_button.clicked.connect(self.load_project)

        # Choose georef out dir
        if not pathlib.Path.exists(pathlib.Path(self.settings_dict["Georef dir"])):
            os.mkdir(self.settings_dict["Georef dir"])
        self.georef_out_picker = FilePicker(
            "Georef output directory:",
            QFileDialog.FileMode.Directory,
            self.settings_dict["Georef dir"],
            main_parent=self,
        )
        self.georef_out_picker.dir_changed.connect(self.change_georef_dir)

        # Choose EGN table file
        start_egn_table_file = (
            self.settings_dict["Working dir"]
            + "/"
            + self.settings_dict["EGN table name"]
        )
        self.egn_table_picker = FilePicker(
            "EGN Table:",
            QFileDialog.FileMode.ExistingFile,
            start_val=start_egn_table_file,
            main_parent=self,
        )
        self.egn_table_picker.dir_changed.connect(self.load_egn_table)

        # Caption
        # TODO: Remove preliminary logo until we got a final one
        # self.title_img = QPixmap("./res/sst_256_cut.png")
        # self.title_img = self.title_img.scaledToWidth(
        #     350, QtCore.Qt.TransformationMode.SmoothTransformation
        # )
        # self.title_label = QLabel()
        # self.title_label.setPixmap(self.title_img)

        # File Info
        self.file_info_checkbox = QCheckBox(
            "Enable meta info by loading file at click (slows down UI)"
        )
        self.file_info_checkbox.setFont(title_font)
        self.file_info_checkbox.setChecked(True)
        self.file_info_checkbox.setToolTip(
            "Currently needs to load the complete file to display the information. This needs too much time and is work in progress."
        )
        self.file_info_text_box = QTextEdit()

        # Bottom line detection
        self.dir_and_path_label = QLabel("Working directory and paths setup")
        self.dir_and_path_label.setFont(title_font)

        self.btm_label = QLabel("Bottom Line Detection")
        self.btm_label.setFont(title_font)
        self.btm_chunk_size_edit = LabeledLineEdit(
            "Chunk Size:",
            QtGui.QIntValidator(100, 5000, self),
            self.settings_dict["Btm chunk size"],
        )
        self.btm_chunk_size_edit.label.setToolTip("Number of pings in one chunk for bottom detection.")
        self.btm_default_thresh = LabeledLineEdit(
            "Default Threshold [0.0 - 1.0]:",
            QtGui.QDoubleValidator(0.0, 1.0, 2, self),
            self.settings_dict["Btm def thresh"],
        )
        self.btm_default_thresh.label.setToolTip("Threshold that is applied to normalized data to find edges between water and ground. Needs to be in range[ 0 - 1].")
        self.btm_downsample_fact = LabeledLineEdit(
            "Downsampling Factor:",
            QtGui.QIntValidator(1, 16, self),
            self.settings_dict["Btm downsampling"],
        )
        self.btm_downsample_fact.label.setToolTip("Integer decimation factor that is used to downsample each ping.")
        self.active_convert_dB_checkbox = QCheckBox("Convert to dB")
        self.active_convert_dB_checkbox.setChecked(
            self.settings_dict["Active convert dB"]
        )
        self.do_btm_detection_btn = QPushButton("Bottom Line Detection")
        self.do_btm_detection_btn.setToolTip("Start Bottom Line Detection")
        self.do_btm_detection_btn.clicked.connect(self.run_bottom_line_detection)

        self.slant_and_egn_label = QLabel("Slant Range Correction and EGN")
        self.slant_and_egn_label.setFont(title_font)
        self.vertical_beam_angle_edit = LabeledLineEdit(
            "Vertical Beam Angle:",
            QtGui.QIntValidator(0, 90, self),
            self.settings_dict["Slant Vertical beam angle"],
        )
        self.vertical_beam_angle_edit.label.setToolTip("Only relevant if internal depth is unknown: Horizontal angle by which the instrument is tilted (usually found in the manual).")
        self.nadir_angle_edit = LabeledLineEdit(
            "Nadir Angle:",
            QtGui.QIntValidator(0, 90, self),
            self.settings_dict["Slant nadir angle"],
        )
        self.nadir_angle_edit.label.setToolTip("Angle between perpendicular and first bottom return (usually unknown, default is 0°)")
        self.use_intern_depth_checkbox = QCheckBox("Use internal Depth")
        self.use_intern_depth_checkbox.setToolTip("Use internal depth information for slant range correction. Otherwise depth is estimated from detected bottom line.")
        self.use_intern_depth_checkbox.setChecked(
            self.settings_dict["Slant use intern depth"]
        )
        self.slant_chunk_size_edit = LabeledLineEdit(
            "Chunk Size:",
            QtGui.QIntValidator(100, 5000, self),
            self.settings_dict["Slant chunk size"],
        )
        self.slant_chunk_size_edit.label.setToolTip("Number of pings in one chunk for for slant range and EGN correction. Is also used to determine the size of the exported waterfall images.")
        self.use_bottom_detection_downsampling_checkbox = QCheckBox(
            "Apply Downsampling"
        )
        self.use_bottom_detection_downsampling_checkbox.setToolTip("Use downsampling factor from bottom line detection to do processing on downsampled data.")
        self.use_bottom_detection_downsampling_checkbox.setChecked(
            self.settings_dict["Slant active use downsampling"]
        )
        self.active_remove_watercol_checkbox = QCheckBox("Remove Watercolumn")
        self.active_remove_watercol_checkbox.setChecked(
            self.settings_dict["Slant active remove wc"]
        )
        self.egn_table_name_edit = LabeledLineEdit(
            "EGN Table Name:",
            validator=None,
            start_val=self.settings_dict["EGN table name"],
        )
        self.egn_table_name_edit.label.setToolTip("Set name of EGN Table that is written as .npz file.")
        self.active_multiprocessing_checkbox = QCheckBox("Active Multiprocessing")
        self.active_multiprocessing_checkbox.setToolTip("Use multiprocessing in python to enable faster processing by multithreading.")
        self.active_multiprocessing_checkbox.setChecked(
            self.settings_dict["Slant active multiprocessing"]
        )
        self.num_worker_edit = LabeledLineEdit(
            "Number of Workers:",
            QtGui.QIntValidator(0, 32, self),
            str(self.settings_dict["Slant num worker"]),
        )
        self.export_slant_correction_checkbox = QCheckBox(
            "Export Slant Range corrected Data"
        )
        self.export_slant_correction_checkbox.setToolTip("Export Slant Range corrected data as .npz file. So it doesn't need to be recalculated for export or viewing.")
        self.export_slant_correction_checkbox.setChecked(
            self.settings_dict["Slant active export slant data"]
        )
        self.export_EGN_correction_checkbox = QCheckBox("Export EGN corrected Data")
        self.export_EGN_correction_checkbox.setToolTip("Export EGN corrected data as .npz file. So it doesn't need to be recalculated for export or viewing.")
        self.export_EGN_correction_checkbox.setChecked(
            self.settings_dict["Slant active export EGN data"]
        )
        self.generate_egn_table = QPushButton("Generate EGN Table")
        self.generate_egn_table.clicked.connect(self.run_generate_slant_and_egn_files)
        self.process_all_btn = QPushButton("Process All Files")
        self.process_all_btn.clicked.connect(self.process_all_files)

        self.napari_label = QLabel("View Results")
        self.napari_label.setFont(title_font)
        self.active_reprocess_file_checkbox = QCheckBox("Reprocess file")
        self.active_reprocess_file_checkbox.setToolTip("Do the slant range and EGN correction processing of the selected file before viewing.")
        self.show_proc_file_btn = QPushButton("View Processed Data")
        self.show_proc_file_btn.clicked.connect(self.show_proc_file_in_napari)

        self.georef_label = QLabel("Georeferencing and Image Generation")
        self.georef_label.setFont(title_font)
        self.active_use_egn_data_checkbox = QCheckBox("Use EGN corrected Data")
        self.active_use_egn_data_checkbox.setToolTip("Export pictures using the EGN corrected data. Otherwise raw data is exported.")
        self.active_use_egn_data_checkbox.setChecked(
            self.settings_dict["Georef active EGN"]
        )
        self.active_dynamic_chunking_checkbox = QCheckBox("Dynamic Chunking")
        self.active_dynamic_chunking_checkbox.setToolTip("Experimental")
        self.active_dynamic_chunking_checkbox.setChecked(
            self.settings_dict["Georef active dynamic chunking"]
        )

        self.active_utm_checkbox = QCheckBox("UTM")
        self.active_utm_checkbox.setToolTip("Coordinates in UTM (default). WGS84 if unchecked.")
        self.active_utm_checkbox.setChecked(
            self.settings_dict["Georef UTM"]
        )

        self.active_colormap_checkbox = QCheckBox("Apply custom Colormap")
        self.active_colormap_checkbox.setToolTip("Applies the colormap used in napari to the exported waterfall images. Otherwise grey scale values are used.")
        self.active_colormap_checkbox.setChecked(True)
        self.generate_single_georef_btn = QPushButton("Selected")
        self.generate_single_georef_btn.clicked.connect(self.run_sidescan_georef)
        self.generate_all_georef_btn = QPushButton("All")
        self.generate_all_georef_btn.clicked.connect(
            lambda: self.run_sidescan_georef(True)
        )
        self.generate_simple_img_btn = QPushButton("Selected")
        self.generate_simple_img_btn.clicked.connect(
            lambda: self.generate_wc_img(False)
        )
        self.generate_all_simple_img_btn = QPushButton("All")
        self.generate_all_simple_img_btn.clicked.connect(
            lambda: self.generate_wc_img(True)
        )

        ## --- Layouts
        self.left_view = QVBoxLayout()
        self.right_view = QVBoxLayout()
        self.right_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.left_view.addWidget(self.file_pick_btn)
        self.left_view.addWidget(self.file_table)

        # TODO: Remove preliminary logo until we got a final one
        # self.right_view.addWidget(self.title_label)
        # self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.file_info_checkbox)
        self.right_view.addWidget(self.file_info_text_box)
        self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.dir_and_path_label)
        self.right_view.addWidget(self.project_save_button)
        self.right_view.addWidget(self.project_load_button)
        self.right_view.addLayout(self.output_picker)
        self.right_view.addLayout(self.georef_out_picker)
        self.right_view.addLayout(self.egn_table_picker)
        self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.btm_label)
        self.right_view.addLayout(self.btm_chunk_size_edit)
        self.right_view.addLayout(self.btm_default_thresh)
        self.right_view.addLayout(self.btm_downsample_fact)
        self.right_view.addWidget(self.active_convert_dB_checkbox)
        self.right_view.addWidget(self.do_btm_detection_btn)
        self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.slant_and_egn_label)
        self.right_view.addLayout(self.vertical_beam_angle_edit)
        self.right_view.addLayout(self.nadir_angle_edit)
        self.right_view.addWidget(self.use_intern_depth_checkbox)
        self.right_view.addLayout(self.slant_chunk_size_edit)
        self.right_view.addWidget(self.use_bottom_detection_downsampling_checkbox)
        self.right_view.addLayout(self.egn_table_name_edit)
        self.right_view.addWidget(self.active_remove_watercol_checkbox)
        self.right_view.addWidget(self.active_multiprocessing_checkbox)
        self.right_view.addLayout(self.num_worker_edit)
        self.right_view.addWidget(self.export_EGN_correction_checkbox)
        self.right_view.addWidget(self.export_slant_correction_checkbox)
        self.right_view.addWidget(self.generate_egn_table)
        self.right_view.addWidget(self.process_all_btn)
        self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.napari_label)
        self.right_view.addWidget(self.active_reprocess_file_checkbox)
        self.right_view.addWidget(self.show_proc_file_btn)
        # self.right_view.addWidget(self.generate_new_egn_table)
        self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.georef_label)
        self.right_view.addWidget(self.active_use_egn_data_checkbox)
        self.right_view.addWidget(self.active_dynamic_chunking_checkbox)
        self.right_view.addWidget(self.active_utm_checkbox)
        self.right_view.addWidget(self.active_colormap_checkbox)

        self.labeled_georef_buttons = Labeled2Buttons(
            "Generate Geotiff:",
            self.generate_single_georef_btn,
            self.generate_all_georef_btn,
        )
        self.right_view.addLayout(self.labeled_georef_buttons)

        self.labeled_img_export_buttons = Labeled2Buttons(
            "Generate Waterfall Image:",
            self.generate_simple_img_btn,
            self.generate_all_simple_img_btn,
        )
        self.right_view.addLayout(self.labeled_img_export_buttons)

        self.base_layout.addLayout(self.left_view)
        self.right_base_widget = QWidget()
        self.right_base_widget.setLayout(self.right_view)
        self.right_base_widget.setMaximumSize(self.right_view.minimumSize())
        # include right base widget in scroll area
        self.right_scroll_area = QScrollArea()
        self.right_scroll_area.setWidget(self.right_base_widget)
        self.right_scroll_area.setWidgetResizable(True)
        # add 10 px for scrollbar
        self.right_scroll_area.setMaximumWidth(
            self.right_view.minimumSize().width() + 10
        )
        self.base_layout.addWidget(self.right_scroll_area)

        # Shortcuts
        self.shortcut_del = QShortcut(QKeySequence("Del"), self.file_table)
        self.shortcut_del.activated.connect(self.delete_file)

    def update_right_view_size(self):
        self.right_base_widget.setMinimumWidth(
            self.right_view.minimumSize().width() + 10
        )
        self.right_base_widget.resize(
            self.right_view.minimumSize().height(),
            self.right_view.minimumSize().width() + 10,
        )
        # add 10 px for scrollbar
        self.right_scroll_area.setMaximumWidth(
            self.right_view.minimumSize().width() + 10
        )
        # TODO: why does this only look as intended when called twice?!
        self.right_base_widget.setMinimumWidth(
            self.right_view.minimumSize().width() + 10
        )
        self.right_base_widget.resize(
            self.right_view.minimumSize().height(),
            self.right_view.minimumSize().width() + 10,
        )
        # add 10 px for scrollbar
        self.right_scroll_area.setMaximumWidth(
            self.right_view.minimumSize().width() + 10
        )

    def pick_new_files(self):
        file_picker = QFileDialog()
        file_picker.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_picker.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_picker.setNameFilter("*.xtf;*.jsf")

        if file_picker.exec_():
            filenames = file_picker.selectedFiles()

            # set dB conversion when the first file of all time is picked
            # TODO: Load first file and decide on the acutal data what to do here
            if self.file_dict["Path"] == []:
                if filenames[0].endswith(".xtf"):
                    self.settings_dict["Active convert dB"] = False
                else:
                    self.settings_dict["Active convert dB"] = True
                self.active_convert_dB_checkbox.setChecked(
                    self.settings_dict["Active convert dB"]
                )

            # check for duplicates adn sort full list
            full_list = self.file_dict["Path"]
            full_list.extend(filenames)
            full_list = list(set(full_list))
            full_list.sort()

            num_files = len(full_list)
            self.file_dict["Path"] = full_list
            self.file_dict["Bottom line"] = ["N"] * num_files
            self.file_dict["File size"] = ["0"] * num_files
            self.file_dict["Slant corrected"] = ["N"] * num_files
            self.file_dict["EGN corrected"] = ["N"] * num_files

            self.update_table()
            self.update_right_view_size()

    def delete_file(self):
        idx_del = self.file_table.selectedIndexes()[0].row()
        self.file_dict["Path"].pop(idx_del)
        self.file_dict["Bottom line"].pop(idx_del)
        self.file_dict["File size"].pop(idx_del)
        self.file_dict["Slant corrected"].pop(idx_del)
        self.file_dict["EGN corrected"].pop(idx_del)

        self.update_table()
        self.update_right_view_size()

    def update_table(self):

        self.check_for_btm_line_data_and_size()

        # Build table
        num_files = len(self.file_dict["Path"])
        self.file_table.clearContents()
        self.file_table.setRowCount(num_files)
        self.file_table.setColumnCount(5)
        self.file_table.setColumnWidth(0, 600)
        self.file_table.setColumnWidth(1, 100)
        self.file_table.setColumnWidth(2, 100)
        self.file_table.setColumnWidth(3, 100)
        self.file_table.setColumnWidth(4, 100)
        self.file_table.setHorizontalHeaderLabels(self.file_dict.keys())

        yes_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        no_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
        for idx, filepath in enumerate(self.file_dict["Path"]):
            new_item = QTableWidgetItem(filepath)
            self.file_table.setItem(idx, 0, new_item)

            if self.file_dict["Bottom line"][idx] == "Y":
                new_item = QTableWidgetItem(yes_icon, "")
            else:
                new_item = QTableWidgetItem(no_icon, "")
            new_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self.file_table.setItem(idx, 1, new_item)

            if self.file_dict["Slant corrected"][idx] == "Y":
                new_item = QTableWidgetItem(yes_icon, "")
            else:
                new_item = QTableWidgetItem(no_icon, "")
            new_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.file_table.setItem(idx, 2, new_item)

            if self.file_dict["EGN corrected"][idx] == "Y":
                new_item = QTableWidgetItem(yes_icon, "")
            else:
                new_item = QTableWidgetItem(no_icon, "")
            new_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.file_table.setItem(idx, 3, new_item)
            new_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            new_item = QTableWidgetItem(self.file_dict["File size"][idx])
            self.file_table.setItem(idx, 4, new_item)

        # check whether a file is selected, if there is none, select index 0
        if self.file_table.selectedIndexes() == []:
            self.file_table.selectRow(0)

    # taken from: https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
    def prettier_size(
        self, n, pow=0, b=1024, u="B", pre=[""] + [p + "i" for p in "KMGTPEZY"]
    ):
        r, f = min(int(log(max(n * b**pow, 1), b)), len(pre) - 1), "{:,.%if} %s%s"
        return (f % (abs(r % (-r - 1)), pre[r], u)).format(n * b**pow / b ** float(r))

    def check_for_btm_line_data_and_size(self):
        for idx, filepath in enumerate(self.file_dict["Path"]):
            filepath = pathlib.Path(filepath)
            work_dir = pathlib.Path(self.settings_dict["Working dir"])
            if filepath.exists():
                self.file_dict["File size"][idx] = self.prettier_size(
                    filepath.stat().st_size
                )

                if (work_dir / (filepath.stem + "_bottom_info.npz")).exists():
                    self.file_dict["Bottom line"][idx] = "Y"
                else:
                    self.file_dict["Bottom line"][idx] = "N"

                if (work_dir / (filepath.stem + "_slant_corrected.npz")).exists():
                    self.file_dict["Slant corrected"][idx] = "Y"
                else:
                    self.file_dict["Slant corrected"][idx] = "N"

                if (work_dir / (filepath.stem + "_egn_corrected.npz")).exists():
                    self.file_dict["EGN corrected"][idx] = "Y"
                else:
                    self.file_dict["EGN corrected"][idx] = "N"
            else:
                self.file_dict["File size"][idx] = "Couldn't read"

    def update_meta_info(self):
        self.check_for_btm_line_data_and_size()
        if self.file_info_checkbox.isChecked():
            file_path = self.file_dict["Path"][
                self.file_table.selectedIndexes()[0].row()
            ]
            sidescan_file = SidescanFile(file_path)
            self.file_info_text_box.clear()
            self.file_info_text_box.insertHtml(
                "<b>Date          :</b> " + str(sidescan_file.timestamp[0]) + "<br />"
            )
            self.file_info_text_box.insertHtml(
                f"<b>Channels        :</b> {sidescan_file.num_ch}<br />"
            )
            self.file_info_text_box.insertHtml(
                f"<b>Number of pings :</b> {sidescan_file.num_ping}<br />"
            )
            self.file_info_text_box.insertHtml(
                f"<b>Samples per ping:</b> {sidescan_file.ping_len}<br />"
            )
            self.file_info_text_box.insertHtml(
                f"<b>Slant ranges    :</b> {np.min(sidescan_file.slant_range)} - {np.max(sidescan_file.slant_range)} m<br />"
            )

    def always_select_row(self):
        self.file_table.selectRow(self.file_table.selectedIndexes()[0].row())

    def run_bottom_line_detection(self):
        run_napari_btm_line(
            self.file_dict["Path"][self.file_table.selectedIndexes()[0].row()],
            chunk_size=int(self.btm_chunk_size_edit.line_edit.text()),
            default_threshold=float(self.btm_default_thresh.line_edit.text()),
            downsampling_factor=int(self.btm_downsample_fact.line_edit.text()),
            work_dir=self.output_picker.cur_dir,
            convert_to_dB=self.active_convert_dB_checkbox.isChecked(),
        )
        self.check_for_btm_line_data_and_size()

    def run_generate_slant_and_egn_files(self):
        # check if EGN file exist
        egn_path = pathlib.Path(
            self.settings_dict["Working dir"]
            + "/"
            + self.egn_table_name_edit.line_edit.text()
        )
        if egn_path.exists():
            dlg = OverwriteWarnDialog(self, str(egn_path))
            if not dlg.exec():
                return

        sonar_file_path_list = []
        for sonar_file in self.file_dict["Path"]:
            sonar_file_path_list.append(pathlib.Path(sonar_file))
        num_worker = int(self.num_worker_edit.line_edit.text())
        pool = multiprocessing.Pool(num_worker)
        generate_slant_and_egn_files(
            sonar_files=sonar_file_path_list,
            out_path=self.settings_dict["Working dir"],
            nadir_angle=int(self.nadir_angle_edit.line_edit.text()),
            use_intern_depth=self.use_intern_depth_checkbox.isChecked(),
            chunk_size=int(self.slant_chunk_size_edit.line_edit.text()),
            generate_final_egn_table=True,
            use_bottom_detection_downsampling=self.use_bottom_detection_downsampling_checkbox.isChecked(),
            egn_table_name=self.egn_table_name_edit.line_edit.text(),
            active_multiprocessing=self.active_multiprocessing_checkbox.isChecked(),
            pool=pool,
            remove_wc=self.active_remove_watercol_checkbox.isChecked(),
        )
        pool.close()

    def do_slant_corr_and_EGN(
        self, filepath, load_slant_data: bool = False, load_egn_data: bool = False
    ):

        sidescan_file = SidescanFile(filepath=filepath)
        bottom_info = np.load(
            pathlib.Path(self.settings_dict["Working dir"])
            / (filepath.stem + "_bottom_info.npz")
        )
        # Check if downsampling was applied
        try:
            downsampling_factor = bottom_info["downsampling_factor"]
        except:
            downsampling_factor = 1

        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=int(self.slant_chunk_size_edit.line_edit.text()),
            convert_to_dB=self.active_convert_dB_checkbox.isChecked(),
            downsampling_factor=downsampling_factor,
        )

        preproc.portside_bottom_dist = bottom_info["bottom_info_port"].flatten()
        preproc.starboard_bottom_dist = bottom_info["bottom_info_star"].flatten()
        preproc.napari_portside_bottom = bottom_info["bottom_info_port"]
        preproc.napari_starboard_bottom = bottom_info["bottom_info_star"]
        preproc.num_chunk = np.shape(bottom_info["bottom_info_star"])[0]

        # slant range correction and EGN data
        if self.export_slant_correction_checkbox.isChecked():
            slant_data_path = pathlib.Path(self.settings_dict["Working dir"]) / (
                filepath.stem + "_slant_corrected.npz"
            )
        else:
            slant_data_path = None

        if load_slant_data:
            slant_data = np.load(slant_data_path)
            preproc.slant_corrected_mat = slant_data["slant_corr"]

        else:
            preproc.slant_range_correction(
                active_interpolation=True,
                nadir_angle=int(self.nadir_angle_edit.line_edit.text()),
                save_to=slant_data_path,
                remove_wc=self.active_remove_watercol_checkbox.isChecked(),
                active_mult_slant_range_resampling=True,
            )

        egn_table_path = self.egn_table_picker.cur_dir
        if not pathlib.Path(egn_table_path).exists():
            dlg = QMessageBox(self)
            dlg.setWindowTitle("EGN table not found")
            dlg.setText(f"The specified EGN table {egn_table_path} doesn't exist.")
            dlg.exec()

        if self.export_slant_correction_checkbox.isChecked():
            egn_data_path = pathlib.Path(self.settings_dict["Working dir"]) / (
                filepath.stem + "_egn_corrected.npz"
            )
        else:
            egn_data_path = None

        if load_egn_data:
            egn_data = np.load(egn_data_path)
            preproc.egn_corrected_mat = egn_data["egn_corrected_mat"]
        else:
            preproc.do_EGN_correction(
                egn_table_path,
                save_to=egn_data_path,
            )

        return sidescan_file, preproc

    def process_all_files(self):
        path_list = []
        for idx, path in enumerate(self.file_dict["Path"]):
            if self.file_dict["Bottom line"][idx] == "Y":
                path_list.append(pathlib.Path(path))

        if self.active_multiprocessing_checkbox.isChecked():
            num_worker = int(self.num_worker_edit.line_edit.text())
            pool = multiprocessing.dummy.Pool(num_worker)
            res = pool.map(
                self.do_slant_corr_and_EGN,
                path_list,
            )
            print(res)
            pool.close()
        else:
            for filepath in path_list:
                self.do_slant_corr_and_EGN(filepath)

    def show_proc_file_in_napari(self):

        filepath = pathlib.Path(
            self.file_dict["Path"][self.file_table.selectedIndexes()[0].row()]
        )
        load_slant = (
            self.file_dict["Slant corrected"][
                self.file_table.selectedIndexes()[0].row()
            ]
            == "Y"
        )
        load_egn = (
            self.file_dict["EGN corrected"][self.file_table.selectedIndexes()[0].row()]
            == "Y"
        )
        if self.active_reprocess_file_checkbox.isChecked():
            load_egn = False
            load_slant = False
        sidescan_file, preproc = self.do_slant_corr_and_EGN(
            filepath, load_slant_data=load_slant, load_egn_data=load_egn
        )

        if self.active_convert_dB_checkbox.isChecked():
            raw_image = np.hstack(
                (preproc.sonar_data_proc[0], preproc.sonar_data_proc[1])
            )
        else:
            raw_image = np.hstack((sidescan_file.data[0], sidescan_file.data[1]))

        colors = [[1, 1, 1, 0], [1, 0, 0, 1]]  # r,g,b,alpha
        bottom_colormap = {
            "colors": colors,
            "name": "bottom_line_cmap",
            "interpolation": "linear",
        }
        preproc.build_bottom_line_map()

        # chunkify all data for plotting
        raw_image_chunk = np.zeros(
            (preproc.num_chunk, preproc.chunk_size, preproc.ping_len * 2)
        )
        slant_corr_chunk = np.zeros(
            (preproc.num_chunk, preproc.chunk_size, preproc.ping_len * 2)
        )
        egn_corr_chunk = np.zeros(
            (preproc.num_chunk, preproc.chunk_size, preproc.ping_len * 2)
        )
        for chunk_idx in range(preproc.num_chunk):
            cur_chunk_shape = np.shape(
                raw_image[
                    chunk_idx
                    * preproc.chunk_size : (chunk_idx + 1)
                    * preproc.chunk_size
                ]
            )
            raw_image_chunk[chunk_idx, 0 : cur_chunk_shape[0]] = raw_image[
                chunk_idx * preproc.chunk_size : (chunk_idx + 1) * preproc.chunk_size
            ]
            slant_corr_chunk[chunk_idx, 0 : cur_chunk_shape[0]] = (
                preproc.slant_corrected_mat[
                    chunk_idx
                    * preproc.chunk_size : (chunk_idx + 1)
                    * preproc.chunk_size
                ]
            )
            egn_corr_chunk[chunk_idx, 0 : cur_chunk_shape[0]] = (
                preproc.egn_corrected_mat[
                    chunk_idx
                    * preproc.chunk_size : (chunk_idx + 1)
                    * preproc.chunk_size
                ]
            )

        viewer, image_layer_1 = napari.imshow(
            raw_image_chunk, colormap=self.sonar_dat_cmap
        )
        image_layer_2 = viewer.add_image(preproc.bottom_map, colormap=bottom_colormap)
        image_layer_3 = viewer.add_image(slant_corr_chunk, colormap=self.sonar_dat_cmap)
        image_layer_4 = viewer.add_image(egn_corr_chunk, colormap=self.sonar_dat_cmap)
        napari.run(max_loop_level=2)

    def run_sidescan_georef(self, active_all_files=False):

        if active_all_files:
            file_list = self.file_dict["Path"]
        else:
            file_list = [
                pathlib.Path(
                    self.file_dict["Path"][self.file_table.selectedIndexes()[0].row()]
                )
            ]

        work_dir = pathlib.Path(self.settings_dict["Working dir"])
        for filepath in file_list:
            load_slant_data = False
            load_egn_data = False
            # check wheter preproc data is present and load or process file
            if (work_dir / (filepath.stem + "_slant_corrected.npz")).exists():
                load_slant_data = True
            if (work_dir / (filepath.stem + "_egn_corrected.npz")).exists():
                load_egn_data = True

            sidescan_file = None
            preproc = None
            sidescan_file, preproc = self.do_slant_corr_and_EGN(
                filepath, load_slant_data=load_slant_data, load_egn_data=load_egn_data
            )

            proc_data_0 = None
            proc_data_1 = None
            if self.active_use_egn_data_checkbox.isChecked():
                proc_data_0 = preproc.egn_corrected_mat[:, 0 : sidescan_file.ping_len]
                proc_data_0 = np.nan_to_num(
                    proc_data_0
                )  # remove nans from excluding far/nadir unknown values
                proc_data_1 = preproc.egn_corrected_mat[:, sidescan_file.ping_len :]
                proc_data_1 = np.nan_to_num(proc_data_1)

            georeferencer = SidescanGeoreferencer(
                filepath=filepath,
                channel=0,
                dynamic_chunking=self.active_dynamic_chunking_checkbox.isChecked(),
                UTM=self.active_utm_checkbox.isChecked(),
                output_folder=self.settings_dict["Georef dir"],
                proc_data=proc_data_0,
                vertical_beam_angle=int(self.vertical_beam_angle_edit.line_edit.text()),
            )
            georeferencer.process()
            georeferencer = SidescanGeoreferencer(
                filepath=filepath,
                channel=1,
                dynamic_chunking=self.active_dynamic_chunking_checkbox.isChecked(),
                UTM=self.active_utm_checkbox.isChecked(),
                output_folder=self.settings_dict["Georef dir"],
                proc_data=proc_data_1,
                vertical_beam_angle=int(self.vertical_beam_angle_edit.line_edit.text()),
            )
            georeferencer.process()

    def generate_wc_img(self, active_generate_all: bool):
        # TODO: this is quite custom for the GNB project, do we want to alter this?
        active_add_raw_img = True
        active_chunkify = True
        active_norm_chunks = False
        filepath = pathlib.Path(
            self.file_dict["Path"][self.file_table.selectedIndexes()[0].row()]
        )
        if active_generate_all:
            pathlist = []
            for path_idx in range(len(self.file_dict["Path"])):
                if self.file_dict["EGN corrected"][path_idx] == "Y":
                    pathlist.append(pathlib.Path(self.file_dict["Path"][path_idx]))
        else:
            pathlist = [filepath]

        work_dir = pathlib.Path(self.settings_dict["Working dir"])
        chunk_size = int(self.slant_chunk_size_edit.line_edit.text())

        for path in pathlist:
            load_slant_data = False
            load_egn_data = False
            # check wheter preproc data is present and load or process file
            if (work_dir / (path.stem + "_slant_corrected.npz")).exists():
                load_slant_data = True
            if (work_dir / (path.stem + "_egn_corrected.npz")).exists():
                load_egn_data = True
            sidescan_file = None
            preproc = None
            sidescan_file, preproc = self.do_slant_corr_and_EGN(
                path, load_slant_data=load_slant_data, load_egn_data=load_egn_data
            )

            data = preproc.egn_corrected_mat
            np.nan_to_num(data, copy=False)
            data /= np.nanmax(np.abs(data)) / 255
            data = np.array(data, dtype=np.uint8)
            if active_add_raw_img:
                raw_data = np.array(
                    np.hstack((sidescan_file.data[0], sidescan_file.data[1])),
                    dtype=float,
                )
                raw_data /= np.nanmax(np.abs(raw_data)) / 255
                raw_data = np.array(raw_data, dtype=np.uint8)
                data = np.hstack((raw_data, data))

            if active_chunkify:
                data_shape = np.shape(data)
                num_chunk = int(np.ceil(data_shape[0] / chunk_size))
                for chunk_idx in range(num_chunk):
                    im_name = str(work_dir / (path.stem + f"_{chunk_idx}.png"))
                    data_out = data[
                        chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size
                    ]
                    if active_norm_chunks:
                        data_out = np.array(data_out, dtype=float)
                        data_out[:, : 2 * preproc.ping_len] /= (
                            np.nanmax(np.abs(data_out[:, : 2 * preproc.ping_len])) / 255
                        )
                        data_out[:, 2 * preproc.ping_len :] /= (
                            np.nanmax(np.abs(data_out[:, 2 * preproc.ping_len :])) / 255
                        )
                        data_out = np.array(data_out, dtype=np.uint8)

                    # Apply colormap
                    if self.active_colormap_checkbox.isChecked():
                        cmap = Colormap(self.sonar_dat_cmap["colors"])
                        data_out = data_out.astype(float) / 255
                        data_out = cmap.map(data_out)
                        data_out *= 255
                    SidescanGeoreferencer.write_img(im_name, data_out.astype(np.uint8))
                    print(f"{im_name} written.")
            else:
                im_name = str(work_dir / (path.stem + ".png"))
                if self.active_colormap_checkbox.isChecked():
                    cmap = Colormap(self.sonar_dat_cmap["colors"])
                    data_out = data_out.astype(float) / 255
                    data_out = cmap.map(data_out)
                    data_out *= 255
                SidescanGeoreferencer.write_img(im_name, data)
                print(f"{im_name} written.")

    def change_output_dir(self):
        self.settings_dict["Working dir"] = self.output_picker.cur_dir
        self.load_project()

    def change_georef_dir(self):
        self.settings_dict["Georef dir"] = self.georef_out_picker.cur_dir

    def load_egn_table(self):
        self.settings_dict["EGN table path"] = str(
            pathlib.Path(self.egn_table_picker.cur_dir).absolute()
        )
        egn_table_info = np.load(self.egn_table_picker.cur_dir)
        nadir_angle = egn_table_info["nadir_angle"]
        self.nadir_angle_edit.line_edit.setText(str(nadir_angle))
        self.settings_dict["Slant nadir angle"] = nadir_angle

        self.update_right_view_size()

    def save_project(self):
        # copy path info to settings dict for saving
        self.update_settings_dict_from_ui()
        filepath = pathlib.Path(self.settings_dict["Working dir"]) / pathlib.Path(
            self.settings_dict["Project filename"]
        )

        # Check if file exists and warn user
        dlg = OverwriteWarnDialog(self)
        if dlg.exec():
            # Save
            try:
                f = open(
                    filepath,
                    "w",
                )
                yaml.dump(self.settings_dict, f)
                f.close()
            except:
                print(f"Can't write to {filepath}")

    def load_project(self):

        filepath = pathlib.Path(self.settings_dict["Working dir"]) / pathlib.Path(
            self.settings_dict["Project filename"]
        )

        if filepath.exists():
            f = open(
                filepath,
                "r",
            )
            loaded_dict = yaml.safe_load(f)
            f.close()
            for key in dict(loaded_dict).keys():
                try:
                    self.settings_dict[key] = loaded_dict[key]
                except:
                    print(f"Couldn't load setting with key: {key}")

            full_list = loaded_dict["Path"]  # downward compatibility
            full_list.sort()
            num_files = len(full_list)
            self.file_dict["Path"] = full_list
            self.file_dict["Bottom line"] = ["N"] * num_files
            self.file_dict["Slant corrected"] = ["N"] * num_files
            self.file_dict["EGN corrected"] = ["N"] * num_files
            self.file_dict["File size"] = ["0"] * num_files
            self.update_table()
            self.update_ui_from_settings()
        else:
            # show message that file doesn't exist
            dlg = QMessageBox(self)
            dlg.setWindowTitle("No settings found!")
            dlg.setText("No settings file found.")
            button = dlg.exec()

        self.update_right_view_size()

    def update_settings_dict_from_ui(self):
        self.settings_dict["Path"] = self.file_dict["Path"]
        self.settings_dict["Working dir"] = str(
            pathlib.Path(self.output_picker.cur_dir).absolute()
        )
        self.settings_dict["Georef dir"] = str(
            pathlib.Path(self.georef_out_picker.cur_dir).absolute()
        )
        self.settings_dict["EGN table name"] = pathlib.Path(
            self.egn_table_picker.cur_dir
        ).name
        self.settings_dict["Btm chunk size"] = int(
            self.btm_chunk_size_edit.line_edit.text()
        )
        self.settings_dict["Btm def thresh"] = float(
            self.btm_default_thresh.line_edit.text()
        )
        self.settings_dict["Btm downsampling"] = int(
            self.btm_downsample_fact.line_edit.text()
        )
        self.settings_dict["Active convert dB"] = (
            self.active_convert_dB_checkbox.isChecked()
        )
        self.settings_dict["Slant Vertical beam angle"] = int(
            self.vertical_beam_angle_edit.line_edit.text()
        )
        self.settings_dict["Slant nadir angle"] = int(
            self.nadir_angle_edit.line_edit.text()
        )
        self.settings_dict["Slant use intern depth"] = (
            self.use_intern_depth_checkbox.isChecked()
        )
        self.settings_dict["Slant chunk size"] = int(
            self.slant_chunk_size_edit.line_edit.text()
        )
        self.settings_dict["Slant active use downsampling"] = (
            self.use_bottom_detection_downsampling_checkbox.isChecked()
        )
        self.settings_dict["Slant active remove wc"] = (
            self.active_remove_watercol_checkbox.isChecked()
        )
        self.settings_dict["Slant active multiprocessing"] = (
            self.active_multiprocessing_checkbox.isChecked()
        )
        self.settings_dict["Slant num worker"] = int(
            self.num_worker_edit.line_edit.text()
        )
        self.settings_dict["Slant active export EGN data"] = (
            self.export_EGN_correction_checkbox.isChecked()
        )
        self.settings_dict["Slant active export slant data"] = (
            self.export_slant_correction_checkbox.isChecked()
        )
        self.settings_dict["View reprocess file"] = (
            self.active_reprocess_file_checkbox.isChecked()
        )
        self.settings_dict["Georef active EGN"] = (
            self.active_use_egn_data_checkbox.isChecked()
        )
        self.settings_dict["Georef active dynamic chunking"] = (
            self.active_dynamic_chunking_checkbox.isChecked()
        )

        self.settings_dict["Georef UTM"] = (
            self.active_dynamic_chunking_checkbox.isChecked()
        )

    def update_ui_from_settings(self):
        self.output_picker.update_dir(self.settings_dict["Working dir"])
        self.georef_out_picker.update_dir(self.settings_dict["Georef dir"])
        try:
            self.egn_table_picker.update_dir(self.settings_dict["EGN table path"])
        except:
            pass
        self.egn_table_name_edit.line_edit.setText(self.settings_dict["EGN table name"])
        self.btm_chunk_size_edit.line_edit.setText(
            str(self.settings_dict["Btm chunk size"])
        )
        self.btm_default_thresh.line_edit.setText(
            str(self.settings_dict["Btm def thresh"])
        )
        self.btm_downsample_fact.line_edit.setText(
            str(self.settings_dict["Btm downsampling"])
        )
        self.active_convert_dB_checkbox.setChecked(
            self.settings_dict["Active convert dB"]
        )
        self.vertical_beam_angle_edit.line_edit.setText(
            str(self.settings_dict["Slant Vertical beam angle"])
        )
        self.nadir_angle_edit.line_edit.setText(
            str(self.settings_dict["Slant nadir angle"])
        )
        self.use_intern_depth_checkbox.setChecked(
            self.settings_dict["Slant use intern depth"]
        )
        self.slant_chunk_size_edit.line_edit.setText(
            str(self.settings_dict["Slant chunk size"])
        )
        self.use_bottom_detection_downsampling_checkbox.setChecked(
            self.settings_dict["Slant active use downsampling"]
        )
        self.active_remove_watercol_checkbox.setChecked(
            self.settings_dict["Slant active remove wc"]
        )
        self.active_multiprocessing_checkbox.setChecked(
            self.settings_dict["Slant active multiprocessing"]
        )
        self.num_worker_edit.line_edit.setText(
            str(self.settings_dict["Slant num worker"])
        )
        self.export_EGN_correction_checkbox.setChecked(
            self.settings_dict["Slant active export EGN data"]
        )
        self.export_slant_correction_checkbox.setChecked(
            self.settings_dict["Slant active export slant data"]
        )
        self.active_reprocess_file_checkbox.setChecked(
            self.settings_dict["View reprocess file"]
        )
        self.active_use_egn_data_checkbox.setChecked(
            self.settings_dict["Georef active EGN"]
        )
        self.active_dynamic_chunking_checkbox.setChecked(
            self.settings_dict["Georef active dynamic chunking"]
        )
        self.active_utm_checkbox.setChecked(
            self.settings_dict["Georef UTM"]
        )

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Styling by custom palette
    palette = QPalette()
    white = QtCore.Qt.white
    red = QColor(255, 0, 0)
    link = QColor(20, 30, 190)
    grey_main = QColor(60, 60, 60)
    palette.setColor(QPalette.Window, grey_main)
    palette.setColor(QPalette.WindowText, white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    palette.setColor(QPalette.Text, white)
    palette.setColor(QPalette.Button, grey_main)
    palette.setColor(QPalette.ButtonText, white)
    palette.setColor(QPalette.BrightText, red)
    palette.setColor(QPalette.Link, link)
    palette.setColor(QPalette.Highlight, link)
    palette.setColor(QPalette.HighlightedText, white)

    # Apply the palette to the application
    app.setPalette(palette)
    app_icon = QtGui.QIcon("./res/icon.ico")
    app.setWindowIcon(app_icon)
    window = SidescanToolsMain()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
