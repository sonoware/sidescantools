from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QMessageBox,
    QStyle,
    QScrollArea,
    QTextEdit,
    QTabWidget,
    QSizePolicy,
    QSpacerItem,
    QRadioButton,
    QButtonGroup,
    QComboBox,
)
from qtpy.QtGui import QPalette, QColor, QShortcut, QKeySequence
import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
import sys, os, pathlib
from bottom_detection_napari_ui import run_napari_btm_line
from sidescan_georeferencer import SidescanGeoreferencer
import yaml, copy
from math import log
import numpy as np
import napari
from napari.utils.colormaps import Colormap
from custom_widgets import (
    QHLine,
    Labeled2Buttons,
    LabeledLineEdit,
    OverwriteWarnDialog,
    ErrorWarnDialog,
    convert_to_dB,
    hist_equalization,
    FilePicker,
)
from custom_threading import FileImportManager, EGNTableBuilder, PreProcManager
from enum import Enum
import scipy.signal as scisig

GAINSTRAT = Enum("GAINSTRAT", [("BAC", 0), ("EGN", 1)])


class SidescanToolsMain(QWidget):
    file_dict = {
        "Path": [],
        "Bottom line": [],
        "Slant corrected": [],
        "Gain corrected": [],
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
        "Active convert dB": True,
        "Btm equal hist": True,
        "Active pie slice filter": True,
        "Active sharpening filter": False,
        "Active gain norm": True,
        "Active hist equal": True,
        "Slant gain norm strategy": GAINSTRAT.BAC.value,
        "Slant vertical beam angle": 60,
        "Slant nadir angle": 0,
        "Slant active intern depth": False,
        "Slant chunk size": 1000,
        "Slant active use downsampling": True,
        "Slant active multiprocessing": True,
        "Slant num worker": 8,
        "Slant active export proc data": True,
        "Slant active export slant data": True,
        "View reprocess file": False,
        "Img chunk size": 1000,
        "Img include raw data": False,
        "Georef active proc data": True,
        "Georef active dynamic chunking": False,
        "Georef UTM": True,
        "Resolution Mode": int,
        "Warp Mode": int,
        "Resampling Method": int,
        "Georef active custom colormap": False,
        "Path": [],
        "Meta info": dict(),
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
        self.file_info_text_box = QTextEdit()
        self.dir_and_path_label = QLabel("Working directory and paths setup")
        self.dir_and_path_label.setFont(title_font)
        # Bottom line detection
        self.bottom_line_detection_widget = BottomLineDetectionWidget(self, title_font)
        self.bottom_line_detection_widget.data_changed.connect(self.update_table)
        # Data processing
        self.processing_widget = ProcessingWidget(self, title_font)
        self.processing_widget.data_changed.connect(self.update_table)
        # View and export
        self.view_and_export_widget = ViewAndExportWidget(self, title_font)
        ## --- Layouts
        self.left_view = QVBoxLayout()
        self.right_view = QVBoxLayout()
        self.right_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        # left side widgets: file picker and table
        self.left_view.addWidget(self.file_pick_btn)
        self.left_view.addWidget(self.file_table)
        # right side widgets: meta info, project settings and all parameter
        # TODO: Remove preliminary logo until we got a final one
        # self.right_view.addWidget(self.title_label)
        # self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.file_info_text_box)
        self.right_view.addWidget(QHLine())
        self.right_view.addWidget(self.dir_and_path_label)
        self.right_view.addLayout(self.output_picker)
        self.right_view.addLayout(self.georef_out_picker)
        self.right_view.addLayout(self.egn_table_picker)
        button_box = QHBoxLayout()
        button_box.addWidget(self.project_save_button)
        button_box.addWidget(self.project_load_button)
        self.right_view.addLayout(button_box)
        self.right_view.addWidget(QHLine())
        # Processing steps are ordered in tabs
        proc_tab = QTabWidget(self)
        tab_bottom = QWidget(self)
        tab_bottom.setLayout(self.bottom_line_detection_widget)
        proc_tab.addTab(tab_bottom, "Bottom Line Detection")
        tab_bottom = QWidget(self)
        tab_bottom.setLayout(self.processing_widget)
        proc_tab.addTab(tab_bottom, "Processing")
        tab_bottom = QWidget(self)
        tab_bottom.setLayout(self.view_and_export_widget)
        proc_tab.addTab(tab_bottom, "View and Export")
        self.right_view.addWidget(proc_tab)
        # right base widget
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
        # Base layout
        self.base_layout.addLayout(self.left_view)
        self.base_layout.addWidget(self.right_scroll_area)
        # Shortcuts
        self.shortcut_del = QShortcut(QKeySequence("Del"), self.file_table)
        self.shortcut_del.activated.connect(self.delete_file)
        # set all std/project dependent parameter
        self.update_ui_from_settings()

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
            # The import manager uses a thread to process new files
            import_manager = FileImportManager()
            # this signal is emitted containing the new results, when import is finished
            import_manager.results_ready.connect(
                lambda meta_info: self.import_new_files_from_manager(
                    filenames, meta_info
                )
            )
            # this signal is emitted if the import failes
            import_manager.aborted_signal.connect(
                lambda err_msg: self.show_import_error_msg(err_msg)
            )
            import_manager.start_import(filenames)

    def import_new_files_from_manager(self, filenames: list, meta_info_list: list):
        # append meta data to settings dict for saving/loading capabilities
        for new_info in meta_info_list:
            self.settings_dict["Meta info"].update(new_info)
        # check for duplicates and sort full list
        full_list = self.file_dict["Path"]
        full_list.extend(filenames)
        full_list = list(set(full_list))
        full_list.sort()
        # fill all the info for the file dict
        num_files = len(full_list)
        self.file_dict["Path"] = full_list
        self.file_dict["Bottom line"] = ["N"] * num_files
        self.file_dict["File size"] = ["0"] * num_files
        self.file_dict["Slant corrected"] = ["N"] * num_files
        self.file_dict["Gain corrected"] = ["N"] * num_files
        # update UI
        self.update_table()
        self.update_right_view_size()

    def show_import_error_msg(self, err_msg: str):
        dlg = ErrorWarnDialog(title="Error while importing files", message=err_msg)
        dlg.exec()

    def delete_file(self):
        idx_del = self.file_table.selectedIndexes()[0].row()
        self.file_dict["Path"].pop(idx_del)
        self.file_dict["Bottom line"].pop(idx_del)
        self.file_dict["File size"].pop(idx_del)
        self.file_dict["Slant corrected"].pop(idx_del)
        self.file_dict["Gain corrected"].pop(idx_del)

        self.update_table()
        self.update_right_view_size()

    def update_table(self):

        self.check_for_btm_line_data_and_size()

        # Build table
        selected_idx = -1
        if len(self.file_table.selectedIndexes()) > 0:
            selected_idx = self.file_table.selectedIndexes()[0].row()
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

            if self.file_dict["Gain corrected"][idx] == "Y":
                new_item = QTableWidgetItem(yes_icon, "")
            else:
                new_item = QTableWidgetItem(no_icon, "")
            new_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.file_table.setItem(idx, 3, new_item)
            new_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            new_item = QTableWidgetItem(self.file_dict["File size"][idx])
            self.file_table.setItem(idx, 4, new_item)

        # check whether a file is selected, if there is none, select index 0
        if len(self.file_table.selectedIndexes()) <= 0:
            if selected_idx == -1:
                selected_idx = 0
            self.file_table.selectRow(selected_idx)

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
                    self.file_dict["Gain corrected"][idx] = "Y"
                else:
                    self.file_dict["Gain corrected"][idx] = "N"
            else:
                self.file_dict["File size"][idx] = "Couldn't read"

    def update_meta_info(self):
        self.check_for_btm_line_data_and_size()
        file_path = self.file_dict["Path"][self.file_table.selectedIndexes()[0].row()]
        self.file_info_text_box.clear()
        self.file_info_text_box.insertHtml(self.settings_dict["Meta info"][file_path])

    def always_select_row(self):
        self.file_table.selectRow(self.file_table.selectedIndexes()[0].row())

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
        self.processing_widget.nadir_angle_edit.line_edit.setText(str(nadir_angle))
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

            # Check whether the dict contains the latest info
            if not "Meta info" in loaded_dict.keys():
                dlg = ErrorWarnDialog(
                    self,
                    title=f"Error while loading settings",
                    message=f"Sorry! The project settings have been written using an old Version of SidescanTools and can't be used anymore.\n You need to import your files again and save the new settings.",
                )
                dlg.exec()
                return

            full_list = loaded_dict["Path"]  # downward compatibility
            full_list.sort()
            num_files = len(full_list)
            self.file_dict["Path"] = full_list
            self.file_dict["Bottom line"] = ["N"] * num_files
            self.file_dict["Slant corrected"] = ["N"] * num_files
            self.file_dict["Gain corrected"] = ["N"] * num_files
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
            self.bottom_line_detection_widget.btm_chunk_size_edit.line_edit.text()
        )
        self.settings_dict["Btm def thresh"] = float(
            self.bottom_line_detection_widget.btm_default_thresh.line_edit.text()
        )
        self.settings_dict["Btm downsampling"] = int(
            self.bottom_line_detection_widget.btm_downsample_fact.line_edit.text()
        )
        self.settings_dict["Active convert dB"] = (
            self.bottom_line_detection_widget.active_convert_dB_checkbox.isChecked()
        )
        self.settings_dict["Btm equal hist"] = (
            self.bottom_line_detection_widget.active_hist_equal_checkbox.isChecked()
        )
        self.settings_dict["Active pie slice filter"] = (
            self.processing_widget.pie_slice_filter_checkbox.isChecked()
        )
        self.settings_dict["Active sharpening filter"] = (
            self.processing_widget.sharpening_filter_checkbox.isChecked()
        )
        self.settings_dict["Active gain norm"] = (
            self.processing_widget.active_gain_norm_checkbox.isChecked()
        )
        self.settings_dict["Slant vertical beam angle"] = int(
            self.processing_widget.vertical_beam_angle_edit.line_edit.text()
        )
        self.settings_dict["Slant nadir angle"] = int(
            self.processing_widget.nadir_angle_edit.line_edit.text()
        )
        self.settings_dict["Slant active intern depth"] = (
            self.processing_widget.active_intern_depth_checkbox.isChecked()
        )
        self.settings_dict["Slant chunk size"] = int(
            self.processing_widget.slant_chunk_size_edit.line_edit.text()
        )
        self.settings_dict["Slant active use downsampling"] = (
            self.processing_widget.active_bottom_detection_downsampling_checkbox.isChecked()
        )
        self.settings_dict["Slant active export proc data"] = (
            self.processing_widget.export_final_proc_checkbox.isChecked()
        )
        self.settings_dict["Slant active export slant data"] = (
            self.processing_widget.export_slant_correction_checkbox.isChecked()
        )
        self.settings_dict["View reprocess file"] = (
            self.view_and_export_widget.active_reprocess_file_checkbox.isChecked()
        )
        self.settings_dict["Active hist equal"] = (
            self.view_and_export_widget.hist_equal_checkbox.isChecked()
        )
        self.settings_dict["Img chunk size"] = (
            self.view_and_export_widget.img_chunk_size_edit.line_edit.text()
        )
        self.settings_dict["Img include raw data"] = (
            self.view_and_export_widget.include_raw_data_checkbox.isChecked()
        )
        self.settings_dict["Georef active proc data"] = (
            self.view_and_export_widget.active_use_proc_data_checkbox.isChecked()
        )
        self.settings_dict["Georef active dynamic chunking"] = (
            self.view_and_export_widget.active_dynamic_chunking_checkbox.isChecked()
        )
        self.settings_dict["Georef UTM"] = (
            self.view_and_export_widget.active_utm_checkbox.isChecked()
        )
        self.settings_dict["Georef active custom colormap"] = (
            self.view_and_export_widget.active_colormap_checkbox.isChecked()
        )

    def update_ui_from_settings(self):
        self.output_picker.update_dir(self.settings_dict["Working dir"])
        self.georef_out_picker.update_dir(self.settings_dict["Georef dir"])
        try:
            self.egn_table_picker.update_dir(self.settings_dict["EGN table path"])
        except:
            pass
        self.processing_widget.egn_table_name_edit.line_edit.setText(
            self.settings_dict["EGN table name"]
        )
        self.bottom_line_detection_widget.btm_chunk_size_edit.line_edit.setText(
            str(self.settings_dict["Btm chunk size"])
        )
        self.bottom_line_detection_widget.btm_default_thresh.line_edit.setText(
            str(self.settings_dict["Btm def thresh"])
        )
        self.bottom_line_detection_widget.btm_downsample_fact.line_edit.setText(
            str(self.settings_dict["Btm downsampling"])
        )
        self.bottom_line_detection_widget.active_convert_dB_checkbox.setChecked(
            self.settings_dict["Active convert dB"]
        )
        self.bottom_line_detection_widget.active_hist_equal_checkbox.setChecked(
            self.settings_dict["Btm equal hist"]
        )
        self.processing_widget.pie_slice_filter_checkbox.setChecked(
            self.settings_dict["Active pie slice filter"]
        )
        self.processing_widget.sharpening_filter_checkbox.setChecked(
            self.settings_dict["Active sharpening filter"]
        )
        self.processing_widget.active_gain_norm_checkbox.setChecked(
            self.settings_dict["Active gain norm"]
        )
        self.processing_widget.vertical_beam_angle_edit.line_edit.setText(
            str(self.settings_dict["Slant vertical beam angle"])
        )
        self.processing_widget.nadir_angle_edit.line_edit.setText(
            str(self.settings_dict["Slant nadir angle"])
        )
        self.processing_widget.active_intern_depth_checkbox.setChecked(
            self.settings_dict["Slant active intern depth"]
        )
        self.processing_widget.slant_chunk_size_edit.line_edit.setText(
            str(self.settings_dict["Slant chunk size"])
        )
        self.processing_widget.active_bottom_detection_downsampling_checkbox.setChecked(
            self.settings_dict["Slant active use downsampling"]
        )
        self.processing_widget.export_final_proc_checkbox.setChecked(
            self.settings_dict["Slant active export proc data"]
        )
        self.processing_widget.export_slant_correction_checkbox.setChecked(
            self.settings_dict["Slant active export slant data"]
        )
        self.view_and_export_widget.active_reprocess_file_checkbox.setChecked(
            self.settings_dict["View reprocess file"]
        )
        self.view_and_export_widget.hist_equal_checkbox.setChecked(
            self.settings_dict["Active hist equal"]
        )
        self.view_and_export_widget.img_chunk_size_edit.line_edit.setText(
            str(self.settings_dict["Img chunk size"])
        )
        self.view_and_export_widget.include_raw_data_checkbox.setChecked(
            self.settings_dict["Img include raw data"]
        )
        self.view_and_export_widget.active_use_proc_data_checkbox.setChecked(
            self.settings_dict["Georef active proc data"]
        )
        self.view_and_export_widget.active_dynamic_chunking_checkbox.setChecked(
            self.settings_dict["Georef active dynamic chunking"]
        )
        self.view_and_export_widget.active_utm_checkbox.setChecked(
            self.settings_dict["Georef UTM"]
        )
        self.view_and_export_widget.active_colormap_checkbox.setChecked(
            self.settings_dict["Georef active custom colormap"]
        )
        self.processing_widget.load_proc_strat()


# Bottom line detection widget
class BottomLineDetectionWidget(QVBoxLayout):
    data_changed = QtCore.Signal()
    """Signal to show that there might be new bottom line detection data present"""

    def __init__(self, parent: SidescanToolsMain, title_font: QtGui.QFont):
        super().__init__()
        self.main_ui = parent
        # define widgets
        self.btm_label = QLabel("Bottom Line Detection")
        self.btm_label.setFont(title_font)
        self.btm_chunk_size_edit = LabeledLineEdit(
            "Chunk Size:",
            QtGui.QIntValidator(100, 9999, self),
            self.main_ui.settings_dict["Btm chunk size"],
        )
        self.btm_chunk_size_edit.line_edit.editingFinished.connect(
            self.validate_chunk_size
        )
        self.btm_chunk_size_edit.label.setToolTip(
            "Number of pings in one chunk for bottom detection."
        )
        self.btm_default_thresh = LabeledLineEdit(
            "Default Threshold [0.0 - 1.0]:",
            QtGui.QDoubleValidator(0.0, 1.0, 2, self),
            self.main_ui.settings_dict["Btm def thresh"],
        )
        self.btm_default_thresh.label.setToolTip(
            "Threshold that is applied to normalised data to find edges between water and ground. Needs to be in range[ 0 - 1]."
        )
        self.btm_downsample_fact = LabeledLineEdit(
            "Downsampling Factor:",
            QtGui.QIntValidator(1, 16, self),
            self.main_ui.settings_dict["Btm downsampling"],
        )
        self.btm_downsample_fact.label.setToolTip(
            "Integer decimation factor that is used to downsample each ping."
        )
        self.active_convert_dB_checkbox = QCheckBox("Convert to dB")
        self.active_convert_dB_checkbox.setToolTip(
            "Convert data to decibel for display (this is usually a good practice)."
        )
        self.active_hist_equal_checkbox = QCheckBox(
            "Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"
        )
        self.active_convert_dB_checkbox.stateChanged.connect(self.db_checkbox_changed)
        self.do_btm_detection_btn = QPushButton("Bottom Line Detection")
        self.do_btm_detection_btn.setToolTip("Start Bottom Line Detection")
        self.do_btm_detection_btn.clicked.connect(self.run_bottom_line_detection)
        # add widgets to layout
        self.addWidget(self.btm_label)
        self.addLayout(self.btm_chunk_size_edit)
        self.addLayout(self.btm_default_thresh)
        self.addLayout(self.btm_downsample_fact)
        self.addWidget(self.active_convert_dB_checkbox)
        self.addWidget(self.active_hist_equal_checkbox)
        self.addWidget(self.do_btm_detection_btn)
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.addItem(verticalSpacer)

    def db_checkbox_changed(self):
        # also update bottom line checkbox
        self.main_ui.view_and_export_widget.active_convert_dB_checkbox.setChecked(
            self.active_convert_dB_checkbox.isChecked()
        )

    def run_bottom_line_detection(self):
        file_idx = 0
        if len(self.main_ui.file_table.selectedIndexes()) > 0:
            file_idx = self.main_ui.file_table.selectedIndexes()[0].row()
        run_napari_btm_line(
            self.main_ui.file_dict["Path"][file_idx],
            chunk_size=int(self.btm_chunk_size_edit.line_edit.text()),
            default_threshold=float(self.btm_default_thresh.line_edit.text()),
            downsampling_factor=int(self.btm_downsample_fact.line_edit.text()),
            work_dir=self.main_ui.output_picker.cur_dir,
            active_dB=self.active_convert_dB_checkbox.isChecked(),
            active_hist_equal=self.active_hist_equal_checkbox.isChecked(),
        )
        self.data_changed.emit()

    def validate_chunk_size(self):
        val = int(self.btm_chunk_size_edit.line_edit.text())
        if val % 2 == 1:
            self.btm_chunk_size_edit.line_edit.setText(str(val - 1))


# Processing widget
class ProcessingWidget(QVBoxLayout):
    data_changed = QtCore.Signal()
    """Signal to show that there might be new preprocessed data present"""

    def __init__(self, parent: SidescanToolsMain, title_font: QtGui.QFont):
        super().__init__()
        self.main_ui = parent
        # define widgets
        self.filter_label = QLabel("Noise Reduction and Sharpening Filter")
        self.filter_label.setFont(title_font)
        self.pie_slice_filter_checkbox = QCheckBox("Filter Stripe Noise (experimental)")
        self.pie_slice_filter_checkbox.setToolTip(
            "Use 2D FFT Pie Slice Filter to remove stripe noise from data."
        )
        self.sharpening_filter_checkbox = QCheckBox(
            "Apply Sharpening Filter (experimental)"
        )
        self.sharpening_filter_checkbox.setToolTip(
            "Use homomorphic filter to sharpen the resulting images."
        )
        self.slant_and_gain_label = QLabel(
            "Slant Range Correction and Gain Normalisation"
        )
        self.slant_and_gain_label.setFont(title_font)
        self.active_gain_norm_checkbox = QCheckBox("Apply Gain Normalisation")
        self.radio_grp_label = QLabel("Gain Normalisation Strategy:")
        self.radio_grp_label.setToolTip(
            "Decide which Gain Normalisation Strategy shall be used: \n"
            "  - Beam Angle Correction: Estimates Beam Pattern from current file and applies normalisation. Works with single files.\n"
            "  - Empirical Gain Normalisation: Estimates a more precise Beam Pattern using all files of the current which is saved as EGN table. Does only work, when enough data is present."
        )
        self.gain_norm_radio_group = QButtonGroup()
        self.beam_ang_corr_radio_btn = QRadioButton(
            "Beam Angle Correction (BAC, works on single file)"
        )
        self.beam_ang_corr_radio_btn.setToolTip(
            "Decide which Gain Normalisation Strategy shall be used: \n"
            "  - Beam Angle Correction: Estimates Beam Pattern from current file and applies normalisation. Works with single files.\n"
            "  - Empirical Gain Normalisation: Estimates a more precise Beam Pattern using all files of the current which is saved as EGN table. Does only work, when enough data is present."
        )
        self.egn_radio_btn = QRadioButton(
            "Empirical Gain Normalisation (EGN, needs precalculated table)"
        )
        self.egn_radio_btn.setToolTip(
            "Decide which Gain Normalisation Strategy shall be used: \n"
            "  - Beam Angle Correction: Estimates Beam Pattern from current file and applies normalisation. Works with single files.\n"
            "  - Empirical Gain Normalisation: Estimates a more precise Beam Pattern using all files of the current which is saved as EGN table. Does only work, when enough data is present."
        )
        self.gain_norm_radio_group.addButton(self.beam_ang_corr_radio_btn)
        self.gain_norm_radio_group.addButton(self.egn_radio_btn)
        self.gain_norm_radio_group.buttonClicked.connect(self.proc_strat_changed)
        self.load_proc_strat()
        self.vertical_beam_angle_edit = LabeledLineEdit(
            "Vertical Beam Angle:",
            QtGui.QIntValidator(0, 90, self),
            self.main_ui.settings_dict["Slant vertical beam angle"],
        )
        self.vertical_beam_angle_edit.label.setToolTip(
            "Only relevant if internal depth is unknown: Horizontal angle by which the instrument is tilted (usually found in the manual)."
        )
        self.nadir_angle_edit = LabeledLineEdit(
            "Nadir Angle:",
            QtGui.QIntValidator(0, 90, self),
            self.main_ui.settings_dict["Slant nadir angle"],
        )
        self.nadir_angle_edit.label.setToolTip(
            "Angle between perpendicular and first bottom return (usually unknown, default is 0°)"
        )
        self.optional_egn_label = QLabel("Advanced Gain Normalisation Parameter")
        self.optional_egn_label.setFont(title_font)
        self.active_intern_depth_checkbox = QCheckBox("Use internal Depth")
        self.active_intern_depth_checkbox.setToolTip(
            "Use internal depth information for slant range correction. Otherwise depth is estimated from detected bottom line."
        )
        self.active_intern_depth_checkbox.setChecked(
            self.main_ui.settings_dict["Slant active intern depth"]
        )
        self.slant_chunk_size_edit = LabeledLineEdit(
            "Chunk Size:",
            QtGui.QIntValidator(100, 9999, self),
            self.main_ui.settings_dict["Slant chunk size"],
        )
        self.slant_chunk_size_edit.line_edit.editingFinished.connect(
            self.validate_chunk_size
        )
        self.slant_chunk_size_edit.label.setToolTip(
            "Number of pings in one chunk for for slant range and EGN correction. Is also used to determine the size of the exported waterfall images."
        )
        self.active_bottom_detection_downsampling_checkbox = QCheckBox(
            "Apply Downsampling"
        )
        self.active_bottom_detection_downsampling_checkbox.setToolTip(
            "Use downsampling factor from bottom line detection to do processing on downsampled data."
        )
        self.active_bottom_detection_downsampling_checkbox.setChecked(
            self.main_ui.settings_dict["Slant active use downsampling"]
        )
        self.egn_table_name_edit = LabeledLineEdit(
            "EGN Table Name:",
            validator=None,
            start_val=self.main_ui.settings_dict["EGN table name"],
        )
        self.egn_table_name_edit.label.setToolTip(
            "Set name of EGN Table that is written as .npz file."
        )
        self.export_slant_correction_checkbox = QCheckBox(
            "Export Slant Range corrected Data"
        )
        self.export_slant_correction_checkbox.setToolTip(
            "Export Slant Range corrected data as .npz file. So it doesn't need to be recalculated for export or viewing."
        )
        self.export_final_proc_checkbox = QCheckBox("Export fully processed Data")
        self.export_final_proc_checkbox.setToolTip(
            "Export fully processed corrected data as .npz file. So it doesn't need to be recalculated for export or viewing."
        )
        self.generate_egn_table = QPushButton("Generate EGN Table")
        self.generate_egn_table.clicked.connect(self.run_generate_slant_and_egn_files)
        self.process_single_btn = QPushButton("Process selected file")
        self.process_single_btn.clicked.connect(self.process_single_file)
        self.process_all_btn = QPushButton("Process All Files")
        self.process_all_btn.clicked.connect(self.process_all_files)
        # Layout
        self.addWidget(self.filter_label)
        self.addWidget(self.pie_slice_filter_checkbox)
        self.addWidget(self.sharpening_filter_checkbox)
        self.addWidget(QHLine())
        self.addWidget(self.slant_and_gain_label)
        self.addWidget(self.active_bottom_detection_downsampling_checkbox)
        self.addWidget(self.active_gain_norm_checkbox)
        self.addWidget(self.radio_grp_label)
        radio_layout = QHBoxLayout()
        radio_layout_btns = QVBoxLayout()
        radio_layout_btns.addWidget(self.beam_ang_corr_radio_btn)
        radio_layout_btns.addWidget(self.egn_radio_btn)
        radio_layout.addItem(
            QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        )
        radio_layout.addLayout(radio_layout_btns)
        self.addLayout(radio_layout)
        self.addWidget(self.active_intern_depth_checkbox)
        self.addLayout(self.vertical_beam_angle_edit)
        self.addWidget(QHLine())
        self.addWidget(self.optional_egn_label)
        self.addLayout(self.nadir_angle_edit)
        self.addLayout(self.slant_chunk_size_edit)
        self.addWidget(self.export_slant_correction_checkbox)
        self.addWidget(self.export_final_proc_checkbox)
        self.addWidget(QHLine())
        self.addLayout(self.egn_table_name_edit)
        self.addWidget(self.generate_egn_table)
        proc_btn_layout = QHBoxLayout()
        proc_btn_layout.addWidget(self.process_single_btn)
        proc_btn_layout.addWidget(self.process_all_btn)
        self.addLayout(proc_btn_layout)
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.addItem(verticalSpacer)

    def run_generate_slant_and_egn_files(self):
        # check if EGN file exist
        egn_path = pathlib.Path(
            self.main_ui.settings_dict["Working dir"]
            + "/"
            + self.egn_table_name_edit.line_edit.text()
        )
        if egn_path.exists():
            dlg = OverwriteWarnDialog(None, str(egn_path))
            if not dlg.exec():
                return

        sonar_file_path_list = []
        for sonar_file in self.main_ui.file_dict["Path"]:
            sonar_file_path_list.append(pathlib.Path(sonar_file))
        egn_table_builder = EGNTableBuilder(egn_path)
        egn_table_builder.table_finished.connect(lambda: self.data_changed.emit())
        egn_table_builder.aborted_signal.connect(
            lambda err_msg: self.show_error_msg(err_msg)
        )
        egn_table_builder.build_egn_table(
            sonar_file_path_list,
            self.main_ui.settings_dict["Working dir"],
            int(self.nadir_angle_edit.line_edit.text()),
            self.active_intern_depth_checkbox.isChecked(),
            int(self.slant_chunk_size_edit.line_edit.text()),
            self.active_bottom_detection_downsampling_checkbox.isChecked(),
        )

    def show_error_msg(self, msg):
        dlg = ErrorWarnDialog(title="Error while processing files", message=msg)
        dlg.exec()

    def process_all_files(self):
        path_list = []
        for idx, path in enumerate(self.main_ui.file_dict["Path"]):
            if self.main_ui.file_dict["Bottom line"][idx] == "Y":
                path_list.append(pathlib.Path(path))

        self.start_intern_processing_manager(path_list)

    def process_single_file(self):
        if len(self.main_ui.file_table.selectedIndexes()) > 0:
            filepath = pathlib.Path(
                self.main_ui.file_dict["Path"][
                    self.main_ui.file_table.selectedIndexes()[0].row()
                ]
            )
        else:
            filepath = pathlib.Path(self.main_ui.file_dict["Path"][0])

        self.start_intern_processing_manager([filepath])

    def start_intern_processing_manager(self, files: list):
        work_dir = pathlib.Path(self.main_ui.settings_dict["Working dir"])
        pre_proc_mng = PreProcManager()
        pre_proc_mng.processing_finished.connect(lambda: self.data_changed.emit())
        pre_proc_mng.aborted_signal.connect(
            lambda err_msg: self.show_error_msg(err_msg)
        )
        pre_proc_mng.proc_files(
            files=files,
            work_dir=work_dir,
            egn_table_path=self.main_ui.egn_table_picker.cur_dir,
            chunk_size=int(self.slant_chunk_size_edit.line_edit.text()),
            nadir_angle=int(self.nadir_angle_edit.line_edit.text()),
            active_export_slant_corr_mat=self.export_slant_correction_checkbox.isChecked(),
            active_export_gain_corr_mat=self.export_final_proc_checkbox.isChecked(),
            load_slant_data=False,
            load_gain_data=False,
            active_pie_slice_filter=self.pie_slice_filter_checkbox.isChecked(),
            active_gain_norm=self.active_gain_norm_checkbox.isChecked(),
            active_egn=self.egn_radio_btn.isChecked(),
            active_bac=self.beam_ang_corr_radio_btn.isChecked(),
            active_sharpening_filter=self.sharpening_filter_checkbox.isChecked(),
        )

    def proc_strat_changed(self, btn_object):
        if self.beam_ang_corr_radio_btn.isChecked():
            self.main_ui.settings_dict["Slant gain norm strategy"] = GAINSTRAT.BAC.value
        elif self.egn_radio_btn.isChecked():
            self.main_ui.settings_dict["Slant gain norm strategy"] = GAINSTRAT.EGN.value

    def load_proc_strat(self):
        if (
            self.main_ui.settings_dict["Slant gain norm strategy"]
            == GAINSTRAT.BAC.value
        ):
            self.beam_ang_corr_radio_btn.setChecked(True)
        elif (
            self.main_ui.settings_dict["Slant gain norm strategy"]
            == GAINSTRAT.EGN.value
        ):
            self.egn_radio_btn.setChecked(True)

    def validate_chunk_size(self):
        val = int(self.slant_chunk_size_edit.line_edit.text())
        if val % 2 == 1:
            self.slant_chunk_size_edit.line_edit.setText(str(val - 1))


# View and export
class ViewAndExportWidget(QVBoxLayout):
    data_changed = QtCore.Signal()
    """Signal to show that there might be new preprocessed data present"""
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

    def __init__(self, parent: SidescanToolsMain, title_font: QtGui.QFont):
        super().__init__()
        self.main_ui = parent
        # define widgets
        self.napari_label = QLabel("View Results")
        self.napari_label.setFont(title_font)
        self.active_reprocess_file_checkbox = QCheckBox("Reprocess file")
        self.active_reprocess_file_checkbox.setToolTip(
            "Do the slant range and EGN correction processing of the selected file before viewing."
        )
        self.active_convert_dB_checkbox = QCheckBox("Convert to dB")
        self.active_convert_dB_checkbox.setToolTip(
            "Convert data to decibel for display (this is usually a good practice)."
        )
        self.active_convert_dB_checkbox.stateChanged.connect(self.db_checkbox_changed)
        self.hist_equal_checkbox = QCheckBox(
            "Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"
        )
        self.hist_equal_checkbox.setToolTip(
            "Use adaptive histogram equalization to improve contrast of the resulting images."
        )
        self.show_proc_file_btn = QPushButton("View Processed Data")
        self.show_proc_file_btn.clicked.connect(self.show_proc_file_in_napari)

        self.img_chunk_size_edit = LabeledLineEdit(
            "Chunk Size:",
            QtGui.QIntValidator(100, 9999, self),
            self.main_ui.settings_dict["Img chunk size"],
        )
        self.img_chunk_size_edit.line_edit.editingFinished.connect(
            self.validate_chunk_size
        )
        self.img_chunk_size_edit.label.setToolTip(
            "Number of pings in one chunk of the exported waterfall images."
        )

        self.georef_label = QLabel("Georeferencing and Image Generation")
        self.georef_label.setFont(title_font)
        self.active_use_proc_data_checkbox = QCheckBox("Use processed Data")
        self.active_use_proc_data_checkbox.setToolTip(
            "Export pictures using the processed (filtered and corrected) data. Otherwise raw data is exported."
        )
        self.res_mode_label = QLabel("Resolution Mode")
        self.res_mode_label.setToolTip("Set mode for final resolution. Leave average, if unsure.")
        self.warp_mode_label = QLabel("Warp Method")
        self.warp_mode_label.setToolTip("Set method for warping algorithm. Leave polynomial 1 if unsure, homography is in expermental state.")
        self.resamp_mode_label = QLabel("Resampling Method")
        self.resamp_mode_label.setToolTip("Select resampling method. Leave near neighbour, if unsure (least interpolation).")

        self.resolution_mode_dropdown = QComboBox()
        self.resolution_mode_dropdown.setToolTip("Set mode for final resolution. Chose average, if unsure.")
        for res_disp, res_int in SidescanGeoreferencer.resolution_options.items():
            self.resolution_mode_dropdown.addItem(res_disp, res_int)
        self.resolution_mode_dropdown.setCurrentIndex(3)
        self.resolution_mode_dropdown.currentIndexChanged.connect(self.change_res_mode)

        self.warp_mode_dropdown = QComboBox()
        self.warp_mode_dropdown.setToolTip("Set method for warping algorithm. Leave polynomial 1 if unsure, homography is in expermental state.")
        for warp_disp, warp_int in SidescanGeoreferencer.warp_options.items():
            self.warp_mode_dropdown.addItem(warp_disp, warp_int)
        self.warp_mode_dropdown.setCurrentIndex(0)
        self.warp_mode_dropdown.currentIndexChanged.connect(self.change_warp_mode)

        self.resamp_mode_dropdown = QComboBox()
        self.resamp_mode_dropdown.setToolTip("Select resampling method. Leave near neighbour, if unsure (least interpolation).")
        for resamp_disp, resamp_int in SidescanGeoreferencer.resampling_options.items():
            self.resamp_mode_dropdown.addItem(resamp_disp, resamp_int)
        self.resamp_mode_dropdown.setCurrentIndex(0)
        self.resamp_mode_dropdown.currentIndexChanged.connect(self.change_warp_mode)

        self.active_dynamic_chunking_checkbox = QCheckBox("Dynamic Chunking")
        self.active_dynamic_chunking_checkbox.setToolTip("Experimental")
        self.active_utm_checkbox = QCheckBox("UTM")
        self.active_utm_checkbox.setToolTip(
            "Coordinates in UTM (default). WGS84 if unchecked."
        )

        self.active_colormap_checkbox = QCheckBox("Apply custom Colormap")
        self.active_colormap_checkbox.setToolTip(
            "Applies the colormap used in napari to the exported waterfall images. Otherwise grey scale values are used."
        )
        self.generate_single_georef_btn = QPushButton("Selected")
        self.generate_single_georef_btn.clicked.connect(self.run_sidescan_georef)
        self.generate_all_georef_btn = QPushButton("All")
        self.generate_all_georef_btn.clicked.connect(
            lambda: self.run_sidescan_georef(True)
        )
        self.include_raw_data_checkbox = QCheckBox(
            "Include raw data in Waterfall Image"
        )
        self.generate_simple_img_btn = QPushButton("Selected")
        self.generate_simple_img_btn.clicked.connect(
            lambda: self.generate_wc_img(False)
        )
        self.generate_all_simple_img_btn = QPushButton("All")
        self.generate_all_simple_img_btn.clicked.connect(
            lambda: self.generate_wc_img(True)
        )
        # layout
        self.addWidget(self.napari_label)
        self.addWidget(self.active_reprocess_file_checkbox)
        self.addWidget(self.active_convert_dB_checkbox)
        self.addWidget(self.hist_equal_checkbox)
        self.addWidget(self.show_proc_file_btn)
        self.addWidget(QHLine())
        self.addWidget(self.georef_label)
        self.addWidget(self.active_use_proc_data_checkbox)
        self.addWidget(self.res_mode_label)
        self.addWidget(self.resolution_mode_dropdown)
        self.addWidget(self.warp_mode_label)
        self.addWidget(self.warp_mode_dropdown)
        self.addWidget(self.resamp_mode_label)
        self.addWidget(self.resamp_mode_dropdown)
        self.addWidget(self.active_dynamic_chunking_checkbox)
        self.addWidget(self.active_utm_checkbox)
        self.addWidget(self.active_colormap_checkbox)
        self.labeled_georef_buttons = Labeled2Buttons(
            "Generate Geotiff:",
            self.generate_single_georef_btn,
            self.generate_all_georef_btn,
        )
        self.addLayout(self.labeled_georef_buttons)
        self.addLayout(self.img_chunk_size_edit)
        self.addWidget(self.include_raw_data_checkbox)
        self.labeled_img_export_buttons = Labeled2Buttons(
            "Generate Waterfall Image:",
            self.generate_simple_img_btn,
            self.generate_all_simple_img_btn,
        )
        self.addLayout(self.labeled_img_export_buttons)
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.addItem(verticalSpacer)

    def db_checkbox_changed(self):
        # also update bottom line checkbox
        self.main_ui.bottom_line_detection_widget.active_convert_dB_checkbox.setChecked(
            self.active_convert_dB_checkbox.isChecked()
        )

    def change_res_mode(self, index):
        selected_res_mode = self.resolution_mode_dropdown.currentText()
        self.main_ui.settings_dict["Resolution Mode"] = selected_res_mode

    def change_warp_mode(self):
        selected_warp_mode = self.warp_mode_dropdown.currentText()
        self.main_ui.settings_dict["Warp Mode"] = selected_warp_mode

    def change_resampling_method(self):
        selected_resamp_mode = self.resamp_mode_dropdown.currentText()
        self.main_ui.settings_dict["Resampling Method"] = selected_resamp_mode   

    def show_proc_file_in_napari(self):
        file_idx = 0
        if len(self.main_ui.file_table.selectedIndexes()) > 0:
            file_idx = self.main_ui.file_table.selectedIndexes()[0].row()
        filepath = pathlib.Path(self.main_ui.file_dict["Path"][file_idx])
        load_slant = self.main_ui.file_dict["Slant corrected"][file_idx] == "Y"
        load_egn = self.main_ui.file_dict["Gain corrected"][file_idx] == "Y"
        if self.active_reprocess_file_checkbox.isChecked():
            load_egn = False
            load_slant = False
        # load data or start processing and trigger napari when data is present
        pre_proc_mng = PreProcManager()
        work_dir = pathlib.Path(self.main_ui.settings_dict["Working dir"])
        pre_proc_mng.processing_finished.connect(
            lambda res_list: self.preproc_to_run_napari(res_list)
        )
        pre_proc_mng.aborted_signal.connect(
            lambda err_msg: self.main_ui.processing_widget.show_error_msg(err_msg)
        )
        pre_proc_mng.proc_files(
            files=[filepath],
            work_dir=work_dir,
            egn_table_path=self.main_ui.egn_table_picker.cur_dir,
            chunk_size=int(
                self.main_ui.processing_widget.slant_chunk_size_edit.line_edit.text()
            ),
            nadir_angle=int(
                self.main_ui.processing_widget.nadir_angle_edit.line_edit.text()
            ),
            active_export_slant_corr_mat=self.main_ui.processing_widget.export_slant_correction_checkbox.isChecked(),
            active_export_gain_corr_mat=self.main_ui.processing_widget.export_final_proc_checkbox.isChecked(),
            load_slant_data=load_slant,
            load_gain_data=load_egn,
            active_pie_slice_filter=self.main_ui.processing_widget.pie_slice_filter_checkbox.isChecked(),
            active_gain_norm=self.main_ui.processing_widget.active_gain_norm_checkbox.isChecked(),
            active_egn=self.main_ui.processing_widget.egn_radio_btn.isChecked(),
            active_bac=self.main_ui.processing_widget.beam_ang_corr_radio_btn.isChecked(),
            active_sharpening_filter=self.main_ui.processing_widget.sharpening_filter_checkbox.isChecked(),
        )

    def preproc_to_run_napari(self, res: list):
        sidescan_file = res[0]
        preproc = res[1]
        data_ch0 = sidescan_file.data[0]
        data_ch1 = sidescan_file.data[1]
        # decimate raw data for plotting in napari
        if preproc.downsampling_factor != 1:
            data_ch0 = scisig.decimate(data_ch0, preproc.downsampling_factor, axis=1)
            data_ch1 = scisig.decimate(data_ch1, preproc.downsampling_factor, axis=1)
        raw_image = np.hstack((data_ch0, data_ch1))
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

        # calculate in dB and do histogram equalization
        raw_image_chunk_plot = copy.copy(raw_image_chunk)
        slant_corr_chunk_plot = copy.copy(slant_corr_chunk)
        egn_corr_chunk_plot = copy.copy(egn_corr_chunk)
        if self.active_convert_dB_checkbox.isChecked():
            raw_image_chunk_plot = convert_to_dB(raw_image_chunk_plot)
            slant_corr_chunk_plot = convert_to_dB(slant_corr_chunk_plot)
            egn_corr_chunk_plot = convert_to_dB(egn_corr_chunk_plot)
        if self.hist_equal_checkbox.isChecked():
            egn_corr_chunk_plot = hist_equalization(egn_corr_chunk_plot)
        viewer, image_layer_1 = napari.imshow(
            raw_image_chunk_plot, colormap=self.sonar_dat_cmap, name="Raw image"
        )
        image_layer_2 = viewer.add_image(preproc.bottom_map, colormap=bottom_colormap)
        image_layer_3 = viewer.add_image(
            slant_corr_chunk_plot,
            colormap=self.sonar_dat_cmap,
            name="Slant corrected image",
        )
        image_layer_4 = viewer.add_image(
            egn_corr_chunk_plot,
            colormap=self.sonar_dat_cmap,
            name="Gain corrected image",
        )
        napari.run(max_loop_level=2)

    def run_sidescan_georef(self, active_all_files=False):

        if active_all_files:
            file_list = self.main_ui.file_dict["Path"]
        else:
            file_idx = 0
            if len(self.main_ui.file_table.selectedIndexes()) > 0:
                file_idx = self.main_ui.file_table.selectedIndexes()[0].row()
            file_list = [pathlib.Path(self.main_ui.file_dict["Path"][file_idx])]

        work_dir = pathlib.Path(self.main_ui.settings_dict["Working dir"])
        for filepath in file_list:
            filepath = pathlib.Path(filepath)
            load_slant_data = False
            load_egn_data = False
            # check wheter preproc data is present and load or process file
            filepath = pathlib.Path(filepath)
            if (work_dir / (filepath.stem + "_slant_corrected.npz")).exists():
                load_slant_data = True
            if (work_dir / (filepath.stem + "_egn_corrected.npz")).exists():
                load_egn_data = True

        pre_proc_mng = PreProcManager()
        pre_proc_mng.new_res_present.connect(
            lambda res_list: self.start_georeferencer(res_list)
        )
        pre_proc_mng.aborted_signal.connect(
            lambda err_msg: self.main_ui.processing_widget.show_error_msg(err_msg)
        )
        pre_proc_mng.proc_files(
            files=file_list,
            work_dir=work_dir,
            egn_table_path=self.main_ui.egn_table_picker.cur_dir,
            chunk_size=int(
                self.main_ui.processing_widget.slant_chunk_size_edit.line_edit.text()
            ),
            nadir_angle=int(
                self.main_ui.processing_widget.nadir_angle_edit.line_edit.text()
            ),
            active_export_slant_corr_mat=self.main_ui.processing_widget.export_slant_correction_checkbox.isChecked(),
            active_export_gain_corr_mat=self.main_ui.processing_widget.export_final_proc_checkbox.isChecked(),
            load_slant_data=load_slant_data,
            load_gain_data=load_egn_data,
            active_pie_slice_filter=self.main_ui.processing_widget.pie_slice_filter_checkbox.isChecked(),
            active_gain_norm=self.main_ui.processing_widget.active_gain_norm_checkbox.isChecked(),
            active_egn=self.main_ui.processing_widget.egn_radio_btn.isChecked(),
            active_bac=self.main_ui.processing_widget.beam_ang_corr_radio_btn.isChecked(),
            active_sharpening_filter=self.main_ui.processing_widget.sharpening_filter_checkbox.isChecked(),
        )

    def start_georeferencer(self, res_list: list):
        sidescan_file = res_list[0]
        preproc = res_list[1]
        filepath = sidescan_file.filepath
        proc_data_0 = None
        proc_data_1 = None
        if self.active_use_proc_data_checkbox.isChecked():
            ping_len = int(np.shape(preproc.egn_corrected_mat)[1]/2)
            proc_data_0 = preproc.egn_corrected_mat[:, 0 : ping_len]
            proc_data_0 = np.nan_to_num(
                proc_data_0
            )  # remove nans from excluding far/nadir unknown values
            proc_data_1 = preproc.egn_corrected_mat[:, ping_len :]
            proc_data_1 = np.nan_to_num(proc_data_1)

        proc_data_out_0 = copy.copy(proc_data_0)
        proc_data_out_1 = copy.copy(proc_data_1)
        if self.active_convert_dB_checkbox.isChecked():
            proc_data_out_0 = convert_to_dB(proc_data_out_0)
            proc_data_out_1 = convert_to_dB(proc_data_out_1)
        if self.hist_equal_checkbox.isChecked():
            proc_data_out_0 = hist_equalization(proc_data_out_0)
            proc_data_out_1 = hist_equalization(proc_data_out_1)

        # start georeferencing
        georeferencer = SidescanGeoreferencer(
            filepath=filepath,
            channel=0,
            dynamic_chunking=self.active_dynamic_chunking_checkbox.isChecked(),
            active_utm=self.active_utm_checkbox.isChecked(),
            output_folder=self.main_ui.settings_dict["Georef dir"],
            proc_data=proc_data_out_0,
            vertical_beam_angle=int(
                self.main_ui.processing_widget.vertical_beam_angle_edit.line_edit.text()
            ),
            resolution_mode = self.resolution_mode_dropdown.currentData(),
            warp_algorithm = self.warp_mode_dropdown.currentData(),
            resampling_method = self.resamp_mode_dropdown.currentData(),
        )
        georeferencer.process()
        georeferencer = SidescanGeoreferencer(
            filepath=filepath,
            channel=1,
            dynamic_chunking=self.active_dynamic_chunking_checkbox.isChecked(),
            active_utm=self.active_utm_checkbox.isChecked(),
            output_folder=self.main_ui.settings_dict["Georef dir"],
            proc_data=proc_data_out_1,
            vertical_beam_angle=int(
                self.main_ui.processing_widget.vertical_beam_angle_edit.line_edit.text()
            ),
            resolution_mode = self.resolution_mode_dropdown.currentData(),
            warp_algorithm = self.warp_mode_dropdown.currentData(),
            resampling_method = self.resamp_mode_dropdown.currentData(),

        )
        georeferencer.process()

    def generate_wc_img(self, active_generate_all: bool):
        if len(self.main_ui.file_table.selectedIndexes()) > 0:
            filepath = pathlib.Path(
                self.main_ui.file_dict["Path"][
                    self.main_ui.file_table.selectedIndexes()[0].row()
                ]
            )
        else:
            filepath = pathlib.Path(self.main_ui.file_dict["Path"][0])
        if active_generate_all:
            pathlist = []
            for path_idx in range(len(self.main_ui.file_dict["Path"])):
                if self.main_ui.file_dict["Gain corrected"][path_idx] == "Y":
                    pathlist.append(
                        pathlib.Path(self.main_ui.file_dict["Path"][path_idx])
                    )
        else:
            pathlist = [filepath]

        work_dir = pathlib.Path(self.main_ui.settings_dict["Working dir"])

        load_slant_data = True
        load_egn_data = True
        pre_proc_mng = PreProcManager()
        pre_proc_mng.new_res_present.connect(
            lambda res_list: self.start_wc_image_export(res_list)
        )
        pre_proc_mng.aborted_signal.connect(
            lambda err_msg: self.main_ui.processing_widget.show_error_msg(err_msg)
        )
        pre_proc_mng.proc_files(
            files=pathlist,
            work_dir=work_dir,
            egn_table_path=self.main_ui.egn_table_picker.cur_dir,
            chunk_size=int(
                self.main_ui.processing_widget.slant_chunk_size_edit.line_edit.text()
            ),
            nadir_angle=int(
                self.main_ui.processing_widget.nadir_angle_edit.line_edit.text()
            ),
            active_export_slant_corr_mat=self.main_ui.processing_widget.export_slant_correction_checkbox.isChecked(),
            active_export_gain_corr_mat=self.main_ui.processing_widget.export_final_proc_checkbox.isChecked(),
            load_slant_data=load_slant_data,
            load_gain_data=load_egn_data,
            active_pie_slice_filter=self.main_ui.processing_widget.pie_slice_filter_checkbox.isChecked(),
            active_gain_norm=self.main_ui.processing_widget.active_gain_norm_checkbox.isChecked(),
            active_egn=self.main_ui.processing_widget.egn_radio_btn.isChecked(),
            active_bac=self.main_ui.processing_widget.beam_ang_corr_radio_btn.isChecked(),
            active_sharpening_filter=self.main_ui.processing_widget.sharpening_filter_checkbox.isChecked(),
        )

    def start_wc_image_export(self, res_list: list):
        # TODO: this is quite custom for the GNB project, do we want to alter this?
        active_add_raw_img = self.include_raw_data_checkbox.isChecked()
        active_chunkify = True
        active_norm_chunks = False
        # get needed data
        sidescan_file = res_list[0]
        preproc = res_list[1]
        chunk_size = int(self.img_chunk_size_edit.line_edit.text())
        work_dir = pathlib.Path(self.main_ui.settings_dict["Working dir"])
        data = copy.copy(preproc.egn_corrected_mat)
        if self.active_convert_dB_checkbox.isChecked():
            data = convert_to_dB(data)
        if self.hist_equal_checkbox.isChecked():
            data = hist_equalization(data)
        np.nan_to_num(data, copy=False)
        data /= np.nanmax(np.abs(data)) / 255
        data = np.array(data, dtype=np.uint8)
        if active_add_raw_img:
            data_ch0 = sidescan_file.data[0]
            data_ch1 = sidescan_file.data[1]
            # decimate raw data for plotting in napari
            if preproc.downsampling_factor != 1:
                data_ch0 = scisig.decimate(
                    data_ch0, preproc.downsampling_factor, axis=1
                )
                data_ch1 = scisig.decimate(
                    data_ch1, preproc.downsampling_factor, axis=1
                )
            raw_data = np.array(
                np.hstack((data_ch0, data_ch1)),
                dtype=float,
            )
            if (
                self.main_ui.bottom_line_detection_widget.active_convert_dB_checkbox.isChecked()
            ):
                raw_data = convert_to_dB(raw_data)
            raw_data /= np.nanmax(np.abs(raw_data)) / 255
            raw_data = np.array(raw_data, dtype=np.uint8)
            data = np.hstack((raw_data, data))

        if active_chunkify:
            data_shape = np.shape(data)
            num_chunk = int(np.ceil(data_shape[0] / chunk_size))
            for chunk_idx in range(num_chunk):
                im_name = str(
                    work_dir / (sidescan_file.filepath.stem + f"_{chunk_idx}.png")
                )
                data_out = data[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
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
            im_name = str(work_dir / (sidescan_file.filepath.stem + ".png"))
            if self.active_colormap_checkbox.isChecked():
                cmap = Colormap(self.sonar_dat_cmap["colors"])
                data_out = data_out.astype(float) / 255
                data_out = cmap.map(data_out)
                data_out *= 255
            SidescanGeoreferencer.write_img(im_name, data)
            print(f"{im_name} written.")

    def validate_chunk_size(self):
        val = int(self.img_chunk_size_edit.line_edit.text())
        if val % 2 == 1:
            self.img_chunk_size_edit.line_edit.setText(str(val - 1))


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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
