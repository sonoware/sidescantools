from qtpy.QtWidgets import (
    QFrame,
    QLineEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QSizePolicy,
)
import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
import pathlib


class QHLine(QFrame):
    """Helper class for a horizontal line"""

    def __init__(self):
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
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.button_2.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
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


class ErrorWarnDialog(QDialog):
    def __init__(self, parent=None, title: str = "Error", message: str = ""):
        """Warning dialog if a file is overwritten"""
        super().__init__(parent)

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.Ok

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        layout = QVBoxLayout()
        message = QLabel(message)
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
