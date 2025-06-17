import pyqtgraph as pg
from magicgui import magicgui
from sidescan_georeferencer import SidescanGeoreferencer
import os
from pathlib import Path
import numpy as np
from sidescan_file import SidescanFile
import napari
from napari.qt import QtViewer
import qtpy.QtCore as QtCore
from qtpy.QtWidgets import QVBoxLayout, QWidget


class NavPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.lola_plot_widget = pg.PlotWidget()
        self.head_plot_widget = pg.PlotWidget()


    def plot_nav(self, filepath: str | os.PathLike):
        """
        Plot navigation data and heading. 

        Parameters:
        ------------------
        filepath: str | os.PathLike
            Path to sidescan file
        """
        filepath = Path(filepath)

        # get nav data
        try:
            get_nav = SidescanGeoreferencer(filepath=filepath)
            get_nav.prep_data()
            print("Getting navigation data...")
            lola_data = get_nav.LOLA_plt
            head_data = get_nav.HEAD_plt
            print(lola_data)

            print("Plotting and Setting title...")
            self.lola_plot_widget.plot(lola_data, pen='r', width=9, title='Navigation')
            self.head_plot_widget.plot(head_data, pen='y', width=9, title='Heading')
            self.lola_plot_widget.setLabel('left', 'Latitude [°]')
            self.lola_plot_widget.setLabel('bottom', 'Longitude [°]')
            self.head_plot_widget.setLabel('left', 'Heading [°]')
            self.head_plot_widget.setLabel('bottom', 'Ping number')
        except Exception as e:
            print(e)
        #pg.plot(lola_data, pen='r', width=9, style=QtCore.Qt.DashLine, title='Navigation')
        #pg.plot(head_data, pen='y', width=9, title='Heading')

    #print(f"np.shape(head_data) {np.shape(head_data)}")
    #print(f"np.shape(lola_data) {np.shape(lola_data)}")
    #print(lola_data)
    #nav_plot = pg.GraphicsLayoutWidget()
    #nav_plot.addPlot(x=lola_data[0], y=lola_data[1], row=0, col=0, title="Navigation")
    #nav_plot.addPlot(x=head_data[0], y=head_data[1], row=0, col=1, title="Heading")
    #nav_plot.addViewBox(row=1, col=0, colspan=2)

    #plt_head = pg.PlotWidget()
    #plt_head.setLabel('left', 'Heading in degree')
    #plt_head.setLabel('bottom', 'Ping Number')
    #plt_lola = pg.PlotWidget()
    #plt_lola.setLabel('left', 'Latitude in degree')
    #plt_lola.setLabel('bottom', 'Longitude in degree')
    #plt_head.plot(head_data, pen='y', width=9, title='Heading')
    #plt_lola.plot(lola_data, pen='r', width=9, style=QtCore.Qt.DashLine, title='Navigation')
    #pg.exec()
    #self.addWidget(self.plot_lola)
    #nav_plot_window = SidescanToolsMain()
    #nav_plot_window.initGUI()
    #nav_plot_window.plot(lola_data)
    #self.plot_nav_show.plot(lola_data)


if __name__ == "__main__":
    filepath = Path("add_path_to_file_here")