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
import qtpy.QtWidgets

#@magicgui
def plot_nav(filepath: str | os.PathLike):
    """
    Plot navigation data and heading. 

    Parameters:
    ------------------
    filepath: str | os.PathLike
        Path to sidescan file
    """
    filepath = Path(filepath)

    # get nav data
    get_nav = SidescanGeoreferencer(filepath=filepath)
    get_nav.prep_data()
    lola_data = get_nav.LOLA_plt
    head_data = get_nav.HEAD_plt
    pg.plot(lola_data, pen='r', width=9, style=QtCore.Qt.DashLine, title='Navigation')
    pg.plot(head_data, pen='y', width=9, title='Heading')

    ########## old #############
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
    #viewer = napari.Viewer(axis_labels=['Longitude','Latitude'])
    #viewer.add_points(lola_data, size=1, name='Navigation', axis_labels=['Longitude','Latitude'])
    #image_layer_2 = viewer.view_points(head_data, name='Heading')
    #viewer, image_layer_1 = napari.imshow(
    #    lola_data, name="Navigation"
    #)
    #image_layer_2 = napari.add_image(
    #    head_data, name="Heading"
    #)
    #napari.run(max_loop_level=2)

if __name__ == "__main__":
    filepath = Path("add_path_to_file_here")