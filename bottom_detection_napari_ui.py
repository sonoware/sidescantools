import napari
from magicgui import magicgui
from pathlib import Path
from sidescan_preproc import SidescanPreprocessor
import numpy as np
from sidescan_file import SidescanFile
import os


def run_napari_btm_line(
    filepath: str | os.PathLike,
    chunk_size=1000,
    default_threshold=0.7,
    downsampling_factor=1,
    work_dir=None,
    active_dB=False,
):
    """Run bottom line detection in napari on a given file

    Parameters
    ----------
    filepath: str | os.PathLike
        Path to sidescan file
    chunk_size: int
        Number of pings per single chunk
    default_threshold: float
        Number in range [0, 1] that is used as threshold for binarization of the image before the edges are detected
    downsampling_factor: int
        Factor used for decimation of ping signals
    work_dir: str | os.PathLike
        Path to desired directory that is used as default directory for saving/loading of results to ``.npz`` files
    active_dB: bool
        If ``True`` data will be converted to dB for display in napari
    """
    filepath = Path(filepath)
    add_line_width = 1  # additional line width for plotting of bottom line

    sidescan_file = SidescanFile(filepath)
    preproc = SidescanPreprocessor(
        sidescan_file=sidescan_file,
        chunk_size=chunk_size,
        downsampling_factor=downsampling_factor,
    )

    # Init bottom detection by doing an initial guess
    preproc.init_napari_bottom_detect(
        default_threshold,
        active_dB=active_dB,
    )

    # build napari GUI
    @magicgui(
        auto_call=True,
        threshold_bin={
            "widget_type": "FloatSlider",
            "min": 0,
            "max": 1.0,
            "step": 0.01,
        },
        choose_strategy={
            "widget_type": "RadioButtons",
            "choices": preproc.bottom_strategy_choices,
        },
        call_button="Recalculate",
    )
    def widget_thresh(
        viewer: napari.Viewer,
        threshold_bin=default_threshold,
        choose_strategy=preproc.bottom_strategy_choices[1],
    ):

        slider_position = viewer.dims.current_step[0]
        preproc.detect_bottom_napari(
            slider_position,
            threshold_bin=threshold_bin,
            bottom_strategy_choice=choose_strategy,
            add_line_width=add_line_width,
        )

        # update bottom plot with new data
        bottom_image_layer.data = preproc.bottom_map

    # Build widget for aux parameters that shall not trigger a recalculation for the current chunk
    @magicgui(auto_call=True, call_button=None)
    def manual_annotation_widget(activate_manual_annotation: bool):
        for layer in viewer.layers:
            if activate_manual_annotation:
                layer.mouse_pan = False
            else:
                layer.mouse_pan = True

    # Build saving and loading widgets using simple npz file
    if work_dir is None:
        default_bottom_path = filepath.parent / (filepath.stem + "_bottom_info.npz")
    else:
        work_dir = Path(work_dir)
        default_bottom_path = work_dir / (filepath.stem + "_bottom_info.npz")

    @magicgui(filename={"mode": "w"}, call_button="Save")
    def filepicker_save(
        filename=default_bottom_path,
    ):
        np.savez(
            filename,
            bottom_info_port=preproc.napari_portside_bottom,
            bottom_info_star=preproc.napari_starboard_bottom,
            downsampling_factor=preproc.downsampling_factor,
        )

    @magicgui(filename={"mode": "r"}, call_button="Load")
    def filepicker_load(
        filename=default_bottom_path,
    ):
        if filename.exists() and filename.suffix == ".npz":
            bottom_info = np.load(filename)
            preproc.napari_portside_bottom = bottom_info["bottom_info_port"]
            preproc.napari_starboard_bottom = bottom_info["bottom_info_star"]

            for chunk_idx in range(preproc.num_chunk):
                preproc.update_bottom_map_napari(
                    chunk_idx, add_line_width=add_line_width
                )
            bottom_image_layer.refresh()

    viewer = napari.Viewer(title="SidescanTools - Bottom line detection")

    # add custom shortcuts
    @viewer.bind_key("m")
    def press_m(viewer):
        manual_annotation_widget.activate_manual_annotation.value = (
            not manual_annotation_widget.activate_manual_annotation.value
        )

    @viewer.bind_key("r")
    def press_r(viewer):
        widget_thresh.changed()

    # add image
    sidescan_image_layer = viewer.add_image(
        preproc.napari_fullmat, name="sidescan image", colormap="copper"
    )

    # show bottom line overlay
    colors = [[1, 1, 1, 0], [1, 0, 0, 1]]  # r,g,b,alpha
    bottom_colormap = {
        "colors": colors,
        "name": "bottom_line_cmap",
        "interpolation": "linear",
    }
    bottom_image_layer = viewer.add_image(
        preproc.bottom_map, name="bottom_map", colormap=bottom_colormap
    )

    # add widgets to main window
    viewer.window.add_dock_widget(widget_thresh, name="Bottom detection parameters")
    widget_thresh.visible = False  # HACK to change size policy...
    viewer.window.add_dock_widget(
        manual_annotation_widget, name="Activate manual annotation"
    )
    viewer.window.add_dock_widget(filepicker_save, name="Save to")
    viewer.window.add_dock_widget(filepicker_load, name="Load from")
    widget_thresh.visible = True
    # label shortcuts
    widget_thresh.threshold_bin.label = "Threshold binarization [0,1]"
    widget_thresh.choose_strategy.label = "Choose strategy"
    widget_thresh.call_button.text = "r: Recalculate"
    manual_annotation_widget.activate_manual_annotation.text = (
        "m: Activate manual annotation"
    )
    filepicker_save.filename.label = "File"
    filepicker_load.filename.label = "File"

    # Handle click or drag events separately
    @bottom_image_layer.mouse_drag_callbacks.append
    def click_drag(layer, event):

        if (
            manual_annotation_widget.activate_manual_annotation.value
            and event.button == 1
            and 0 <= event.position[1] < layer.data.shape[1]
            and 0 <= event.position[2] < layer.data.shape[2]
        ):

            # print('mouse down')
            dragged = False
            yield

            # on move
            last_pos = np.zeros(3)
            while event.type == "mouse_move":
                dragged = True

                cur_pos = np.array(np.round(event.position), dtype=int)
                if cur_pos[2] < layer.data.shape[2] / 2:
                    preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = cur_pos[2]
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] / 2 - cur_pos[2]
                        )
                else:
                    preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                        cur_pos[2] - layer.data.shape[2] / 2
                    )
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] - cur_pos[2]
                        )

                # check whether movement skipped points and do linear interpolation
                if (last_pos > 0).all():
                    if last_pos[1] - cur_pos[1] > 1:
                        if cur_pos[2] < layer.data.shape[2] / 2:
                            preproc.napari_portside_bottom[
                                cur_pos[0], cur_pos[1] : last_pos[1]
                            ] = cur_pos[2]
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_starboard_bottom[
                                    cur_pos[0], cur_pos[1] : last_pos[1]
                                ] = (layer.data.shape[2] / 2 - cur_pos[2])
                        else:
                            preproc.napari_starboard_bottom[
                                cur_pos[0], cur_pos[1] : last_pos[1]
                            ] = (cur_pos[2] - layer.data.shape[2] / 2)
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_portside_bottom[
                                    cur_pos[0], cur_pos[1] : last_pos[1]
                                ] = (layer.data.shape[2] - cur_pos[2])
                    elif last_pos[1] - cur_pos[1] < 1:
                        if cur_pos[2] < layer.data.shape[2] / 2:
                            preproc.napari_portside_bottom[
                                cur_pos[0], last_pos[1] : cur_pos[1]
                            ] = cur_pos[2]
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_starboard_bottom[
                                    cur_pos[0], last_pos[1] : cur_pos[1]
                                ] = (layer.data.shape[2] / 2 - cur_pos[2])
                        else:
                            preproc.napari_starboard_bottom[
                                cur_pos[0], last_pos[1] : cur_pos[1]
                            ] = (cur_pos[2] - layer.data.shape[2] / 2)
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_portside_bottom[
                                    cur_pos[0], last_pos[1] : cur_pos[1]
                                ] = (layer.data.shape[2] - cur_pos[2])

                last_pos = cur_pos
                preproc.update_bottom_map_napari(cur_pos[0], add_line_width=0)
                bottom_image_layer.data = preproc.bottom_map

                yield
            # on release
            if dragged:
                dragged = False
            else:
                cur_pos = np.array(np.round(event.position), dtype=int)
                if event.position[2] < layer.data.shape[2] / 2:
                    preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = cur_pos[2]
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] / 2 - cur_pos[2]
                        )
                else:
                    preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                        cur_pos[2] - layer.data.shape[2] / 2
                    )
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] - cur_pos[2]
                        )
            # set map to trigger drawing
            preproc.update_bottom_map_napari(int(event.position[0]), add_line_width=0)
            bottom_image_layer.data = preproc.bottom_map

    # run main loop
    viewer.show(block=True)


if __name__ == "__main__":
    chunk_size = 1000
    default_threshold = (
        0.07  # [0.0, 1.0] -> threshold to make sonar img binary for edge detection
    )
    downsampling_factor = 1
    active_dB = False

    filepath = Path("add_path_to_file_here")
    work_dir = "./sidescan_out"
    run_napari_btm_line(
        filepath,
        chunk_size,
        default_threshold,
        downsampling_factor,
        work_dir=work_dir,
        active_dB=active_dB,
    )
