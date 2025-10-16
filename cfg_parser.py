from enum import Enum
import os
import yaml
import json
from pydantic import BaseModel, Field
from pathlib import Path

GAINSTRAT = Enum("GAINSTRAT", [("BAC", 0), ("EGN", 1)])
RESOLUTION_OPTIONS = Enum(
    "RESOLUTION_OPTIONS",
    [("Same", 0), ("Highest", 1), ("Lowest", 2), ("Average", 3), ("Common", 4)],
)
WARP_OPTIONS = Enum("WARP_OPTIONS", [("Polynomial1", 0), ("Homography", 1)])
RESAMPLING_OPTIONS = Enum(
    "RESAMPLING_OPTIONS",
    [
        ("Near", 0),
        ("Bilinear", 1),
        ("Cubic", 2),
        ("Lanczos", 3),
        ("Average", 4),
        ("RMS", 5),
        ("Mode", 6),
        ("Maximum", 7),
        ("Minimum", 8),
        ("Median", 9),
        ("1. Quartile", 10),
        ("3. Quartile", 11),
        ("Weighted Sum", 12),
    ],
)


class CFGHelper:

    @staticmethod
    def enum_options_with_index(enum_cls):
        return "\n".join(f"{i}: {member.name}" for i, member in enumerate(enum_cls))


class MainProcParameter(BaseModel):
    georef_dir: str = Field(
        default="./georef_out",
        description="Path to directory that will store georeferencing results.",
    )
    active_convert_dB: bool = Field(
        default=True, description="Convert data to decibel for display/export"
    )
    egn_table_path: str = Field(
        default="./sidescan_out/EGN_table.npz", description="Path to .npz EGN Table"
    )
    active_bottom_line_refinement: bool = Field(
        default=True,
        description="Uses the internal primary sensor altitude information to find the bottom line in an area around the known altitude",
    )
    active_btm_refinement_shift_by_offset: bool = Field(
        default=True,
        description=" The mean distance of the raw altitude and the detected bottom line is calculated. If this parameter is true, the raw altitude information is shifted by this value and the result is used in the following as bottom line. Otherwise the result of the edge detection itself is used as bottom line.",
    )
    btm_refinement_search_range: float = Field(
        default=0.06,
        ge=0.01,
        le=1.0,
        description="Fraction in float (default=0.06). Defines the range around the sensor altitude which is used for the bottom line refinement.",
    )
    active_bottom_line_smoothing: bool = Field(
        default=True, description="Smooth detected bottom line (highly recommended)"
    )
    additional_bottom_line_inset: int = Field(
        default=0,
        ge=0,
        description="Offset in integer (default=0). Move the bottom line by this amount inwards. May be useful to exclude remaining samples of the watercolumn, but should usually not be needed.Offset in integer (default=0). Move the bottom line by this amount inwards. May be useful to exclude remaining samples of the watercolumn, but should usually not be needed.",
    )
    active_pie_slice_filter: bool = Field(
        default=False,
        description="A 2D FFT based filter is applied to remove the horizontal stripes that often occur in sidescan images.",
    )
    active_sharpening_filter: bool = Field(
        default=False,
        description="A homomorphic filter is applied to amplify the high frequency information. This feature is highly experimental and currently not advised.",
    )
    active_gain_normalization: bool = Field(
        default=True,
        description="Perform Gain Normalization via BAC or EGN. Choose strategy via gain_norm_strategy parameter.",
    )
    active_hist_equalization: bool = Field(
        default=True,
        description="Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the data to improve contrast.",
    )
    gain_norm_strategy: int = Field(
        default=GAINSTRAT.BAC.value,
        ge=0,
        le=len(GAINSTRAT),
        description=f"Enum to choose gain normalization strategy.\n Options: \n {CFGHelper.enum_options_with_index(GAINSTRAT)}",
    )
    vertical_beam_angle: int = Field(
        default=60,
        ge=1,
        le=90,
        description="Degrees as integer. Angle in the vertical plane that the sound waves cover (usually found in the manual).",
    )
    nadir_angle: int = Field(
        default=0,
        ge=0,
        lt=90,
        description="Degrees as integer. Angle between perpendicular and first bottom return (usually needs to be estimated, leave 0 if unsure).",
    )


class BottomlineDetectionParameter(BaseModel):
    chunk_size: int = Field(
        default=1000, gt=0, description="Number of pings in a single view"
    )
    default_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold used on normalized ping data to make data binary for bottom detection",
    )
    downsampling_factor: int = Field(
        default=1,
        description="Integer number to reduce samples per each ping (Data is decimated using this factor)",
    )
    active_clahe: bool = Field(
        default=True, description="Apply CLAHE to the data to improve contrast"
    )


class SlantAndGainCorrectionParameter(BaseModel):
    active_intern_depth: bool = Field(
        default=False,
        description="Use internal altitude information instead of detected bottom line.",
    )
    chunk_size: int = Field(
        default=1000, gt=0, description="Internal chunk size used for processing."
    )
    active_use_downsampling: bool = Field(
        default=True,
        description="Process data with the same downsampling that has been applied while the bottom line detection. Otherwise data is processed with the original resolution.",
    )
    active_multiprocessing: bool = Field(
        default=True,
        description="Use multiprocessing for parallel processing of multiple files. (Currently UI only)",
    )
    num_worker: int = Field(
        default=8, gt=0, description="Number of workers used for multiprocessing."
    )
    active_export_proc_data: bool = Field(
        default=True,
        description="Export fully processed data as .npz file. This data is saved to the working dir and is loaded for later postprocessing/viewing the data.",
    )
    active_export_slant_data: bool = Field(
        default=True,
        description="Export slant corrected data as .npz file. This data is saved to the working dir and is loaded for later postprocessing/viewing the data.",
    )
    bac_resolution: int = Field(
        default=360,
        gt=0,
        description="Number of quantized values of the estimated beam pattern.",
    )
    egn_table_resolution_angle: int = Field(
        default=360,
        gt=0,
        description="Number of quantized values of the estimated beam pattern in angle direction",
    )
    egn_table_resolution_range_factor: int = Field(
        default=2,
        gt=0,
        description="Range reduction factor. This defines by resolution in range direction of the resulting EGN table by dividing the ping length by this factor.",
    )


class GeoreferencingAndViewingParameter(BaseModel):
    active_view_reprocess_file: bool = Field(
        default=False,
        description="Reprocess file before viewing results in napari. (UI only)",
    )
    img_chunk_size: int = Field(
        default=1000, gt=0, description="Chunk size used for PNG export"
    )
    img_include_raw_data: bool = Field(
        default=False, description="Produces PNG with additional raw unprocessed data"
    )
    active_proc_data: bool = Field(
        default=True,
        description="Export processed data in the georeferenced image. Otherwise a waterfall image based on the raw data will be created.",
    )
    active_utm: bool = Field(
        default=True,
        description="Use UTM for export, otherwise WGS84 (unprojected) is used",
    )
    active_export_navigation: bool = Field(
        default=True, description="Export additional navigation data file"
    )
    resolution_mode: int = Field(
        default=RESOLUTION_OPTIONS.Average.value,
        ge=0,
        le=len(RESOLUTION_OPTIONS),
        description=f"Set output file resolution. \n Options: \n {CFGHelper.enum_options_with_index(RESOLUTION_OPTIONS)} \n Find more infos at gdal doc.",
    )
    warp_mode: int = Field(
        default=WARP_OPTIONS.Polynomial1.value,
        ge=0,
        le=len(WARP_OPTIONS),
        description=f"Set transformation method for output chunks. \n Options: \n {CFGHelper.enum_options_with_index(WARP_OPTIONS)} \n Find more infos at gdal doc.",
    )
    resampling_mode: int = Field(
        default=RESAMPLING_OPTIONS.Near.value,
        ge=0,
        le=len(RESAMPLING_OPTIONS),
        description=f"Set resampling method for output chunks. \n Options: \n {CFGHelper.enum_options_with_index(RESAMPLING_OPTIONS)} \n Find more infos at gdal doc.",
    )
    active_custom_colormap: bool = Field(
        default=False,
        description="Use the SidescanTools custom colormap that is used in napari also for image export.",
    )


class MetaInfo(BaseModel):
    working_dir: str = Field(
        default="./sidescan_out",
        description="Main directory to store results and exported PNGs.",
    )
    project_filename: str = Field(
        default="project_info.yml", description="Filename of this cfg"
    )
    egn_table_name: str = Field(
        default="EGN_table.npz",
        description="Name that will be used when a new EGN table is generated.",
    )
    paths: list = Field(default=list(), description="List of all imported file paths.")
    meta_info: dict = Field(
        default=dict(),
        description="Dictionary of meta information for each imported file.",
    )


class CFG(BaseModel):
    main_proc_params: MainProcParameter = MainProcParameter()
    bottomline_params: BottomlineDetectionParameter = BottomlineDetectionParameter()
    slant_gain_params: SlantAndGainCorrectionParameter = (
        SlantAndGainCorrectionParameter()
    )
    georef_view_params: GeoreferencingAndViewingParameter = (
        GeoreferencingAndViewingParameter()
    )
    meta_infos: MetaInfo = MetaInfo()

    @staticmethod
    def save_cfg_and_schema(model, dst_path: os.PathLike):
        dst_path = Path(dst_path)
        if dst_path.exists():
            # schema
            with open(dst_path / "project_info_schema.json", "w") as fd:
                json.dump(default_model.model_json_schema(), fd, indent=4)

            # cfg
            with open(dst_path / "projecT_info.yml", "w") as fd:
                # schema usage
                fd.write(
                    f"# yaml-language-server: $schema=project_info_schema.json \n \n"
                )
                # dump all to yml
                model = default_model.model_dump(by_alias=True)
                yaml.safe_dump(model, fd, sort_keys=False)
        else:
            raise FileNotFoundError(
                f"Trying to save CFG to {dst_path}. Directory does not exist."
            )

    def load_cfg(self, src_path: os.PathLike):
        with open(src_path) as fd:
            data = fd.read()

        self = CFG(**yaml.safe_load(data))


class CFGParser:
    cfg: dict
    categories: dict

    def __init__(self):
        self.cfg = {
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
            "Georef UTM": True,
            "Georef Navigation": True,
            "Resolution Mode": 3,
            "Warp Mode": 0,
            "Resampling Method": 0,
            "Georef active custom colormap": False,
            "Path": [],
            "Meta info": dict(),
            "BAC resolution": 360,
            "EGN table resolution parameters": [360, 2],
            "Bottom line refinement search range": 0.06,
            "Active bottom line refinement": True,
            "Active btm refinement shift by offset": True,
            "Active bottom line smoothing": True,
            "Additional bottom line inset": 0,
        }
        self.categorized_dict = {
            k: dict()
            for k in [
                "0: Main Processing Parameters",
                "1: Bottom Detection",
                "2: Slant and Gain Correction",
                "3: Georeferencing and Viewing",
                "4: Meta info",
            ]
        }

    def save_cfg(self, filepath: os.PathLike, cfg: dict):
        """Save CFG in a structure that makes working with the CLI version more easy."""
        self.categorized_dict["0: Main Processing Parameters"].update(
            [
                CFGParser.get_kv_pair(cfg, "Georef dir"),
                CFGParser.get_kv_pair(cfg, "Active convert dB"),
                CFGParser.get_kv_pair(cfg, "EGN table path"),
                CFGParser.get_kv_pair(cfg, "Active bottom line refinement"),
                CFGParser.get_kv_pair(cfg, "Active btm refinement shift by offset"),
                CFGParser.get_kv_pair(cfg, "Bottom line refinement search range"),
                CFGParser.get_kv_pair(cfg, "Active bottom line smoothing"),
                CFGParser.get_kv_pair(cfg, "Additional bottom line inset"),
                CFGParser.get_kv_pair(cfg, "Active pie slice filter"),
                CFGParser.get_kv_pair(cfg, "Active sharpening filter"),
                CFGParser.get_kv_pair(cfg, "Active gain norm"),
                CFGParser.get_kv_pair(cfg, "Active hist equal"),
                CFGParser.get_kv_pair(cfg, "Slant gain norm strategy"),
                CFGParser.get_kv_pair(cfg, "Slant vertical beam angle"),
                CFGParser.get_kv_pair(cfg, "Slant nadir angle"),
            ]
        )
        self.categorized_dict["1: Bottom Detection"].update(
            [
                CFGParser.get_kv_pair(cfg, "Btm chunk size"),
                CFGParser.get_kv_pair(cfg, "Btm def thresh"),
                CFGParser.get_kv_pair(cfg, "Btm downsampling"),
                CFGParser.get_kv_pair(cfg, "Btm equal hist"),
            ]
        )
        self.categorized_dict["2: Slant and Gain Correction"].update(
            [
                CFGParser.get_kv_pair(cfg, "Slant active intern depth"),
                CFGParser.get_kv_pair(cfg, "Slant chunk size"),
                CFGParser.get_kv_pair(cfg, "Slant active use downsampling"),
                CFGParser.get_kv_pair(cfg, "Slant active multiprocessing"),
                CFGParser.get_kv_pair(cfg, "Slant num worker"),
                CFGParser.get_kv_pair(cfg, "Slant active export proc data"),
                CFGParser.get_kv_pair(cfg, "Slant active export slant data"),
                CFGParser.get_kv_pair(cfg, "BAC resolution"),
                CFGParser.get_kv_pair(cfg, "EGN table resolution parameters"),
            ]
        )
        self.categorized_dict["3: Georeferencing and Viewing"].update(
            [
                CFGParser.get_kv_pair(cfg, "View reprocess file"),
                CFGParser.get_kv_pair(cfg, "Img chunk size"),
                CFGParser.get_kv_pair(cfg, "Img include raw data"),
                CFGParser.get_kv_pair(cfg, "Georef active proc data"),
                CFGParser.get_kv_pair(cfg, "Georef UTM"),
                CFGParser.get_kv_pair(cfg, "Georef Navigation"),
                CFGParser.get_kv_pair(cfg, "Resolution Mode"),
                CFGParser.get_kv_pair(cfg, "Warp Mode"),
                CFGParser.get_kv_pair(cfg, "Resampling Method"),
                CFGParser.get_kv_pair(cfg, "Georef active custom colormap"),
            ]
        )
        self.categorized_dict["4: Meta info"].update(
            [
                CFGParser.get_kv_pair(cfg, "Working dir"),
                CFGParser.get_kv_pair(cfg, "Project filename"),
                CFGParser.get_kv_pair(cfg, "EGN table name"),
                CFGParser.get_kv_pair(cfg, "Path"),
                CFGParser.get_kv_pair(cfg, "Meta info"),
            ]
        )

        try:
            f = open(
                filepath,
                "w",
            )
            yaml.dump(self.categorized_dict, f)
            f.close()
        except:
            print(f"Can't write to {filepath}")

    def load_cfg(self, path: os.PathLike):
        f = open(
            path,
            "r",
        )
        loaded_dict = yaml.safe_load(f)
        f.close()
        if "0: Main Processing Parameters" in dict(loaded_dict).keys():
            # load categorized cfg
            for key in dict(loaded_dict).keys():
                try:
                    for intern_key in loaded_dict[key]:
                        self.cfg[intern_key] = loaded_dict[key][intern_key]
                except:
                    print(f"Couldn't load setting with key: [{key}][{intern_key}]")
        else:
            # load legacy cfg
            for key in dict(loaded_dict).keys():
                try:
                    self.cfg[key] = loaded_dict[key]
                except:
                    print(f"Couldn't load setting with key: {key}")

        return self.cfg

    @staticmethod
    def get_kv_pair(cfg, key):
        return (key, cfg[key])


if __name__ == "__main__":
    default_model = CFG()
    dst = "./sidescan_out"
    CFG.save_cfg_and_schema(default_model, dst)

    default_model.load_cfg(dst + "/project_info.yml")
    print(default_model)
