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
    paths: list[str] = Field(
        default_factory=list, description="List of all imported file paths."
    )
    meta_info: dict = Field(
        default_factory=dict,
        description="Dictionary of meta information for each imported file.",
    )


class CFG(BaseModel):
    main_proc_params: MainProcParameter = Field(default_factory=MainProcParameter)
    bottomline_params: BottomlineDetectionParameter = Field(
        default_factory=BottomlineDetectionParameter
    )
    slant_gain_params: SlantAndGainCorrectionParameter = Field(
        default_factory=SlantAndGainCorrectionParameter
    )
    georef_view_params: GeoreferencingAndViewingParameter = Field(
        default_factory=GeoreferencingAndViewingParameter
    )
    meta_infos: MetaInfo = Field(default_factory=MetaInfo)

    def save_cfg_and_schema(self, dst_path: os.PathLike):
        dst_path = Path(dst_path)
        if dst_path.exists():
            # schema
            with open(dst_path / "project_info_schema.json", "w") as fd:
                json.dump(self.model_json_schema(), fd, indent=4)

            # cfg
            with open(dst_path / "project_info.yml", "w") as fd:
                # schema usage
                fd.write(
                    f"# yaml-language-server: $schema=project_info_schema.json \n \n"
                )
                # dump all to yml
                model = self.model_dump(by_alias=True)
                yaml.safe_dump(model, fd, sort_keys=False)
        else:
            raise FileNotFoundError(
                f"Trying to save CFG to {dst_path}. Directory does not exist."
            )

    @classmethod
    def load_cfg(cls, src_path: os.PathLike):
        with open(Path(src_path) / "project_info.yml") as fd:
            data = fd.read()

        return cls(**yaml.full_load(data))


if __name__ == "__main__":
    default_model = CFG()
    dst = "./sidescan_out"
    default_model.meta_infos.paths = ["1", "2"]
    default_model.meta_infos.meta_info = {"a": "b"}
    default_model.save_cfg_and_schema(dst)

    loaded_model = CFG.load_cfg(dst)
    print(loaded_model)
