
# SidescanTools

Welcome to SidescanTools, an open-source software to read and <br /> 
<img align="right" width="250" height="250" src="./res/sidescantools_logo_rund.png" hspace="25" title="Logo design and artwork by Aili Xue">
process data from side-scan sonar instruments. <br /> 
This tool can be used to create high-resolution 2D images of the sea floor.
The data can be processed to reduce noise, apply slant-range correction and gain normalisation on a set of side-scan files.
They can then be exported as `.geotiff` or simple `.png` files.
As of now, SidescanTools can process and read data from two formats:
- .jsf: open file format by EdgeTech
- .xtf: cross-platform readable file format

# Main Processing Steps
1. Detect **bottom line** in waterfall view (required for any following processing step).
2. **Geometric Corrections**: Slant-range correction: Calculate ground range by projecting slant ranges onto bottom, assuming flat seafloor.
3. **Radiometric Corrections**:
- Filter stripe noise using a Filter in 2DFFT domain
- Apply a sharpening filter 
- Apply one of two **Gain Normalization** Strategies:

  - Beam Angle Correction (BAC, works on a single file)
  - Empirical Gain Normalization (EGN, works by analyzing all files in the project)

   **_Note_**: EGN needs quite some data for good performance. If only few data exist, use BAC!
   BAC sums & averages intensities per beam angle over all pings in a file. 
   EGN sums and averages amplitudes of all pings by beam angle and distance over all loaded files to correct for intensity.
   **_Note_**: that this only works for files from the *same* instrument.
   A good approach is one EGN table per survey/day and per instrument.

4. **View and Export**
  - View data of the different steps in napari to examine procesing results.
  - Export data as **georeferenced image** to view on a map.
  - If only a simple image is needed, a waterfall image can also be exported.

# Issues and Planned Features
The following features are still under development and will be improved in future releases:
- **Bottom line detection** sometimes failes, especially when there are a lot of reflections in the water coloumn. Therefore a strategy to counter this should be examined.
- When creating a docker image, add **- conda-forge::sqlite=3.50.0** & **- conda-forge::libsqlite=3.50.0** to the environment.yaml

# Getting Started
SidescanTools may be used with **full feature extent** as **GUI Tool**. Also there is the option to use a **CLI only** implementation when there is no graphical interface available.
In the following the usage of the GUI version is described. At the end of this readme the CLI variant is explained.

1. Currently we use Anaconda/Miniconda for platform indepent installation using Python 3.12. This is preferred because the installation of GDAL is essential and often doesn't work using pip. 
2. Clone this git repository
3. Install required packages from `environment.yml`: `conda env create -f environment.yml`
   Using a virtual (conda!) environment is recommended.
   Currently packages are listed without minimum version.
4. Start GUI by executing `python main_gui.py`


# Usage
The tool works based on a project directory, which can be set via the `Working directory` button. 
Pressing `Save Project Info` will save all settings and information about the imported files to this directory where it then can be loaded from.
In the following all GUI elements are explained in more detail.

## Add Sidescan Data
- Add sidescan data by pressing `Add XTF/JSF` in the top left panel
- The data is imported into the current project and it is checked whether the data can be interpreted

## Bottom Line Detection (BLD)
- `Bottomline Detection` initiates the bottom-line detection window
- `Chunk Size`: Number of pings in a single view
- `Default Threshold`: Threshold used on normalized ping data to make data binary for bottom detection
- `Downsampling Factor`: Integer number to reduce samples per each ping (Data is decimated using this factor)
- `Convert to dB`: Convert data to decibels instead of raw intensities
- `Apply Contrast Limited Adaptive Histogram Equalization`: Apply CLAHE to the data to improve contrast

### Interactive BLD
- The selected file is read and divided into chunks.
  An initial depth detection is done for the full file which can be adjusted for each frame/chunk.
- Threshold and side strategies that can be selected: `Each Side Individually`, `Combine Both Sides`, `Only Use Portside`, `Only Use Starboard`
- The depth detection result can be saved or loaded to a `.npz` file.
  Files need so be saved to the current working directory to be accessible for the next processing steps.
  - `Chunk Size`: Number of pings per chunk to visualise in bottom detection window
  - `Apply Downsampling`: Use downsampling factor as defined in bottom line detection. If unchecked, data will only be downsampled for bottom line detection but not for final image/geotiff generation.

## Processing
### Noise Reduction and Sharpening Filter
- `Filter Stripe Noise (experimental)`: A 2D FFT based filter is applied to remove the horizontal stripes that often occur in sidescan images.
- `Apply Sharpening Filter (experimental)`: A homomorphic filter is applied to amplify the high frequency information.

### Slant Range Correction and Gain Normalisation
- `Apply Downsampling`: Use downsampling factor as defined in bottom line detection. If unchecked, data will only be downsampled for bottom line detection but not for final image/geotiff generation.
- `Apply Gain Normalisation`: Apply BAC or EGN to the data.
- `Vertical Beam Angle` (only relevant if internal depth is unknown or shall be omitted): Angle in the vertical plane that the sound waves cover(usually found in the manual)
- Tick `Use Internal Depth` if the flying altitude of the side scan instrument is known & has been logged correctly

### Advanced Gain Normalisation Filter
- `Nadir Angle` (only relevant if internal depth is unknown or shall be omitted): Angle between perpendicular and first bottom return (usually need to be estimated, leave 0° if unsure)
- `Chunk Size`: Number of pings per chunk to calculate EGN table and use for waterfall image generation
- `Generate EGN table`: Initiates EGN table generation. All files loaded in the project that have bottom line information available will be processed. For each sonar file, the required information is saved to individual EGN info files. In a last step, all these info files are combined into one EGN table that can be applied to gain normalise all data of this side scan sonar type (see next step). This process needs quite some time (check console outputs).
- `Process All Files`: Applies previously calculated slant range correction & EGN to all loaded files at once. This will take some time depending on the amount of data (check console outputs).

#### Parameters that are only exposed via `project_info.yml`
When using BAC or EGN for Gain Normalisation, the resolution of the estimated beam/beam and range pattern is usually fixed. It can be adjusted by these parameters:
- `BAC resolution`: Number of quantized values of the estimated beam pattern.
- `EGN table resolution parameters`: Two integer values. The first is the number of quantized values of the estimated beam pattern in angle direction. The second parameter is the range reduction factor. This defines by resolution in range direction of the resulting EGN table by dividing the ping length by this factor.

## View and Export
### View Results
- Tick the `Reprocess File` option to apply slant range correction and EGN only to the selected file when viewing the results.
- `Convert to dB`: Convert data to decibels instead of raw intensities
- `Apply Contrast Limited Adaptive Histogram Equalization`: Apply CLAHE to the data to improve contrast
- `View Processed Data`: Initiates data viewer to inspect the raw input data, bottom line detection, slant range and EGN corrected data of the currently selected file.

### Georeferencing and image generation
- Tick `Use processed Data` if above processing steps should be applied, otherwise a waterfall image based on the raw data will be created
- `Resolution`: Set output file resolution. Currently, default is 0.2m. **Planned:** Use sample size as resolution default.
- `Search Radius`: Set value to include <search_radius> neighbours for nearneighbor algorithm. Default: 2 * resolutiopn. More info [here:](https://www.pygmt.org/latest/api/generated/pygmt.nearneighbor.html)
- Use `Blockmedian` to reduce noise and data size. More info [here:](https://www.pygmt.org/latest/api/generated/pygmt.blockmedian.html#pygmt.blockmedian)
- Untick `UTM` if you prefer WGS84 (unprojected) 
- `Apply Custom Colormap`: Select from a range of colormaps; if unticked, greyscale values are used
- `Generate Geotiff for selected file`: Uses [pygmt: 0.17.0](https://www.pygmt.org/latest/index.html) with `blockmedian` (optional) and `nearneighbour` as gridding algortihm of xyz data (x/y: lon/lat, z: backscatter as amplitudes or greyscale). Use blockmedian to further noise and output grid size. Output raster if saved as 1-band geotiff with intensities as values.
- `Include raw data in waterfall image`: produces additional png with raw undprocessed data
- `Generate Waterfall Image`: Generates a non-georeferenced png file from processed data. Adjust chunk size if you need one file instead of several.

# Usage as CLI Tool
If no graphical interface is desired or accessible, the CLI variant of SidescanTools can be used. Therefore only the python packages defined in `environment_cli.yml` are required. 

To process a file or a directory, use the following command:
```
python main_cli.py file_or_folder_path project_info.yml
```

This command processes the specified file or all files within the folder, using the settings defined in the provided `project_info.yml` file.

## Output
The tool produces:

- A **fully processed waterfall-like image** (.png format) stored in the same folder as the input data.
- An **GeoTIFF** of the processed data, saved in a separate folder as specified in `project_info.yml`.

**_Note_**: This workflow is intended for datasets where the sensor altitude is known during acquisition. This altitude information is used and refined during processing.

- **Mandatory Arguments**:
  - `file_or_folder_path`:

    Path to a single `.xtf` or `.jsf` file, or a folder containing multiple such files.
  - `project_info.yml`: 

    Path to the CFG file.

- **optional flags**:
  - `-w`: **Write default CFG**:

    This option expects only one path to a valid directory as argument. A default CFG is written as `project_info.yml` to the given directory.
  - `-g`: **Generate EGN Table**
  
    This option generates an EGN table by analyzing all sidescan files in the provided `file_or_folder_path`. The result is written to a numpy file `egn_table_<timestamp>.npz` containing all info for SidescanTools to use this table for later EGN processing of these or other files.

    To use this EGN table you need to adjust the `EGN table path` in your `project_info.yml` to point to the latest generated EGN table (and have EGN gain normalisation enabled by setting `Slant gain norm strategy` to `1`).
  - `-n`: **No GeoTiff**
  
    Skips GeoTIFF generation and only creates the .png image(s).

## Additional details about main parameters in `project_info.yml`
In the following the most important CFG parameters which are relevant for  processing with the CLI variant are explained.

- `Active bottom line refinement`: If `true`: Uses the internal primary sensor altitude information to find the bottom line in an area around the known altitude. This bottom line is found via an edge detection algorithm and its result can be used in two different ways. See `Active btm refinement shift by offset` for more information.

- `Active bottom line smoothing`: If `true`: Smooth detected bottom line.

- `Active btm refinement shift by offset`: This parameter only has effect when `Active bottom line refinement`=`true`. The mean distance of the raw altitude and the detected bottom line is calculated. If this parameter is `true`, the raw altitude information is shifted by this value and the result is used in the following as bottom line. Otherwise the result of the edge detection itself is used as bottom line

- `Active convert dB`: If `true`: Convert data to decibels instead of raw intensities.

- `Active hist equal`: If `true`: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the data to improve contrast.

- `Active pie slice filter`: If `true`: A 2D FFT based filter is applied to remove the horizontal stripes that often occur in sidescan images.

- `Active sharpening filter` If `true`: A homomorphic filter is applied to amplify the high frequency information. This feature is highly experimental and currently not advised.

- `Additional bottom line inset`: Offset in integer (default=0). Move the bottom line by this amount inwards. May be useful to exclude remaining samples of the watercolumn, but should usually not be needed.

- `Bottom line refinement search range`: Fraction in float (default=0.06). Defines the range around the sensor altitude which is used for the bottom line refinement.

- `EGN table path`: Path to the EGN Table which is used for gain normalisation.

- `Georef dir`: Directory which is used by the georeferencing to hold interim files and where the resulting GeoTIFFs are saved.

- `Slant gain norm strategy`: Enum. 

  When set to `0`: BAC is used for gain normalisation. 

  When set to `1`: EGN is used for gain normalisation. 

- `Slant nadir angle`: Degrees as integer. Angle between perpendicular and first bottom return (usually needs to be estimated, leave `0` if unsure)

- `Slant vertical beam angle`: Degrees as integer. Angle in the vertical plane that the sound waves cover (usually found in the manual)

# About
SidescanTools is an open-source software project by [GEOMAR](https://www.geomar.de/ghostnetbusters) and [sonoware](https://www.sonoware.de/news/2024-12-06_uebergabe_foerderbescheid/) funded by the AI Fund of the State of Schleswig-Holstein. The logo design and artwork has been done by [Aili Xue](https://ailixue.myportfolio.com/work).
