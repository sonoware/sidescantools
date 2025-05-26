# SidescanTools
Welcome to SidescanTools, an open-source software to read and process data from side-scan sonar instruments.
This tool can be used to create high-resolution 2D images of the sea floor.
The data can be processed to reduce noise, apply slant-range correction and gain normalisation on a set of side-scan files.
They can then be exported as `.geotiff` or simple `.png` files.
As of now, SidescanTools can process and read data from two formats:
- .jsf: open file format by EdgeTech
- .xtf: cross-platform readable file format

# Main Processing Steps
1. Detect **bottom line** in waterfall view (required for any following processing step).
2. **Geometric Corrections**: Slant-range correction: Calculate ground range by projecting slant ranges onto bottom, assuming flat seafloor.
3. **Radiometric Corrections** Baam angle correction (BAC) or Empirical Gain Normalisation (EGN): is used to correct for intensity changes by angle (BAC) or angle and distance (EGN).
   Note: EGN Needs mtultiple files for good performance. If only few files exist, use BAC!
   BAC sums & averages intensities per beam angle over all pings in a file. 
   EGN sums and averages amplitudes of all pings by beam angle and distance over all loaded files to correct for intensity.
   Note that this only works for files from the *same* instrument.
   A good approach is one EGN table per survey/day and per instrument.
4. Export data as **georeferenced image** to view on a map.

# Issues and Planned Features
The following features are still under development and will be improved in future releases:
- **Georeferencing** currently uses [gdal v3.11](https://gdal.org/) and `homography` or `polynomial 1` as warping algortihm along with ground control points. Usually, homography is more precise but can in some cases produce wavy structures and/or shifts in the data. In this case, use `Polynomial` (preserves parallel lines) -- _custom georeferencing to be implemented_
- **Data processing** currently freezes the GUI, information about the current state is only shown in the console -- _background processing to be implemented_
- **Loading XTF/JSF files** always reads the entire file -- _loading only headers to show meta information while keeping the GUI responsive to be implemented_
- **Recommendations** _for downsampling and processing based on loaded data to be implemented_
- _Implement a standard case per possible imported datatype for optimal visibility of objects in the images_

# Getting Started
1. Make sure to have Python 3.11 - 3.13 installed. SidescanTools may or may not work with other versions of Python.
2. Clone this git repository
3. Install required packages from `requirements.txt`.
   Using a virtual (conda!) environment is recommended.
   Currently packages are listed without minimum version.
   - Recommended: Use anaconda or miniconda and install gdal v3.11 using: conda install -c confa-forge gdal=3.11
   - If you do not use conda and are on Windows and have trouble installing GDAL, check [cgohlke's wheels](https://github.com/cgohlke/geospatial-wheels/releases) (not recommended! Currently does not work for gdal v3.11!)
   - On Linux, if not using conda, you may need to install your system's gdal package (e.g., `apt install libgdal-dev` or `yum install libgdal-dev`)
4. Start GUI by running the typing `main.py` script from within your environment


# Usage
In the following section all GUI elements are described in more detail.

## Add Sidescan Data
- Add sidescan data by pressing `Add XTF/JSF` in the top left panel

## Bottom Line Detection (BLD)
- `Bottomline Detection` initiates the bottom-line detection window
- `Chunk Size`: Number of pings in a single view
- `Default Threshold`: Threshold used on normalized ping data to make data binary for bottom detection
- `Downsampling Factor`: Integer number to reduce samples per each ping (Data is decimated using this factor)
- Tick `Convert to dB` to convert data to decibels instead of intensities

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
- `Filter Stripe Noise (experimental)`
- `Apply Sharpening Filter (experimental)`

### Slant Range Correction and Gain Normalisation
- `Apply Downsampling`
- `Apply Gain Normalisation`
- `Vertical Beam Angle` (only relevant if internal depth is unknown or shall be omitted): Horizontal angle by which the instrument is tilted (usually found in the manual)
- Tick `Use Internal Depth` if the flying alitude of the side scan instrument is known & has been logged correctly

### Advanced Gain Normalisation Filter
- `Nadir Angle` (only relevant if internal depth is unknown or shall be omitted): Angle between perpendicular and first bottom return (usually need to be estimated, leave 0° if unsure)
- `Chunk Size`: Number of pings per chunk to calculate EGN table and use for waterfall image generation
- `Remove Watercolumn`: Removes watercolumn prior to slant range correction and EGN calculation
- `Active Multiprocessing`: Activate multi-threading
- `Number of Workers`: Set number of parallel processing workers
- `Generate EGN table`: Initiates EGN table generation. All files loaded in the project that have bottom line information available will be processed. For each sonar file, the required information is saved to individual EGN info files. In a last step, all these info files are combined into one EGN table that can be applied to gain normalise all data of this side scan sonar type (see next step). This process needs quite some time (check console outputs).
- `Process All Files`: Applies previously calculated slant range correction & EGN to all loaded files at once. This will take some time depending on the amount of data (check console outputs).

## View and Export
### View Results
- Tick the `Reprocess File` option to apply slant range correction and EGN only to the selected file when viewing the results.
- `Convert to dB`
- `Apply CLAHE`
- `View Processed Data`: Initiates data viewer to inspect the raw input data, bottom line detection, slant range and EGN corrected data of the currently selected file.

### Georeferencing and image generation
- Tick `Use processed Data` if above processing steps should be applied, otherwise a waterfall image based on the raw data will be created
- `Dynamic Chunking` chooses number of pings within one chunk for georeferencing based on distance between GPS points. Only apply when GPS data are bad! If unticked, chunk size is 5 pings.
- Untick `UTM` if you prefer WGS84 (unprojected)
- Tick `Polynomial` if homographic transformation doesn't work. Sometimes, if e.g. vessel movement is irregular, the polynomial tranformation type works more reliable. If in doubt, compare geotiff with the (ungeoreferenced) waterfall image. 
- `Apply Custom Colormap`: Select from a range of colormaps; if unticked, greyscale values are used
- `Generate Geotiff`: Uses gdal reproject (with either homography or polynomial order 1) and ground control points (gcps) to georeference data chunk wise and export as Geotiff
- `Include raw data in waterfall image`: produces additional png with raw undprocessed data
- `Generate Waterfall Image`: Generates a non-georeferenced png file from processed data. Adjust chunk size if you need one file instead of several.
#### Important Note: If the file is shorter than wide (i.e., number of pings > number of samples for both channels) the georeferencing doesn't work!

# About
SidescanTools is an open-source software project by [GEOMAR](https://www.geomar.de/ghostnetbusters) and [sonoware](https://www.sonoware.de/news/2024-12-06_uebergabe_foerderbescheid/) funded by the AI Fund of the State of Schleswig-Holstein.
