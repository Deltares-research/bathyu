from bathyu import TiledRasters
from bathyu.io import export, nlho_grids_io

nlho_grids_folder = r"p:\tgg-mariene-data\__UPDATES\GRIDS5M"


# Find all NLHO surveys in the given bounding box and export them to geotiffs
surveys = TiledRasters.from_nlho_grid_files(
    nlho_grids_folder, 560000, 5760000, 580000, 5765000
)
export.to_geotiff(
    surveys,
    r"p:\11211195-co2-noordzee\Data\NLHO\maasgeul.tiff",
    compress=True,
)

for survey in surveys:
    survey_data = nlho_grids_io.reconstruct_nlho_survey(
        nlho_grids_folder,
        survey,
        within_bbox=True,
        xmin=560000,
        ymin=5760000,
        xmax=580000,
        ymax=5765000,
    )
    export.to_geotiff(
        survey_data,
        rf"p:\11211195-co2-noordzee\Data\NLHO\alle_nlho_surveys_maasvlakte\{survey}.tiff",
        compress=True,
    )
