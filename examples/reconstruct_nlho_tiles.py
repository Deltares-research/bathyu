from bathyu import TiledRasters
from bathyu.io import export, nlho_grids_io

nlho_grids_folder = r"p:\tgg-mariene-data\__UPDATES\GRIDS5M"

# Find all NLHO surveys in the given bounding box and export them to geotiffs
surveys = nlho_grids_io.find_nlho_surveys_in_bbox(
    nlho_grids_folder, 565000, 5755000, 575000, 5765000
)
for survey in surveys:
    survey_data = nlho_grids_io.reconstruct_nlho_survey(
        nlho_grids_folder,
        survey,
        within_bbox=True,
        xmin=565000,
        ymin=5750000,
        xmax=575000,
        ymax=5765000,
    )
    export.to_geotiff(
        survey_data,
        rf"p:\11211195-co2-noordzee\Data\NLHO\alle_nlho_surveys_maasvlakte\{survey}.tiff",
        compress=True,
    )
