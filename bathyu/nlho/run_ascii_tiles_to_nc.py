import json
from pathlib import Path

from tqdm import tqdm

from bathyu.nlho.nlho import (
    get_nlho_tiles_and_files,
    nlho_from_opendap,
    parse_survey_metadata,
    tile_surveys_to_netcdf,
    validate_nlho_metadata,
)

METADATA_JSON = Path(__file__).parent / "metadata.json"

if __name__ == "__main__":
    example = nlho_from_opendap(
        url="https://opendap.deltares.nl/thredds/dodsC/opendap/hydrografie/surveys/catalog.nc"
    )
    tiles, files = get_nlho_tiles_and_files(
        r"p:\tgg-mariene-data\__UPDATES\GRIDS5M",
        bbox=[555000, 5765000, 575000, 5790000],
    )
    metadata = parse_survey_metadata(
        r"p:\tgg-mariene-data\__UPDATES\SVN_CHECKOUTS\metadata_SVN\METADATA_ALL.xlsx"
    )
    metadata = validate_nlho_metadata(metadata, files)
    metadata_fields = json.loads(METADATA_JSON.read_text())
    for tile in tqdm(tiles):
        tile_surveys_to_netcdf(
            tile,
            files,
            metadata,
            metadata_fields,
            r"p:\tgg-mariene-data\__UPDATES\NetCDF5M",
        )
