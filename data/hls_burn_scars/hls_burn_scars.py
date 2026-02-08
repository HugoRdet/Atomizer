import os
from glob import glob
import datasets

_CITATION = """\
@software{HLS_Foundation_2023,
    author = {Phillips, Christopher and Roy, Sujit and Ankur, Kumar and Ramachandran, Rahul},
    doi    = {10.57967/hf/0956},
    month  = aug,
    title  = {{HLS Foundation Burnscars Dataset}},
    url    = {https://huggingface.co/ibm-nasa-geospatial/hls_burn_scars},
    year   = {2023}
}
"""

_DESCRIPTION = """\
This dataset contains Harmonized Landsat and Sentinel-2 imagery of burn scars and the associated masks for the years 2018-2021 over the contiguous United States. There are 804 512x512 scenes. Its primary purpose is for training geospatial machine learning models.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars"

_LICENSE = "cc-by-4.0"

_URLS = {
    "hls_burn_scars": {
        "train/val": "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/resolve/main/hls_burn_scars.tar.gz"
    }
}

class HLSBurnScars(datasets.GeneratorBasedBuilder):
    """MIT Scene Parsing Benchmark dataset."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="hls_burn_scars", version=VERSION, description=_DESCRIPTION),
    ]

    def _info(self):    
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "annotation": datasets.Image(),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]

        data_dirs = dl_manager.download_and_extract(urls)
        train_data = os.path.join(data_dirs['train/val'], "training")
        val_data = os.path.join(data_dirs['train/val'], "validation")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": train_data,
                    "split": "training",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data": val_data,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data": val_data,
                    "split": "testing",
                },
            )
        ]

    def _generate_examples(self, data, split):
        files = glob(f"{data}/*_merged.tif")
        for idx, filename in enumerate(files):
            if filename.endswith("_merged.tif"):
                annotation_filename = filename.replace('_merged.tif', '.mask.tif')
                yield idx, {
                    "image": {"path": filename}, 
                    "annotation": {"path": annotation_filename}
                }