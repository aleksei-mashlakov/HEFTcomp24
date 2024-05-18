import argparse

from src.pipeline.baseline import BaselinePipeline
from src.pipeline.gluonts import GluonTsPipeline
from src.pipeline.lgbm import LGBPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", choices=["gluonts", "baseline", "lgbm"], default="baseline")


if __name__ == "__main__":
    args = parser.parse_args()
    match args.pipeline:
        case "gluonts":
            pipeline = GluonTsPipeline.default()
        case "baseline":
            pipeline = BaselinePipeline.default()
        case "lgbm":
            pipeline = LGBPipeline.default()
        case _:
            raise ValueError("The pipeline name doesnt match")

    pipeline.run()
