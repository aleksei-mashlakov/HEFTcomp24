import argparse

from prefect import flow

from src.pipeline.baseline import BaselinePipeline
from src.pipeline.gluonts import GluonTsPipeline
from src.pipeline.lgbm import LGBPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", choices=["gluonts", "baseline", "lgbm"], default="baseline")


@flow(log_prints=True)
def run(name: str) -> None:
    match name:
        case "gluonts":
            pipeline = GluonTsPipeline.default()
        case "baseline":
            pipeline = BaselinePipeline.default()
        case "lgbm":
            pipeline = LGBPipeline.default()
        case _:
            raise ValueError("The pipeline name doesn't match available")

    pipeline.run()


if __name__ == "__main__":
    args = parser.parse_args()

    run.serve(name="submission_pipeline", cron="0 9 * jan-may *", parameters={"name": args.pipeline})
