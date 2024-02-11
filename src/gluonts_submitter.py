from src.pipeline.competition import GluonTsPipeline

if __name__ == "__main__":
    pipeline = GluonTsPipeline.default()
    pipeline.run()
