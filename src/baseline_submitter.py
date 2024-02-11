from src.pipeline.competition import BaselinePipeline

if __name__ == "__main__":
    pipeline = BaselinePipeline.default()
    pipeline.run()
