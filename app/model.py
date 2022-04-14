import logging
from pathlib import Path
from typing import Optional

import typer

from inference import evaluate_test_dataset, run_demo_video
from MRCNN.rocks import train as model_train

logging.basicConfig(filename='./data/logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

app = typer.Typer()


@app.command()
def train(dataset: Optional[Path] = Path("../data/dataset/train/")):
    """
    API call for model.train()
    :param dataset: path to train dataset
    :return: None
    """
    model_train()


@app.command()
def evaluate(dataset: Optional[Path] = Path("../data/dataset/val/")):
    """
    API call to model.evaluate()
    :param dataset: path to validation dataset
    :return: None
    """

    evaluate_test_dataset(dataset)


@app.command()
def demo(video_path: Optional[Path] = Path("../data/demo/demo_video.avi")):
    """
    API call to run video and show Model work in Runtime
    :param video_path: path to demo video
    :return: None
    """
    run_demo_video(video_path)
















if __name__ == '__main__':
    app()
