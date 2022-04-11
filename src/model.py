import logging

import typer

from typing import Optional
from pathlib import Path

logging.basicConfig(filename='../data/logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

app = typer.Typer()


@app.command()
def train(dataset: Optional[Path] = Path("../data/dataset/train/")):
    """
    API call for model.train()
    :param dataset: path to train dataset
    :return: None
    """


@app.command()
def evaluate(dataset: Optional[Path] = Path("../data/dataset/val/")):
    """
    API call to model.evaluate()
    :param dataset: path to validation dataset
    :return: None
    """


@app.command()
def demo(video_path: Optional[Path] = Path("../data/demo_video.avi")):
    """
    API call to run video and show Model work in Runtime
    :param video_path: path to demo video
    :return: None
    """




if __name__ == '__main__':
    app()