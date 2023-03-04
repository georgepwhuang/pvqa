import os
import pathlib

CODE_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = pathlib.Path(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = ROOT_DIR.joinpath("data")
LOG_DIR = ROOT_DIR.joinpath("logs")
MODEL_DIR = ROOT_DIR.joinpath("models")
CONFIG_DIR = ROOT_DIR.joinpath("config")
