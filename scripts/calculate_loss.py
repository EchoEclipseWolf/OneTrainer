"""
This script calculates the loss of a model.

It uses the GenerateLossesModel to calculate the loss.
It reads a JSON config file to configure the training process.
This script is used by the train.py script to calculate the loss of a model.
"""
from util.import_util import script_imports

script_imports()

import json

from modules.module.GenerateLossesModel import GenerateLossesModel
from modules.util.args.CalculateLossArgs import CalculateLossArgs
from modules.util.config.TrainConfig import TrainConfig


def main():
    """
    Calculates the loss of a model.

    Reads a JSON configuration file to configure the training process.
    Parses command line arguments using CalculateLossArgs.
    Initializes and starts the GenerateLossesModel trainer.
    """
    args = CalculateLossArgs.parse_args()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenerateLossesModel(train_config, args.output_path)
    trainer.start()


if __name__ == '__main__':
    main()
