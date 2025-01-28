"""
This script trains a model.

It uses the GenericTrainer to train the model.
It uses the TrainArgs to parse the command line arguments.
It reads a JSON config file to configure the training process.
This is the main training script, and is used in conjunction with create_train_files.py.
"""
from util.import_util import script_imports

script_imports()

import json

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig


def main():
    """
    Trains a model.

    Parses command line arguments using TrainArgs.
    Reads a JSON configuration file to configure the training process.
    Initializes and starts the GenericTrainer.
    Starts and ends the training process, handling keyboard interrupts gracefully.
    """
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenericTrainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()


if __name__ == '__main__':
    main()
