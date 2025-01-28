"""
This script starts the Train UI.

It uses the TrainUI to create the UI.
This script is a standalone UI that allows the user to train models.
"""
from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI


def main():
    """
    Starts the Train UI.

    Initializes and starts the TrainUI.
    Closes the UI after the main loop finishes.
    """
    ui = TrainUI()
    ui.mainloop()
    ui.close()


if __name__ == '__main__':
    main()
