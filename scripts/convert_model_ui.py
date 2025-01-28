"""
This script starts the Convert Model UI.

It uses the ConvertModelUI to create the UI.
This script is a standalone UI that allows the user to convert models.
It is connected to the convert_model.py script, as it provides a UI for it.
"""
from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUI import ConvertModelUI


def main():
    """
    Starts the Convert Model UI.

    Initializes and starts the ConvertModelUI.
    """
    ui = ConvertModelUI(None)
    ui.mainloop()


if __name__ == '__main__':
    main()
