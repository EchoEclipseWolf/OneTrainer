"""
This script starts the Caption UI.

It uses the CaptionUI to create the UI.
This script is a standalone UI that allows the user to caption images.
It can work independently, or can be used in conjunction with other scripts, such as generate_captions.py.
"""
from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUI import CaptionUI
from modules.util.args.CaptionUIArgs import CaptionUIArgs


def main():
    """
    Starts the Caption UI.

    Parses command line arguments using CaptionUIArgs.
    Initializes and starts the CaptionUI.
    """
    args = CaptionUIArgs.parse_args()

    ui = CaptionUI(None, args.dir, args.include_subdirectories)
    ui.mainloop()


if __name__ == '__main__':
    main()
