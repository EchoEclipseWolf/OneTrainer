"""
This script generates masks for images.

It uses the ClipSegModel, RembgHumanModel, RembgModel, or MaskByColor to generate the masks.
It uses the GenerateMasksArgs to parse the command line arguments.
It is used to generate masks for images, which can then be used for inpainting.
"""
from util.import_util import script_imports

script_imports()

from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor
from modules.module.RembgHumanModel import RembgHumanModel
from modules.module.RembgModel import RembgModel
from modules.util.args.GenerateMasksArgs import GenerateMasksArgs
from modules.util.enum.GenerateMasksModel import GenerateMasksModel

import torch


def main():
    """
    Generates masks for images.

    Parses command line arguments using GenerateMasksArgs.
    Initializes the specified model for generating masks.
    Masks the folder of images using the specified model.
    Prints an error message if there is an error while processing the image.
    """
    args = GenerateMasksArgs.parse_args()

    model = None
    if args.model == GenerateMasksModel.CLIPSEG:
        model = ClipSegModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.REMBG:
        model = RembgModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.REMBG_HUMAN:
        model = RembgHumanModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.COLOR:
        model = MaskByColor(torch.device(args.device), args.dtype.torch_dtype())

    model.mask_folder(
        sample_dir=args.sample_dir,
        prompts=args.prompts,
        mode=args.mode,
        threshold=args.threshold,
        smooth_pixels=args.smooth_pixels,
        expand_pixels=args.expand_pixels,
        alpha=args.alpha,
        error_callback=lambda filename: print("Error while processing image " + filename),
        include_subdirectories=args.include_subdirectories
    )


if __name__ == "__main__":
    main()
