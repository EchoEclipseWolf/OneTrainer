"""
This script installs ZLUDA.

It uses the ZLUDAInstaller to install ZLUDA.
It is a standalone script that installs ZLUDA, allowing the other scripts to utilize it.
"""
from util.import_util import script_imports

script_imports(allow_zluda=False)

import sys

from modules.zluda import ZLUDAInstaller

"""
Installs ZLUDA.

Retrieves the path to the ZLUDA installation directory.
Installs ZLUDA and makes a copy of the installation.
Prints an error message if there is an error while installing.
Prints the path of the installed ZLUDA if the installation is successfull.
"""
if __name__ == '__main__':
    try:
        zluda_path = ZLUDAInstaller.get_path()
        ZLUDAInstaller.install(zluda_path)
        ZLUDAInstaller.make_copy(zluda_path)
    except Exception as e:
        print(f'Failed to install ZLUDA: {e}')
        sys.exit(1)

    print(f'ZLUDA installed: {zluda_path}')
