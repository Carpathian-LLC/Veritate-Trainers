# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_200m: 200M vanilla Veritate, standalone single-size trainer. Shape and
#   training presets live in manifest.json; training loop is
#   plugins/common/vanilla_trainer.py.
# plugins/veritate_200m/plugin.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys

HERE      = os.path.dirname(os.path.abspath(__file__))
COMMON    = os.path.normpath(os.path.join(HERE, "..", "common"))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COMMON)
sys.path.insert(0, REPO_ROOT)

import vanilla_trainer


# ------------------------------------------------------------------------------------
# Constants

PLUGIN_ID = "veritate_200m"


# ------------------------------------------------------------------------------------
# Functions

if __name__ == "__main__":
    vanilla_trainer.run(plugin_id=PLUGIN_ID, here=HERE)
