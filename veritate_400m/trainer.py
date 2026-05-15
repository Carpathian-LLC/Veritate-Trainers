# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_400m: medium-large vanilla Veritate (200-800M range). Same
#   canonical trunk as the 80M / 200M plugins; tuned defaults for the band
#   where activation checkpointing earns its keep. Shape / LR / batch presets
#   live in manifest.json. Training loop is plugins/common/vanilla_trainer.py.
# plugins/veritate_400m/plugin.py
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

PLUGIN_ID = "veritate_400m"


# ------------------------------------------------------------------------------------
# Functions

if __name__ == "__main__":
    vanilla_trainer.run(plugin_id=PLUGIN_ID, here=HERE)
