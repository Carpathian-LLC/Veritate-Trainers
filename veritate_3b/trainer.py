# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_3b: 1B-3B vanilla Veritate. Defaults turn on activation
#   checkpointing and 8-bit AdamW so the optimizer fits a 24-48 GB GPU.
#   Shape / LR / batch presets live in manifest.json; training loop is
#   plugins/common/vanilla_trainer.py.
# plugins/veritate_3b/plugin.py
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

PLUGIN_ID = "veritate_3b"


# ------------------------------------------------------------------------------------
# Functions

if __name__ == "__main__":
    vanilla_trainer.run(plugin_id=PLUGIN_ID, here=HERE)
