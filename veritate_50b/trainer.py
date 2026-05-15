# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_50b: 13B-50B vanilla Veritate. Cluster-scale shapes; activation
#   checkpointing and 8-bit AdamW are mandatory and still not sufficient on
#   a single card. Shape / LR / batch presets live in manifest.json;
#   training loop is plugins/common/vanilla_trainer.py.
# plugins/veritate_50b/plugin.py
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

PLUGIN_ID = "veritate_50b"


# ------------------------------------------------------------------------------------
# Functions

if __name__ == "__main__":
    vanilla_trainer.run(plugin_id=PLUGIN_ID, here=HERE)
