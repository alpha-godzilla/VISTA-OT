"""Default paths for the V100 server deployment.

Each path can be overridden with an environment variable when the repository is
used on another machine.
"""

import os


LLAVA_MODEL_PATH = os.environ.get(
    "VISTA_LLAVA_MODEL_PATH",
    "/data/sun_yuxi/models/llava-1.5-7b-hf",
)
COCO_ROOT = os.environ.get(
    "VISTA_COCO_ROOT",
    "/data/sun_yuxi/datasets/coco",
)
COCO_TRAIN2014_PATH = os.path.join(COCO_ROOT, "train2014")
COCO_VAL2014_PATH = os.path.join(COCO_ROOT, "val2014")
COCO_ANNOTATIONS_PATH = os.path.join(COCO_ROOT, "annotations")
