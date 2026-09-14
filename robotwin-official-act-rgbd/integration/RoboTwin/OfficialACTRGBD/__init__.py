import sys
from pathlib import Path

REPOSITORY = Path("/ssd/hhw/depth-model/repos/official-act-rgbd")
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from official_act_rgbd.deploy_policy import *  # noqa: F401,F403,E402

