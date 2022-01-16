import os, sys
from pathlib import Path, PurePosixPath

from lib_utils import nltkconfig

reporoot = nltkconfig.getRepoRoot()
print(reporoot)
print(nltkconfig.getDataFolder())
