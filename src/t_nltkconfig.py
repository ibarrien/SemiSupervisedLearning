from lib_utils import projconfig
from lib_utils import nltkconfig

if __name__ == '__main__':
	reporoot = projconfig.getRepoRoot()
	print(f"reporoot {reporoot}")
	print(f"data {projconfig.getDataFolder()}")
	print(f"nltkconfig.getDataFolder() {nltkconfig.getDataFolder()}")
