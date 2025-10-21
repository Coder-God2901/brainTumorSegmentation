from monai.apps import DecathlonDataset
from pathlib import Path
root = Path("data/Task01_BrainTumour")
root.mkdir(parents=True, exist_ok=True)
try:
	DecathlonDataset(
		root_dir=str(root),
		task="Task01_BrainTumour",
		section='training',
		download=True,
		cache_rate=0.0,
		runtime_cache=True,
		num_workers=1,
	)
	print("✅ Dataset prepared (downloaded). MONAI will avoid preloading all images into RAM.")
except MemoryError:
	print("⚠️ MemoryError: MONAI attempted to load too much data into memory. Try setting cache_rate=0.0 and runtime_cache=True, or run on a machine with more RAM.")
	raise
