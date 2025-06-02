import os

photo_folder = "data/only_photos"
illus_folder = "data/only_illustrations"
# 1) Create/clear output dirs
os.makedirs(photo_folder, exist_ok=True)
os.makedirs(illus_folder, exist_ok=True)
for folder in (photo_folder, illus_folder):
    for fname in os.listdir(folder):
        fp = os.path.join(folder, fname)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass