import os
import sys
import shutil
import zipfile
import urllib.request

ZIP_URL = "https://github.com/kimkim00/UIT-ViSD4SA/archive/refs/heads/master.zip"
OUT_DIR = os.path.join("data", "raw", "uit_visd4sa")
TMP_ZIP = os.path.join("data", "raw", "uit_visd4sa_master.zip")
TMP_EXTRACT = os.path.join("data", "raw", "_tmp_uit_visd4sa")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TMP_ZIP), exist_ok=True)

print("Downloading:", ZIP_URL)
urllib.request.urlretrieve(ZIP_URL, TMP_ZIP)
print("Saved zip:", TMP_ZIP)

# Extract zip
if os.path.exists(TMP_EXTRACT):
    shutil.rmtree(TMP_EXTRACT)
os.makedirs(TMP_EXTRACT, exist_ok=True)

print("Extracting zip...")
with zipfile.ZipFile(TMP_ZIP, "r") as z:
    z.extractall(TMP_EXTRACT)

# 🔥 Tự động tìm thư mục con đầu tiên
subdirs = [d for d in os.listdir(TMP_EXTRACT) if os.path.isdir(os.path.join(TMP_EXTRACT, d))]
if not subdirs:
    print("ERROR: No extracted subdirectory found.")
    sys.exit(1)

root = os.path.join(TMP_EXTRACT, subdirs[0])
print("Detected extracted folder:", root)

wanted_ext = {".json", ".jsonl", ".txt", ".tsv", ".csv"}
copied = 0

for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext in wanted_ext:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, root)
            dst = os.path.join(OUT_DIR, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

print(f"Copied {copied} files into: {OUT_DIR}")

print("\nListing top-level of OUT_DIR:")
for name in sorted(os.listdir(OUT_DIR))[:50]:
    print(" -", name)

print("\nDONE.")
