"""
ShapeStacks 외부 데이터 다운로드 및 전처리 스크립트
"""
import os
import subprocess
import sys
import tarfile
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "shapestacks")


def download_file(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    fname = url.split("/")[-1]
    dest_path = os.path.join(dest_dir, fname)
    if os.path.exists(dest_path):
        print(f"  [skip] {fname} already exists")
        return dest_path
    print(f"  Downloading {fname} ...")
    subprocess.run(
        ["curl", "-L", "-C", "-", "-o", dest_path, url],
        check=True,
    )
    return dest_path


def extract_tar_gz(path, dest_dir):
    print(f"  Extracting {os.path.basename(path)} ...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")
    print(f"  Done extracting.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    urls = {
        "manual": "http://shapestacks-file.robots.ox.ac.uk/static/download/v1/ShapeStacks-Manual.md",
        "meta": "http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-meta.tar.gz",
        "rgb": "http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-rgb.tar.gz",
    }

    print("=" * 60)
    print(" ShapeStacks Dataset Download")
    print("=" * 60)

    # Download manual
    print("\n[1/3] Manual...")
    download_file(urls["manual"], DATA_DIR)

    # Download & extract metadata
    print("\n[2/3] Metadata (labels)...")
    meta_path = download_file(urls["meta"], DATA_DIR)
    if meta_path.endswith(".tar.gz") and not os.path.exists(
        os.path.join(DATA_DIR, "shapestacks-meta")
    ):
        extract_tar_gz(meta_path, DATA_DIR)

    # Download & extract RGB images
    print("\n[3/3] RGB images (~33GB) ...")
    rgb_path = download_file(urls["rgb"], DATA_DIR)
    if rgb_path.endswith(".tar.gz") and not os.path.exists(
        os.path.join(DATA_DIR, "shapestacks-rgb")
    ):
        extract_tar_gz(rgb_path, DATA_DIR)

    print("\n" + "=" * 60)
    print(" Download complete!")
    print(f" Data location: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
