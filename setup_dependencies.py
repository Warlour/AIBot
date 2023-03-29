#!/usr/bin/env python3

# For ZIP archives
import urllib.request, zipfile, io
# For pip
import sys, subprocess

def main():
    archives = [
        ("https://www.unicode.org/Public/UCD/latest/ucd/UCD.zip", "ucd"),
        ("https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip", "unihan"),
    ]

    python_packages = ["flask"]

    print("Downloading and extracting archives...")

    for archive_info in archives:
        url, extract_path = archive_info
        archive_name = url.split('/')[-1]

        print(f"Downloading {archive_name} from {url}")

        with urllib.request.urlopen(url) as req:
            zip_data = io.BytesIO(req.read())

        print(f"Extracting {archive_name} into {extract_path}")

        with zipfile.ZipFile(zip_data) as archive:
            # SECURITY: It appears that zipfile.extractall attempts to sanitize
            # the ZIP file's filenames to prevent writing files outside of the
            # path given to it. However, it may still be risky to use it on
            # untrusted files. See https://bugs.python.org/issue40763 and
            # https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extractall
            archive.extractall(path=extract_path)

    print("Installing packages...")

    for package_name in python_packages:
        # This is the recommended way to use pip, per
        # https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

    print("Done!")

if __name__ == "__main__":
    main()
