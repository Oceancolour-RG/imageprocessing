#!/usr/bin/env python3


import re
from pathlib import Path


def restructure(camera_path: Path):
    """
    Restructure the MicaSense image data directories to
    minimise the number of subfolders. Use this script
    with caution!!

    The RedEdge-MX and RedEdge-MX-Blue natively stores the
    image data (IMG_YYYY_B.tif) in the following directories.
    SYNCXXXXSET/
         |--> diag0.dat
         |--> gpslog0.dat
         |--> hostlog0.dat
         |--> paramlog0.dat
         |--> 000/
               |--> IMG_0000_B.tif
               ...
               |--> IMG_0199_B.tif
         |--> 001/
               |--> IMG_0200_B.tif
               ...
               |--> IMG_0399_B.tif
    Where B represents the band number;
        B = 1 -> 5 for RedEdge-MX
        B = 6 -> 10 for RedEdge-MX-Blue

    Each 000/, 001/ etc folder contain 200 image acquisitons
    per band resulting in a total of 1000 tif files per folder.

    This script restructures the directories to the following:
    SYNCXXXXSET/
        |--> dat/
              |--> diag0.dat
              |--> gpslog0.dat
              |--> hostlog0.dat
              |--> paramlog0.dat
        |--> IMG_0000_B.tif
        |--> IMG_0001_B.tif
        ...
        |--> IMG_YYYY_B.tif


    Parameters
    ----------
    camera_path : Path
        Path to the camera directory,
        e.g.
        Path("/path/to/20211126_WoodmanPoint/micasense/red_cam/") or
        Path("/path/to/20211126_WoodmanPoint/micasense/blue_cam/")
    """

    def move_file(fn: Path, opath: Path) -> None:
        """move file to opath"""
        target_fn = opath / fn.name
        opath.mkdir(exist_ok=True)
        fn.rename(target_fn)  # move file to target_fn

    def match_syncset(dname: str) -> bool:
        """match with SYNCXXXXSET folder, where XXXX are digits"""
        return True if re.search(r"SYNC\d{4,}SET", dname, re.I) else False

    def match_imgtif(fname: str) -> bool:
        """
        match with IMG_XXXX_Z.tif; where, XXXX => digits; z ==> 1 to 10
        """
        return True if re.search(r"^IMG_\d{4,}_([1-9]|10).tif$", fname, re.I) else False

    for d in camera_path.iterdir():
        if d.is_file() or not match_syncset(d.name):
            continue
        # e.g. d = camera_path / "SYNC0009SET"
        dpth = d / "dat"  # doesn't exist yet
        mpth = d / "misc"  # doesn't exist yet
        for subd in d.iterdir():
            # subd will either be the 000/, 001/, ... folders or
            # the .dat files (if present). If this script is run
            # twice then subd will .tif files or dat/ or misc/ folders
            if subd.is_file():
                # In case this script is run twice, keep any IMG_XXXX_Z.tif
                # files in the SYNCXXXXSET folders
                if not match_imgtif(subd.name):
                    opth_ = dpth if ".dat" in subd.suffix else mpth
                    # print(f"moving {subd} to {opth_}")
                    move_file(fn=subd, opath=opth_)

            else:
                # Check that the folders are 000/, 001/ etc
                if re.fullmatch(r"\d{0,3}", subd.name):  # skip dat/, misc/ folders
                    for tif in subd.iterdir():
                        if tif.is_file() and match_imgtif(tif.name):
                            # print(f"moving {tif} to {d}")
                            move_file(fn=tif, opath=d)  # move tif files

                    # remove 000/, 001/ folders if they are empty (best to
                    # play it safe when removing directories)
                    if not any(subd.iterdir()):
                        # print(f"removing {subd}")
                        subd.rmdir()  # remove directory
