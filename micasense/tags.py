#!/usr/bin/env python3

import pyexiv2
from pathlib import Path
from fractions import Fraction
from typing import Tuple, Optional

from .utils import get_ave_focallen, get_ave_pc


def ddeg_to_fraction(
    ddeg: float, islat: bool, precision: int = 6
) -> Tuple[str, Tuple[Fraction]]:
    """
    Convert decimal degrees to a List[Fraction]

    Parameters
    ----------
    ddeg : float
        decimal degree (latitude or longitude)
    islat : bool
        Whether `ddeg` is latitude (True) or longitude (False)
    precision : int
        The precision of the seconds (from degree, minutes, seconds).
        Max. precision is 6

    Returns
    -------
    ref : str
        "N" or "S" for latitude
        "E" or "W" for longitude
    frac_d : List[Fraction]:
        Rational representation of the decimal degrees
    """

    if islat:
        ref = "S" if ddeg < 0 else "N"
    else:
        ref = "W" if ddeg < 0 else "E"

    base = abs(ddeg) - abs(int(ddeg))
    dd = abs(int(ddeg))
    mm = int(60.0 * base)
    ss = 60.0 * (60.0 * base - mm)

    # apply the necessary precision on the seconds
    denom_ss = 10**precision
    numer_ss = int(ss * denom_ss)

    # see https://exiv2.org/tags.html for documentation:
    # "When degrees, minutes and seconds are expressed, the format is dd/1,mm/1,ss/1"
    frac_d = [
        Fraction(numerator=dd, denominator=1),  # add degress as dd/1
        Fraction(numerator=mm, denominator=1),  # add minutes as mm/1
        Fraction(numerator=numer_ss, denominator=denom_ss),  # add seconds
    ]

    return ref, frac_d


def add_exif(
    acq_meta: dict,
    tiff_fn: Path,
    image_pp: int,
    compression: int,
    imshape: Optional[Tuple[int, int]] = None,
    image_name: Optional[str] = None,
    principal_point: Optional[str] = None,
) -> None:
    """
    Add Exif tags to individual tiff files.

    Parameters
    ----------
    acq_meta : dict
        acquisition metadata
    tiff_fn : Path
        filename of tiff image

    image_pp : int
        image preprocessing,
        1 = raw (vignetting + dark current)
        2 = undistorted (vignetting + dark current + undistorting)
        3 = aligned (vignetting + dark current + undistorting + alignment)
        4 = HOPE retrieval product

    compression : int
        image compression exif tag
            "jpeg": 7
            "lzw": 5
            "packbits": 3277,
            "deflate": 32946
            "webp": 34927
            "none": 1

    imshape : Tuple[int, int] [Optional]
        image shape (nrows, ncols) - specify if downsampling has occurred.

    image_name : str [Optional]
        image name (e.g. IMG_0274_1.tif) - this is only used if image_pp < 4

    principal_point : str [Optional]
        The principal point (mm) - required for image_pp = 2
        e.g. principal_point = "2.47158,1.80417"
    """
    if not image_name and image_pp < 4:
        raise ValueError("`image_name` is required when `image_pp` < 4")
    if not principal_point and image_pp == 2:
        raise ValueError("`principal_point` is required when `image_pp` == 2")

    # Add the relevant Exif metadata to be used by Agisoft Metashape
    out_md = pyexiv2.metadata.ImageMetadata(str(tiff_fn))
    out_md.read()

    # add tags
    d1 = 1000000
    dfx = acq_meta["focalplane_xres"]  # pixels / mm
    dfy = acq_meta["focalplane_yres"]  # pixels / mm
    latref, lat_fdeg = ddeg_to_fraction(ddeg=acq_meta["ppk_lat"], islat=True)
    lonref, lon_fdeg = ddeg_to_fraction(ddeg=acq_meta["ppk_lon"], islat=False)
    alt_f = Fraction(int(acq_meta["dls_altitde"] * d1), d1, _normalize=False)

    if imshape:
        nrows, ncols = imshape
    else:
        ncols, nrows = acq_meta["image_size"]

    exiftags_md = {
        "Exif.Photo.FocalPlaneXResolution": Fraction(int(dfx * d1), d1),
        "Exif.Photo.FocalPlaneYResolution": Fraction(int(dfy * d1), d1),
        "Exif.Photo.FocalPlaneResolutionUnit": 4,
        "Exif.GPSInfo.GPSAltitude": alt_f,
        "Exif.GPSInfo.GPSAltitudeRef": f"{int(acq_meta['GPSAltitudeRef'])}",
        "Exif.GPSInfo.GPSDOP": Fraction(int(acq_meta["GPSDOP"] * d1), d1),
        "Exif.GPSInfo.GPSLatitude": lat_fdeg,
        "Exif.GPSInfo.GPSLatitudeRef": latref,
        "Exif.GPSInfo.GPSLongitude": lon_fdeg,
        "Exif.GPSInfo.GPSLongitudeRef": lonref,
        "Exif.Image.DateTime": acq_meta["dls_utctime"].strftime("%Y:%m:%d %H:%M:%S"),
        "Exif.Photo.SubSecTime": f"{1000*acq_meta['dls_utctime'].microsecond}",
        "Exif.GPSInfo.GPSVersionID": "2 2 0 0",
        "Exif.Image.BitsPerSample": 16,
        "Exif.Image.ImageLength": nrows,
        "Exif.Image.ImageWidth": ncols,
        "Exif.Image.Make": "MicaSense",
        "Exif.Image.Model": "RedEdge-M",
        "Exif.Image.Orientation": 1,
        "Exif.Image.PhotometricInterpretation": 1,
        "Exif.Image.PlanarConfiguration": 1,
        "Exif.Photo.FocalLength": Fraction(550000000, 100000000, _normalize=False),
        "Exif.Image.Compression": compression,
    }

    xmptags_md = {
        "Xmp.xmp.Camera.PerspectiveFocalLengthUnits": "mm",
        "Xmp.xmp.Camera.GPSXYAccuracy": f"{acq_meta['Xmp.Camera.GPSXYAccuracy']}",
        "Xmp.xmp.Camera.GPSZAccuracy": f"{acq_meta['Xmp.Camera.GPSZAccuracy']}",
        "Xmp.xmp.Camera.ModelType": "perspective",
        "Xmp.xmp.Camera.RigName": "RedEdge-M",
    }

    if image_pp >= 3:  # aligned image or retrieval product
        # Unclear what the homography does to the principal point and focal
        # length. Let's assume that the average is taken across all images.
        xmptags_md["Xmp.xmp.Camera.PrincipalPoint"] = get_ave_pc(acq_meta)
        xmptags_md["Xmp.xmp.Camera.PerspectiveFocalLength"] = get_ave_focallen(acq_meta)

    else:
        # copy over the perspective focal length
        xmptags_md["Xmp.xmp.Camera.PerspectiveFocalLength"] = acq_meta["image_data"][
            image_name
        ]["focal_length_mm"]

        xmptags_md["Xmp.xmp.Camera.RigRelatives"] = ",".join(
            [f"{_}" for _ in acq_meta["image_data"][image_name]["rig_relatives"]]
        )
        xmptags_md["Xmp.xmp.Camera.RigCameraIndex"] = str(
            acq_meta["image_data"][image_name]["rig_camera_index"]
        )
        xmptags_md["Xmp.xmp.Camera.RigRelativesReferenceRigCameraIndex"] = "1"

        if image_pp == 1:  # raw image
            # copy the principal point
            xmptags_md["Xmp.xmp.Camera.PrincipalPoint"] = ",".join(
                [f"{_}" for _ in acq_meta["image_data"][image_name]["principal_point"]]
            )

            xmptags_md["Xmp.xmp.Camera.PerspectiveDistortion"] = ",".join(
                [f"{_}" for _ in acq_meta["image_data"][image_name]["distortion_params"]]
            )

        if image_pp == 2:  # undistored image
            # Undistortion uses cv2.getOptimalNewCameraMatrix, which optimises
            # the intrinsic camera parameters, such as, fx, fy, cx and cy.
            # cv2 outputs these parameters in units of pixels, but exif requires
            # units of inches, cm, mm or micrometres. We cannot convert units
            # as additional parameters are needed. Hence, we assumed that the
            # focal length was conserved.
            xmptags_md["Xmp.xmp.Camera.PrincipalPoint"] = principal_point

    if image_pp < 4:
        xmptags_md["Xmp.xmp.Camera.BandName"] = acq_meta["image_data"][image_name][
            "band_name"
        ]
        xmptags_md["Xmp.xmp.Camera.CentralWavelength"] = str(
            acq_meta["image_data"][image_name]["wavelength_center"]
        )
        xmptags_md["Xmp.xmp.Camera.WavelengthFWHM"] = str(
            acq_meta["image_data"][image_name]["wavelength_fwhm"]
        )

    for k in exiftags_md:
        out_md[k] = pyexiv2.ExifTag(k, exiftags_md[k])
    for k in xmptags_md:
        out_md[k] = pyexiv2.XmpTag(k, xmptags_md[k])

    # write tags
    out_md.write()

    return
