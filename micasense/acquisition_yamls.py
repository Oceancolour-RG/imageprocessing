#!/usr/bin/env python3

import pytz
import yaml
import pyexiv2

from pathlib import Path
from packaging import version
from numpy import array, argsort
from typing import List, Tuple, Union, Optional
from datetime import datetime, timedelta

from pprint import pprint  # noqa


def assert_num_acqui(red_path: Path, blue_path: Path) -> Tuple[int, List[str]]:
    """
    Ensure that the number of acquisitions for each band
    is the same.
    Parameters
    ----------
    red_path : Path
        Path to red camera tif image data
    blue_path : Path
        Path to blue camera tif image data
    """
    list_acqui = sorted([f.name.split("_")[1] for f in red_path.glob("**/*_1.tif")])
    nacqi = len(list_acqui)
    test = (
        nacqi
        == len([f for f in red_path.glob("**/*_2.tif")])
        == len([f for f in red_path.glob("**/*_3.tif")])
        == len([f for f in red_path.glob("**/*_4.tif")])
        == len([f for f in red_path.glob("**/*_5.tif")])
        == len([f for f in blue_path.glob("**/*_6.tif")])
        == len([f for f in blue_path.glob("**/*_7.tif")])
        == len([f for f in blue_path.glob("**/*_8.tif")])
        == len([f for f in blue_path.glob("**/*_9.tif")])
        == len([f for f in blue_path.glob("**/*_10.tif")])
    )
    err_msg = (
        "Number of acquisitions between bands are not the same. "
        "Please Check image data"
    )
    assert test, err_msg

    return nacqi, list_acqui


def get_md(f: Union[Path, str]) -> pyexiv2.metadata.ImageMetadata:
    """return the pyexiv2 EXIF/XMP metadata object"""
    try:
        md = pyexiv2.ImageMetadata(str(f))
        md.read()
    except OSError:
        md = None
    except TypeError:
        md = None
    return md


def get_dls2_position(
    md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
) -> Tuple[Union[float, None], Union[float, None], Union[float, None]]:
    """get the WGS-84 latitude, longitude tuple as signed decimal degrees"""

    def convert_deg2decimal(coord: List) -> float:
        return (
            coord[0].__float__()
            + (coord[1].__float__() / 60.0)
            + (coord[2].__float__() / 3600.0)
        )

    lat_key = "Exif.GPSInfo.GPSLatitude"
    lon_key = "Exif.GPSInfo.GPSLongitude"
    alt_key = "Exif.GPSInfo.GPSAltitude"
    lat, lon, alt = None, None, None

    if (lat_key in md_keys) and (lon_key in md_keys) and (alt_key in md_keys):
        lat = convert_deg2decimal(md[lat_key].value)
        latref = md["Exif.GPSInfo.GPSLatitudeRef"].value
        lat = lat if latref == "N" else -lat

        lon = convert_deg2decimal(md[lon_key].value)
        lonref = md["Exif.GPSInfo.GPSLongitudeRef"].value
        lon = lon if lonref == "E" else -lon

        alt = md[alt_key].value.__float__()

    return lat, lon, alt


def get_utctime(
    md_keys: List[str], md: Union[pyexiv2.metadata.ImageMetadata, None]
) -> Union[datetime, None]:
    """Extract the datetime (to the nearest millisecond)"""

    utctime = None
    dt_key = "Exif.Image.DateTime"

    if md is not None:
        if dt_key in md_keys:
            utctime = datetime.strptime(md[dt_key].raw_value, "%Y:%m:%d %H:%M:%S")
            # utctime can also be obtained with DateTimeOriginal:
            # utctime = datetime.strptime(
            #     md["Exif.Photo.DateTimeOriginal"].raw_value, "%Y:%m:%d %H:%M:%S"
            # )

            # extract the millisecond from the EXIF metadata:
            subsec = int(md["Exif.Photo.SubSecTime"].raw_value)

            sign = -1.0 if subsec < 0 else 1.0
            millisec = sign * 1e3 * float("0.{}".format(abs(subsec)))

            utctime += timedelta(milliseconds=millisec)
            timezone = pytz.timezone("UTC")
            utctime = timezone.localize(utctime)

    return utctime


def find_imgfile(fpath: Path, fkey: str) -> Union[Path, None]:
    """
    Find file give it directrory (fpath) and file keywords (fkey).
    Returns a Path object if exists/found else None
    """
    flist = [f for f in fpath.glob(f"**/{fkey}")]
    imgf = None if not flist else flist[0]
    return imgf


def mean_level(bl_list: list) -> float:
    bl_sum, bl_num = 0.0, 0.0
    for x in bl_list:
        bl_sum += float(x)
        bl_num += 1.0
    return bl_sum / bl_num


def dls_present(md_keys: List[str], md: pyexiv2.metadata.ImageMetadata) -> bool:
    dls_keys = [
        "Xmp.DLS.HorizontalIrradiance",
        "Xmp.DLS.DirectIrradiance",
        "Xmp.DLS.SpectralIrradiance",
    ]
    return all([v in md_keys for v in dls_keys])


def supports_radiometric_calibration(md_keys: List[str]) -> bool:
    return "Xmp.MicaSense.RadiometricCalibration" in md_keys


def irradiance_scale_factor(
    md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
) -> float:
    """
    Get the calibration scale factor for the irradiance measurements
    in this image metadata. Due to calibration differences between
    DLS1 and DLS2, we need to account for a scale factor change in
    their respective units. This scale factor is pulled from the image
    metadata, or, if the metadata doesn't give us the scale, we assume
    one based on a known combination of tags
    """
    scale_factor = 0.0
    key = "Xmp.DLS.IrradianceScaleToSIUnits"
    # key = "Xmp.Camera.IrradianceScaleToSIUnits"

    if key in md_keys:
        # the metadata contains the scale
        scale_factor = float(md[key].value)
    elif "Xmp.DLS.HorizontalIrradiance" in md_keys:
        # DLS2 but the metadata is missing the scale, assume 0.01
        # For some reason, the DLS2 outputs processed irradiance
        # with units of micro-W/cm^2/nm;
        # Hence scale factor = 0.01
        # W/m^2/nm = 0.01 micro-W/cm^2/nm
        scale_factor = 0.01
    else:
        # DLS1, so we use a scale of 1
        scale_factor = 1.0
    return scale_factor


def horizontal_irradiance_valid(
    md_keys: List[str],
    firmware_version: str,
    camera_model: str,
) -> bool:
    """
    Defines if horizontal irradiance tag contains a value that can be trusted
    some firmware versions had a bug whereby the direct and scattered irradiance
    were correct, but the horizontal irradiance was calculated incorrectly
    """
    if "Xmp.DLS.HorizontalIrradiance" not in md_keys:
        return False

    if camera_model == "Altum":
        good_version = "1.2.3"
    elif camera_model == "RedEdge" or camera_model == "RedEdge-M":
        good_version = "5.1.7"
    else:
        raise ValueError(
            "Camera model is required to be RedEdge or Altum, not {} ".format(
                camera_model
            )
        )
    return version.parse(firmware_version) >= version.parse(good_version)


def focal_length_mm(md: pyexiv2.metadata.ImageMetadata, fp_xres: float) -> float:
    units = md["Xmp.Camera.PerspectiveFocalLengthUnits"].value
    focal_len_mm = 0.0
    if units == "mm":
        focal_len_mm = float(md["Xmp.Camera.PerspectiveFocalLength"].value)
    else:
        focal_len_px = float(md["Xmp.Camera.PerspectiveFocalLength"].value)
        focal_len_mm = focal_len_px / fp_xres
    return focal_len_mm


def get_auto_calibration_image(
    md_keys: List[str],
    md: pyexiv2.metadata.ImageMetadata,
    panel_albedo: Union[float, None],
    panel_region: Union[List[int], None],
    panel_serial: Union[str, None],
) -> bool:
    """
    True if this image is an auto-calibration image, where the camera has
    found and identified a calibration panel
    """
    # print("PLEASE TEST auto_calibration_image()")
    key = None
    for k in md_keys:
        k_ = k.lower()
        if ("xmp." in k_) and (".calibrationpicture" in k_):
            key = k

    if key is None:
        cal_tag = None
    else:
        cal_tag = int(md[key].value)

    return (
        cal_tag is not None
        and cal_tag == 2
        and panel_albedo is not None
        and panel_region is not None
        and panel_serial is not None
    )


def get_panel_albedo(
    md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
) -> Union[float, None]:
    """
    Surface albedo of the active portion of the reflectance panel as
    calculated by the camera (usually from the informatoin in the panel QR code)
    """
    # print("PLEASE TEST panel_albedo()")
    key = None
    for k in md_keys:
        if ("xmp." in k.lower()) and (".albedo" in k.lower()):
            key = k

    return None if key is None else float(md[key].value)


def get_panel_region(
    md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
) -> Union[None, List[int]]:
    """A 4-tuple containing image x,y coordinates of the panel active area"""
    # print("PLEASE TEST panel_region()")
    key, coords = None, None
    for k in md_keys:
        if ("xmp." in k.lower()) and (".reflectarea" in k.lower()):
            key = k

    if key is not None:
        c_ = [float(i) for i in md[key].value[0].split(",")]
        coords = list(zip(c_[0::2], c_[1::2]))

    return coords


def get_panel_serial(
    md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
) -> Union[str, None]:
    """The panel serial number as extracted from the image by the camera"""
    # print("PLEASE TEST panel_serial()")
    key = None
    for k in md_keys:
        if ("xmp." in k.lower()) and (".panelserial" in k.lower()):
            key = k

    return None if key is None else md[key].value


def band_dict_from_file(
    f: Path,
) -> Tuple[
    Union[dict, None],
    Union[dict, None],
    Union[float, None],
    Union[float, None],
    Union[float, None],
]:
    """Extract EXIF, XMP metadata from image file & return as a dict"""

    def str_or_none(
        key: str, md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
    ) -> Union[str, None]:
        return None if key not in md_keys else md[key].value

    def float_or_zero(
        key: str, md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
    ) -> float:
        return 0.0 if key not in md_keys else float(md[key].value)

    def asfloats(vlist: List[str]) -> List[float]:
        return [float(x) for x in vlist]

    def asfloats_or_none(
        key: str, md_keys: List[str], md: pyexiv2.metadata.ImageMetadata
    ) -> Union[List[float], None]:
        olist = None
        if key in md_keys:
            olist = [float(x) for x in md[key].value]
        return olist

    md = get_md(str(f))
    if md:

        md_keys = []
        md_keys.extend(md.exif_keys)
        md_keys.extend(md.iptc_keys)
        md_keys.extend(md.xmp_keys)
        # pprint(md_keys)

        kw = {"md_keys": md_keys, "md": md}
        panel_albedo = get_panel_albedo(**kw)
        panel_region = get_panel_region(**kw)
        panel_serial = get_panel_serial(**kw)
        auto_cal_image = get_auto_calibration_image(
            md_keys, md, panel_albedo, panel_region, panel_serial
        )

        firmware_version = md["Exif.Image.Software"].value.strip("v")
        camera_model = md["Exif.Image.Model"].value
        lat, lon, alt = get_dls2_position(**kw)

        misc_dict = {
            "dls_serialnum": str_or_none("Xmp.DLS.Serial", **kw),
            "dls_yaw": float_or_zero("Xmp.DLS.Yaw", **kw),  # radians
            "dls_pitch": float_or_zero("Xmp.DLS.Pitch", **kw),  # radians
            "dls_roll": float_or_zero("Xmp.DLS.Roll", **kw),  # radians
            "dls_latitude": lat,  # float or None
            "dls_longitude": lon,  # float or None
            "dls_altitde": alt,  # float or None
            "dls_utctime": get_utctime(**kw),  # datetime or None
            "dls_solarazi": float_or_zero("Xmp.DLS.SolarAzimuth", **kw),
            "dls_solarelevation": float_or_zero("Xmp.DLS.SolarElevation", **kw),
            "GPSAltitudeRef": float_or_zero("Exif.GPSInfo.GPSAltitudeRef", **kw),
            "GPSDOP": float_or_zero("Exif.GPSInfo.GPSDOP", **kw),
            "Xmp.Camera.GPSXYAccuracy": float_or_zero("Xmp.Camera.GPSXYAccuracy", **kw),
            "Xmp.Camera.GPSZAccuracy": float_or_zero("Xmp.Camera.GPSZAccuracy", **kw),
            "camera_model": camera_model,
            "firmware_version": firmware_version,
            "camera_make": md["Exif.Image.Make"].value,
            "flight_id": md["Xmp.MicaSense.FlightId"].value,
            "capture_id": md["Xmp.MicaSense.CaptureId"].value,
            "dls_firmware_version": str_or_none("Xmp.DLS.SwVersion", **kw),
            "image_size": [
                int(md["Exif.Image.ImageWidth"].value),
                int(md["Exif.Image.ImageLength"].value),
            ],
        }
        bps = float(md["Exif.Image.BitsPerSample"].value)
        fp_xres = float(md["Exif.Photo.FocalPlaneXResolution"].value)
        fp_yres = float(md["Exif.Photo.FocalPlaneYResolution"].value)

        md_dict = {
            "panel_albedo": panel_albedo,
            "panel_region": panel_region,
            "panel_serial": panel_serial,
            "auto_calibration_image": auto_cal_image,
            "dls_present": dls_present(**kw),
            "band_name": md["Xmp.Camera.BandName"].value,
            "dls_Ed": float_or_zero("Xmp.DLS.SpectralIrradiance", **kw),
            "dls_Ed_h": float_or_zero("Xmp.DLS.HorizontalIrradiance", **kw),
            "dls_Ed_d": float_or_zero("Xmp.DLS.DirectIrradiance", **kw),
            "dls_Ed_s": float_or_zero("Xmp.DLS.ScatteredIrradiance", **kw),
            "dls_EstimatedDirectLightVector": asfloats_or_none(
                "Xmp.DLS.EstimatedDirectLightVector", **kw
            ),
            "dls_wavelength": float_or_zero("Xmp.DLS.CenterWavelength", **kw),
            "dls_bandwidth": float_or_zero("Xmp.DLS.Bandwidth", **kw),
            "blacklevel": mean_level(md["Exif.Image.BlackLevel"].value),
            "darkpixels": mean_level(md["Xmp.MicaSense.DarkRowValue"].value),
            "exposure": float(md["Exif.Photo.ExposureTime"].value),
            "gain": float(md["Exif.Photo.ISOSpeed"].value) / 100.0,
            "rad_calibration": asfloats(md["Xmp.MicaSense.RadiometricCalibration"].value),
            "vignette_xy": asfloats(md["Xmp.Camera.VignettingCenter"].value),
            "vignette_poly": asfloats(md["Xmp.Camera.VignettingPolynomial"].value),
            "focal_length": float(md["Exif.Photo.FocalLength"].value),
            "persp_focal_length": float(md["Xmp.Camera.PerspectiveFocalLength"].value),
            "persp_focal_length_units": md[
                "Xmp.Camera.PerspectiveFocalLengthUnits"
            ].value,
            "focal_length_mm": focal_length_mm(md, fp_xres),
            "distortion_params": asfloats(md["Xmp.Camera.PerspectiveDistortion"].value),
            "principal_point": asfloats(md["Xmp.Camera.PrincipalPoint"].value.split(",")),
            "rig_relatives": asfloats(md["Xmp.Camera.RigRelatives"].value.split(",")),
            "rig_camera_index": int(md["Xmp.Camera.RigCameraIndex"].value),
            "wavelength_center": float(md["Xmp.Camera.CentralWavelength"].value),
            "wavelength_fwhm": float(md["Xmp.Camera.WavelengthFWHM"].value),
            "band_sensitivity": float(md["Xmp.Camera.BandSensitivity"].value),
            "camera_serialnum": md["Exif.Photo.BodySerialNumber"].value,
            "supports_radiometric_cal": supports_radiometric_calibration(md_keys),
            "irradiance_scale_factor": irradiance_scale_factor(**kw),
            "horizontal_irradiance_valid": horizontal_irradiance_valid(
                md_keys, firmware_version, camera_model
            ),
        }
    else:
        md_dict, misc_dict, bps, fp_xres, fp_yres = None, None, None, None, None

    return md_dict, misc_dict, bps, fp_xres, fp_yres


def get_syncxxxxset_folders(mpath: Path) -> Tuple[List[str], List[str]]:
    """
    Return the names of the SYNCXXXXSET folders and the parent
    folder names, which should be "red_cam" and "blue_cam".
    """
    sync_d, cam_d = [], []
    for d in sorted(mpath.glob("**/SYNC*SET")):
        if d.is_file():
            continue

        parent_d = d.parts[-2]
        if "cam" in parent_d.lower():
            if "red" in parent_d.lower():
                cam_d.append(parent_d)
            if "blue" in parent_d.lower():
                cam_d.append(parent_d)

        sync_d.append(d.name)

    return list(set(sync_d)), list(set(cam_d))


def get_acqui_id(mpath: Path, cam_folders: List[str], sync_folder: str) -> List[str]:
    """
    Return the names of all acqusition ids within sync_dir across
    both camera's

    Parameters
    ----------
    mpath : Path
        The parent micasense data directory that contains the cam_folders
    cam_folders : List[str]
        A list of the camera folder names, e.g. ["red_cam", "blue_cam"]
    sync_folder : str
        The name of the "SYNCXXXXSET" folder
    Returns
    -------
    acqi_ids : List[str]
        List of acquisition id's, e.g.
        ["0000", "0001", "0002", ....., "XXXX"]
    """
    acqi_ids = []
    for cam in cam_folders:
        # cam = "red_cam" or "blue_cam"
        for f in (mpath / cam / sync_folder).glob("**/*.tif"):
            acqi_ids.append(f.stem.split("_")[1])

    return sorted(list(set(acqi_ids)))


def get_bandnum(tif: Path) -> int:
    """Return the band number from the filename"""
    return int(tif.stem.split("_")[-1])


def get_dls2_ed(md_dict: dict) -> Tuple[List[float], List[float], str]:
    """
    Return the DLS2 downwelling solar irradiance at the sorted
    (ascending) wavelengths and the units

    Parameters
    ----------
    md_dict : dict
        Preliminary image-set metadata dictionary

    Returns
    -------
    dls2_ed : List[float]
        The DLS2 downwelling solar irradiance at the sorted (ascending)
        wavelengths
    dls2_wvl : List[float]
        The sorted (ascending) DLS2 wavelengths
    dls2_units : str
        The SI units of solar irradiance
    """
    id_, isf_ = "image_data", "irradiance_scale_factor"
    dls2_wvl, dls2_ed = [], []
    for k in md_dict[id_]:
        dls2_wvl.append(md_dict[id_][k]["dls_wavelength"])
        dls2_ed.append(md_dict[id_][k]["dls_Ed_h"] * md_dict[id_][k][isf_])
    dls2_units = "W/m^2/nm"

    # There probably is a way to do the sorting without using
    # numpy, but, this is quick and easy to implement
    dls2_wvl = array(dls2_wvl, order="C")
    dls2_ed = array(dls2_ed, order="C")
    s_ix = argsort(dls2_wvl)

    # DO NOT USE list(nd.adrray) as yaml still thinks its a numpy binary
    return dls2_ed[s_ix].tolist(), dls2_wvl[s_ix].tolist(), dls2_units


def create_img_acqi_yamls(
    dpath: Union[Path, str],
    opath: Optional[Union[Path, str]] = None,
) -> None:
    """
    Create a yaml file for each image acquisition containing the
    relevant metadata needed to process image data into radiance
    and reflectance.

    Parameters
    ----------
    dpath : Path
        Parent directory containing a set of folders named "red_cam"
        and "blue_cam". Within each of these folders there should be
        a set of sub-directories named "SYNCXXXXSET". Filenames
        will be relative to `dpath`
    opath : Path [Optional]
        The output path where the yamls are stored. If not provided,
        then a folder named "metadata" will be created within the
        specified `dpath` folder.

    Notes
    -----
    ** Each yaml file contains the following metadata:
       yaml_dict = {
           "base_path": "/path/to/somewhere/micasense"
           "band_1": {
               "file": str,  # e.g. /metadata/IMG_0000_1.tif
               "band"
           }
       }
    ** The absolute path of "file" can be recovered by:
       fn = Path(base_path) / file

    ** This code assumes that the IMG_XXXX_*.tif files have been
       moved from their native 000/, 001/, ... folders directly
       into the SYNCXXXXSET folder, e.g.
       red_cam/
          |--> SYNC0009SET/
                  |--> dat/
                        |--> diag0.dat
                        |--> gpslog0.dat
                        |--> hostlog0.dat
                        |--> paramlog0.dat
                  |--> IMG_0000_1.tif
                  |--> IMG_0000_2.tif
                  ...
                  |--> IMG_0000_5.tif
                  ...
                  ...
                  |--> IMG_XXXX_*.tif
       blue_cam/
          |--> SYNC00009SET/
                  |--> dat/
                        |--> diag0.dat
                        |--> gpslog0.dat
                        |--> hostlog0.dat
                        |--> paramlog0.dat
                  |--> IMG_0000_6.tif
                  |--> IMG_0000_7.tif
                  ...
                  |--> IMG_0000_10.tif
                  ...
                  ...
                  |--> IMG_XXXX_*.tif

        see micasense.restructure_dirs.restructure()
    """

    def add2dict(md_dict: dict, fkey: str, val_list: List[float]):
        if val_list:
            if len(val_list) == 1:
                md_dict[f"{fkey}"] = val_list[0]
            else:
                raise Exception(
                    f"more than one unique instance of {fkey} = {val_list} - recode"
                )
        else:
            md_dict[f"{fkey}"] = None

    # ensure dpath is a Path object
    dpath = Path(dpath) if isinstance(dpath, str) else dpath

    if opath:
        o_ypath = Path(opath) if isinstance(opath, str) else opath
        o_ypath.mkdir(exist_ok=True)

    # 1) The "SYNCXXXXSET" sub-directories in the "red_cam" folder
    #    should match those in the "blue_cam" directory. Thus, get
    #    a list of all SYNCXXXXSET folders.
    sync_d, cam_d = get_syncxxxxset_folders(dpath)

    # 2) Iterate through each "SYNCXXXXSET" folder and extract the
    #    relevant image metadata,  including whether all images in
    #    a set are valid (i.e. not corrupt)
    print("processing:")
    for d in sync_d:
        print(f"   SYNC folder: {d}")
        # get a sorted list of all acquisition ids in `d`
        acqi_ids = get_acqui_id(dpath, cam_d, d)

        # create output yaml path for each `d`
        if not opath:
            o_ypath = dpath / "metadata" / d
            o_ypath.mkdir(exist_ok=True, parents=True)

        # 3) iterate through the acquisition id's and extrac metadata
        for acq in acqi_ids:

            # initiate the output dict
            md_dict = {"image_data": dict(), "base_path": str(dpath)}

            # get a sorted list of tif files for this acquisition
            unsorted_tifs = []
            for c_pth in cam_d:
                for f in (dpath / c_pth / d).glob(f"**/IMG_{acq}_*.tif"):
                    unsorted_tifs.append(f)

            # sort based on band number
            tifs = sorted(unsorted_tifs, key=lambda f: get_bandnum(f))

            # Note that some tifs may be corrupt.
            done = False
            valid = True if len(tifs) == 10 else False
            bps_ls, fp_xres_ls, fp_yres_ls = [], [], []
            for f in tifs:
                band_dict, misc_dict, bps, fp_xres, fp_yres = band_dict_from_file(f)
                # bname = f"band_{get_bandnum(f):02d}"  # e.g. "band_01" -> "band_10"
                bname = f.name  # e.g. "IMG_0000_1.tif"
                md_dict["image_data"][bname] = band_dict

                if not band_dict:
                    # flag set if any tiff are corrupt
                    valid = False
                else:
                    # valid = True
                    md_dict["image_data"][bname]["filename"] = str(f.relative_to(dpath))

                if misc_dict and not done:
                    for key in misc_dict:
                        md_dict[key] = misc_dict[key]
                    done = True

                # unsure if bps, fp_xres or fp_yres are band-dependent
                bps_ls.append(bps) if bps else None
                fp_xres_ls.append(fp_xres) if fp_xres else None
                fp_yres_ls.append(fp_yres) if fp_yres else None

            # append the unique values of bps, fp_xres and fp_yres
            add2dict(md_dict, "bits_persample", list(set(bps_ls)))
            add2dict(md_dict, "focalplane_xres", list(set(fp_xres_ls)))
            add2dict(md_dict, "focalplane_yres", list(set(fp_yres_ls)))

            md_dict["valid_set"] = valid
            md_dict["ppk_lat"] = None
            md_dict["ppk_lon"] = None
            md_dict["ppk_height"] = None
            md_dict["ppk_lat_uncert"] = None
            md_dict["ppk_lon_uncert"] = None
            md_dict["ppk_alt_uncert"] = None

            # Add the DLS2 solar irradiance spectra into the main portion
            # of the dictionary.
            if valid:
                dls2_ed, dls2_wvl, dls2_units = get_dls2_ed(md_dict)
            else:
                dls2_ed, dls2_wvl, dls2_units = None, None, "W/m^2/nm"

            md_dict["dls2_ed"] = dls2_ed  # List[float]
            md_dict["dls2_ed_wvl"] = dls2_wvl  # List[float]
            md_dict["dls2_ed_units"] = dls2_units  # str

            # write to yaml
            yaml_ofile = o_ypath / f"IMG_{acq}.yaml"
            with open(yaml_ofile, "w") as fid:
                yaml.dump(md_dict, fid, default_flow_style=False)

    return
