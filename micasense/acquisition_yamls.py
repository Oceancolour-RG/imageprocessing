#!/usr/bin/env python3

import sys  # noqa
import pytz
import yaml
import pyexiv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from packaging import version
from typing import List, Tuple, Union, Optional
from datetime import datetime, timedelta, timezone
from scipy.signal import correlate, correlation_lags

from pprint import pprint  # noqa

import micasense.load_yaml as ms_yaml


def add_ppk_to_yaml(
    yml_f: Union[Path, str],
    ppk_lat: Union[float, None],
    ppk_lon: Union[float, None],
    ppk_height: Union[float, None],
) -> None:
    """add ppk lat/lon/height to yaml document"""
    acq_dict = ms_yaml.load_all(yaml_file=yml_f)

    acq_dict["ppk_lat"] = ppk_lat
    acq_dict["ppk_lon"] = ppk_lon
    acq_dict["ppk_height"] = ppk_height

    with open(yml_f, "w") as fid:
        yaml.dump(acq_dict, fid, default_flow_style=False)


def datetime_from_event_text(date: str, time: str, leap_sec: float) -> datetime:
    """
    Get a datetime object from a row of an *_events.pos file

    Parameters
    ----------
    date : str
        YYYY/MM/DD e.g. "2021/11/26"
    time : str
        HH:MM:SS.SSS e.g. "02:12:51.4"
    leap_sec : float
        The current leap second used to convert GPS time to UTC time

    Returns
    -------
    dt : datetime
        datetime object associated with the date and time
    """

    year, month, day = [int(x) for x in date.split("/")]
    hour, minute = [int(x) for x in time.split(":")[0:2]]
    dec_sec = float(time.split(":")[-1])
    microsec = 1e6 * (dec_sec - int(dec_sec))

    return datetime(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=int(dec_sec),
        microsecond=int(microsec),
        tzinfo=timezone.utc,
    ) - timedelta(seconds=leap_sec)


def get_obs_start_end_events(
    fname: Union[Path, str], leap_sec: float
) -> Tuple[datetime, datetime]:
    """
    Get the observation start and end UTC time from the
    header of the *_events.pos file without having to
    read a potentially large file

    Parameters
    ----------
    fname : Path or str
        filename of *_events.pos
    leap_sec : float
        The current leap second used to convert GPS time to UTC time

    Returns
    -------
    obs_start : datetime
        observation start datetime object
    obs_end : datetime
        observation end datetime object
    """

    with open(fname, "r", encoding="utf-8") as fid:
        for i, row in enumerate(fid):
            if not row.startswith("%"):
                break

            if "% obs start" in row:
                date, time = row.strip().split()[4:6]
                obs_start = datetime_from_event_text(date, time, leap_sec)

            if "% obs end" in row:
                date, time = row.strip().split()[4:6]
                obs_end = datetime_from_event_text(date, time, leap_sec)

    return obs_start, obs_end


def open_events(
    fname: Union[Path, str], leap_sec: float, get_frame_rate: bool = False
) -> Tuple[
    List[float], List[float], List[float], List[datetime], Union[List[float], None]
]:
    """
    Parameters
    ----------
    fname : Path or str
        filename of *_events.pos file
    leap_sec : float
        The current leap second used to convert GPS time to UTC time
    get_frame_rate : bool [default=False]
        Whether to return the frame rate of sequential trigger events

    Returns
    -------
    lat : List[float]
        Latitudes (decimal degrees) of trigger events recorded by Reach M2
    lon : List[float]
        Longitudes (decimal degrees) of trigger events recorded by Reach M2
    height : List[float]
        Ellipsoid heights of trigger events recorded by Reach M2
    dt_ls : List[datetime]
        datetime (UTC) of trigger events recorded by Reach M2
    reach_frate : List[float] or None
        if get_frame_rate is True:
            reach_frate -> frame rate (seconds) of trigger events recorded
                           by Reach M2
        if get_frame_rate is False:
            reach_frate = None
    """
    with open(fname, encoding="utf-8") as fid:
        contents = fid.readlines()

    lat, lon, height, dt_ls = [], [], [], []
    reach_frate = [] if get_frame_rate else None

    cnt = 0
    for i in range(len(contents)):
        if contents[i].startswith("%"):
            continue

        row = contents[i].strip().split()
        dt = datetime_from_event_text(row[0], row[1], leap_sec)
        if cnt > 0:
            reach_frate.append((dt - prev_dt).total_seconds())  # noqa

        lat.append(float(row[2]))
        lon.append(float(row[3]))
        height.append(float(row[4]))
        dt_ls.append(dt)

        prev_dt = dt  # noqa
        cnt += 1

    return lat, lon, height, dt_ls, reach_frate


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
    for d in sync_d:
        # get a sorted list of all acquisition ids in `d`
        acqi_ids = get_acqui_id(dpath, cam_d, d)

        # create output yaml path for each `d`
        if not opath:
            o_ypath = dpath / "metadata" / d
            o_ypath.mkdir(exist_ok=True, parents=True)

        # 3) iterate through the acquisition id's and extrac metadata
        for acq in acqi_ids:

            # initiate the output dict
            md_dict = {
                "image_data": dict(),
                "base_path": str(dpath),
            }

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

            # write to yaml
            yaml_ofile = o_ypath / f"IMG_{acq}.yaml"
            with open(yaml_ofile, "w") as fid:
                yaml.dump(md_dict, fid, default_flow_style=False)


def plot_frate_comp(
    reach_frate: np.ndarray,
    ms_frate: np.ndarray,
    corr: np.ndarray,
    lags: np.ndarray,
    opng: Optional[Union[Path, str]] = None,
) -> None:
    start = 0.50
    stop = 1.50
    bwidth = 0.02

    bedges = np.arange(start=start, stop=stop + bwidth, step=bwidth)
    cbins = 0.5 * (bedges[0:-1] + (np.roll(bedges, -1))[0:-1])

    rfreq, _ = np.histogram(reach_frate, bins=bedges, density=True)
    msfreq, _ = np.histogram(ms_frate, bins=bedges, density=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
    axes[0].plot(cbins, rfreq * np.diff(bedges), "r.-", label="Reach M2 trigger events")
    axes[0].plot(cbins, msfreq * np.diff(bedges), "k.--", label="Micasense image-sets")
    axes[0].set_xlabel("frame rate (seconds)")
    axes[0].set_ylabel("PDF")
    axes[0].legend(loc=1)

    max_ix = np.argmax(corr)
    axes[1].plot(lags, corr, "k.")
    axes[1].plot(lags[max_ix], corr[max_ix], "rs")
    axes[1].set_ylabel("Correlation")
    axes[1].set_xlabel("lag indices")
    axes[1].annotate(
        "Micasense vs Reach UTC times\n",
        xy=(0.01, 0.95),
        xycoords="axes fraction",
        fontsize=10,
    )
    axes[1].annotate(
        f"Index = {lags[max_ix]}",
        xy=(lags[max_ix], corr[max_ix]),
        xycoords="data",
        fontsize=10,
        color="r",
    )

    fig.subplots_adjust(hspace=0.2)

    if opng is not None:
        fig.savefig(opng, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)


def get_crosscorr(in1: np.ndarray, in2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corr = (
        correlate(in1=in1, in2=in2, mode="full")
        / ((in1**2).sum() * (in2**2).sum()) ** 0.5
    )

    lags = correlation_lags(in1_len=in1.size, in2_len=in2.size, mode="full")
    return corr, lags


def append_reach2yaml(
    yaml_path: Union[Path, str],
    events_file: Union[Path, str],
    leap_sec: float = 18,
) -> None:
    """
    Append Reach M2 lat/lon/height into the yamls created by the
    create_img_acqi_yamls() function.

    Parameters
    ----------
    yaml_path : Path or str
        The directory containing the yaml's for a given flight
    events_file : Path or str
        The events.pos file associated with the flight
    leap_sec : float (default = 18)
        The leap seconds used to convert the Reach M2 GPS time to UTC.
        see https://endruntechnologies.com/support/leap-seconds
        As of 13 Dec. 2016, the current GPS-UTC leap seconds is 18

    Notes
    -----
      +++ The number of trigger events in the  *_events.pos file may not
          match the number tiff's acquired during the flight. The reason
          for this is unclear at the moment, but may be caused by captu-
          ring image data as the reach  m2 is turning on or in the proc-
          ess of acquiring a satellite fix.

          Histograms of the Reach M2 trigger event frame rate and the
          Micasense frame rate are nearly identical, where a mode exists
          at 1.0 seconds with ~90% of the frame rates existing between
          0.90 to 1.10 seconds. A cross-correlation between the frame
          rates can be then be used so sync the lat/lon/height

          There is no correlation between the Micasense frame rate and
          the max. exposure time for an acquisition.
    """
    print(f"Appending Reach M2 GPS data to yamls in {yaml_path}")
    obs_start, obs_end = get_obs_start_end_events(events_file, leap_sec=leap_sec)
    ms_start = ms_yaml.load_all(yaml_file=yaml_path / "IMG_0000.yaml", key="dls_utctime")
    epoch_time = obs_start if obs_start < ms_start else ms_start

    # load the events_file
    lat, lon, height, reach_dt, reach_frate = open_events(
        events_file, leap_sec=leap_sec, get_frame_rate=True
    )
    reach_dt_since_epoch = [(dt - epoch_time).total_seconds() for dt in reach_dt]

    tifs = sorted(yaml_path.glob("**/IMG_*.yaml"))
    n_ms = len(tifs)
    n_trg = len(lat)
    n_missing = n_ms - n_trg

    ms_dt = []
    ms_dt_since_epoch = []
    nodata_ix = []
    for i, f in enumerate(tifs):
        ms_time = ms_yaml.load_all(yaml_file=f, key="dls_utctime")
        if ms_time is None:
            nodata_ix.append(i)
            continue

        ms_dt_since_epoch.append((ms_time - epoch_time).total_seconds())
        ms_dt.append(ms_time)

    # ---------------------- #
    #         PLOT           #
    # ---------------------- #
    ms_dt_since_epoch = np.array(ms_dt_since_epoch, order="C", dtype=np.float64)
    reach_dt_since_epoch = np.array(reach_dt_since_epoch, order="C", dtype=np.float64)

    corr, lags = get_crosscorr(in1=ms_dt_since_epoch, in2=reach_dt_since_epoch)
    start_ix = lags[np.argmax(corr)]
    print(f"    Number of recorded trigger events in Reach M2: {n_trg}")
    print(f"    Number of Micasense image sets: {n_ms}")
    print(f"    Number of missing trigger events in Reach M2: {n_missing}")
    print(f"    Micasense alignment index from cross-correlation: {start_ix}")
    print(f"    Number of invalid Micasense image sets: {len(nodata_ix)}")

    ms_frate = [(ms_dt[i] - ms_dt[i - 1]).total_seconds() for i in range(1, len(ms_dt))]
    ms_frate = np.array(ms_frate, order="C", dtype=np.float64)
    reach_frate = np.array(reach_frate, order="C", dtype=np.float64)

    opng = yaml_path / f"{yaml_path.parts[-1]}_frame_rate_crosscorr.png"
    plot_frate_comp(reach_frate, ms_frate, corr, lags, opng)

    # ------------------------- #
    #  APPEND REACH M2 TO YAML  #
    # ------------------------- #
    for i in range(n_trg):
        f_ix = i + start_ix
        if i in nodata_ix:
            continue
        add_ppk_to_yaml(
            yml_f=tifs[f_ix], ppk_lat=lat[i], ppk_lon=lon[i], ppk_height=height[i]
        )
