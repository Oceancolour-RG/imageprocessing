#!/usr/bin/env python3
# coding: utf-8
"""
RedEdge Metadata Management Utilities

Copyright 2017 MicaSense, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# Support strings in Python 2 and 3
from __future__ import unicode_literals

import pytz
import math
import pyexiv2

from os.path import isfile
from pathlib2 import Path
from packaging import version
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple, List


class MetadataFromExif(object):
    """Container for Micasense image metadata extracted from EXIF metadata"""

    def __init__(
        self,
        filename: Union[str, Path],
    ):

        if not isfile(filename):
            raise IOError("Input path is not a file")

        md = pyexiv2.ImageMetadata(str(filename))
        md.read()

        self.exif = md

        meta_dict = dict()
        if md.exif_keys:
            for keys in md.exif_keys:
                meta_dict[keys] = md[keys].raw_value

        if md.iptc_keys:
            for keys in md.iptc_keys:
                meta_dict[keys] = md[keys].raw_value

        if md.xmp_keys:
            for keys in md.xmp_keys:
                meta_dict[keys] = md[keys].raw_value

        self.meta = meta_dict

    def get_all(self) -> dict:
        """Get all extracted metadata items"""
        return self.meta

    def get_item(
        self, item: str, index: Optional[Union[None, int]] = None
    ) -> Union[float, int, str, None]:
        """Get metadata item by Namespace:Parameter"""
        val = None
        try:
            val = self.exif[item]
            if index is not None:
                try:
                    if isinstance(val, unicode):
                        val = val.encode("ascii", "ignore")
                except NameError:
                    # throws on python 3 where unicode is undefined
                    pass
                if isinstance(val, str) and len(val.split(",")) > 1:
                    val = val.split(",")
                val = val[index]
        except KeyError:
            # print ("Item "+item+" not found")
            pass
        except IndexError:
            print(
                "Item {0} is length {1}, index {2} is outside this range.".format(
                    item, len(self.exif[item]), index
                )
            )
        return val

    def print_all(self) -> None:
        for item in self.get_all():
            print("{}: {}".format(item, self.get_item(item)))

    def dls_present(self) -> bool:
        dls_keys = [
            "Xmp.DLS.HorizontalIrradiance",
            "Xmp.DLS.DirectIrradiance",
            "Xmp.DLS.SpectralIrradiance",
        ]
        return any([v in self.meta.keys() for v in dls_keys])

    def supports_radiometric_calibration(self) -> bool:
        return "Xmp.MicaSense.RadiometricCalibration" in self.meta.keys()

    def position(self) -> Tuple[float, float, float]:
        """get the WGS-84 latitude, longitude tuple as signed decimal degrees"""

        def convert_deg2decimal(coord: List) -> float:
            return (
                coord[0].__float__()
                + (coord[1].__float__() / 60.0)
                + (coord[2].__float__() / 3600.0)
            )

        lat = convert_deg2decimal(self.exif["Exif.GPSInfo.GPSLatitude"].value)
        latref = self.exif["Exif.GPSInfo.GPSLatitudeRef"].value
        lat = lat if latref == "N" else -lat

        lon = convert_deg2decimal(self.exif["Exif.GPSInfo.GPSLongitude"].value)
        lonref = self.exif["Exif.GPSInfo.GPSLongitudeRef"].value
        lon = lon if lonref == "E" else -lon

        alt = float(self.exif["Exif.GPSInfo.GPSAltitude"].value)
        return lat, lon, alt

    def utc_time(self) -> datetime:
        """Get the timezone-aware datetime of the image capture"""
        utctime = datetime.strptime(
            self.exif["Exif.Image.DateTime"].raw_value, "%Y:%m:%d %H:%M:%S"
        )
        # utctime can also be obtained with DateTimeOriginal:
        # utctime = datetime.strptime(
        #     self.exif["Exif.Photo.DateTimeOriginal"].raw_value, "%Y:%m:%d %H:%M:%S"
        # )

        # extract the millisecond from the EXIF metadata:
        subsec = int(self.exif["Exif.Photo.SubSecTime"].raw_value)

        sign = -1.0 if subsec < 0 else 1.0
        millisec = sign * 1e3 * float("0.{}".format(abs(subsec)))

        utctime += timedelta(milliseconds=millisec)
        timezone = pytz.timezone("UTC")
        utctime = timezone.localize(utctime)
        return utctime

    def dls_pose(self) -> Tuple[float, float, float]:
        """get DLS pose as local earth-fixed yaw, pitch, roll in radians"""
        pose_keys = ["Xmp.DLS.Yaw", "Xmp.DLS.Pitch", "Xmp.DLS.Roll"]
        if all([v in self.meta.keys() for v in pose_keys]):
            yaw = float(self.exif["Xmp.DLS.Yaw"].value)
            pitch = float(self.exif["Xmp.DLS.Pitch"].value)
            roll = float(self.exif["Xmp.DLS.Roll"].value)
        else:
            yaw = pitch = roll = 0.0
        return yaw, pitch, roll

    def rig_relatives(self) -> Union[List[float], None]:
        if "Xmp.Camera.RigRelatives" in self.meta.keys():
            return [
                float(i) for i in self.exif["Xmp.Camera.RigRelatives"].value.split(",")
            ]
        else:
            return None

    def capture_id(self) -> Union[None, str]:
        return self.exif["Xmp.MicaSense.CaptureId"].value

    def flight_id(self) -> Union[None, str]:
        return self.exif["Xmp.MicaSense.FlightId"].value

    def camera_make(self) -> Union[None, str]:
        # this should return "micasense"
        return self.exif["Exif.Image.Make"].value

    def camera_model(self) -> Union[None, str]:
        # this should return "RedEdge-M"
        return self.exif["Exif.Image.Model"].value

    def firmware_version(self) -> Union[None, str]:
        return self.exif["Exif.Image.Software"].value

    def band_name(self) -> Union[None, str]:
        # for Red-Camera:
        #     Blue, Green, Red, NIR, Red edge
        # for Blue-Camera:
        #     Blue-444, Green-531, Red-650, Red edge-705, Red edge-740
        return self.exif["Xmp.Camera.BandName"].value

    def band_index(self) -> Union[None, int]:
        # for Red-Camera
        #     0, 1, 2, 3, 4
        # for Blue-Camera
        #     5, 6, 7, 8, 9
        k = "Xmp.Camera.RigCameraIndex"
        return int(self.exif[k].value) if k in self.meta.keys() else None

    def exposure(self) -> float:
        """extract the exposure (integration time) in seconds"""
        exp = float(self.exif["Exif.Photo.ExposureTime"].value)
        # correct for incorrect exposure in some legacy RedEdge firmware versions
        if self.camera_model() != "Altum":
            if math.fabs(exp - (1.0 / 6329.0)) < 1e-6:
                exp = 0.000274
        return exp

    def gain(self) -> float:
        """extract the image gain"""
        return float(self.exif["Exif.Photo.ISOSpeed"].value) / 100.0

    def image_size(self) -> Tuple[int, int]:
        """extract the image size (ncols, nrows)"""
        return int(self.exif["Exif.Image.ImageWidth"].value), int(
            self.exif["Exif.Image.ImageLength"].value
        )

    def center_wavelength(self) -> float:
        """extract the central wavelength (nm)"""
        return float(self.exif["Xmp.Camera.CentralWavelength"].value)

    def bandwidth(self) -> float:
        """extract the bandwidth (nm)"""
        return float(self.exif["Xmp.Camera.WavelengthFWHM"].value)

    def radiometric_cal(self) -> List[float]:
        """extract the radiometric calibration coefficients"""
        return [float(v) for v in self.exif["Xmp.MicaSense.RadiometricCalibration"].value]

    def black_level(self) -> float:
        """Extract the mean dark current"""
        if "Exif.Image.BlackLevel" in self.meta.keys():
            bl_sum, bl_num = 0.0, 0.0
            for x in self.exif["Exif.Image.BlackLevel"].value:
                bl_sum += float(x)
                bl_num += 1.0
            mean_bl = bl_sum / bl_num

        else:
            mean_bl = 0.0
        return mean_bl

    def dark_pixels(self) -> float:
        """
        Get the average of the optically covered pixel values
        Note: these pixels are raw, and have not been radiometrically
              corrected. Use the black_level() method for all
              radiomentric calibrations
        """
        total, num = 0.0, 0.0
        for pixel in self.exif["Xmp.MicaSense.DarkRowValue"].value:
            total += float(pixel)
            num += 1.0
        return total / float(num)

    def bits_per_pixel(self) -> int:
        """
        get the number of bits per pixel, which defines the
        maximum digital number value in the image
        """
        return int(self.exif["Exif.Image.BitsPerSample"].value)

    def vignette_center(self) -> List[float]:
        """get the vignette center in X and Y image coordinates"""
        k = "Xmp.Camera.VignettingCenter"
        return [float(v) for v in self.exif[k].value] if k in self.meta.keys() else None

    def vignette_polynomial(self) -> List[float]:
        """
        get the radial vignette polynomial in the order
        it's defined within the metadata
        """
        k = "Xmp.Camera.VignettingPolynomial"
        return [float(v) for v in self.exif[k].value] if k in self.meta.keys() else None

    def distortion_parameters(self) -> List[float]:
        return [float(v) for v in self.exif["Xmp.Camera.PerspectiveDistortion"].value]

    def principal_point(self) -> List[float]:
        return [float(v) for v in self.exif["Xmp.Camera.PrincipalPoint"].value.split(",")]

    def focal_plane_resolution_px_per_mm(self) -> Tuple[float, float]:
        fp_x_resolution = float(self.exif["Exif.Photo.FocalPlaneXResolution"].value)
        fp_y_resolution = float(self.exif["Exif.Photo.FocalPlaneYResolution"].value)
        return fp_x_resolution, fp_y_resolution

    def focal_length_mm(self) -> float:
        key = "Xmp.Camera.PerspectiveFocalLengthUnits"
        units = None if key not in self.meta.keys() else self.exif[key].value
        focal_len_mm = 0.0
        if units == "mm":
            focal_len_mm = float(self.exif["Xmp.Camera.PerspectiveFocalLength"].value)
        else:
            focal_len_px = float(self.exif["Xmp.Camera.PerspectiveFocalLength"].value)
            focal_len_mm = focal_len_px / self.focal_plane_resolution_px_per_mm()[0]
        return focal_len_mm

    def sfactor_35mm(self) -> float:
        """
        extract the 35mm scale factor, sf[35], such that,
        35 mm equivalent focal length = sf[35] * focal_length
        """

        def __calc_backend(pxl_um: float, ncols: int, nrows: int) -> float:
            """
            Backend calculation of the 35 mm scale factor

            Parameters
            ----------
            pxl_um : pixel size (micrometers, um) [float]
            ncols : image width (or number of image columns) [int]
            nrows : image height (or number of image rows) [int]
            """
            pxl_mm = pxl_um / 1000.0  # pixel size in mm
            diag_img = ((ncols * pxl_mm) ** 2 + (nrows * pxl_mm) ** 2) ** 0.5
            diag_35mm = ((36.0) ** 2 + (24.0) ** 2) ** 0.5
            return diag_35mm / diag_img

        w, h = self.image_size()
        pxl_size = 3.75  # micrometers
        if self.camera_model() == "Altum":
            if self.band_name().lower() == "lwir":
                pxl_size = 12.0  # micrometers
            else:
                pxl_size = 3.45

        return __calc_backend(pxl_size, w, h)

    def focal_length_35_mm_eq(self) -> float:
        # pyexiv2 cannot access the Composite keys including:
        # Composite:FocalLength35efl
        return float(self.exif["Exif.Photo.FocalLength"].value) * self.sfactor_35mm()

    def irradiance_scale_factor(self) -> float:
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

        if key in self.meta.keys():
            # the metadata contains the scale
            scale_factor = float(self.exif[key].value)
        elif "Xmp.DLS.HorizontalIrradiance" in self.meta.keys():
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

    def horizontal_irradiance_valid(self) -> bool:
        """
        Defines if horizontal irradiance tag contains a value that can be trusted
        some firmware versions had a bug whereby the direct and scattered irradiance
        were correct, but the horizontal irradiance was calculated incorrectly
        """
        if "Xmp.DLS.HorizontalIrradiance" not in self.meta.keys():
            return False

        version_string = self.firmware_version().strip("v")
        if self.camera_model() == "Altum":
            good_version = "1.2.3"
        elif self.camera_model() == "RedEdge" or self.camera_model() == "RedEdge-M":
            good_version = "5.1.7"
        else:
            raise ValueError(
                "Camera model is required to be RedEdge or Altum, not {} ".format(
                    self.camera_model()
                )
            )
        return version.parse(version_string) >= version.parse(good_version)

    def __float_or_zero(self, key: str) -> float:
        return 0.0 if key not in self.meta.keys() else float(self.exif[key].value)

    def __get_irrad(self, key: str) -> float:
        return self.__float_or_zero(key) * self.irradiance_scale_factor()

    def spectral_irradiance(self) -> float:
        """
        Raw spectral irradiance measured by an irradiance sensor.
        Calibrated to W/m^2/nm using irradiance_scale_factor, but
        not corrected for angles
        """
        return self.__get_irrad("Xmp.DLS.SpectralIrradiance")

    def horizontal_irradiance(self) -> float:
        """
        Horizontal irradiance at the earth's surface below the DLS on the
        plane normal to the gravity vector at the location (local flat
        plane spectral irradiance)
        """
        return self.__get_irrad("Xmp.DLS.HorizontalIrradiance")

    def scattered_irradiance(self) -> float:
        """scattered component of the spectral irradiance"""
        return self.__get_irrad("Xmp.DLS.ScatteredIrradiance")

    def direct_irradiance(self) -> float:
        """
        direct component of the spectral irradiance on a ploane normal
        to the vector towards the sun
        """
        return self.__get_irrad("Xmp.DLS.DirectIrradiance")

    def solar_azimuth(self) -> float:
        """solar azimuth at the time of capture, as calculated by the camera system"""
        return self.__float_or_zero("Xmp.DLS.SolarAzimuth")

    def solar_elevation(self) -> float:
        """solar elevation at the time of capture, as calculated by the camera system"""
        return self.__float_or_zero("Xmp.DLS.SolarElevation")

    def estimated_direct_vector(self) -> Union[List[float], None]:
        """estimated direct light vector relative to the DLS2 reference frame"""
        ed_vect = None
        key = "Xmp.DLS.EstimatedDirectLightVector"
        if key in self.meta.keys():
            ed_vect = [float(v) for v in self.exif[key].value]
        return ed_vect

    def auto_calibration_image(self) -> bool:
        """
        True if this image is an auto-calibration image, where the camera has
        found and identified a calibration panel
        """
        # print("PLEASE TEST auto_calibration_image()")
        key = None
        for k in self.meta.keys():
            k_ = k.lower()
            if ("xmp." in k_) and (".calibrationpicture" in k_):
                key = k

        if key is None:
            cal_tag = None
        else:
            cal_tag = int(self.exif[key].value)

        return (
            cal_tag is not None
            and cal_tag == 2
            and self.panel_albedo() is not None
            and self.panel_region() is not None
            and self.panel_serial() is not None
        )

    def panel_albedo(self) -> Union[float, None]:
        """
        Surface albedo of the active portion of the reflectance panel as
        calculated by the camera (usually from the informatoin in the panel QR code)
        """
        # print("PLEASE TEST panel_albedo()")
        key = None
        for k in self.meta.keys():
            if ("xmp." in k.lower()) and (".albedo" in k.lower()):
                key = k

        return None if key is None else float(self.exif[key].value)

    def panel_region(self) -> Union[None, List[int]]:
        """A 4-tuple containing image x,y coordinates of the panel active area"""
        # print("PLEASE TEST panel_region()")
        key, coords = None, None
        for k in self.meta.keys():
            if ("xmp." in k.lower()) and (".reflectarea" in k.lower()):
                key = k

        if key is not None:
            c_ = [float(i) for i in self.exif[key].value[0].split(",")]
            coords = list(zip(c_[0::2], c_[1::2]))

        return coords

    def panel_serial(self) -> Union[str, None]:
        """The panel serial number as extracted from the image by the camera"""
        # print("PLEASE TEST panel_serial()")
        key = None
        for k in self.meta.keys():
            if ("xmp." in k.lower()) and (".panelserial" in k.lower()):
                key = k

        return None if key is None else self.exif[key].value


class MetadataFromDict(object):
    """
    Container for Micasense image metadata extracted from yaml dictionary
    """

    def __init__(
        self,
        filename: Union[str, Path],
        metadata_dict: dict,
    ):
        """
        Parameters
        ----------
        filename : Path or str
            The image (.tif) filename, used to extract the relevant
            metadata from the metadata_dict
        metadata_dict : dict
            The image-set acquisition metadata dictionary extracted from a
            yaml file.
        """

        if isinstance(filename, str):
            filename = Path(filename)

        self.im_name = filename.name
        self.meta = metadata_dict

    def print_all(self) -> None:
        for item in self.meta():
            print("{}: {}".format(item, self.meta[item]))

    def dls_present(self) -> bool:
        return self.meta["image_data"][self.im_name]["dls_present"]

    def supports_radiometric_calibration(self) -> bool:
        return self.meta["image_data"][self.im_name]["supports_radiometric_cal"]

    def position(self) -> Tuple[float, float, float]:
        """get the WGS-84 latitude, longitude tuple as signed decimal degrees"""
        return (
            self.meta["dls_latitude"],
            self.meta["dls_longitude"],
            self.meta["dls_altitde"],
        )

    def utc_time(self) -> datetime:
        return self.meta["dls_utctime"]

    def dls_pose(self) -> Tuple[float, float, float]:
        """get DLS pose as local earth-fixed yaw, pitch, roll in radians"""
        return (
            self.meta["dls_yaw"],
            self.meta["dls_pitch"],
            self.meta["dls_roll"],
        )

    def rig_relatives(self) -> Union[List[float], None]:
        return self.meta["image_data"][self.im_name]["rig_relatives"]

    def capture_id(self) -> Union[None, str]:
        return self.meta["capture_id"]

    def flight_id(self) -> Union[None, str]:
        return self.meta["flight_id"]

    def camera_make(self) -> Union[None, str]:
        # this should return "micasense"
        return self.meta["camera_make"]

    def camera_model(self) -> Union[None, str]:
        # this should return "RedEdge-M"
        return self.meta["camera_model"]

    def firmware_version(self) -> Union[None, str]:
        return self.meta["firmware_version"]

    def band_name(self) -> Union[None, str]:
        # for Red-Camera:
        #     Blue, Green, Red, NIR, Red edge
        # for Blue-Camera:
        #     Blue-444, Green-531, Red-650, Red edge-705, Red edge-740
        return self.meta["image_data"][self.im_name]["band_name"]

    def band_index(self) -> Union[None, int]:
        # for Red-Camera
        #     0, 1, 2, 3, 4
        # for Blue-Camera
        #     5, 6, 7, 8, 9
        return self.meta["image_data"][self.im_name]["rig_camera_index"]

    def exposure(self) -> float:
        """extract the exposure (integration time) in seconds"""
        return self.meta["image_data"][self.im_name]["exposure"]

    def gain(self) -> float:
        """extract the image gain"""
        return self.meta["image_data"][self.im_name]["gain"]

    def image_size(self) -> Tuple[int, int]:
        """extract the image size (ncols, nrows)"""
        return self.meta["image_size"]

    def center_wavelength(self) -> float:
        """extract the central wavelength (nm)"""
        return self.meta["image_data"][self.im_name]["wavelength_center"]

    def bandwidth(self) -> float:
        """extract the bandwidth (nm)"""
        return self.meta["image_data"][self.im_name]["wavelength_fwhm"]

    def radiometric_cal(self) -> List[float]:
        """extract the radiometric calibration coefficients"""
        return self.meta["image_data"][self.im_name]["rad_calibration"]

    def black_level(self) -> float:
        """Extract the mean dark current"""
        return self.meta["image_data"][self.im_name]["blacklevel"]

    def dark_pixels(self) -> float:
        """
        Get the average of the optically covered pixel values
        Note: these pixels are raw, and have not been radiometrically
              corrected. Use the black_level() method for all
              radiomentric calibrations
        """
        return self.meta["image_data"][self.im_name]["darkpixels"]

    def bits_per_pixel(self) -> int:
        """
        get the number of bits per pixel, which defines the
        maximum digital number value in the image
        """
        return self.meta["bits_persample"]

    def vignette_center(self) -> List[float]:
        """get the vignette center in X and Y image coordinates"""
        return self.meta["image_data"][self.im_name]["vignette_xy"]

    def vignette_polynomial(self) -> List[float]:
        """
        get the radial vignette polynomial in the order
        it's defined within the metadata
        """
        return self.meta["image_data"][self.im_name]["vignette_poly"]

    def distortion_parameters(self) -> List[float]:
        return self.meta["image_data"][self.im_name]["distortion_params"]

    def principal_point(self) -> List[float]:
        return self.meta["image_data"][self.im_name]["principal_point"]

    def focal_plane_resolution_px_per_mm(self) -> Tuple[float, float]:
        return self.meta["focalplane_xres"], self.meta["focalplane_yres"]

    def focal_length_mm(self) -> float:
        return self.meta["image_data"][self.im_name]["focal_length_mm"]

    def sfactor_35mm(self) -> float:
        """
        extract the 35mm scale factor, sf[35], such that,
        35 mm equivalent focal length = sf[35] * focal_length
        """
        if self.camera_model() == "Altum":
            raise Exception("Have yet to write code for Altum")
        else:
            pxl_mm = 3.75 / 1000.0  # pixel size in mm
            w, h = self.image_size()
            diag_img = ((w * pxl_mm) ** 2 + (h * pxl_mm) ** 2) ** 0.5
            diag_35mm = ((36.0) ** 2 + (24.0) ** 2) ** 0.5
            return diag_35mm / diag_img

    def focal_length_35_mm_eq(self) -> float:
        # pyexiv2 cannot access the Composite keys including:
        # Composite:FocalLength35efl
        return self.meta["image_data"][self.im_name]["focal_length"] * self.sfactor_35mm()

    def irradiance_scale_factor(self) -> float:
        # return the conversion factor to get irradiance as SI units
        return self.meta["image_data"][self.im_name]["irradiance_scale_factor"]

    def horizontal_irradiance_valid(self) -> bool:
        return self.meta["image_data"][self.im_name]["horizontal_irradiance_valid"]

    def spectral_irradiance(self) -> float:
        """
        LEGACY FUNCTION
        Raw spectral irradiance measured by an irradiance sensor.
        Calibrated to W/m^2/nm using irradiance_scale_factor, but
        not corrected for angles
        """
        return (
            self.meta["image_data"][self.im_name]["dls_Ed"]
            * self.irradiance_scale_factor()
        )

    def horizontal_irradiance(self) -> float:
        """
        Horizontal irradiance at the earth's surface below the DLS on the
        plane normal to the gravity vector at the location (local flat
        plane spectral irradiance)
        """
        return (
            self.meta["image_data"][self.im_name]["dls_Ed_h"]
            * self.irradiance_scale_factor()
        )

    def scattered_irradiance(self) -> float:
        """scattered component of the spectral irradiance"""
        return (
            self.meta["image_data"][self.im_name]["dls_Ed_s"]
            * self.irradiance_scale_factor()
        )

    def direct_irradiance(self) -> float:
        """
        direct component of the spectral irradiance on a ploane normal
        to the vector towards the sun
        """
        return (
            self.meta["image_data"][self.im_name]["dls_Ed_d"]
            * self.irradiance_scale_factor()
        )

    def solar_azimuth(self) -> float:
        """solar azimuth at the time of capture, as calculated by the camera system"""
        return self.meta["dls_solarazi"]

    def solar_elevation(self) -> float:
        """solar elevation at the time of capture, as calculated by the camera system"""
        return self.meta["dls_solarzen"]

    def estimated_direct_vector(self) -> Union[List[float], None]:
        """estimated direct light vector relative to the DLS2 reference frame"""
        return self.meta["image_data"][self.im_name]["dls_EstimatedDirectLightVector"]

    def auto_calibration_image(self) -> bool:
        return self.meta["image_data"][self.im_name]["auto_calibration_image"]

    def panel_albedo(self) -> Union[float, None]:
        return self.meta["image_data"][self.im_name]["panel_albedo"]

    def panel_region(self) -> Union[None, List[int]]:
        return self.meta["image_data"][self.im_name]["panel_region"]

    def panel_serial(self) -> Union[str, None]:
        return self.meta["image_data"][self.im_name]["panel_serial"]
