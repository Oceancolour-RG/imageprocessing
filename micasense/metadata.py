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

import os
import pytz
import math
import exiftool  # replace exiftool with pyexiv2

from pathlib2 import Path
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple, List


class MetadataFromExif(object):
    """Container for Micasense image metadata"""

    def __init__(
        self,
        filename: Union[str, Path],
    ):

        if not os.path.isfile(filename):
            raise IOError("Input path is not a file")

        with exiftool.ExifTool(None) as exift:
            self.exif = exift.get_metadata(str(filename))

    def get_all(self) -> dict:
        """Get all extracted metadata items"""
        return self.exif

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

    def size(self, item) -> int:
        """get the size (length) of a metadata item"""
        val = self.get_item(item)
        try:
            if isinstance(val, unicode):
                val = val.encode("ascii", "ignore")
        except NameError:
            # throws on python 3 where unicode is undefined
            pass

        if isinstance(val, str) and len(val.split(",")) > 1:
            val = val.split(",")
        if val is not None:
            return len(val)
        else:
            return 0

    def print_all(self) -> None:
        for item in self.get_all():
            print("{}: {}".format(item, self.get_item(item)))

    def dls_present(self) -> bool:
        return (
            self.get_item("XMP:Irradiance") is not None
            or self.get_item("XMP:HorizontalIrradiance") is not None
            or self.get_item("XMP:DirectIrradiance") is not None
        )

    def supports_radiometric_calibration(self) -> bool:
        if (self.get_item("XMP:RadiometricCalibration")) is None:
            return False
        return True

    def position(self) -> Tuple[float, float, float]:
        """get the WGS-84 latitude, longitude tuple as signed decimal degrees"""
        lat = self.get_item("EXIF:GPSLatitude")
        lon = self.get_item("EXIF:GPSLongitude")

        lat = -lat if self.get_item("EXIF:GPSLatitudeRef") == "S" else lat
        lon = -lon if self.get_item("EXIF:GPSLongitudeRef") == "W" else lon

        alt = self.get_item("EXIF:GPSAltitude")

        return lat, lon, alt

    def utc_time(self) -> datetime:
        """Get the timezone-aware datetime of the image capture"""
        str_time = self.get_item("EXIF:DateTimeOriginal")
        utc_time = datetime.strptime(str_time, "%Y:%m:%d %H:%M:%S")
        subsec = int(self.get_item("EXIF:SubSecTime"))

        sign = -1.0 if subsec < 0 else 1.0
        ms = sign * 1e3 * float("0.{}".format(abs(subsec)))

        utc_time += timedelta(milliseconds=ms)
        timezone = pytz.timezone("UTC")
        utc_time = timezone.localize(utc_time)
        return utc_time

    def dls_pose(self) -> Tuple[float, float, float]:
        """get DLS pose as local earth-fixed yaw, pitch, roll in radians"""
        if self.get_item("XMP:Yaw") is not None:
            # should be XMP.DLS.Yaw, but exiftool doesn't expose it that way
            yaw = float(self.get_item("XMP:Yaw"))
            pitch = float(self.get_item("XMP:Pitch"))
            roll = float(self.get_item("XMP:Roll"))
        else:
            yaw = pitch = roll = 0.0
        return yaw, pitch, roll

    def rig_relatives(self) -> Union[List[float], None]:
        if self.get_item("XMP:RigRelatives") is not None:
            nelem = self.size("XMP:RigRelatives")
            return [float(self.get_item("XMP:RigRelatives", i)) for i in range(nelem)]
        else:
            return None

    def capture_id(self) -> Union[None, str]:
        return self.get_item("XMP:CaptureId")

    def flight_id(self) -> Union[None, str]:
        return self.get_item("XMP:FlightId")

    def camera_make(self) -> Union[None, str]:
        # this should return "micasense"
        return self.get_item("EXIF:Make")

    def camera_model(self) -> Union[None, str]:
        # this should return "RedEdge-M"
        return self.get_item("EXIF:Model")

    def firmware_version(self) -> Union[None, str]:
        return self.get_item("EXIF:Software")

    def band_name(self) -> Union[None, str]:
        # for Red-Camera:
        #     Blue, Green, Red, NIR, Red edge
        # for Blue-Camera:
        #     Blue-444, Green-531, Red-650, Red edge-705, Red edge-740
        return self.get_item("XMP:BandName")

    def band_index(self) -> Union[None, int]:
        # for Red-Camera
        #     0, 1, 2, 3, 4
        # for Blue-Camera
        #     5, 6, 7, 8, 9
        return self.get_item("XMP:RigCameraIndex")

    def exposure(self) -> float:
        exp = self.get_item("EXIF:ExposureTime")
        # correct for incorrect exposure in some legacy RedEdge firmware versions
        if self.camera_model() != "Altum":
            if math.fabs(exp - (1.0 / 6329.0)) < 1e-6:
                exp = 0.000274
        return exp

    def gain(self) -> float:
        return self.get_item("EXIF:ISOSpeed") / 100.0

    def image_size(self) -> Tuple[int, int]:
        return self.get_item("EXIF:ImageWidth"), self.get_item("EXIF:ImageHeight")

    def center_wavelength(self) -> int:
        return self.get_item("XMP:CentralWavelength")

    def bandwidth(self) -> int:
        return self.get_item("XMP:WavelengthFWHM")

    def radiometric_cal(self) -> List[float]:
        nelem = self.size("XMP:RadiometricCalibration")
        return [
            float(self.get_item("XMP:RadiometricCalibration", i)) for i in range(nelem)
        ]

    def black_level(self) -> float:
        """Extract the mean dark current"""
        if self.get_item("EXIF:BlackLevel") is None:
            return 0

        total, num = 0.0, 0.0
        for bl in self.get_item("EXIF:BlackLevel").split(" "):
            total += float(bl)
            num += 1.0
        return total / float(num)

    def dark_pixels(self) -> float:
        """
        Get the average of the optically covered pixel values
        Note: these pixels are raw, and have not been radiometrically
              corrected. Use the black_level() method for all
              radiomentric calibrations
        """
        total, num = 0.0, 0.0
        for pixel in self.get_item("XMP:DarkRowValue"):
            total += float(pixel)
            num += 1.0
        return total / float(num)

    def bits_per_pixel(self) -> int:
        """
        get the number of bits per pixel, which defines the
        maximum digital number value in the image
        """
        return self.get_item("EXIF:BitsPerSample")

    def vignette_center(self) -> List[float]:
        """get the vignette center in X and Y image coordinates"""
        nelem = self.size("XMP:VignettingCenter")
        return [float(self.get_item("XMP:VignettingCenter", i)) for i in range(nelem)]

    def vignette_polynomial(self) -> List[float]:
        """
        get the radial vignette polynomial in the order
        it's defined within the metadata
        """
        nelem = self.size("XMP:VignettingPolynomial")
        return [float(self.get_item("XMP:VignettingPolynomial", i)) for i in range(nelem)]

    def distortion_parameters(self) -> List[float]:
        nelem = self.size("XMP:PerspectiveDistortion")
        return [
            float(self.get_item("XMP:PerspectiveDistortion", i)) for i in range(nelem)
        ]

    def principal_point(self) -> List[float]:
        return [float(item) for item in self.get_item("XMP:PrincipalPoint").split(",")]

    def focal_plane_resolution_px_per_mm(self) -> Tuple[float, float]:
        fp_x_resolution = float(self.get_item("EXIF:FocalPlaneXResolution"))
        fp_y_resolution = float(self.get_item("EXIF:FocalPlaneYResolution"))
        return fp_x_resolution, fp_y_resolution

    def focal_length_mm(self) -> float:
        units = self.get_item("XMP:PerspectiveFocalLengthUnits")
        focal_length_mm = 0.0
        if units == "mm":
            focal_length_mm = float(self.get_item("XMP:PerspectiveFocalLength"))
        else:
            focal_length_px = float(self.get_item("XMP:PerspectiveFocalLength"))
            focal_length_mm = focal_length_px / self.focal_plane_resolution_px_per_mm()[0]
        return focal_length_mm

    def focal_length_35_mm_eq(self) -> float:
        return float(self.get_item("Composite:FocalLength35efl"))

    def __float_or_zero(self, str) -> float:
        if str is not None:
            return float(str)
        else:
            return 0.0

    def irradiance_scale_factor(self) -> float:
        """
        Get the calibration scale factor for the irradiance measurements
        in this image metadata. Due to calibration differences between
        DLS1 and DLS2, we need to account for a scale factor change in
        their respective units. This scale factor is pulled from the image
        metadata, or, if the metadata doesn't give us the scale, we assume
        one based on a known combination of tags
        """
        if self.get_item("XMP:IrradianceScaleToSIUnits") is not None:
            # the metadata contains the scale
            scale_factor = self.__float_or_zero(
                self.get_item("XMP:IrradianceScaleToSIUnits")
            )
        elif self.get_item("XMP:HorizontalIrradiance") is not None:
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
        if self.get_item("XMP:HorizontalIrradiance") is None:
            return False
        from packaging import version

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

    def spectral_irradiance(self) -> float:
        """
        Raw spectral irradiance measured by an irradiance sensor.
        Calibrated to W/m^2/nm using irradiance_scale_factor, but
        not corrected for angles
        """
        return (
            self.__float_or_zero(self.get_item("XMP:SpectralIrradiance"))
            * self.irradiance_scale_factor()
        )

    def horizontal_irradiance(self) -> float:
        """
        Horizontal irradiance at the earth's surface below the DLS on the
        plane normal to the gravity vector at the location (local flat
        plane spectral irradiance)
        """
        return (
            self.__float_or_zero(self.get_item("XMP:HorizontalIrradiance"))
            * self.irradiance_scale_factor()
        )

    def scattered_irradiance(self) -> float:
        """scattered component of the spectral irradiance"""
        return (
            self.__float_or_zero(self.get_item("XMP:ScatteredIrradiance"))
            * self.irradiance_scale_factor()
        )

    def direct_irradiance(self) -> float:
        """
        direct component of the spectral irradiance on a ploane normal
        to the vector towards the sun
        """
        return (
            self.__float_or_zero(self.get_item("XMP:DirectIrradiance"))
            * self.irradiance_scale_factor()
        )

    def solar_azimuth(self) -> float:
        """solar azimuth at the time of capture, as calculated by the camera system"""
        return self.__float_or_zero(self.get_item("XMP:SolarAzimuth"))

    def solar_elevation(self) -> float:
        """solar elevation at the time of capture, as calculated by the camera system"""
        return self.__float_or_zero(self.get_item("XMP:SolarElevation"))

    def estimated_direct_vector(self) -> Union[List[float], None]:
        """estimated direct light vector relative to the DLS2 reference frame"""
        if self.get_item("XMP:EstimatedDirectLightVector") is not None:
            return [
                self.__float_or_zero(item)  # This doesn't make sense!
                for item in self.get_item("XMP:EstimatedDirectLightVector")
            ]
        else:
            return None

    def auto_calibration_image(self) -> bool:
        """
        True if this image is an auto-calibration image, where the camera has
        found and idetified a calibration panel
        """
        cal_tag = self.get_item("XMP:CalibrationPicture")
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
        albedo = self.get_item("XMP:Albedo")
        if albedo is not None:
            return self.__float_or_zero(albedo)
        return albedo

    def panel_region(self) -> Union[None, List[int]]:
        """A 4-tuple containing image x,y coordinates of the panel active area"""
        if self.get_item("XMP:ReflectArea") is not None:
            coords = [int(item) for item in self.get_item("XMP:ReflectArea").split(",")]
            return list(zip(coords[0::2], coords[1::2]))
        else:
            return None

    def panel_serial(self) -> Union[str, None]:
        """The panel serial number as extracted from the image by the camera"""
        return self.get_item("XMP:PanelSerial")
