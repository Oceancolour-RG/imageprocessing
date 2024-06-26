#!/usr/bin/env python3
# coding: utf-8
"""
RedEdge Image Class

    An Image is a single file taken by a RedEdge camera representing one
    band of multispectral information

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

import cv2
import math
import numpy as np

import micasense.load_yaml as ms_yaml
import micasense.plotutils as plotutils
import micasense.metadata2 as metadata

# import micasense.metadata as metadata
import micasense.dls as dls

from os.path import isfile
from pathlib import Path
from typing import Optional, Iterable, Union, Tuple


# helper function to convert euler angles to a rotation matrix
def rotations_degrees_to_rotation_matrix(rotation_degrees: np.ndarray) -> np.matrix:
    cx = np.cos(np.deg2rad(rotation_degrees[0]))
    cy = np.cos(np.deg2rad(rotation_degrees[1]))
    cz = np.cos(np.deg2rad(rotation_degrees[2]))
    sx = np.sin(np.deg2rad(rotation_degrees[0]))
    sy = np.sin(np.deg2rad(rotation_degrees[1]))
    sz = np.sin(np.deg2rad(rotation_degrees[2]))

    r_x = np.mat([1, 0, 0, 0, cx, -sx, 0, sx, cx]).reshape(3, 3)
    r_y = np.mat([cy, 0, sy, 0, 1, 0, -sy, 0, cy]).reshape(3, 3)
    r_z = np.mat([cz, -sz, 0, sz, cz, 0, 0, 0, 1]).reshape(3, 3)

    return r_x * r_y * r_z


def check_vigparms(vig_params: Union[dict, None], order: int = 6) -> None:
    """
    check the user-specified vignetting parameters. This function is
    primarily used in the image.Image class

    Parameters
    ----------
    vig_params : dict or None
        vignetting parameter dictionary with the following keys
        {
            "vignette_center": [float, float],  # x, y
            "vignette_polynomial": List[float],  # vignetting polynomials
        }
        Here, the 'vignette_polynomial' must be an Iterable
        with n-order of elements

    order : int [default = 6]
        Polynomial order of vignetting model
    """

    def _checknum(v: float) -> bool:
        """Check whether the variable `v` is float or int"""
        return isinstance(v, (int, float))

    err = "{0} must be an iterable with {1} elements\n{0}: {2}"

    if vig_params is not None:
        if not isinstance(vig_params, dict):
            raise TypeError("vig_params must be a dictionary (or None)")

        # --- check "vignette_centre" --- #
        if "vignette_center" not in vig_params:
            raise ValueError("vig_params must contain 'vignette_center'")
        else:
            tmp1 = vig_params["vignette_center"]
            err1 = err.format(
                "vig_params['vignette_center']", "two positive numeric", tmp1
            )
            if isinstance(tmp1, Iterable):
                if (len(tmp1) != 2) or (not all([_checknum(v) and v > 0 for v in tmp1])):
                    # check if vignette_centre has two positive numeric elements
                    raise ValueError(err1)
            else:
                # vignette_center is not an Iterable
                raise ValueError(err1)

        # --- check "vignette_polynomial" --- #
        if "vignette_polynomial" not in vig_params:
            raise ValueError("vig_params must contain 'vignette_polynomial'")
        else:
            tmp2 = vig_params["vignette_polynomial"]
            err2 = err.format(
                "vig_params['vignette_polynomial']", f"{order} numeric", tmp2
            )
            if isinstance(tmp2, Iterable):
                if (len(tmp2) != order) or (not all([_checknum(v) for v in tmp2])):
                    # check if vignette_polynomial has {order} numeric elements
                    raise ValueError(err2)
            else:
                # vignette_polynomial is not an Iterable
                raise ValueError(err2)

    return


def check_dc(dark_current: Union[float, int, None], bits_per_pixel: int) -> None:
    """
    Check the dark current value. This function is primarily
    used in the image.Image class


    Parameters
    ----------
    dark_current : float, int or None
        dark current value (must be greater than 0 and less
        than the maximum DN value)

    bits_per_pixel : int
        pixel bits (used to get the maximum DN value)
    """
    max_dn = (2**bits_per_pixel) - 1

    if dark_current is not None:
        if isinstance(dark_current, (float, int)):
            err = f"`dark_current` ({dark_current}) must range between 0 and {max_dn}"
            if (dark_current < 0) or (dark_current >= max_dn):
                raise ValueError(err)
        else:
            raise TypeError("`dark_current` must be float, int or None")

    return


class Image(object):
    """
    An Image is a single file taken by a RedEdge camera representing one
    band of multispectral information
    """

    def __init__(
        self,
        image_path: Union[Path, str],
        yaml_path: Optional[Union[Path, str]] = None,
        metadata_dict: Optional[dict] = None,
        vig_params: Optional[dict] = None,
        dark_current: Optional[Union[float, int]] = None,
    ):
        """
        Create an Image object from a single band.
        Parameters
        ----------
        image_path : Path or str
            The filename of the single band (.tif)
        yaml_path : Path, str or None [Optional]
            The yaml filename containing image metadata
            for an acquisition image set.

            A yaml file is useful, for instance, if a set of image
            calibration or vignetting coefficients are used that are
            different from that in the tif exif metadata.
        metadata_dict : dict [Optional]
            The metadata dictionary of the image_path

        vig_params : dict [Optional]
            User defined vignetting parameters. This dictionary
            must have the following keys,
            {
                "vignette_center": [float, float],  # x, y
                "vignette_polynomial": List[float],  # vignetting polynomials
            }
            Here, the vignetting polynomials must have six values
            for the model given in:
            https://support.micasense.com/hc/en-us/articles/
               115000351194-Radiometric-Calibration-Model-for-MicaSense-Sensors

        dark_current : float or int [Optional]
            User defined dark current value (must be greater than 0 and
            less than the maximum DN value)

        Notes
        -----
          1) `vig_params` and  `dark_current` should be used for testing
             purposes. For operational workflows, it is recommended that
             the default vignetting and dark current parameters stored
             in the `yaml_path` be replaced with calibrated values.
          2) if both `metadata_dict` and `yaml_path` are specified then
             `metadata_dict` is used, while ignoring `yaml_path`
          3) if metadata_dict=None while the `yaml_path` is specified,
             then the `metadata_dict` is loaded from `yaml_path`
        """
        if not isfile(image_path):
            raise IOError(f"Provided path is not a file: {image_path}")

        if (metadata_dict is None) or (not isinstance(metadata_dict, dict)):
            # metadata_dict is None or not a dict, check if yaml_path was
            # specified so that metadata_dict can be loaded.
            if yaml_path and isfile(yaml_path):
                # load yaml as metadata_dict
                metadata_dict = ms_yaml.load_all(yaml_file=yaml_path)

        self.path = image_path
        if metadata_dict is None:
            self.meta = metadata.MetadataFromExif(filename=self.path)
            # md = self.meta.exif
        else:
            self.meta = metadata.MetadataFromDict(
                filename=self.path, metadata_dict=metadata_dict
            )
            # self.meta.exif = None

        if self.meta.band_name() is None:
            raise ValueError("Provided file path does not have a band name: {image_path}")
        if (
            self.meta.band_name().upper() != "LWIR"
            and not self.meta.supports_radiometric_calibration()
        ):
            raise ValueError(
                "Library requires images taken with RedEdge-(3/M/MX) camera firmware "
                "v2.1.0 or later. Upgrade your camera firmware to at least version "
                "2.1.0 to use this library with RedEdge-(3/M/MX) cameras."
            )

        self.utc_time = self.meta.utc_time()
        self.latitude, self.longitude, self.altitude = self.meta.position()
        self.location = (self.latitude, self.longitude, self.altitude)
        self.dls_present = self.meta.dls_present()
        self.dls_yaw, self.dls_pitch, self.dls_roll = self.meta.dls_pose()
        self.capture_id = self.meta.capture_id()
        self.flight_id = self.meta.flight_id()
        self.band_name = self.meta.band_name()
        self.band_index = self.meta.band_index()
        if self.meta.supports_radiometric_calibration():
            self.radiometric_cal = self.meta.radiometric_cal()
        self.exposure_time = self.meta.exposure()
        self.gain = self.meta.gain()
        self.bits_per_pixel = self.meta.bits_per_pixel()

        # ---------- dark current ---------- #
        self.black_level = self.meta.black_level()
        self.dark_pixels = self.meta.dark_pixels()
        self.user_dark_current = dark_current  # float or None
        check_dc(self.user_dark_current, self.bits_per_pixel)  # run check

        # ---------- vignetting ---------- #
        self.vignette_center = self.meta.vignette_center()
        self.vignette_polynomial = self.meta.vignette_polynomial()

        # Get the user vignetting parameters (if provided)
        self.user_vig_params = vig_params  # dict or None
        check_vigparms(self.user_vig_params)  # run checks

        # ---------- lens calibration ---------- #
        self.distortion_parameters = self.meta.distortion_parameters()
        self.principal_point = self.meta.principal_point()
        self.focal_plane_resolution_px_per_mm = (
            self.meta.focal_plane_resolution_px_per_mm()
        )
        self.focal_length = self.meta.focal_length_mm()
        self.focal_length_35 = self.meta.focal_length_35_mm_eq()

        self.center_wavelength = self.meta.center_wavelength()
        self.bandwidth = self.meta.bandwidth()
        self.rig_relatives = self.meta.rig_relatives()
        self.spectral_irradiance = self.meta.spectral_irradiance()

        self.auto_calibration_image = self.meta.auto_calibration_image()
        self.panel_albedo = self.meta.panel_albedo()
        self.panel_region = self.meta.panel_region()
        self.panel_serial = self.meta.panel_serial()

        # New camera matrix computed with getOptimalNewCameraMatrix()
        # during undistortion
        self.newcammat_dfx = None
        self.newcammat_dfy = None
        self.newcammat_ppx = None
        self.newcammat_ppy = None

        # Note that dls_orientation_vector is only used
        # for processing the DLS1 data. According to the
        # Micasense website:
        # https://support.micasense.com/hc/en-us/articles/115005084647-Reading-DLS-Irradiance-Metadata
        # "The Horizontal Irradiance tag [DLS2] is not impacted by IMU errors since
        #  it uses the directional light sensors for angle compensation".
        self.dls_orientation_vector = np.array([0, 0, -1])
        if self.dls_present:
            (
                self.sun_vector_ned,
                self.sensor_vector_ned,
                self.sun_sensor_angle,
                self.solar_elevation,
                self.solar_azimuth,
            ) = dls.compute_sun_angle(
                self.location,
                self.meta.dls_pose(),
                self.utc_time,
                self.dls_orientation_vector,
            )
            self.angular_correction = dls.fresnel(self.sun_sensor_angle)

            # when we have good horizontal irradiance the camera
            # provides the solar az and el also
            if (
                self.meta.scattered_irradiance() != 0
                and self.meta.direct_irradiance() != 0
            ):
                self.solar_azimuth = self.meta.solar_azimuth()
                self.solar_elevation = self.meta.solar_elevation()
                self.scattered_irradiance = self.meta.scattered_irradiance()
                self.direct_irradiance = self.meta.direct_irradiance()
                self.direct_to_diffuse_ratio = (
                    self.meta.direct_irradiance() / self.meta.scattered_irradiance()
                )
                self.estimated_direct_vector = self.meta.estimated_direct_vector()
                if self.meta.horizontal_irradiance_valid():
                    self.horizontal_irradiance = self.meta.horizontal_irradiance()
                else:
                    self.horizontal_irradiance = self.compute_horizontal_irradiance_dls2()
            else:
                self.direct_to_diffuse_ratio = 6.0  # assumption
                self.horizontal_irradiance = self.compute_horizontal_irradiance_dls1()

            self.spectral_irradiance = self.meta.spectral_irradiance()
        else:
            # no dls present or LWIR band: compute what we can, set the rest to 0
            (
                self.sun_vector_ned,
                self.sensor_vector_ned,
                self.sun_sensor_angle,
                self.solar_elevation,
                self.solar_azimuth,
            ) = dls.compute_sun_angle(
                self.location, (0, 0, 0), self.utc_time, self.dls_orientation_vector
            )
            self.angular_correction = dls.fresnel(self.sun_sensor_angle)
            self.horizontal_irradiance = 0
            self.scattered_irradiance = 0
            self.direct_irradiance = 0
            self.direct_to_diffuse_ratio = 0

        # Internal image containers; these can use a lot of memory,
        # clear with Image.clear_images
        self.__vc_g = None  # [float] -> vicarious calibration gains
        self.__raw_image = None  # pure raw pixels
        # black level and gain-exposure/radiometric compensated
        self.__intensity_image = None
        self.__radiance_image = None  # calibrated to radiance
        self.__reflectance_image = None  # calibrated to reflectance (0-1)
        self.__reflectance_irradiance = None
        self.__undistorted_source = None  # can be any of raw, intensity, radiance
        # current undistorted image, depending on source
        self.__undistorted_image = None

    # solar elevation is defined as the angle betwee the horizon and the sun,
    # so it is 0 when the sun is at the horizon and pi/2 when the sun is
    # directly overhead
    def horizontal_irradiance_from_direct_scattered(self) -> float:
        return (
            self.direct_irradiance * np.sin(self.solar_elevation)
            + self.scattered_irradiance
        )

    def compute_horizontal_irradiance_dls1(self) -> float:
        percent_diffuse = 1.0 / self.direct_to_diffuse_ratio
        # percent_diffuse = 5e4/(img.center_wavelength**2)
        sensor_irradiance = self.spectral_irradiance / self.angular_correction
        # find direct irradiance in the plane normal to the sun
        untilted_direct_irr = sensor_irradiance / (
            percent_diffuse + np.cos(self.sun_sensor_angle)
        )
        self.direct_irradiance = untilted_direct_irr
        self.scattered_irradiance = untilted_direct_irr * percent_diffuse
        # compute irradiance on the ground using the solar altitude angle
        return self.horizontal_irradiance_from_direct_scattered()

    def compute_horizontal_irradiance_dls2(self) -> float:
        """
        Compute the proper solar elevation, solar azimuth, and horizontal
        irradiance for cases where the camera system did not do it correctly
        """
        _, _, _, self.solar_elevation, self.solar_azimuth = dls.compute_sun_angle(
            self.location, (0, 0, 0), self.utc_time, np.array([0, 0, -1])
        )
        return self.horizontal_irradiance_from_direct_scattered()

    def __lt__(self, other):
        return self.band_index < other.band_index

    def __gt__(self, other):
        return self.band_index > other.band_index

    def __eq__(self, other):
        return (self.band_index == other.band_index) and (
            self.capture_id == other.capture_id
        )

    def __ne__(self, other):
        return (self.band_index != other.band_index) or (
            self.capture_id != other.capture_id
        )

    def raw(self) -> np.ndarray:
        """Lazy load the raw image"""
        if self.__raw_image is None:
            try:
                self.__raw_image = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
            except IOError:
                raise Exception(f"Could not open image at path {self.path}")
        return self.__raw_image

    def set_raw(self, img: np.ndarray) -> None:
        """set raw image from input img"""
        self.__raw_image = img.astype(np.uint16)
        return

    def set_undistorted(self, img: np.ndarray) -> None:
        """set undistorted image from input img"""
        self.__undistorted_image = img.astype(np.uint16)
        return

    def set_external_rig_relatives(self, external_rig_relatives) -> None:
        self.rig_translations = external_rig_relatives["rig_translations"]
        # external rig relatives are in rad
        self.rig_relatives = [
            np.rad2deg(a) for a in external_rig_relatives["rig_relatives"]
        ]
        px, py = external_rig_relatives["cx"], external_rig_relatives["cy"]
        fx, fy = external_rig_relatives["fx"], external_rig_relatives["fy"]
        rx = self.focal_plane_resolution_px_per_mm[0]
        ry = self.focal_plane_resolution_px_per_mm[1]
        self.principal_point = [px / rx, py / ry]
        self.focal_length = (fx + fy) * 0.5 / rx
        # TODO - set the distortion etc.
        return

    def clear_image_data(self) -> None:
        """clear all computed images to reduce memory overhead"""
        self.__raw_image = None
        self.__intensity_image = None
        self.__radiance_image = None
        self.__reflectance_image = None
        self.__reflectance_irradiance = None
        self.__undistorted_source = None
        self.__undistorted_image = None
        self.__vc_g = None
        return

    def size(self) -> Tuple[int, int]:
        width, height = self.meta.image_size()
        return width, height

    def get_saturated_pxls(self, undistort: bool = True) -> np.ndarray:
        """
        Get the index locations of pixels considered saturated.
        Here, a pixel is saturated if it's DN > (99 % of max. DN),
        where max. DN = (2^16 - 1) = 65535,

        Hence, saturated pixel: DN > 64880

        Parameters
        ----------
        undistort : bool
            Whether to undistort the raw image

        Returns
        -------
        sat_ix : ndarray {dtype=bool}
            Boolean array in which saturated pixel are True
        """

        if undistort:
            return self.raw() > 64880
        else:
            return self.undistorted(image=self.raw()) > 64880

    def vignette(
        self, user_params: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the vignetting correction derived from a six-ordered
        polynomial model.

        Parameters
        ----------
        user_params : dict [Optional]
            If provided then user-defined vignetting centre and polynom-
            ials are used to model the vignetting. This is only used for
            testing of new model coefficients and debugging purposes. To
            apply new vignetting coefficients into the radiance and ref-
            lectance calculations run,
            >>> Image(filename, vig_params)

            Where `vig_params` will be used instead of the default values

            `user_params` must have the following keys:
            {
                "vignette_center": [float, float],  # x, y
                "vignette_polynomial": List[float],  # vignetting polynomials
            }
            Here, the vignetting polynomials must have six values
            https://support.micasense.com/hc/en-us/articles/
               115000351194-Radiometric-Calibration-Model-for-MicaSense-Sensors

        Returns
        -------
        vigcorr : np.ndarray
            Vignetting correction
        x, y : np.ndarray's
            x and y meshgrid

        Note: vigcorr, x and y are transposed from normal image orientation
        """
        if isinstance(user_params, dict):
            # get vignette center
            print("Testing mode in `vignette()`: user_params specified")
            check_vigparms(user_params)
            vignette_center_x = user_params["vignette_center"][0]
            vignette_center_y = user_params["vignette_center"][1]
            v_poly_list = user_params["vignette_polynomial"].copy()

        else:
            # `self.user_vig_params` has already been checked in __init__()
            if self.user_vig_params:
                vignette_center_x = self.user_vig_params["vignette_center"][0]
                vignette_center_y = self.user_vig_params["vignette_center"][1]
                v_poly_list = self.user_vig_params["vignette_polynomial"].copy()

            else:
                # use default values if `self.user_vig_params` is None
                vignette_center_x, vignette_center_y = self.vignette_center
                v_poly_list = list(self.vignette_polynomial)

        # reverse list and append 1., so that we can call with numpy polyval
        v_poly_list.reverse()
        v_poly_list.append(1.0)
        v_polynomial = np.array(v_poly_list)

        # perform vignette correction
        # get coordinate grid across image, seem swapped because of transposed vignette
        x_dim, y_dim = self.raw().shape[1], self.raw().shape[0]
        x, y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))

        # meshgrid returns transposed arrays
        x = x.T
        y = y.T

        # compute matrix of distances from image center
        r = np.hypot((x - vignette_center_x), (y - vignette_center_y))

        # compute the vignette polynomial for each distance - we divide by the
        # polynomial so that the corrected image is,
        # image_corrected = image_original * vignetteCorrection
        vigcorr = 1.0 / np.polyval(v_polynomial, r)
        return vigcorr, x, y

    def select_dc(self, which_dc: str, func_name: str) -> float:
        """
        Select the dark current offset to apply

        Parameters
        ----------
        which_dc : str
            Which dark current offset to apply ('dark', 'black' or 'user')

        func_name : str
            Function name that is using `select_dc` for useful warning
            print-out

        Returns
        -------
        dc : float
           The dark current offset
        """
        if which_dc not in ["dark", "black", "user"]:
            raise ValueError(
                f"`which_dc` ({which_dc}) must be 'dark', 'black', or  'user'"
            )
        if which_dc == "user":
            if self.user_dark_current is None:
                print(
                    f"{func_name}: user-specified dark current value not supplied when "
                    "initiating Image class. Using dark pixels instead"
                )
                dc = self.dark_pixels
            else:
                dc = self.user_dark_current

        elif which_dc == "black":  # use black-level
            dc = self.black_level
        else:  # use dark pixels
            dc = self.dark_pixels

        return dc

    def intensity(
        self, force_recompute: bool = False, which_dc: str = "dark"
    ) -> np.ndarray:
        """
        Computes and return the intensity image after black level, vignette,
        and row correction applied. Intensity is in units of DN*Seconds without
        a radiance correction

        Parameters
        ----------
        force_recompute : bool
            Recompute the reflectance image even if already done so.

        which_dc : str
            Whether to use the `dark_pixels` ("dark"), `black_level` ("black"), or
            `user_defined` ("user")
            Note:
            `black_level` has a temporally constant value of 4800 across all bands.
            This is unrealistic as the dark current increases with sensor temperature.
            `dark_pixels` the averaged DN of the optically covered pixel values. This
            value is different for each band and varies across an acquisition, presu-
            mably from increases in temperature.
            `user_defined` temperature invariant value acquired through a dark
            current assessment
        """
        dc = self.select_dc(which_dc=which_dc.lower(), func_name="intensity")

        if self.__intensity_image is not None and force_recompute is False:
            return self.__intensity_image

        # get image dimensions
        image_raw = np.copy(self.raw()).T

        #  get radiometric calibration factors
        _, a2, a3 = (
            self.radiometric_cal[0],
            self.radiometric_cal[1],
            self.radiometric_cal[2],
        )

        # apply image correction methods to raw image

        vig, x, y = self.vignette()
        r_cal = 1.0 / (1.0 + a2 * y / self.exposure_time - a3 * y)

        lt_im = vig * r_cal * (image_raw - dc)
        lt_im[lt_im < 0] = 0

        max_raw_dn = float(2**self.bits_per_pixel)
        intensity_image = lt_im.astype("float64") / (
            self.gain * self.exposure_time * max_raw_dn
        )

        self.__intensity_image = intensity_image.T
        return self.__intensity_image

    def radiance(
        self,
        force_recompute: bool = False,
        which_dc: str = "dark",
        vc_g: float = 1.0,
        apply_vig: bool = True,
    ) -> np.ndarray:
        """
        Computes and returns the radiance image after all radiometric
        corrections have been applied

        Parameters
        ----------
        force_recompute : bool
            Recompute the reflectance image even if already done so.

        which_dc : str
            Whether to use the `dark_pixels` ("dark"), `black_level` ("black"), or
            `user_defined` ("user")
            Note:
            `black_level` has a temporally constant value of 4800 across all bands.
            This is unrealistic as the dark current increases with sensor temperature.
            `dark_pixels` the averaged DN of the optically covered pixel values. This
            value is different for each band and varies across an acquisition, presu-
            mably from increases in temperature.
            `user_defined` temperature invariant value acquired through a dark
            current assessment

        vc_g : float
            Vicarious calibration gain to apply [default=1]

        apply_vig : bool
            Whether to perform vignettig (True) or not (False)
        """
        if force_recompute is False:
            if (self.__radiance_image is not None) and (self.__vc_g == vc_g):
                # return precomputed radiance if it exists
                return self.__radiance_image

        # get image dimensions
        image_raw = np.copy(self.raw()).T

        # update the internal vc_g
        self.__vc_g = vc_g

        if self.band_name != "LWIR":
            #  get radiometric calibration factors
            a1, a2, a3 = (
                self.radiometric_cal[0],
                self.radiometric_cal[1],
                self.radiometric_cal[2],
            )
            g_ = self.gain
            te_ = self.exposure_time
            max_raw_dn = float(2**self.bits_per_pixel)

            # apply image correction methods to raw image
            vig, x, y = self.vignette()
            dc = self.select_dc(which_dc=which_dc.lower(), func_name="radiance")

            # original code (below - commented out) is hard to follow:
            # r_cal = 1.0 / (1.0 + a2 * y / self.exposure_time - a3 * y)
            # lt_im = vig * r_cal * (image_raw - dc)
            # lt_im[lt_im < 0] = 0

            # radiance_image = (
            #     lt_im.astype(float) / (self.gain * self.exposure_time) * a1 / max_raw_dn
            # )

            # new code following the equation of:
            # https://support.micasense.com/hc/en-us/articles/
            #    115000351194-Radiometric-Calibration-Model-for-MicaSense-Sensors
            normcorr_dn = (image_raw.astype("float64") - float(dc)) / max_raw_dn
            r_cal = a1 / (g_ * (te_ + (a2 * y) - (a3 * te_ * y)))  # float64

            if apply_vig:  # apply vignetting
                radiance_image = vc_g * vig * r_cal * normcorr_dn
            else:
                radiance_image = vc_g * r_cal * normcorr_dn

            radiance_image[radiance_image < 0] = 0

        else:
            # convert to deg.C from K
            lt_im = vc_g * 0.01 * image_raw.astype("float64") - 273.15
            radiance_image = lt_im

        self.__radiance_image = radiance_image.T
        return self.__radiance_image

    def reflectance(
        self,
        irradiance: Optional[float] = None,
        force_recompute: bool = False,
        which_dc: str = "dark",
        vc_g: float = 1.0,
        return_rrs: bool = False,
    ) -> np.ndarray:
        """
        Lazy-compute and return a reflectance image
        provided an irradiance reference

        Parameters
        ----------
        irradiance : float [Optional]
            The irradiance used to normalise the radiance image.
            If None then the horizontal irradiance from the DLS2
            will be used.

        force_recompute : bool
            Recompute the reflectance image even if already done so.

        which_dc : str
            Whether to use the `dark_pixels` ("dark"), `black_level` ("black"), or
            `user_defined` ("user")
            Note:
            `black_level` has a temporally constant value of 4800 across all bands.
            This is unrealistic as the dark current increases with sensor temperature.
            `dark_pixels` the averaged DN of the optically covered pixel values. This
            value is different for each band and varies across an acquisition, presu-
            mably from increases in temperature.
            `user_defined` temperature invariant value acquired through a dark
            current assessment

        vc_g : float
            Vicarious calibration gains to apply during radiance computation

        return_rrs : bool
            if True : returns remote sensing reflectance, Rrs, (units: 1/sr)
            if False: returns Reflectance (units: a.u.)
            This only applies to VIS-NIR bands not LWIR

        Returns
        -------
        reflectance : np.ndarray
            Reflectance or Rrs image
        """
        # print(
        #     f"Computing reflectance for {self.band_name} ({self.band_index}),"
        #     f"{self.center_wavelength}, {self.dark_pixels}, {self.black_level}"
        # )
        if not force_recompute:
            if (
                self.__reflectance_image is not None
                and self.__vc_g == vc_g
                and (self.__reflectance_irradiance == irradiance or irradiance is None)
            ):
                # DO NOT recompute if:
                # 1) ``self.__reflectance_image`` exists, and
                # 2) ``self.__vc_g`` equals user-specified ``vc_g``
                # 3) ``self.__reflectance_irradiance`` == user specified ``irradiance``
                #    or the input irradiance is None
                return self.__reflectance_image

        rad_kw = {"which_dc": which_dc.lower(), "vc_g": vc_g}

        if self.band_name != "LWIR":
            if irradiance is None:
                # User has not provided an irradiance value - get the
                # DLS2 irradiance value (if it exists)
                if self.horizontal_irradiance != 0.0:
                    irradiance = self.horizontal_irradiance
                    # self.__reflectance_irradiance remains None - is this for a reason?
                else:
                    raise ValueError(
                        "horizontal irradiance value is 0.0 - please provide "
                        "a band-specific spectral irradiance to compute reflectance"
                    )
            else:
                # Apply the user-specified irradiance value
                self.__reflectance_irradiance = irradiance

            # print(f"irradiance at {self.center_wavelength}: {irradiance}")
            # compute the Reflectance or Remote sensing reflectance
            if return_rrs:
                self.__reflectance_image = self.radiance(**rad_kw) / irradiance
            else:
                self.__reflectance_image = math.pi * self.radiance(**rad_kw) / irradiance
        else:
            # Return the LongWave IR radiance image as-is
            self.__reflectance_image = self.radiance(**rad_kw)

        return self.__reflectance_image

    def undistorted_radiance(
        self,
        force_recompute: bool = False,
        which_dc: str = "dark",
        vc_g: float = 1.0,
    ) -> np.ndarray:
        return self.undistorted(
            self.radiance(force_recompute=force_recompute, which_dc=which_dc, vc_g=vc_g)
        )

    def undistorted_reflectance(
        self,
        irradiance: Optional[float] = None,
        vc_g: float = 1.0,
        force_recompute: bool = False,
        which_dc: str = "dark",
        return_rrs: bool = False,
    ) -> np.ndarray:
        return self.undistorted(
            self.reflectance(
                irradiance=irradiance,
                force_recompute=force_recompute,
                which_dc=which_dc,
                vc_g=vc_g,
                return_rrs=return_rrs,
            )
        )

    def plottable_vignette(self) -> np.ndarray:
        return self.vignette()[0].T

    def cv2_distortion_coeff(self) -> np.ndarray:
        # dist_coeffs = np.array(k[0],k[1],p[0],p[1],k[2]])
        return np.array(self.distortion_parameters)[[0, 1, 3, 4, 2]]

    # values in pp are in [mm], rescale to pixels
    def principal_point_px(self) -> Tuple[float, float]:
        center_x = self.principal_point[0] * self.focal_plane_resolution_px_per_mm[0]
        center_y = self.principal_point[1] * self.focal_plane_resolution_px_per_mm[1]
        return (center_x, center_y)

    def cv2_camera_matrix(self) -> np.ndarray:
        center_x, center_y = self.principal_point_px()

        # set up camera matrix for cv2
        cam_mat = np.zeros((3, 3))
        cam_mat[0, 0] = self.focal_length * self.focal_plane_resolution_px_per_mm[0]
        cam_mat[1, 1] = self.focal_length * self.focal_plane_resolution_px_per_mm[1]
        cam_mat[2, 2] = 1.0
        cam_mat[0, 2] = center_x
        cam_mat[1, 2] = center_y

        # set up distortion coefficients for cv2
        return cam_mat

    def rig_xy_offset_in_px(self) -> Tuple[float, float]:
        pixel_pitch_mm_x = 1.0 / self.focal_plane_resolution_px_per_mm[0]
        pixel_pitch_mm_y = 1.0 / self.focal_plane_resolution_px_per_mm[1]
        px_fov_x = 2.0 * math.atan2(pixel_pitch_mm_x / 2.0, self.focal_length)
        px_fov_y = 2.0 * math.atan2(pixel_pitch_mm_y / 2.0, self.focal_length)
        t_x = math.radians(self.rig_relatives[0]) / px_fov_x
        t_y = math.radians(self.rig_relatives[1]) / px_fov_y
        return (t_x, t_y)

    def undistorted(self, image: np.ndarray, centre_pp: bool = False) -> np.ndarray:
        """return the undistorted image from input image"""
        # If we have already undistorted the same source, just return that here
        # otherwise, lazy compute the undstorted image
        if (
            self.__undistorted_source is not None
            and image.data == self.__undistorted_source.data
        ):
            return self.__undistorted_image

        self.__undistorted_source = image

        new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(
            self.cv2_camera_matrix(),  # cameraMatrix
            self.cv2_distortion_coeff(),  # distCoeffs
            self.size(),  # imageSize
            1,  # alpha
            self.size(),  # newImgSize
            centre_pp,  # centerPrincipalPoint
        )

        # We have an issue. The undistorted image has intrinsic camera
        # parameters defined in `new_cam_mat`, which have units of 'pixels'.
        # Unfortunately, the EXIF tags require units of inches, cm, mm or
        # micrometres. In order to convert between pixels and, say, mm,
        # we need to know the focal plane x and y resolutions (units: px/mm).
        # These parameters in `new_cam_mat` are also unknown. Additionally,
        # `new_cam_mat` generally has different fx and fy. Yet a pinhole
        # camera model only has a single focal length. This implies that
        # the focal plane x and y resolutions (dfx and dfx) are different.
        # fx (pixels) = f (mm) * dfx (pixels / mm)
        # fy (pixels) = f (mm) * dfy (pixels / mm)
        # cx (pixels) = X-Principal point (mm) * dfx (pixels / mm)
        # cy (pixels) = Y-Principal point (mm) * dfy (pixels / mm)

        # From fx, fy, cx and cy in `new_cam_mat`, we cannot solve for
        # f, dfx, dfy, X-PP, Y-PP as there are five unknowns from four
        # known values. Thus we have to assume that the focal length
        # has been conserved.

        dfx = new_cam_mat[0, 0] / self.focal_length
        dfy = new_cam_mat[1, 1] / self.focal_length
        self.newcammat_dfx = dfx
        self.newcammat_dfy = dfy
        self.newcammat_ppx = new_cam_mat[0, 2] / dfx
        self.newcammat_ppy = new_cam_mat[1, 2] / dfy

        map1, map2 = cv2.initUndistortRectifyMap(
            self.cv2_camera_matrix(),
            self.cv2_distortion_coeff(),
            np.eye(3),
            new_cam_mat,
            self.size(),
            cv2.CV_32F,
        )  # cv2.CV_32F for 32 bit floats
        # compute the undistorted 16 bit image
        self.__undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        return self.__undistorted_image

    def plot_raw(self, title=None, figsize=None):
        """Create a single plot of the raw image"""
        if title is None:
            title = "{} Band {} Raw DN".format(self.band_name, self.band_index)
        return plotutils.plotwithcolorbar(self.raw(), title=title, figsize=figsize)

    def plot_intensity(self, title=None, figsize=None):
        """Create a single plot of the image converted to uncalibrated intensity"""
        if title is None:
            title = "{} Band {} Intensity (DN*sec)".format(
                self.band_name, self.band_index
            )
        return plotutils.plotwithcolorbar(self.intensity(), title=title, figsize=figsize)

    def plot_radiance(self, title=None, figsize=None):
        """Create a single plot of the image converted to radiance"""
        if title is None:
            title = "{} Band {} Radiance".format(self.band_name, self.band_index)
        return plotutils.plotwithcolorbar(self.radiance(), title=title, figsize=figsize)

    def plot_vignette(self, title=None, figsize=None):
        """Create a single plot of the vignette"""
        if title is None:
            title = "{} Band {} Vignette".format(self.band_name, self.band_index)
        return plotutils.plotwithcolorbar(
            self.plottable_vignette(), title=title, figsize=figsize
        )

    def plot_undistorted_radiance(self, title=None, figsize=None):
        """Create a single plot of the undistorted radiance"""
        if title is None:
            title = "{} Band {} Undistorted Radiance".format(
                self.band_name, self.band_index
            )
        return plotutils.plotwithcolorbar(
            self.undistorted(self.radiance()), title=title, figsize=figsize
        )

    def plot_all(self, figsize=(13, 10)) -> None:
        plots = [
            self.raw(),
            self.plottable_vignette(),
            self.radiance(),
            self.undistorted(self.radiance()),
        ]
        plot_types = ["Raw", "Vignette", "Radiance", "Undistorted Radiance"]
        titles = [
            "{} Band {} {}".format(str(self.band_name), str(self.band_index), tpe)
            for tpe in plot_types
        ]
        plotutils.subplotwithcolorbar(2, 2, plots, titles, figsize=figsize)

        # get the homography that maps from this image to the reference image

    def get_homography(
        self,
        ref,
        r_mat: Optional[np.matrix] = None,
        t_mat: Optional[np.matrix] = None,
    ) -> np.ndarray:
        # if we have externally supplied rotations/translations for the rig use these
        # otherwise use the rig-relatives intrinsic to the image
        if r_mat is None:
            r_mat = rotations_degrees_to_rotation_matrix(self.rig_relatives)
        if t_mat is None:
            t_mat = np.zeros(3)

        r_ref = rotations_degrees_to_rotation_matrix(ref.rig_relatives)
        a_arr = np.zeros((4, 4))
        a_arr[0:3, 0:3] = np.dot(r_ref.T, r_mat)
        a_arr[0:3, 3] = t_mat
        a_arr[3, 3] = 1.0

        c_arr, _ = cv2.getOptimalNewCameraMatrix(
            self.cv2_camera_matrix(), self.cv2_distortion_coeff(), self.size(), 1
        )

        cr_arr, _ = cv2.getOptimalNewCameraMatrix(
            ref.cv2_camera_matrix(), ref.cv2_distortion_coeff(), ref.size(), 1
        )

        cc = np.zeros((4, 4))
        cc[0:3, 0:3] = c_arr
        cc[3, 3] = 1.0

        ccr = np.zeros((4, 4))
        ccr[0:3, 0:3] = cr_arr
        ccr[3, 3] = 1.0

        b_arr = np.array(np.dot(ccr, np.dot(a_arr, np.linalg.inv(cc))))
        b_arr[:, 2] = b_arr[:, 2] - b_arr[:, 3]
        b_arr = b_arr[0:3, 0:3]
        b_arr = b_arr / b_arr[2, 2]

        return np.array(b_arr)
