#!/usr/bin/env python3
# coding: utf-8
"""
MicaSense Capture Class

    A Capture is a set of Images taken by one camera which share the same unique
    capture identifier (capture_id). Generally these images will be found in the
    same folder and also share the same filename prefix, such as IMG_0000_*.tif,
    but this is not required.

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

import math
import numpy as np
from pathlib import Path
from os.path import isfile
from typing import Union, List, Optional, Tuple
from pysolar.solar import get_altitude, get_azimuth

import micasense.image as image
import micasense.load_yaml as ms_yaml
import micasense.plotutils as plotutils

from micasense.panel import Panel


class Capture(object):
    """
    A Capture is a set of Images taken by one MicaSense camera which share
    the same unique capture identifier (capture_id). Generally these images will be
    found in the same folder and also share the same filename prefix, such
    as IMG_0000_*.tif, but this is not required.
    """

    def __init__(
        self,
        images: Union[image.Image, List[image.Image]],
        panel_corners: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        images: image.Image, List[image.Image]

            system file paths.
            Class is typically created using from_file(str or Path),
            from_file_list(List[str] or List[Path]), or from_yaml(str
            or Path) methods. Captures are also created automatically
            using ImageSet.from_directory()
        panel_corners: List[int] [Optional]
            List of int coordinates
            e.g.
            [
                [[873, 1089], [767, 1083], [763, 1187], [869, 1193]],
                [[993, 1105], [885, 1101], [881, 1205], [989, 1209]],
                [[1000, 1030], [892, 1026], [888, 1130], [996, 1134]],
                [[892, 989], [786, 983], [780, 1087], [886, 1093]],
                [[948, 1061], [842, 1057], [836, 1161], [942, 1165]]
            ]

            The camera should automatically detect panel corners. This instance
            variable will be None for aerial captures. You can populate this for
            panel captures by calling detect_panels().
        """
        err_msg = "Provide an image.Image or List[image.Image] to create a Capture."
        if isinstance(images, image.Image):
            self.images = [images]
        elif isinstance(images, list):
            # ensure that the list only contains image.Image objects
            if all(type(i) is image.Image for i in images):
                self.images = images
            else:
                raise RuntimeError(err_msg)
        else:
            raise RuntimeError(err_msg)

        self.num_bands = len(self.images)
        self.images.sort()
        capture_ids = [img.capture_id for img in self.images]
        if len(set(capture_ids)) != 1:
            raise RuntimeError("Images provided must have the same capture_id.")
        self.uuid = self.images[0].capture_id
        self.panels = None
        self.detected_panel_count = 0
        if panel_corners is None:
            self.panel_corners = [None] * len(self.eo_indices())
        else:
            self.panel_corners = panel_corners

    def set_panel_corners(self, panel_corners: List[int]):
        """
        Define panel corners by hand.
        :param panel_corners: 2d List of int coordinates.
            e.g. [[536, 667], [535, 750], [441, 755], [444, 672]]
        :return: None
        """
        self.panel_corners = panel_corners
        self.panels = None
        self.detect_panels()

    def append_image(self, img: np.ndarray):
        """
        Add an Image to the Capture.
        :param img: An Image object.
        :return: None
        """
        if self.uuid != img.capture_id:
            raise RuntimeError("Added images must have the same capture_id.")
        self.images.append(img)
        self.images.sort()

    def append_images(self, images: List[np.ndarray]):
        """
        Add multiple Images to the Capture.
        :param images: List of Image objects.
        """
        [self.append_image(img) for img in images]

    def append_file(self, filename: str):
        """
        Add an Image to the Capture using a file path.
        :param filename: str system file path.
        """
        self.append_image(image.Image(filename))

    @classmethod
    def from_file(cls, filename: Union[Path, str]):
        """
        Create Capture instance from file path.
        Parameters
        ----------
        filename: Path or str
            system file path
        Returns
        -------
        Capture object.
        """
        return cls(image.Image(image_path=filename))

    @classmethod
    def from_filelist(cls, file_list: Union[List[Path], List[str]]):
        """
        Create Capture instance from List of file paths.
        Parameters
        ----------
        file_list: List of str
            File paths for each band in a single capture
        Returns
        -------
        Capture object.
        """
        if len(file_list) == 0:
            raise IOError("No files provided. Check your file paths.")
        for f in file_list:
            if not isfile(f):
                raise IOError(
                    "All files in file list must be a file. "
                    f"The following file is not:\n{f}"
                )
        images = [image.Image(image_path=f) for f in file_list]
        return cls(images)

    @classmethod
    def from_yaml(cls, yaml_file: Union[Path, str]):
        """
        Create Capture object from a yaml file containing the
        relevant metadata for each band.

        Parameters
        ----------
        yaml_file : Path or str
            The yaml file
        Returns
        -------
        Capture object.
        """
        d = ms_yaml.load_all(yaml_file=yaml_file)

        images = [
            image.Image(
                image_path=d["image_data"][key]["filename"],
                metadata_dict=d,
            )
            for key in d["image_data"]
        ]

        return cls(images)

    def __get_reference_index(self):
        """
        Find the reference image which has the smallest rig offsets,
        they should be (0,0).
        Returns
        -------
        ndarray of ints,
            The indices of the minimum values along an axis.
        """
        return np.argmin(
            (np.array([i.rig_xy_offset_in_px() for i in self.images]) ** 2).sum(1)
        )

    def __plot(
        self, images, num_cols=2, plot_type=None, color_bar=True, fig_size=(14, 14)
    ):
        """
        Plot the Images from the Capture.
        :param images: List of Image objects
        :param num_cols: int number of columns
        :param plot_type: str for plot title formatting
        :param color_bar: boolean to determine color bar inclusion
        :param fig_size: Tuple size of the figure
        :return: plotutils result. matplotlib Figure and Axis in both cases.
        """
        if plot_type is None:
            plot_type = ""
        else:
            titles = [
                "{} Band {} {}".format(
                    str(img.band_name),
                    str(img.band_index),
                    plot_type
                    if img.band_name.upper() != "LWIR"
                    else "Brightness Temperature",
                )
                for img in self.images
            ]
        num_rows = int(math.ceil(float(len(self.images)) / float(num_cols)))
        if color_bar:
            return plotutils.subplotwithcolorbar(
                num_rows, num_cols, images, titles, fig_size
            )
        else:
            return plotutils.subplot(num_rows, num_cols, images, titles, fig_size)

    def __lt__(self, other):
        return self.utc_time() < other.utc_time()

    def __gt__(self, other):
        return self.utc_time() > other.utc_time()

    def __eq__(self, other):
        return self.uuid == other.uuid

    def location(self):
        """(lat, lon, alt) tuple of WGS-84 location units are radians, meters msl"""
        # TODO: These units are "signed decimal degrees" per metadata.py comments?
        return self.images[0].location

    def utc_time(self):
        """Returns a timezone-aware datetime object of the capture time."""
        return self.images[0].utc_time

    def solar_geoms(self) -> Tuple[float, float]:
        """Returns the solar zenith and azimuth at the time of capture"""
        lat, lon, _ = self.location()
        dt = self.utc_time()
        return 90.0 - get_altitude(lat, lon, dt), get_azimuth(lat, lon, dt)

    def clear_image_data(self):
        """
        Clears (dereferences to allow garbage collection) all internal
        image data stored in this class. Call this after processing-heavy
        image calls to manage program memory footprint. When processing
        many images, such as iterating over the Captures in an ImageSet,
        it may be necessary to call this after Capture is processed.
        """
        for img in self.images:
            img.clear_image_data()

    def center_wavelengths(self):
        """Returns a list of the image center wavelengths in nanometers."""
        return [img.center_wavelength for img in self.images]

    def band_names(self):
        """Returns a list of the image band names as they are in the image metadata."""
        return [img.band_name for img in self.images]

    def band_names_lower(self):
        """
        Returns a list of the Image band names in all
        lower case for easier comparisons
        """
        return [img.band_name.lower() for img in self.images]

    def dls_present(self) -> bool:
        """Returns true if DLS metadata is present in the images."""
        return self.images[0].dls_present

    def dls_irradiance_raw(self) -> List[float]:
        """Returns a list of the raw DLS measurements from the image metadata."""
        return [img.spectral_irradiance for img in self.images]

    def dls_irradiance(self) -> List[float]:
        """
        Returns a list of the corrected earth-surface (horizontal)
        DLS irradiance in W/m^2/nm
        """
        return [img.horizontal_irradiance for img in self.images]

    def direct_irradiance(self) -> List[float]:
        """
        Returns a list of the DLS irradiance from the direct source in W/m^2/nm
        """
        return [img.direct_irradiance for img in self.images]

    def scattered_irradiance(self) -> List[float]:
        """
        Returns a list of the DLS scattered irradiance
        from the direct source in W/m^2/nm
        """
        return [img.scattered_irradiance for img in self.images]

    def dls_pose(self):
        """
        Returns (yaw, pitch, roll) tuples in radians of the earth-fixed DLS pose
        """
        return self.images[0].dls_yaw, self.images[0].dls_pitch, self.images[0].dls_roll

    def plot_raw(self):
        """Plot raw images as the data came from the camera."""
        self.__plot([img.raw() for img in self.images], plot_type="Raw")

    def plot_vignette(self):
        """Compute (if necessary) and plot vignette correction images."""
        self.__plot([img.vignette()[0].T for img in self.images], plot_type="Vignette")

    def plot_radiance(self):
        """Compute (if necessary) and plot radiance images."""
        self.__plot([img.radiance() for img in self.images], plot_type="Radiance")

    def plot_undistorted_radiance(self):
        """Compute (if necessary) and plot undistorted radiance images."""
        self.__plot(
            [img.undistorted(img.radiance()) for img in self.images],
            plot_type="Undistorted Radiance",
        )

    def plot_undistorted_reflectance(self, irradiance_list: List[float]) -> None:
        """
        Compute (if necessary) and plot reflectances given a list
        of irradiances.

        Parameters
        ----------
        irradiance_list: List
            A list returned from Capture.dls_irradiance() or
            Capture.panel_irradiance()
        """
        self.__plot(
            self.undistorted_reflectance(irradiance_list),
            plot_type="Undistorted Reflectance",
        )

    def compute_radiance(self) -> None:
        """Compute Image radiances"""
        [img.radiance() for img in self.images]

    def compute_undistorted_radiance(self) -> None:
        """Compute Image undistorted radiance."""
        [img.undistorted_radiance() for img in self.images]

    def compute_reflectance(self, irradiance_list=None, force_recompute=True) -> None:
        """
        Compute Image reflectance from irradiance list, but don't return.

        Parameters
        ----------
        irradiance_list: List
            A list returned from Capture.dls_irradiance() or Capture.panel_irradiance()
        force_recompute: boolean
            Specifies whether reflectance is to be recomputed.

        Returns
        -------
        None
        """
        if irradiance_list is not None:
            [
                img.reflectance(irradiance_list[i], force_recompute=force_recompute)
                for i, img in enumerate(self.images)
            ]
        else:
            [img.reflectance(force_recompute=force_recompute) for img in self.images]

    def compute_undistorted_reflectance(
        self, irradiance_list=None, force_recompute=True
    ) -> None:
        """
        Compute undistorted image reflectance from irradiance list.

        Parameters
        ----------
        irradiance_list: List
            A list of returned from Capture.dls_irradiance() or Capture.panel_irradiance()
            TODO: improve this docstring
        force_recompute: boolean
           Specifies whether reflectance is to be recomputed.

        Returns
        -------
        None
        """
        if irradiance_list is not None:
            [
                img.undistorted_reflectance(
                    irradiance_list[i], force_recompute=force_recompute
                )
                for i, img in enumerate(self.images)
            ]
        else:
            [
                img.undistorted_reflectance(force_recompute=force_recompute)
                for img in self.images
            ]

    def eo_images(self) -> List[np.ndarray]:
        """Returns a list of the EO Images in the Capture."""
        return [img for img in self.images if img.band_name != "LWIR"]

    def lw_images(self) -> List[np.ndarray]:
        """Returns a list of the longwave infrared Images in the Capture."""
        return [img for img in self.images if img.band_name == "LWIR"]

    def eo_indices(self) -> List[int]:
        """Returns a list of the indexes of the EO Images in the Capture."""
        return [index for index, img in enumerate(self.images) if img.band_name != "LWIR"]

    def lw_indices(self) -> List[int]:
        """
        Returns a list of the indexes of the longwave infrared Images in the Capture
        """
        return [index for index, img in enumerate(self.images) if img.band_name == "LWIR"]

    def reflectance(self, irradiance_list: List[float]) -> List[np.ndarray]:
        """
        Compute reflectance Images.

        Parameters
        ----------
        irradiance_list: List
           A list returned from Capture.dls_irradiance() or Capture.panel_irradiance()
           TODO: improve this docstring
        Returns
        -------
        List of reflectance EO and long wave infrared Images for given irradiance.
        """
        eo_imgs = [
            img.reflectance(irradiance_list[i]) for i, img in enumerate(self.eo_images())
        ]
        lw_imgs = [img.reflectance() for i, img in enumerate(self.lw_images())]
        return eo_imgs + lw_imgs

    def undistorted_reflectance(self, irradiance_list: List[float]) -> List[np.ndarray]:
        """
        Compute undistorted reflectance Images.

        Parameters
        ----------
        irradiance_list: List
            A list returned from Capture.dls_irradiance() or Capture.panel_irradiance()
            TODO: improve this docstring
        Returns
        -------
        List of undistorted reflectance images for given irradiance.
        """
        eo_imgs = [
            img.undistorted(img.reflectance(irradiance_list[i]))
            for i, img in enumerate(self.eo_images())
        ]
        lw_imgs = [
            img.undistorted(img.reflectance()) for i, img in enumerate(self.lw_images())
        ]
        return eo_imgs + lw_imgs

    def panels_in_all_expected_images(self):
        """
        Check if all expected reflectance panels are
        detected in the EO Images in the Capture.

        Returns
        -------
        True if reflectance panels are detected.
        """
        expected_panels = sum(str(img.band_name).upper() != "LWIR" for img in self.images)
        return self.detect_panels() == expected_panels

    def panel_raw(self) -> List[float]:
        """Return a list of mean panel region values for raw images."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        raw_list = []
        for p in self.panels:
            mean, _, _, _ = p.raw()
            raw_list.append(mean)
        return raw_list

    def panel_radiance(self) -> List[float]:
        """Return a list of mean panel region values for converted radiance Images."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        radiance_list = []
        for p in self.panels:
            mean, _, _, _ = p.radiance()
            radiance_list.append(mean)
        return radiance_list

    def panel_irradiance(self, reflectances=None) -> List[float]:
        """Return a list of mean panel region values for irradiance values."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        if reflectances is None:
            reflectances = [
                panel.reflectance_from_panel_serial() for panel in self.panels
            ]
        if len(reflectances) != len(self.panels):
            raise ValueError("Length of panel reflectances must match length of Images.")
        irradiance_list = []
        for i, p in enumerate(self.panels):
            mean_irr = p.irradiance_mean(reflectances[i])
            irradiance_list.append(mean_irr)
        return irradiance_list

    def panel_reflectance(self, panel_refl_by_band=None) -> List[float]:
        # FIXME: panel_refl_by_band parameter isn't used?
        """Return a list of mean panel reflectance values."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        reflectance_list = []
        for i, p in enumerate(self.panels):
            self.images[i].reflectance()
            mean_refl = p.reflectance_mean()
            reflectance_list.append(mean_refl)
        return reflectance_list

    def panel_albedo(self) -> Union[List[float], None]:
        """Return a list of panel reflectance values from metadata."""
        if self.panels_in_all_expected_images():
            albedos = [panel.reflectance_from_panel_serial() for panel in self.panels]
            if None in albedos:
                albedos = None
        else:
            albedos = None
        return albedos

    def detect_panels(self) -> int:
        """Detect reflectance panels in the Capture, and return a count."""

        if self.panels is not None and self.detected_panel_count == len(self.images):
            return self.detected_panel_count
        self.panels = [
            Panel(img, panel_corners=pc)
            for img, pc in zip(self.images, self.panel_corners)
        ]
        self.detected_panel_count = 0
        for p in self.panels:
            if p.panel_detected():
                self.detected_panel_count += 1
        # if panel_corners are defined by hand
        if self.panel_corners is not None and all(
            corner is not None for corner in self.panel_corners
        ):
            self.detected_panel_count = len(self.panel_corners)
        return self.detected_panel_count

    def plot_panels(self):
        """Plot Panel images."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        self.__plot(
            [p.plot_image() for p in self.panels], plot_type="Panels", color_bar=False
        )

    def set_external_rig_relatives(self, external_rig_relatives) -> None:
        """
        Set external rig relatives.
        :param external_rig_relatives: TODO: Write this parameter docstring
        :return: None
        """
        for i, img in enumerate(self.images):
            img.set_external_rig_relatives(external_rig_relatives[str(i)])

    def has_rig_relatives(self) -> bool:
        """
        Check if Images in Capture have rig relatives.
        :return: boolean True if all Images have rig relatives metadata.
        """
        for img in self.images:
            if img.meta.rig_relatives() is None:
                return False
        return True

    def get_warp_matrices(self, ref_index=None) -> List[np.ndarray]:
        """
        Get warp matrices. Used in imageutils.refine_alignment_warp

        Parameters
        ----------
        ref_index: int to specify image for homography

        Returns
        -------
        2d List of warp matrices
        """
        if ref_index is None:
            ref = self.images[self.__get_reference_index()]
        else:
            ref = self.images[ref_index]
        warp_matrices = [np.linalg.inv(im.get_homography(ref)) for im in self.images]
        return [w / w[2, 2] for w in warp_matrices]
