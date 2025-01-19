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
from matplotlib.figure import Figure
from pysolar.solar import get_altitude, get_azimuth
from typing import Union, List, Optional, Tuple, Iterable

from .image import Image
from .panel import Panel
from .yaml_handler import load_yaml
from .plotutils import subplotwithcolorbar


class Capture(object):
    """
    A Capture is a set of Images taken by one MicaSense camera which share
    the same unique capture identifier (capture_id). Generally these images will be
    found in the same folder and also share the same filename prefix, such
    as IMG_0000_*.tif, but this is not required.
    """

    def __init__(
        self,
        images: Union[Image, List[Image]],
        panel_corners: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        images: Image, List[Image]

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
        err_msg = "Provide an Image or List[Image] to create a Capture."
        if isinstance(images, Image):
            self.images = [images]  # a single image
        elif isinstance(images, list):
            # ensure that the list only contains Image objects
            if all(type(i) is Image for i in images):
                self.images = images  # a list of images
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
        self.append_image(Image(filename))

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
        return cls(Image(image_path=filename))

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
        images = [Image(image_path=f) for f in file_list]
        return cls(images)

    @classmethod
    def from_yaml(
        cls, yaml_file: Union[Path, str], base_path: Optional[Union[Path, str]] = None
    ):
        """
        Create Capture object from a yaml file containing the
        relevant metadata for each band.

        Parameters
        ----------
        yaml_file : Path or str
            The micasense acquistion yaml file
        base_path : Optional[Path or str]
            The Base path of the relative filenames given in the yaml file.
            If None, then the base_path provided in the yaml file will be used.

        Returns
        -------
        Capture object.
        """
        d = load_yaml(yaml_file=yaml_file)
        if base_path is None:
            base_path = Path(d["base_path"])
        elif isinstance(base_path, (str, Path)):
            base_path = Path(base_path)
        else:
            raise ValueError("`base_path` must be None, str or Path")

        if not base_path.exists():
            raise FileNotFoundError(f"'{base_path}' does not exist")

        images = [
            Image(
                image_path=base_path / d["image_data"][key]["filename"],
                metadata_dict=d,
            )
            for key in d["image_data"]
        ]

        return cls(images)

    @classmethod
    def from_yaml_special(
        cls,
        yaml_file: Union[Path, str],
        base_path: Optional[Union[Path, str]] = None,
        vig_yaml: Optional[Path] = None,
    ):
        """
        Create Capture object from a yaml file containing the
        relevant metadata for each band.

        Parameters
        ----------
        yaml_file : Path or str
            The micasense acquisition yaml file
        base_path : Optional[Path or str]
            The Base path of the relative filenames given in the yaml file.
            If None, then the base_path provided in the yaml file will be used.
        vig_yaml : Optional[Path]
            The yaml file containing paths to the vignetting image (.npy)
            binary files for each band specified in `yaml_file`. If not specified,
            then the default vignetting model is used.

        Returns
        -------
        Capture object.
        """

        def get_bandnum(tif: Path) -> int:
            """Return the band number from the filename"""
            return int(tif.stem.split("_")[-1])

        def load_vig_image(vig_params: dict, band_num: int) -> Union[None, np.ndarray]:
            # Note: vig_params could be an empty dict, {}, in which case return None
            out = None
            if vig_params:
                if band_num not in vig_params:
                    raise KeyError(
                        f"'{band_num}' is not a key in 'vig_params'\n"
                        f"vig_params: {vig_params}"
                    )
                if "vig_model_npy" not in vig_params[band_num]:
                    raise KeyError(
                        f"'vig_model_npy' not in `vig_params[{band_num}]`\n"
                        f"{vig_params[band_num]}"
                    )

                # check that vignetting image file exists
                vig_model_path = Path(vig_params[band_num]["vig_model_npy"])
                if not vig_model_path.exists():
                    raise FileNotFoundError(f"File '{vig_model_path}' not found.")

                # Attempt to load npy; raise Exception if fails to load.
                try:
                    out = np.load(vig_model_path)
                except Exception as e:
                    raise IOError(f"Error loading file '{vig_model_path}': {e}")

            return out

        # ------------------------------- #
        d = load_yaml(yaml_file=yaml_file)
        if base_path is None:
            base_path = Path(d["base_path"])
        elif isinstance(base_path, (str, Path)):
            base_path = Path(base_path)
        else:
            raise ValueError("`base_path` must be None, str or Path")

        if not base_path.exists():
            raise FileNotFoundError(f"'{base_path}' does not exist")

        vig_params = load_yaml(vig_yaml) if vig_yaml else {}

        images = []
        for k in d["image_data"]:
            image_fn = base_path / d["image_data"][k]["filename"]
            bn = get_bandnum(image_fn)

            kw = {
                "image_path": image_fn,
                "metadata_dict": d,
                "dark_current": None,
                "vig_params": None,
                "vig_image": load_vig_image(vig_params=vig_params, band_num=bn),
            }
            images.append(Image(**kw))

        return cls(images=images)

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
        self,
        images: np.ndarray,
        num_cols: int = 2,
        title: str = "",
        cmap: str = "jet",
        cbar: bool = True,
        figsize: Iterable[int] = (14, 14),
        vrange: Optional[Iterable[float]] = None,
        selected_bands: Optional[Iterable[int]] = None,
        show: bool = False,
    ):
        """
        Plot a list images from the Capture object

        Parameters
        ----------
        images : np.ndarray, {dims=(nbands, nrows, ncols)}
            Image cube
        num_cols : int
            Number of columns in the figure
        title : str
            plot title
        cmap : str
            registered colormap name used to map scalar data to colors
        cbar : bool
            boolean to determine color bar inclusion
        figsize : Iterable[int] {Optional}
            Tuple size of the figure
        vrange: Iterable[float] {Optional}
            common data range [vmin, vmax] that the colour map covers
        selected_bands : Iterable[int] {Optional}
            user selected bands (given by their indices) to display.
            Here, indexing starts at 0
        show : bool
            whether to show the image with plt.show()

        Returns
        -------
        matplotlib Figure and Axis in both cases.
        """
        if selected_bands is None:
            selected_bands = [i for i in range(len(images))]

        axtitles = []
        for sb in selected_bands:
            for i in self.images:
                if i.band_index == sb:
                    axtitles.append(f"Band {i.band_index} ({i.center_wavelength} nm)")

        num_rows = int(math.ceil(float(len(selected_bands)) / float(num_cols)))
        return subplotwithcolorbar(
            rows=num_rows,
            cols=num_cols,
            images=images[selected_bands, :, :],
            suptitle=title,
            cmap=cmap,
            cbar=cbar,
            titles=axtitles,
            figsize=figsize,
            vrange=vrange,
            show=show,
        )

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
        """
        Get the solar geometry

        Returns
        -------
        sza : float
            solar zenith angle
        saa : float
            solar azimuth anlge, where,
            0  = north, 90 = east, 180 = south, 270 = west
        """
        lat, lon, _ = self.location()
        dt = self.utc_time()

        sza = 90.0 - get_altitude(lat, lon, dt)

        saa = get_azimuth(lat, lon, dt)  # negative angles are west of North
        saa = saa % 360

        return sza, saa

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

    def plot_raw(self):
        """Plot raw images as the data came from the camera."""
        self.__plot([img.raw() for img in self.images], title="Raw", show=True)

    def plot_vignette(
        self,
        cmap: str = "jet",
        vrange: Optional[Iterable[float]] = None,
        figsize: Optional[Iterable[float]] = None,
        selected_bands: Optional[Iterable[int]] = None,
        show: bool = False,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Compute (if necessary) and plot vignette correction images

        Parameters
        ----------
        cmap : str
            registered colormap name used to map scalar data to colors
        vrange : Iterable[float] {Optional}
            common data range [vmin, vmax] that the colour map covers
        figsize : Iterable[float] {Optional}
            Tuple size of the figure
        selected_bands : Iterable[int] {Optional}
            user selected bands (given by their indices) to display.
            Here, indexing starts at 0
        show : bool
            whether to show the image with plt.show()

        Returns
        -------
        fig, axes
        """
        return self.__plot(
            images=np.array([img.vignette()[0].T for img in self.images]),
            title="Vignetting correction",
            figsize=figsize,
            vrange=vrange,
            cmap=cmap,
            selected_bands=selected_bands,
        )

    def plot_radiance(self, **kw):
        """Compute (if necessary) and plot radiance images."""
        self.__plot(
            images=np.array([img.radiance(**kw) for img in self.images]),
            title="Radiance",
            show=True,
        )

    def plot_undistorted_radiance(self, **kw):
        """Compute (if necessary) and plot undistorted radiance images."""
        self.__plot(
            images=np.array([img.undistorted(img.radiance(**kw)) for img in self.images]),
            title="Undistorted Radiance",
            show=True,
        )

    def plot_undistorted_reflectance(
        self,
        irradiance_list: List[float],
        vcg_list: List[float],
        cmap: str = "jet",
        figsize: Optional[Iterable[float]] = None,
        vrange: Optional[Iterable[float]] = None,
        selected_bands: Optional[Iterable[float]] = None,
        show: bool = False,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Compute (if necessary) and plot reflectances given a list
        of irradiances.

        Parameters
        ----------
        irradiance_list: List[float]
            A list of downwelling solar irradiance values either measured by the DLS2 (see
            Capture.dls_irradiance() or Capture.panel_irradiance()) or an independent
            irradiance sensor.

        vcg_list : List[float]
            A list of vicarious calibration gains

        cmap : str
            registered colormap name used to map scalar data to colors
        figsize : Iterable[float] {Optional}
            Tuple size of the figure
        vrange : Iterable[float] {Optional}
            common data range [vmin, vmax] that the colour map covers
        selected_bands : Iterable[int] {Optional}
            user selected bands (given by their indices) to display.
            Here, indexing starts at 0
        show : bool
            whether to show the image with plt.show()


        Returns
        -------
        fig, axes
        """
        images = np.array(
            self.undistorted_reflectance(irradiance=irradiance_list, vc_g=vcg_list)
        )
        return self.__plot(
            images=images,
            title="Undistorted Reflectance",
            figsize=figsize,
            vrange=vrange,
            cmap=cmap,
            selected_bands=selected_bands,
            show=show,
        )

    def compute_radiance(
        self,
        vc_g: Optional[List[float]] = None,
        force_recompute: bool = True,
        which_dc: str = "dark",
    ) -> None:
        """Compute Image radiances"""
        vcg_ = [1] * len(self.images) if vc_g is None else vc_g
        [
            img.radiance(
                vc_g=vcg_[i],
                force_recompute=force_recompute,
                which_dc=which_dc,
            )
            for i, img in enumerate(self.images)
        ]

    def compute_undistorted_radiance(
        self,
        vc_g: Optional[List[float]] = None,
        force_recompute=True,
        which_dc: str = "dark",
    ) -> None:
        """Compute Image undistorted radiance."""
        vcg_ = [1] * len(self.images) if vc_g is None else vc_g

        [
            img.undistorted_radiance(
                vc_g=vcg_[i],
                force_recompute=force_recompute,
                which_dc=which_dc,
            )
            for i, img in enumerate(self.images)
        ]

    def compute_reflectance(
        self,
        irradiance: Optional[List[float]] = None,
        vc_g: Optional[List[None]] = None,
        force_recompute: bool = True,
        which_dc: str = "dark",
    ) -> None:
        """
        Compute Image reflectance from irradiance list, but don't return.

        Parameters
        ----------
        irradiance: List[float] or None
            A list of downwelling solar irradiance values either measured by the DLS2 (see
            Capture.dls_irradiance() or Capture.panel_irradiance()) or an independent
            irradiance sensor.

        vc_g : List[float] or None
            A list of vicarious calibration gains

        force_recompute: bool
            Specifies whether reflectance is to be recomputed.

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

        Returns
        -------
        None
        """
        ed_ = [None] * len(self.images) if irradiance is None else irradiance
        vcg_ = [1] * len(self.images) if vc_g is None else vc_g

        [
            img.reflectance(
                irradiance=ed_[i],
                vc_g=vcg_[i],
                force_recompute=force_recompute,
                which_dc=which_dc,
            )
            for i, img in enumerate(self.images)
        ]

    def compute_undistorted_reflectance(
        self,
        irradiance: Optional[List[float]] = None,
        vc_g: Optional[List[float]] = None,
        force_recompute: bool = True,
        which_dc: str = "dark",
    ) -> None:
        """
        Compute undistorted image reflectance from irradiance list.

        Parameters
        ----------
        irradiance: List[float] or None
            A list of downwelling solar irradiance values either measured by the DLS2 (see
            Capture.dls_irradiance() or Capture.panel_irradiance()) or an independent
            irradiance sensor.

        vc_g : List[float] or None
            A list of vicarious calibration gains

        force_recompute: bool
           Specifies whether reflectance is to be recomputed.

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

        Returns
        -------
        None
        """
        ed_ = [None] * len(self.images) if irradiance is None else irradiance
        vcg_ = [1] * len(self.images) if vc_g is None else vc_g

        [
            img.undistorted_reflectance(
                irradiance=ed_[i],
                vc_g=vcg_[i],
                force_recompute=force_recompute,
                which_dc=which_dc,
            )
            for i, img in enumerate(self.images)
        ]

    def reflectance(
        self,
        irradiance: List[float],
        vc_g: List[float],
        which_dc: str = "dark",
    ) -> List[np.ndarray]:
        """
        Compute reflectance Images.

        Parameters
        ----------
        irradiance : List[float]
            A list of downwelling solar irradiance values either measured by the DLS2 (see
            Capture.dls_irradiance() or Capture.panel_irradiance()) or an independent
            irradiance sensor.

        vc_g : List[float]
            A list of vicarious calibration gains

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

        Returns
        -------
        List of reflectance EO and long wave infrared Images for given irradiance.
        """
        eo_imgs = [
            img.reflectance(irradiance=irradiance[i], vc_g=vc_g[i], which_dc=which_dc)
            for i, img in enumerate(self.eo_images())
        ]
        lw_imgs = [  # Note that `lw_imgs` are radiance images
            img.reflectance(vc_g=vc_g[i], which_dc=which_dc)
            for i, img in enumerate(self.lw_images())
        ]

        return eo_imgs + lw_imgs  # append `lw_imgs` to `eo_imgs`

    def undistorted_reflectance(
        self,
        irradiance: List[float],
        vc_g: List[float],
        which_dc: str = "dark",
    ) -> List[np.ndarray]:
        """
        Compute undistorted reflectance Images.

        Parameters
        ----------
        irradiance: List[float] or None
            A list of downwelling solar irradiance values either measured by the DLS2 (see
            Capture.dls_irradiance() or Capture.panel_irradiance()) or an independent
            irradiance sensor.

        vc_g : List[float] or None
            A list of vicarious calibration gains

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

        Returns
        -------
        List of undistorted reflectance images for given irradiance.
        """
        eo_imgs = [
            img.undistorted(
                img.reflectance(irradiance=irradiance[i], vc_g=vc_g[i], which_dc=which_dc)
            )
            for i, img in enumerate(self.eo_images())
        ]

        lw_imgs = [  # Note that `lw_imgs` are radiance images
            img.undistorted(img.reflectance(vc_g=vc_g[i], which_dc=which_dc))
            for i, img in enumerate(self.lw_images())
        ]
        return eo_imgs + lw_imgs  # append `lw_imgs` to `eo_imgs`

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
        self.__plot([p.plot_image() for p in self.panels], title="Panels", cbar=False)

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
