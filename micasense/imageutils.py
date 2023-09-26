#!/usr/bin/env python3
# coding: utf-8
"""
Misc. image processing utilities

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
import rasterio
import numpy as np
import numexpr as ne
import multiprocessing

from pathlib import Path
from imageio import imwrite
from warnings import filterwarnings
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank, gaussian
from typing import Union, List, Tuple, Optional

from micasense.tags import add_exif
from micasense.capture import Capture
from micasense.load_yaml import load_all
from micasense.plotutils import plotwithcolorbar, plot_overlay_withcolorbar


AVAIL_COMP = ["jpeg", "lzw", "packbits", "deflate", "webp", "none"]
EXIF_COMP = {
    "jpeg": 7,
    "lzw": 5,
    "packbits": 32773,
    "deflate": 32946,
    "webp": 34927,
    "none": 1,
}


def closest_ix(arr: np.ndarray, v: float) -> int:
    return int(abs(arr - v).argmin())


def normalize(im, min=None, max=None):
    width, height = im.shape
    norm = np.zeros((width, height), dtype="float32")
    if min is not None and max is not None:
        norm = (im - min) / (max - min)
    else:
        cv2.normalize(
            im,
            dst=norm,
            alpha=0.0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    norm[norm < 0.0] = 0.0
    norm[norm > 1.0] = 1.0
    return norm


def local_normalize(im):
    # TODO: mainly using this as a type conversion, but it's expensive
    norm = img_as_ubyte(normalize(im))
    width, _ = im.shape
    disksize = int(width / 5)
    if disksize % 2 == 0:
        disksize = disksize + 1
    selem = disk(disksize)
    # norm2 = rank.equalize(norm, selem=selem)  # `selem` is a deprecated argument
    norm2 = rank.equalize(norm, footprint=selem)
    return norm2


def gradient(im, ksize=5):
    im = local_normalize(im)
    # im = normalize(im)
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=ksize)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def compute_nonlinear_index(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Compute a non-linear index having the formulation of:
    index = (b1 - b2) / (b1 + b2)

    Parameters
    ----------
    b1 : np.ndarray {dims=(nrows, ncols)}
        band 1
    b2 : np.ndarray {dims=(nrows, ncols)}
        band 2

    Returns
    -------
    index : np.ndarray {dims=(nrows, ncols)}
        index
    """
    return ne.evaluate("(b1 - b2) / (b1 + b2)")


def get_ndwi(
    ms_capture: Capture,
    im_aligned: np.ndarray,
    green_wvl: Union[float, int] = 560,
    nir_wvl: Union[float, int] = 842,
    nodata: float = -9999.0,
) -> np.ndarray:
    """
    Compute the NDWI (normalised difference water index)
    NDWI = (green - nir) / (green + nir)

    see: https://eos.com/make-an-analysis/ndwi/

    Parameters
    ----------
    ms_capture : Capture
        Capture object (a set of micasense.image.Images) taken by a single or
        a pair (e.g. dual camera) of Micasense camera(s), which share the same
        unique identifier (capture id).
    im_aligned : np.ndarray, (dtype=float32)
        The stack of aligned (or unaligned) images.
    green_wvl : float, int
        wavelength of green band (default=560 nm)
    nir_wvl : np.ndarray, dims=(nrows, ncols)
        wavelength of NIR band (default=842 nm)
    nodata : float
        Nodata value in the NDWI product (default=-9999.0)

    Returns
    -------
    ndwi : np.ndarray {dims=(nrows, ncols), dtype=float64}
        Normalised difference water index

    """
    wavels = np.array(ms_capture.center_wavelengths())
    nir_ix = closest_ix(wavels, nir_wvl)
    grn_ix = closest_ix(wavels, green_wvl)

    ndwi = compute_nonlinear_index(b1=im_aligned[grn_ix], b2=im_aligned[nir_ix])
    ndwi[(im_aligned[nir_ix] == 0) & (im_aligned[grn_ix] == 0)] = nodata
    return np.array(ndwi, order="C", dtype="float64")


def get_ndvi(
    ms_capture: Capture,
    im_aligned: np.ndarray,
    red_wvl: Union[float, int] = 668,
    nir_wvl: Union[float, int] = 842,
    nodata: float = -9999.0,
) -> np.ndarray:
    """
    Compute the NDVI (normalised difference vegetation index)

    https://eos.com/make-an-analysis/ndvi/

    Parameters
    ----------
    ms_capture : Capture
        Capture object (a set of micasense.image.Images) taken by a single or
        a pair (e.g. dual camera) of Micasense camera(s), which share the same
        unique identifier (capture id).
    im_aligned : np.ndarray, (dtype=float32)
        The stack of aligned (or unaligned) images.
    red_wvl : float, int
        wavelength of green band (default=668 nm)
    nir_wvl : np.ndarray, dims=(nrows, ncols)
        wavelength of NIR band (default=842 nm)
    nodata : float
        Nodata value in the NDWI product (default=-9999.0)


    Returns
    -------
    ndvi : np.ndarray {dims=(nrows, ncols), dtype=float64}
        Normalised difference vegetation index

    """
    wavels = np.array(ms_capture.center_wavelengths())
    nir_ix = closest_ix(wavels, nir_wvl)
    red_ix = closest_ix(wavels, red_wvl)

    ndvi = compute_nonlinear_index(b1=im_aligned[nir_ix], b2=im_aligned[red_ix])
    ndvi[(im_aligned[nir_ix] == 0) & (im_aligned[red_ix] == 0)] = nodata
    return np.array(ndvi, order="C", dtype="float64")


def relatives_ref_band(ms_capture: Capture) -> int:
    for img in ms_capture.images:
        if img.rig_xy_offset_in_px() == (0, 0):
            return img.band_index()
    return 0


def translation_from_ref(ms_capture: Capture, band, ref=4) -> None:
    x, y = ms_capture.images[band].rig_xy_offset_in_px()
    rx, ry = ms_capture.images[ref].rig_xy_offset_in_px()
    return None


def align(pair: dict) -> dict:
    """Determine an alignment matrix between two images
    @input:
    Dictionary of the following form:
    {
        'warp_mode':  cv2.MOTION_* (MOTION_AFFINE, MOTION_HOMOGRAPHY)
        'max_iterations': Maximum number of solver iterations
        'epsilon_threshold': Solver stopping threshold
        'ref_index': index of reference image
        'match_index': index of image to match to reference
    }
    @returns:
    Dictionary of the following form:
    {
        'ref_index': index of reference image
        'match_index': index of image to match to reference
        'warp_matrix': transformation matrix to use to map match
                       image to reference image frame
    }

    Major props to Alexander Reynolds,
    https://stackoverflow.com/users/5087436/alexander-reynolds, for his
    insight into the pyramided matching process found at
    https://stackoverflow.com/questions/45997891/
    cv2-motion-euclidean-for-the-warp-mode-in-ecc-image-alignment-method

    """
    warp_mode = pair["warp_mode"]
    max_iterations = pair["max_iterations"]
    epsilon_threshold = pair["epsilon_threshold"]
    ref_index = pair["ref_index"]
    match_index = pair["match_index"]
    translations = pair["translations"]

    # Initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # warp_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float32)
        warp_matrix = pair["warp_matrix_init"]
    else:
        # warp_matrix = np.array([[1,0,0],[0,1,0]], dtype=float32)
        warp_matrix = np.array(
            [[1, 0, translations[1]], [0, 1, translations[0]]], dtype="float32"
        )

    w = pair["ref_image"].shape[1]

    if pair["pyramid_levels"] is None:
        nol = int(w / (1280 / 3)) - 1
    else:
        nol = pair["pyramid_levels"]

    if pair["debug"]:
        print("number of pyramid levels: {}".format(nol))

    warp_matrix[0][2] /= 2**nol
    warp_matrix[1][2] /= 2**nol

    if ref_index != match_index:

        show_debug_images = pair["debug"]
        # construct grayscale pyramid
        gray1 = pair["ref_image"]
        gray2 = pair["match_image"]
        if gray2.shape[0] < gray1.shape[0]:
            cv2.resize(
                gray2,
                None,
                fx=gray1.shape[0] / gray2.shape[0],
                fy=gray1.shape[0] / gray2.shape[0],
                interpolation=cv2.INTER_AREA,
            )
        gray1_pyr = [gray1]
        gray2_pyr = [gray2]

        for level in range(nol):
            gray1_pyr[0] = gaussian(normalize(gray1_pyr[0]))
            gray1_pyr.insert(
                0,
                cv2.resize(
                    gray1_pyr[0], None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA
                ),
            )
            gray2_pyr[0] = gaussian(normalize(gray2_pyr[0]))
            gray2_pyr.insert(
                0,
                cv2.resize(
                    gray2_pyr[0], None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA
                ),
            )

        # Terminate the optimizer if either the max
        # iterations or the threshold are reached
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iterations,
            epsilon_threshold,
        )

        # run pyramid ECC.  Here, we estimate the warp_matrix from low-res
        # imagery to it's native resolution. Note the warp matrix from the
        # lower res are passed to the higher res.  This approach is useful
        # when misaligments are large. It should be noted that "the motion
        # model is not adequate when there is local motion in the images (
        # e.g. the subject has moved a bit in the two images). An additional
        # local alignment needs to be done using say an optical flow based
        # approach" https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/
        for level in range(nol + 1):
            grad1 = gradient(gray1_pyr[level])
            grad2 = gradient(gray2_pyr[level])

            if show_debug_images:

                plotwithcolorbar(gray1_pyr[level], "ref level {}".format(level))
                plotwithcolorbar(gray2_pyr[level], "match level {}".format(level))
                plotwithcolorbar(grad1, "ref grad level {}".format(level))
                plotwithcolorbar(grad2, "match grad level {}".format(level))
                print("Starting warp for level {} is:\n {}".format(level, warp_matrix))

            try:
                cc, warp_matrix = cv2.findTransformECC(
                    grad1,
                    grad2,
                    warp_matrix,
                    warp_mode,
                    criteria,
                    inputMask=None,
                    gaussFiltSize=1,
                )
            except TypeError:
                cc, warp_matrix = cv2.findTransformECC(
                    grad1, grad2, warp_matrix, warp_mode, criteria
                )

            if show_debug_images:
                print("Warp after alignment level {} is \n{}".format(level, warp_matrix))

            if level != nol:
                # scale up only the offset by a factor of 2 for
                # the next (larger image) pyramid level
                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    warp_matrix = warp_matrix * np.array(
                        [[1, 1, 2], [1, 1, 2], [0.5, 0.5, 1]], dtype="float32"
                    )
                else:
                    warp_matrix = warp_matrix * np.array(
                        [[1, 1, 2], [1, 1, 2]], dtype="float32"
                    )

    return {
        "ref_index": pair["ref_index"],
        "match_index": pair["match_index"],
        "warp_matrix": warp_matrix,
    }


def default_warp_matrix(warp_mode):
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    else:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")


def refine_alignment_warp(
    ms_capture: Capture,
    ref_index: int = 4,
    warp_mode: int = cv2.MOTION_HOMOGRAPHY,
    max_iterations: int = 2500,
    epsilon_threshold: float = 1e-9,
    multithreaded: bool = True,
    debug: bool = False,
    pyramid_levels: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Extract the alignment warp matrices and alignment pairs in capture using openCV

    Parameters
    ----------
    ms_capture : Capture
        Capture object (a set of micasense.image.Images) taken by a single or
        a pair (e.g. dual camera) of Micasense camera(s), which share the same
        unique identifier (capture id).
    ref_index : int [Optional]
        For RedEdge-MX, ref_index=1 is best for alignment
        For Dual Camera, ref_index=4 is best for alignment
    warp_mode : int [Optional]
        cv2.MOTION_TRANSLATION, 0
            sets a translational motion model; warp_matrix is 2x3 with the
            first 2x2 part being the unity matrix and the rest two parame-
            ters being estimated.
        cv2.MOTION_EUCLIDEAN, 1
            sets a Euclidean (rigid) transformation as motion model; three
            parameters are estimated; `warp_matrix` is 2x3.
        cv2.MOTION_AFFINE, 2
            sets an affine motion model (DEFAULT); six parameters are estim-
            ated; `warp_matrix` is 2x3.
        cv2.MOTION_HOMOGRAPHY, 3
            sets a homography as a motion model; eight parameters are estim-
            ated; `warp_matrix` is 3x3.
        Note: best results will be AFFINE and HOMOGRAPHY, at the expense of speed
    max_iterations : int [default = 2500]
        Maximum iterations
    epsilon_threshold : float [default = 1.0e-9]
        Epsilon threshold
    debug : bool [default = False]
        Debug mode
    pyramid_levels : int or None [default = None]
        Pyramid level
    """
    # Match other bands to this reference image (index into ms_capture.images[])
    ref_img = (
        ms_capture.images[ref_index]
        .undistorted(ms_capture.images[ref_index].radiance())
        .astype("float32")
    )

    if ms_capture.has_rig_relatives():
        warp_matrices_init = ms_capture.get_warp_matrices(ref_index=ref_index)
    else:
        warp_matrices_init = [default_warp_matrix(warp_mode)] * len(ms_capture.images)

    alignment_pairs = []  # this will be a list of dict (not used by LWIR)
    for i, img in enumerate(ms_capture.images):
        if img.rig_relatives is not None:
            translations = img.rig_xy_offset_in_px()
        else:
            translations = (0, 0)
        if img.band_name != "LWIR":
            # alignment dict for LWIR appended at the end
            alignment_pairs.append(
                {
                    "warp_mode": warp_mode,
                    "max_iterations": max_iterations,
                    "epsilon_threshold": epsilon_threshold,
                    "ref_index": ref_index,
                    "ref_image": ref_img,
                    "match_index": i,
                    "match_image": img.undistorted(img.radiance()).astype("float32"),
                    "translations": translations,
                    "warp_matrix_init": np.array(warp_matrices_init[i], dtype="float32"),
                    "debug": debug,
                    "pyramid_levels": pyramid_levels,
                }
            )
    # warp_matrices creates a list of None with nelem = len(alignment_pairs)
    warp_matrices = [None] * len(alignment_pairs)

    # required to work across linux/mac/windows,
    # see https://stackoverflow.com/questions/47852237
    if multithreaded and multiprocessing.get_start_method() != "spawn":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except ValueError:
            multithreaded = False

    if multithreaded:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for _, mat in enumerate(pool.imap_unordered(align, alignment_pairs)):
            warp_matrices[mat["match_index"]] = mat["warp_matrix"]
            print("Finished aligning band {}".format(mat["match_index"]))
        pool.close()
        pool.join()
    else:
        # Single-threaded alternative
        for pair in alignment_pairs:
            # align() optimizes/refines the warp matrices
            mat = align(pair)  # mat is a dict
            warp_matrices[mat["match_index"]] = mat["warp_matrix"]
            print("Finished aligning band {}".format(mat["match_index"]))

    if ms_capture.images[-1].band_name == "LWIR":
        img = ms_capture.images[-1]
        alignment_pairs.append(
            {
                "warp_mode": warp_mode,
                "max_iterations": max_iterations,
                "epsilon_threshold": epsilon_threshold,
                "ref_index": ref_index,
                "ref_image": ref_img,
                "match_index": img.band_index,
                "match_image": img.undistorted(img.radiance()).astype("float32"),
                "translations": translations,
                "debug": debug,
            }
        )
        warp_matrices.append(ms_capture.get_warp_matrices(ref_index)[-1])
    return warp_matrices, alignment_pairs


def load_warp_matrices(warp_file: Path) -> List[np.ndarray]:
    """load the warp matrices and return a list of np.ndarray's"""
    tmp_wm = np.load(warp_file)
    # convert to a list of np.ndarray's
    warp_matrices = [tmp_wm[i, :, :] for i in range(tmp_wm.shape[0])]
    del tmp_wm

    return warp_matrices


def warp_matrices_wrapper(
    warp_npy_file: Path,
    uav_yaml_file: Path,
    match_index: int,
    max_align_iter: int,
    warp_mode: int,
    pyramid_levels: int,
) -> List[np.ndarray]:
    """wrapper to create/save or load the warp matrices"""
    if not warp_npy_file.exists():
        wrp_capture = Capture.from_yaml(uav_yaml_file)
        warp_matrices, _ = refine_alignment_warp(
            ms_capture=wrp_capture,
            ref_index=match_index,
            max_iterations=max_align_iter,
            warp_mode=warp_mode,
            pyramid_levels=pyramid_levels,
        )
        np.save(warp_npy_file, np.array(warp_matrices, order="C", dtype="float64"))

    else:
        warp_matrices = load_warp_matrices(warp_npy_file)

    return warp_matrices


# apply homography to create an aligned stack
def aligned_capture_backend(
    ms_capture: Capture,
    warp_matrices: List[np.ndarray],
    warp_mode: int = cv2.MOTION_HOMOGRAPHY,
    valid_ix: Optional[List[int]] = None,
    img_type: str = "reflectance",
    interpolation_mode: int = cv2.INTER_LANCZOS4,
    crop_edges: bool = True,
) -> np.ndarray:
    width, height = ms_capture.images[0].size()

    im_aligned = np.zeros((height, width, len(warp_matrices)), dtype="float32")

    for i in range(0, len(warp_matrices)):

        # the undistored radiance or reflectance has been calculated in
        # `aligned_capture()`, thus enforce force_recompute=False
        if img_type == "reflectance":
            img = ms_capture.images[i].undistorted_reflectance(force_recompute=False)
        else:
            img = ms_capture.images[i].undistorted_radiance(force_recompute=False)

        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            im_aligned[:, :, i] = cv2.warpAffine(
                img,  # ms_capture.images[i]
                warp_matrices[i],
                (width, height),
                flags=interpolation_mode + cv2.WARP_INVERSE_MAP,
            )
        else:
            im_aligned[:, :, i] = cv2.warpPerspective(
                img,  # ms_capture.images[i]
                warp_matrices[i],
                (width, height),
                flags=interpolation_mode + cv2.WARP_INVERSE_MAP,
            )

    s_cix, s_rix, e_cix, e_rix = valid_ix
    if not crop_edges:
        # flag
        im_aligned[:, 0 : s_cix + 1, :] = -1
        im_aligned[:, e_cix:, :] = -1

        im_aligned[0 : s_rix + 1, :, :] = -1
        im_aligned[e_rix:, :, :] = -1

        return im_aligned
    else:
        # crop the edges
        return im_aligned[s_rix : e_rix + 1, s_cix : e_cix + 1, :]


def aligned_capture(
    ms_capture: Capture,
    warp_matrices: Optional[Union[List[float], List[np.ndarray]]] = None,
    img_type: Optional[str] = None,
    warp_mode: int = cv2.MOTION_HOMOGRAPHY,
    crop_edges: bool = True,
    irradiance: Optional[List[float]] = None,
    use_darkpixels: bool = True,
) -> np.ndarray:
    """
    Creates aligned Capture. Computes undistorted radiance
    or reflectance images if necessary.

    Parameters
    ----------
    ms_capture : Capture
        .
    warp_matrices : List[np.ndarray]
        List of warp matrices derived from Capture.get_warp_matrices()
    img_type : str
        'radiance' or 'reflectance' depending on image metadata.
    warp_mode : int
        Also known as warp_mode. MOTION_HOMOGRAPHY or MOTION_AFFINE.
        For Altum images only use HOMOGRAPHY.
    crop_edges : bool
        .
    irradiance : List[float] or None
        Irradiance spectrum (band-ordered not wavelength-ordered) or None
    use_darkpixels : bool
        Whether to use the `dark_pixels` (True) or `black_level` (False).
        Note:
        `black_level` has a temporally constant value of 4800 across all bands.
        This is unrealistic as the dark current increases with sensor temperature.
        `dark_pixels` the averaged DN of the optically covered pixel values. This
        value is different for each band and varies across an acquisition, presu-
        mably from increases in temperature.


    Returns
    -------
    np.ndarray with alignment changes
    """
    if not img_type and irradiance is None:
        img_type = "radiance" if ms_capture.dls_irradiance() is None else "reflectance"

    # compute the radiance or reflectance
    if img_type == "radiance":
        ms_capture.compute_undistorted_radiance(
            use_darkpixels=use_darkpixels, force_recompute=True
        )  # radiance is stored in ms_capture.image.__radiance_image

    elif img_type == "reflectance":
        if irradiance is None:
            # why is [0] appended to the dls irradiance??
            irradiance = ms_capture.dls_irradiance() + [0]

        ms_capture.compute_undistorted_reflectance(
            irradiance=irradiance, use_darkpixels=use_darkpixels, force_recompute=True
        )  # reflectance is stored in ms_capture.image.__reflectance_image
    else:
        raise ValueError("`img_type` must either be 'radiance', 'reflectance' or None")

    if warp_matrices is None:
        warp_matrices = ms_capture.get_warp_matrices()

    valid_ix, _ = find_crop_bounds(
        ms_capture=ms_capture, registration_transforms=warp_matrices, warp_mode=warp_mode
    )

    im_aligned = aligned_capture_backend(
        ms_capture=ms_capture,
        warp_matrices=warp_matrices,
        warp_mode=warp_mode,
        valid_ix=valid_ix,
        img_type=img_type,
        interpolation_mode=cv2.INTER_LANCZOS4,
        crop_edges=crop_edges,
    )

    return im_aligned


class BoundPoint(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "(%f, %f)" % (self.x, self.y)

    def __repr__(self):
        return self.__str__()


class Bounds(object):
    def __init__(self):
        arbitrary_large_value = 100000000
        self.max = BoundPoint(-arbitrary_large_value, -arbitrary_large_value)
        self.min = BoundPoint(arbitrary_large_value, arbitrary_large_value)

    def __str__(self):
        return "Bounds min: %s, max: %s" % (str(self.min), str(self.max))

    def __repr__(self):
        return self.__str__()


def find_crop_bounds(
    ms_capture: Capture,
    registration_transforms: Union[List[float], List[np.ndarray]],
    warp_mode: int = cv2.MOTION_HOMOGRAPHY,
) -> Tuple[List[int], List[float]]:
    """
    Compute the crop rectangle to be applied to a set of images after
    registration such  that no pixel in the resulting stack of images
    will include a blank value for any of the bands

    Parameters
    ----------
    ms_capture : Capture
        Capture object (a set of micasense.image.Images) taken by a single or
        a pair (e.g. dual camera) of Micasense camera(s), which share the same
        unique identifier (capture id).
    registration_transforms : List[float] or List[np.ndarray]
        A list of affine transforms applied to register the image.
    warp_mode : int [Optional]
        cv2.MOTION_TRANSLATION, 0
            sets a translational motion model; warp_matrix is 2x3 with the
            first 2x2 part being the unity matrix and the rest two parame-
            ters being estimated.
        cv2.MOTION_EUCLIDEAN, 1
            sets a Euclidean (rigid) transformation as motion model; three
            parameters are estimated; `warp_matrix` is 2x3.
        cv2.MOTION_AFFINE, 2
            sets an affine motion model (DEFAULT); six parameters are estim-
            ated; `warp_matrix` is 2x3.
        cv2.MOTION_HOMOGRAPHY, 3
            sets a homography as a motion model; eight parameters are estim-
            ated; `warp_matrix` is 3x3.

    Returns
    -------
    valid_ix : List[int]
        valid_ix the smallest overlapping rectangle
        start_cix, start_rix, end_cix, end_rix = valid_ix
    edges : List[float]
        The mapped edges of the images
    """
    image_sizes = [image.size() for image in ms_capture.images]
    lens_distortions = [image.cv2_distortion_coeff() for image in ms_capture.images]
    camera_matrices = [image.cv2_camera_matrix() for image in ms_capture.images]

    bounds = [
        get_inner_rect(s, a, d, c, warp_mode=warp_mode)[0]
        for s, a, d, c in zip(
            image_sizes, registration_transforms, lens_distortions, camera_matrices
        )
    ]
    edges = [
        get_inner_rect(s, a, d, c, warp_mode=warp_mode)[1]
        for s, a, d, c in zip(
            image_sizes, registration_transforms, lens_distortions, camera_matrices
        )
    ]
    combined_bounds = get_combined_bounds(bounds, image_sizes[0])

    buff = 2  # increase cropping by `buff` to be on the safe side
    start_cix = int(np.ceil(combined_bounds.min.x)) + buff
    start_rix = int(np.ceil(combined_bounds.min.y)) + buff
    end_cix = int(np.floor(combined_bounds.max.x)) - buff
    end_rix = int(np.floor(combined_bounds.max.y)) - buff

    # width = np.floor(combined_bounds.max.x - combined_bounds.min.x)
    # height = np.floor(combined_bounds.max.y - combined_bounds.min.y)
    # return [left, top, width, height], edges

    return [start_cix, start_rix, end_cix, end_rix], edges


def get_inner_rect(
    image_size,
    affine,
    distortion_coeffs,
    camera_matrix,
    warp_mode=cv2.MOTION_HOMOGRAPHY,
):
    w = image_size[0]
    h = image_size[1]

    left_edge = np.array([np.ones(h) * 0, np.arange(0, h)]).T
    right_edge = np.array([np.ones(h) * (w - 1), np.arange(0, h)]).T
    top_edge = np.array([np.arange(0, w), np.ones(w) * 0]).T
    bottom_edge = np.array([np.arange(0, w), np.ones(w) * (h - 1)]).T

    left_map = map_points(
        left_edge,
        image_size,
        affine,
        distortion_coeffs,
        camera_matrix,
        warp_mode=warp_mode,
    )
    left_bounds = min_max(left_map)
    right_map = map_points(
        right_edge,
        image_size,
        affine,
        distortion_coeffs,
        camera_matrix,
        warp_mode=warp_mode,
    )
    right_bounds = min_max(right_map)
    top_map = map_points(
        top_edge,
        image_size,
        affine,
        distortion_coeffs,
        camera_matrix,
        warp_mode=warp_mode,
    )
    top_bounds = min_max(top_map)
    bottom_map = map_points(
        bottom_edge,
        image_size,
        affine,
        distortion_coeffs,
        camera_matrix,
        warp_mode=warp_mode,
    )
    bottom_bounds = min_max(bottom_map)

    bounds = Bounds()
    bounds.max.x = right_bounds.min.x
    bounds.max.y = bottom_bounds.min.y
    bounds.min.x = left_bounds.max.x
    bounds.min.y = top_bounds.max.y
    edges = (left_map, right_map, top_map, bottom_map)
    return bounds, edges


def get_combined_bounds(bounds, image_size):
    w = image_size[0]
    h = image_size[1]

    final = Bounds()

    final.min.x = final.min.y = 0
    final.max.x = w
    final.max.y = h

    for b in bounds:
        final.min.x = max(final.min.x, b.min.x)
        final.min.y = max(final.min.y, b.min.y)
        final.max.x = min(final.max.x, b.max.x)
        final.max.y = min(final.max.y, b.max.y)

    # limit to image size
    final.min.x = max(final.min.x, 0)
    final.min.y = max(final.min.y, 0)
    final.max.x = min(final.max.x, w - 1)
    final.max.y = min(final.max.y, h - 1)
    # Add 1 px of margin (remove one pixel on all sides)
    final.min.x += 1
    final.min.y += 1
    final.max.x -= 1
    final.max.y -= 1

    return final


def min_max(pts):
    bounds = Bounds()
    for p in pts:
        if p[0] > bounds.max.x:
            bounds.max.x = p[0]
        if p[1] > bounds.max.y:
            bounds.max.y = p[1]
        if p[0] < bounds.min.x:
            bounds.min.x = p[0]
        if p[1] < bounds.min.y:
            bounds.min.y = p[1]
    return bounds


def map_points(
    pts,
    image_size,
    warp_matrix,
    distortion_coeffs,
    camera_matrix,
    warp_mode=cv2.MOTION_HOMOGRAPHY,
):
    # extra dimension makes opencv happy
    pts = np.array([pts], dtype="float64")
    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coeffs, image_size, 1
    )
    new_pts = cv2.undistortPoints(pts, camera_matrix, distortion_coeffs, P=new_cam_mat)
    if warp_mode == cv2.MOTION_AFFINE:
        new_pts = cv2.transform(new_pts, cv2.invertAffineTransform(warp_matrix))
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        new_pts = cv2.perspectiveTransform(
            new_pts, np.linalg.inv(warp_matrix).astype("float32")
        )
    # apparently the output order has changed in 4.1.1 (possibly earlier from 3.4.3)
    if cv2.__version__ <= "3.4.4":
        return new_pts[0]
    else:
        return new_pts[:, 0, :]


def save_capture_as_stack(
    ms_capture: Capture,
    im_aligned: np.ndarray,
    out_filename: Union[str, Path],
    img_type: str = "reflectance",
    photometric: str = "MINISBLACK",
    compression: str = "lzw",
    odtype: str = "uint16",
    yml_fn: Optional[Path] = None,
    other_ds: Optional[dict] = None,
    other_wvl: Optional[np.ndarray] = None,
) -> None:
    filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    """
    Write a geotif (without a defined Affine and CRS projection)
    of a stack of aligned/unaligned images.

    Parameters
    ----------
    ms_capture : Capture
        Capture object (a set of micasense.image.Images) taken by a single or
        a pair (e.g. dual camera) of Micasense camera(s), which share the same
        unique identifier (capture id).
    im_aligned : np.ndarray, {dtype=float32, dims=(nrows, ncols, nbands)}
        The stack of aligned (or unaligned) images.
    out_filename : str or Path
        The output geotif filename
    photometric : str [Optional]
        GDAL argument (see https://gdal.org/drivers/raster/gtiff.html)
    compression : str [Optional]
        "jpeg", "lzw", "packbits", "deflate", "webp", "none"
        compression that are used by GDAL and EXIF
        * see https://gdal.org/drivers/raster/gtiff.html for information
          on the different compression algorithms. Note though, PACKBITS
          DEFLATE and LZW are lossless approaches. Default is lzw
        * see https://exiftool.org/TagNames/EXIF.html#Compression
    odtype : str
        output dtype, options include "uint16", "float32"

    yml_fn : Path [Optional]
        The image set metadata yaml. This is needed to write metadata into
        the exif tags. If not provided, then exif metadata will not be
        written to output tifs
    other_ds : Dict [Optional]
        A dictionary of additional dataset to be added as bands in the
        tif file, e.g.
        other_ds = {
            "vzen": {  uint16 = float_data * scale + offset
                "data": vzen,  # np.ndarray, float32/64
                "info": "view zenith angle",
                "scale": 728.1555555555556,  # float
                "offset": 1,  # float
            }
        }
    other_wvl : np.ndarray
        user-supplied wavelength (ascending order).
    """

    def raise_err(user_parm: str, pname: str, allowable: List[str]) -> None:
        if user_parm not in allowable:
            raise ValueError(f"specified {pname} ('{user_parm}') not in {allowable}")

    odtype = odtype.lower()
    avail_odt = ["uint16", "float32", "float64"]
    raise_err(odtype, "odtype", avail_odt)

    ocomp = compression.lower()
    raise_err(ocomp, "compression", AVAIL_COMP)

    vis_sfactor, thermal_sfactor, thermal_offset = 1.0, 1.0, 0.0
    np_odt = np.dtype(odtype)
    nodata = np_odt.type(-9999.0)
    if odtype == "uint16":
        vis_sfactor = np.iinfo(np.dtype(odtype)).max
        thermal_sfactor, thermal_offset = 100.0, 273.15
        nodata = np_odt.type(0)

    nrows, ncols, nbands = im_aligned.shape

    wavel = ms_capture.center_wavelengths()
    if other_wvl is not None:
        if not isinstance(other_wvl, (np.ndarray, list, tuple)):
            raise ValueError("`other_wavel` must be np.ndarray, list, tuple")
        if len(other_wvl) != len(wavel):
            raise ValueError(
                "`other_wvl` must have the same number of wavelengths as bands"
            )
        other_wvl = np.array(sorted(other_wvl), dtype="float64")

    else:
        other_wvl = np.array(sorted(wavel))
    eo_list = list(np.argsort(np.array(wavel)[ms_capture.eo_indices()]))

    # To conserve memory, the geotiff will be saved as uint16
    meta = {
        "driver": "GTiff",
        "dtype": odtype,
        "nodata": nodata,
        "width": ncols,
        "height": nrows,
        "count": nbands if not other_ds else nbands + len(other_ds),
        "compress": ocomp,
        "tiled": True,
        "blockxsize": int(ncols) // 5,
        "blockysize": int(nrows) // 5,
        "interleave": "band",
    }

    with rasterio.open(str(out_filename), "w", **meta) as dst:

        # iterate through the visible bands
        vis_wavel = ""
        for out_bix, in_bix in enumerate(eo_list):
            bandim = im_aligned[:, :, in_bix]

            # identify flagged pixels (<=0.0) if img_type == "reflectance"
            # then pixels with values > 1.0 will also be masked.
            if img_type == "reflectance":
                flagged_ix = (bandim <= 0) | (bandim > 1.0)
            else:
                flagged_ix = bandim <= 0

            bandim[flagged_ix] = nodata

            # convert bandim from float32 to uint16.
            z_idx = out_bix + 1
            dst.write(np.array(bandim * vis_sfactor, dtype=odtype), indexes=z_idx)

            dst.set_band_description(z_idx, f"B{z_idx}_R({int(other_wvl[out_bix])} nm)")
            vis_wavel += f"{other_wvl[out_bix]},"
            eval(f"dst.update_tags(B{z_idx}_scale={vis_sfactor})")
            eval(f"dst.update_tags(B{z_idx}_offset=0)")

        # iterate through the thermal bands
        for out_bix, in_bix in enumerate(ms_capture.lw_indices()):
            bandim = (im_aligned[:, :, in_bix] + thermal_offset) * thermal_sfactor
            bandim[bandim < 0] = nodata
            if odtype == "uint16":
                bandim[bandim > vis_sfactor] = vis_sfactor

            z_idx = len(eo_list) + out_bix + 1
            dst.write(bandim.asdtype(odtype), indexes=z_idx)
            dst.set_band_description(z_idx, f"LWIR {in_bix+1}")

        # append any additional datasets
        if other_ds:
            out_bix += 1
            for i, k in enumerate(other_ds):
                z_idx = out_bix + i + 1
                # perform datatype conversion using the scale and offset values
                scale = other_ds[k]["scale"]
                offset = other_ds[k]["offset"]

                scaled_d = other_ds[k]["data"] * scale + offset
                scaled_d[scaled_d > vis_sfactor] = vis_sfactor

                dst.write(scaled_d.astype(odtype), indexes=z_idx)
                dst.set_band_description(z_idx, f"B{z_idx}_{other_ds[k]['info']}")
                eval(f"dst.update_tags(B{z_idx}_scale=scale)")
                eval(f"dst.update_tags(B{z_idx}_offset=offset)")

        # NOTE: The following tags are written into the EXIF/XMP "GDAL Metadata"
        #       metadata tag. Annoyingly, these tags are not accessible with
        #       pyexiv2, but can be accessed with rasterio
        #       >>> with rasterio.open(tfile, "r") as src:
        #       >>>     custom_tags = src.tags()  # dict
        #       >>> solar_zenith = custom_tags["solarzenith"]  # float
        if vis_sfactor:
            dst.update_tags(vis_sfactor=vis_sfactor)

        if vis_wavel:
            if vis_wavel.endswith(","):
                vis_wavel = vis_wavel[0:-1]
            dst.update_tags(vis_wavel=vis_wavel)

        if thermal_sfactor:
            dst.update_tags(thermal_sfactor=thermal_sfactor)
        if thermal_offset:
            dst.update_tags(thermal_offset=thermal_offset)

        szen, sazi = ms_capture.solar_geoms()

        dst.update_tags(solarzenith=szen)
        dst.update_tags(solarazimuth=sazi)

    if not Path(out_filename).exists():
        raise Exception(f"issue with writing {out_filename}")

    if yml_fn:
        add_exif(
            acq_meta=load_all(yml_fn),
            tiff_fn=out_filename,
            compression=EXIF_COMP[ocomp],
            imshape=(nrows, ncols),
            image_pp=4,  # hope retrievals - hack for multilayered tiffs
            image_name=None,
            principal_point=None,
        )


def save_aligned_individual(
    ms_capture: Capture,
    im_aligned: np.ndarray,
    out_basename: str,
    photometric: str = "MINISBLACK",
    compression: str = "lzw",
    odtype: str = "uint16",
    image_pp: int = 1,
    flag_opt: int = 0,
    yml_fn: Optional[Path] = None,
) -> None:
    filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    """
    Write a geotif (without a defined Affine and CRS projection)
    for each band in `im_aligned`

    Parameters
    ----------
    ms_capture : Capture
        Capture object (a set of micasense.image.Images) taken by a single or
        a pair (e.g. dual camera) of Micasense camera(s), which share the same
        unique identifier (capture id).
    im_aligned : np.ndarray, {dtype=float32, dims=(nrows, ncols, nbands)}
        The stack of aligned (or unaligned) images.
    out_basename : str
        The output geotif basename, e.g. /path/to/output/IMG_ALIGNED_0251
    photometric : str [Optional]
        GDAL argument (see https://gdal.org/drivers/raster/gtiff.html)
    compression : str [Optional]
        "jpeg", "lzw", "packbits", "deflate", "webp", "none"
        compression that are used by GDAL and EXIF
        * see https://gdal.org/drivers/raster/gtiff.html for information
          on the different compression algorithms. Note though, PACKBITS
          DEFLATE and LZW are lossless approaches. Default is lzw
        * see https://exiftool.org/TagNames/EXIF.html#Compression
    odtype : str
        output dtype, options include "uint16", "float32"

    image_pp : int
        image preprocessing,
        1 = raw (vignetting + dark current)
        2 = undistorted (vignetting + dark current + undistorting)
        3 = aligned (vignetting + dark current + undistorting + alignment)

    flag_opt : int
        The type of flagging
        0 = flag pixels that have values <= 0 (bandim <= 0)
        1 = flag pixels with values <= 0 or > 1 ((bandim <= 0) | (bandim > 1.0))

    yml_fn : Path [Optional]
        The image set metadata yaml. This is needed to write metadata into
        the exif tags. If not provided, then exif metadata will not be
        written to output tifs
    """

    def raise_err(user_parm: str, pname: str, allowable: List[str]) -> None:
        if user_parm not in allowable:
            raise ValueError(f"specified {pname} ('{user_parm}') not in {allowable}")

    odtype = odtype.lower()
    avail_odt = ["uint16", "float32", "float64"]
    raise_err(odtype, "odtype", avail_odt)

    ocomp = compression.lower()
    raise_err(ocomp, "compression", AVAIL_COMP)

    vis_sfactor = 1.0
    np_odt = np.dtype(odtype)
    nodata = np_odt.type(-9999.0)
    if odtype == "uint16":
        vis_sfactor = np.iinfo(np.dtype(odtype)).max
        nodata = np_odt.type(0)

    nrows, ncols, nbands = im_aligned.shape

    # To conserve memory, the geotiff will be saved as uint16
    meta = {
        "driver": "GTiff",
        "dtype": odtype,
        "nodata": nodata,
        "width": ncols,
        "height": nrows,
        "count": 1,
        "compress": ocomp,
    }

    for i in range(nbands):
        img = ms_capture.images[i]

        # get output name
        bnum = img.path.stem.split("_")[-1]
        ofn = str(out_basename) + f"_{bnum}.tif"

        bandim = im_aligned[:, :, i]
        # mask flags according to `flag_opt`
        flagged_ix = (bandim <= 0) if flag_opt == 0 else (bandim <= 0) | (bandim > 1.0)
        bandim[flagged_ix] = nodata

        # avoid wrap-around when scaling from float to uint16
        bandim[bandim > 1.0] = 1.0

        # This scaling only works for reflectance
        with rasterio.open(ofn, "w", **meta) as dst:
            dst.write(np.array(bandim * vis_sfactor, order="C", dtype=odtype), 1)

        if yml_fn:
            add_exif(
                acq_meta=load_all(yml_fn),
                tiff_fn=ofn,
                compression=EXIF_COMP[ocomp],
                imshape=(nrows, ncols),
                image_pp=image_pp,
                image_name=img.path.name,
                principal_point=f"{img.newcammat_ppx},{img.newcammat_ppy}",
            )

    return


def save_capture_as_rgb(
    im_aligned: np.ndarray,
    out_filename: Union[str, Path],
    gamma: float = 1.4,
    downsample: int = 1,
    white_balance: str = "norm",
    hist_min_percent: float = 0.5,
    hist_max_percent: float = 99.5,
    sharpen: bool = True,
    rgb_band_indices: Tuple[int, int, int] = (2, 1, 0),
):
    """
    Output the Images in the Capture object as RGB.
    Parameters
    ----------
    out_filename: str system file path
    gamma: float gamma correction
    downsample: int downsample for cv2.resize()
    white_balance: str (default="norm")
        Specifies whether to normalize across bands using hist_min_percent
        and hist_max_percent. Else this parameter is ignored.
    hist_min_percent: float for min histogram stretch
    hist_max_percent: float for max histogram stretch
    sharpen: boolean
    rgb_band_indices: List band order
    """
    im_display = np.zeros(im_aligned.shape, dtype="float32")

    # modify these percentiles to adjust contrast.
    # For many images, 0.5 and 99.5 are good values
    im_min = np.percentile(im_aligned[:, :, rgb_band_indices].flatten(), hist_min_percent)
    im_max = np.percentile(im_aligned[:, :, rgb_band_indices].flatten(), hist_max_percent)

    for i in rgb_band_indices:
        # For rgb true color, we usually want to use the same min and max
        # scaling across the 3 bands to maintain the "white balance" of
        # the calibrated image
        if white_balance == "norm":
            im_display[:, :, i] = normalize(im_aligned[:, :, i], im_min, im_max)
        else:
            im_display[:, :, i] = normalize(im_aligned[:, :, i])

    rgb = im_display[:, :, rgb_band_indices]
    rgb = cv2.resize(
        rgb,
        None,
        fx=1 / downsample,
        fy=1 / downsample,
        interpolation=cv2.INTER_AREA,
    )

    if sharpen:
        gaussian_rgb = cv2.GaussianBlur(rgb, (9, 9), 10.0)
        gaussian_rgb[gaussian_rgb < 0] = 0
        gaussian_rgb[gaussian_rgb > 1] = 1
        unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
        unsharp_rgb[unsharp_rgb < 0] = 0
        unsharp_rgb[unsharp_rgb > 1] = 1
    else:
        unsharp_rgb = rgb

    # Apply a gamma correction to make the render appear
    # closer to what our eyes would see
    if gamma != 0:
        gamma_corr_rgb = unsharp_rgb ** (1.0 / gamma)
        imwrite(out_filename, (255 * gamma_corr_rgb).astype("uint8"))
    else:
        imwrite(out_filename, (255 * unsharp_rgb).astype("uint8"))


def save_thermal_over_rgb(
    ms_capture: Capture,
    im_aligned: np.ndarray,
    out_filename: Union[str, Path],
    fig_size: Tuple[int, int] = (30, 23),
    lw_index: Optional[int] = None,
    hist_min_percent: float = 0.2,
    hist_max_percent: float = 99.8,
):
    """
    Output the Images in the Capture object as thermal over RGB.
    :param out_filename: str system file path.
    :param fig_size: Tuple dimensions of the figure.
    :param lw_index: int Index of LWIR Image in Capture.
    :param hist_min_percent: float Minimum histogram percentile.
    :param hist_max_percent: float Maximum histogram percentile.
    """
    # by default we don't mask the thermal, since it's native
    # resolution is much lower than the MS
    if lw_index is None:
        lw_index = ms_capture.lw_indices()[0]
    masked_thermal = im_aligned[:, :, lw_index]

    im_display = np.zeros(
        (im_aligned.shape[0], im_aligned.shape[1], 3),
        dtype="float32",
    )
    rgb_band_indices = [
        ms_capture.band_names_lower().index("red"),
        ms_capture.band_names_lower().index("green"),
        ms_capture.band_names_lower().index("blue"),
    ]

    # for rgb true color, we usually want to use the same min and max
    # scaling across the 3 bands to maintain the "white balance" of
    # the calibrated image

    # modify these percentiles to adjust contrast
    im_min = np.percentile(im_aligned[:, :, rgb_band_indices].flatten(), hist_min_percent)

    # for many images, 0.5 and 99.5 are good values
    im_max = np.percentile(im_aligned[:, :, rgb_band_indices].flatten(), hist_max_percent)

    for dst_band, src_band in enumerate(rgb_band_indices):
        im_display[:, :, dst_band] = normalize(im_aligned[:, :, src_band], im_min, im_max)

    # Compute a histogram
    min_display_therm = np.percentile(masked_thermal, hist_min_percent)
    max_display_therm = np.percentile(masked_thermal, hist_max_percent)

    fig, _ = plot_overlay_withcolorbar(
        im_display,
        masked_thermal,
        figsize=fig_size,
        title="Temperature over True Color",
        vmin=min_display_therm,
        vmax=max_display_therm,
        overlay_alpha=0.25,
        overlay_colormap="jet",
        overlay_steps=16,
        display_contours=True,
        contour_steps=16,
        contour_alpha=0.4,
        contour_fmt="%.0fC",
        show=False,
    )
    fig.savefig(out_filename)
