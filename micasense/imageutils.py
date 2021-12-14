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
import imageio
import warnings
import rasterio
import numpy as np
import multiprocessing

from pathlib2 import Path
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank, gaussian
from typing import Union, List, Tuple, Optional


import micasense.capture as capture
import micasense.plotutils as plotutils


def normalize(im, min=None, max=None):
    width, height = im.shape
    norm = np.zeros((width, height), dtype=np.float32)
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


def relatives_ref_band(ms_capture: capture.Capture) -> int:
    for img in ms_capture.images:
        if img.rig_xy_offset_in_px() == (0, 0):
            return img.band_index()
    return 0


def translation_from_ref(ms_capture: capture.Capture, band, ref=4) -> None:
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
        # warp_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        warp_matrix = pair["warp_matrix_init"]
    else:
        # warp_matrix = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        warp_matrix = np.array(
            [[1, 0, translations[1]], [0, 1, translations[0]]], dtype=np.float32
        )

    w = pair["ref_image"].shape[1]

    if pair["pyramid_levels"] is None:
        nol = int(w / (1280 / 3)) - 1
    else:
        nol = pair["pyramid_levels"]

    if pair["debug"]:
        print("number of pyramid levels: {}".format(nol))

    warp_matrix[0][2] /= 2 ** nol
    warp_matrix[1][2] /= 2 ** nol

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
        # run pyramid ECC
        for level in range(nol + 1):
            grad1 = gradient(gray1_pyr[level])
            grad2 = gradient(gray2_pyr[level])

            if show_debug_images:

                plotutils.plotwithcolorbar(gray1_pyr[level], "ref level {}".format(level))
                plotutils.plotwithcolorbar(
                    gray2_pyr[level], "match level {}".format(level)
                )
                plotutils.plotwithcolorbar(grad1, "ref grad level {}".format(level))
                plotutils.plotwithcolorbar(grad2, "match grad level {}".format(level))
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
                        [[1, 1, 2], [1, 1, 2], [0.5, 0.5, 1]], dtype=np.float32
                    )
                else:
                    warp_matrix = warp_matrix * np.array(
                        [[1, 1, 2], [1, 1, 2]], dtype=np.float32
                    )

    return {
        "ref_index": pair["ref_index"],
        "match_index": pair["match_index"],
        "warp_matrix": warp_matrix,
    }


def default_warp_matrix(warp_mode):
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    else:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def refine_alignment_warp(
    ms_capture: capture.Capture,
    ref_index: Optional[int] = 4,
    warp_mode: Optional[int] = cv2.MOTION_HOMOGRAPHY,
    max_iterations: Optional[int] = 2500,
    epsilon_threshold: Optional[float] = 1e-9,
    multithreaded: Optional[bool] = True,
    debug: Optional[bool] = False,
    pyramid_levels: Optional[Union[int, None]] = None,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Extract the alignment warp matrices and alignment pairs in capture using openCV

    Parameters
    ----------
    ms_capture : capture.Capture
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
                    "warp_matrix_init": np.array(warp_matrices_init[i], dtype=np.float32),
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


# apply homography to create an aligned stack
def aligned_capture_backend(
    ms_capture: capture.Capture,
    warp_matrices: List[np.ndarray],
    warp_mode: Optional[int] = cv2.MOTION_HOMOGRAPHY,
    valid_ix: Optional[Union[List[int], None]] = None,
    img_type: Optional[str] = "reflectance",
    interpolation_mode: Optional[int] = cv2.INTER_LANCZOS4,
    crop_edges: Optional[bool] = True,
) -> np.ndarray:
    width, height = ms_capture.images[0].size()

    im_aligned = np.zeros((height, width, len(warp_matrices)), dtype=np.float32)

    for i in range(0, len(warp_matrices)):
        if img_type == "reflectance":
            img = ms_capture.images[i].undistorted_reflectance()
        else:
            img = ms_capture.images[i].undistorted_radiance()

        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            im_aligned[:, :, i] = cv2.warpAffine(
                img,
                warp_matrices[i],
                (width, height),
                flags=interpolation_mode + cv2.WARP_INVERSE_MAP,
            )
        else:
            im_aligned[:, :, i] = cv2.warpPerspective(
                img,
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
    ms_capture: capture.Capture,
    warp_matrices: Optional[Union[List[float], List[np.ndarray], None]] = None,
    img_type: Optional[Union[None, str]] = None,
    warp_mode: Optional[int] = cv2.MOTION_HOMOGRAPHY,
    irradiance_list: Optional[Union[List[float], None]] = None,
    crop_edges: Optional[bool] = True,
) -> np.ndarray:
    """
    Creates aligned Capture. Computes undistorted radiance
    or reflectance images if necessary.

    Parameters
    ----------
    irradiance_list: List of mean panel region irradiance.
    warp_matrices: 2d List of warp matrices derived from Capture.get_warp_matrices()
    img_type: str 'radiance' or 'reflectance' depending on image metadata.
    warp_mode: OpenCV import.
        Also known as warp_mode. MOTION_HOMOGRAPHY or MOTION_AFFINE.
        For Altum images only use HOMOGRAPHY.

    Returns
    -------
    np.ndarray with alignment changes
    """
    if (
        img_type is None
        and irradiance_list is None
        and ms_capture.dls_irradiance() is None
    ):
        ms_capture.compute_undistorted_radiance()
        img_type = "radiance"

    elif img_type is None:
        if irradiance_list is None:
            # why 0 is appended to the dls irradiance??
            irradiance_list = ms_capture.dls_irradiance() + [0]
        ms_capture.compute_undistorted_reflectance(irradiance_list)
        img_type = "reflectance"

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
    ms_capture: capture.Capture,
    registration_transforms: Union[List[float], List[np.ndarray]],
    warp_mode: Optional[int] = cv2.MOTION_HOMOGRAPHY,
) -> Tuple[List[int], List[float]]:
    """
    Compute the crop rectangle to be applied to a set of images after
    registration such  that no pixel in the resulting stack of images
    will include a blank value for any of the bands

    Parameters
    ----------
    ms_capture : capture.Capture
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

    start_cix = int(np.ceil(combined_bounds.min.x))
    start_rix = int(np.ceil(combined_bounds.min.y))
    end_cix = int(np.floor(combined_bounds.max.x))
    end_rix = int(np.floor(combined_bounds.max.y))

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
    pts = np.array([pts], dtype=np.float64)
    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coeffs, image_size, 1
    )
    new_pts = cv2.undistortPoints(pts, camera_matrix, distortion_coeffs, P=new_cam_mat)
    if warp_mode == cv2.MOTION_AFFINE:
        new_pts = cv2.transform(new_pts, cv2.invertAffineTransform(warp_matrix))
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        new_pts = cv2.perspectiveTransform(
            new_pts, np.linalg.inv(warp_matrix).astype(np.float32)
        )
    # apparently the output order has changed in 4.1.1 (possibly earlier from 3.4.3)
    if cv2.__version__ <= "3.4.4":
        return new_pts[0]
    else:
        return new_pts[:, 0, :]


def save_capture_as_stack(
    ms_capture: capture.Capture,
    im_aligned: np.ndarray,
    out_filename: Union[str, Path],
    img_type: Optional[str] = "reflectance",
    sort_by_wavelength: Optional[bool] = True,
    photometric: Optional[str] = "MINISBLACK",
    compression: Optional[str] = "lzw",
) -> None:
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    """
    Write a geotif (without a defined Affine and CRS projection)
    of a stack of aligned/unaligned images.

    Parameters
    ----------
    ms_capture : capture.Capture
        Capture object (a set of micasense.image.Images) taken by a single or
        a pair (e.g. dual camera) of Micasense camera(s), which share the same
        unique identifier (capture id).
    im_aligned : np.ndarray, (dtype=np.float32)
        The stack of aligned (or unaligned) images.
    out_filename : str or Path
        The output geotif filename
    sort_by_wavelength : bool [Optional]
        Specifies whether to save the image stack with ordered
        wavelength (ascending order), default = True
    photometric : str [Optional]
        GDAL argument (see https://gdal.org/drivers/raster/gtiff.html)
    compression : str [Optional]
        "jpeg", "lzw", "packbits", "deflate", "ccittrle", "ccittfax3",
         "ccittfax4", "lzma", "zstd", "lerc", "lerc_deflate", "lerc_zstd",
         "webp", "jxl", "none"
         see https://gdal.org/drivers/raster/gtiff.html for information
         on the different compression algorithms. Note though, PACKBITS
         DEFLATE and LZW are lossless approaches. Default is lzw
    """
    nrows, ncols, nbands = im_aligned.shape
    nodata = 0
    odtype = np.dtype(np.uint16)
    vis_sfactor = None
    thermal_sfactor, thermal_offset = None, None

    wavel = ms_capture.center_wavelengths()
    if sort_by_wavelength:
        eo_list = list(np.argsort(np.array(wavel)[ms_capture.eo_indices()]))
    else:
        eo_list = ms_capture.eo_indices()

    # To conserve memory, the geotiff will be saved as uint16
    meta = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": nodata,
        "width": ncols,
        "height": nrows,
        "count": nbands,
    }
    blockxsize = int(ncols) // 5
    blockysize = int(nrows) // 5

    with rasterio.open(
        str(out_filename),
        "w",
        **meta,
        compress=compression,
        tiled=True,
        blockxsize=blockxsize,
        blockysize=blockysize,
        interleave="band",  # equivalent to band-sequential interleave
    ) as dst:

        # iterate through the visible bands
        vis_wavel = ""
        for out_bix, in_bix in enumerate(eo_list):
            vis_sfactor = np.iinfo(odtype).max
            bandim = im_aligned[:, :, in_bix]

            # identify flagged pixels (<=0.0) if img_type == "reflectance"
            # then pixels with values > 1.0 will also be masked.
            if img_type == "reflectance":
                flagged_ix = (bandim <= 0) | (bandim > 1.0)
            else:
                flagged_ix = bandim <= 0

            bandim[flagged_ix] = 0.0

            # convert bandim from float32 to uint16.
            dst.write(
                np.array(bandim * vis_sfactor, order="C", dtype=odtype),
                indexes=out_bix + 1,
            )
            dst.set_band_description(out_bix + 1, f"{wavel[in_bix]} (Band{in_bix+1:02d})")
            vis_wavel += f"{wavel[in_bix]},"
        if vis_wavel.endswith(","):
            vis_wavel = vis_wavel[0:-1]

        # iterate through the thermal bands
        for out_bix, in_bix in enumerate(ms_capture.lw_indices()):
            thermal_sfactor = 100.0
            thermal_offset = 273.15
            bandim = (im_aligned[:, :, in_bix] + thermal_offset) * thermal_sfactor
            bandim[bandim < 0] = 0
            bandim[bandim > np.iinfo(odtype).max] = np.iinfo(odtype).max

            dst.write(bandim.asdtype(odtype), indexes=len(eo_list) + out_bix + 1)
            # dst.set_band_description(len(eo_list) + out_bix + 1, f"LWIR {out_bix+1}")

        # NOTE: The following tags are written into the EXIF/XMP
        #       "GDAL Metadata" metadata tag.  Annoyingly, these
        #       tags are not accessible with pyexiv2, but can be
        #       accessed with rasterio;
        #       >>> with rasterio.open(tfile, "r") as src:
        #       >>>     custom_tags = src.tags()  # dict
        #       >>> solar_zenith = custom_tags["solarzenith"]  # float
        if vis_sfactor:
            dst.update_tags(vis_sfactor=vis_sfactor)
        if vis_wavel:
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


def save_capture_as_rgb(
    im_aligned: np.ndarray,
    out_filename: Union[str, Path],
    gamma: Optional[float] = 1.4,
    downsample: Optional[int] = 1,
    white_balance: Optional[str] = "norm",
    hist_min_percent: Optional[float] = 0.5,
    hist_max_percent: Optional[float] = 99.5,
    sharpen: Optional[bool] = True,
    rgb_band_indices: Optional[Tuple[int]] = (2, 1, 0),
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
    im_display = np.zeros(im_aligned.shape, dtype=np.float32)

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
        imageio.imwrite(out_filename, (255 * gamma_corr_rgb).astype("uint8"))
    else:
        imageio.imwrite(out_filename, (255 * unsharp_rgb).astype("uint8"))


def save_thermal_over_rgb(
    ms_capture: capture.Capture,
    im_aligned: np.ndarray,
    out_filename: Union[str, Path],
    fig_size: Optional[Tuple[int]] = (30, 23),
    lw_index: Optional[int] = None,
    hist_min_percent: Optional[float] = 0.2,
    hist_max_percent: Optional[float] = 99.8,
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
        dtype=np.float32,
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

    fig, _ = plotutils.plot_overlay_withcolorbar(
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
