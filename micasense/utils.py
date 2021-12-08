#!/usr/bin/env python3
# coding: utf-8
"""
MicaSense Image Processing Utilities
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
import numpy as np
from typing import Tuple
import micasense.metadata2 as metadata


def raw_image_to_radiance(
    image_raw: np.ndarray, meta: metadata.MetadataFromExif
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # get image dimensions
    image_raw = image_raw.T
    ncols = image_raw.shape[0]
    nrows = image_raw.shape[1]

    if not meta.supports_radiometric_calibration():
        raise Exception(
            "Radiometric calibration factors not present in metadata. "
            "Cannot Convert to Radiance."
        )

    #  get radiometric calibration factors
    a1, a2, a3 = meta.radiometric_cal()

    # get dark current pixel values
    darklevel = meta.black_level()

    # get exposure time & gain (gain = ISO/100)
    exposure_time = meta.exposure()
    gain = meta.gain()

    # apply image correction methods to raw image
    # step 1 - row gradient correction, vignette & radiometric calibration:
    # compute the vignette map image
    vig, x, y = vignette_map(meta, ncols, nrows)

    # row gradient correction
    r_cal = 1.0 / (1.0 + a2 * y / exposure_time - a3 * y)

    # subtract the dark level and adjust for vignette and row gradient
    lt_im = vig * r_cal * (image_raw - darklevel)

    # Floor any negative radiances to zero (can happend due to noise around blacklevel)
    lt_im[lt_im < 0] = 0

    # lt_im = np.round(lt_im).astype(np.uint16)

    # apply the radiometric calibration - i.e. scale by the gain-exposure product and
    # multiply with the radiometric calibration coefficient
    # need to normalize by 2^16 for 16 bit images
    # because coefficients are scaled to work with input values of max 1.0
    bit_depthmax = float(2 ** meta.bits_per_pixel())
    radiance_image = lt_im.astype(float) / (gain * exposure_time) * a1 / bit_depthmax

    # return both the radiance compensated image and the DN corrected image, for the
    # sake of the tutorial and visualization
    return radiance_image.T, lt_im.T, vig.T, r_cal.T


def vignette_map(
    meta: metadata.MetadataFromExif, ncols: int, nrows: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get vignette center
    x_vignette, y_vignette = meta.vignette_center()

    # get vignette polynomial
    vignette_poly_list = meta.vignette_polynomial()

    # reverse list and append 1., so that we can call with numpy polyval
    vignette_poly_list.reverse()
    vignette_poly_list.append(1.0)
    vignette_poly = np.array(vignette_poly_list)

    # perform vignette correction
    # get coordinate grid across image
    x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))

    # meshgrid returns transposed arrays
    x = x.T
    y = y.T

    # compute matrix of distances from image center
    r = np.hypot((x - x_vignette), (y - y_vignette))

    # compute the vignette polynomial for each distance. So
    # we divide by the polynomial so that the corrected image,
    #     image_corrected = image_original * vignetteCorrection
    vignette = 1.0 / np.polyval(vignette_poly, r)
    return vignette, x, y


def correct_lens_distortion(image: np.ndarray, meta: metadata.MetadataFromExif) -> np.ndarray:
    # get lens distortion parameters
    distortion_params = np.array(meta.distortion_parameters())

    # get the two principal points
    pp = np.array(meta.principal_point())
    # values in pp are in [mm] and need to be rescaled to pixels
    focalplane_xres, focalplane_yres = meta.focal_plane_resolution_px_per_mm()

    c_x = pp[0] * focalplane_xres
    c_y = pp[1] * focalplane_yres
    # k = distortion_params[0:3] # seperate out k -parameters
    # p = distortion_params[3::] # separate out p - parameters
    fx = fy = meta.focal_length_mm() * focalplane_xres

    # apply perspective distortion
    h, w = image.shape

    # set up camera matrix for cv2
    cam_mat = np.zeros((3, 3))
    cam_mat[0, 0] = fx
    cam_mat[1, 1] = fy
    cam_mat[2, 2] = 1.0
    cam_mat[0, 2] = c_x
    cam_mat[1, 2] = c_y

    # set up distortion coefficients for cv2
    # dist_coeffs = np.array(k[0],k[1],p[0],p[1],k[2]])
    dist_coeffs = distortion_params[[0, 1, 3, 4, 2]]

    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w, h), 1)

    map1, map2 = cv2.initUndistortRectifyMap(
        cam_mat, dist_coeffs, np.eye(3), new_cam_mat, (w, h), cv2.CV_32F
    )  # cv2.CV_32F for 32 bit floats

    # compute the undistorted 16 bit image
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
