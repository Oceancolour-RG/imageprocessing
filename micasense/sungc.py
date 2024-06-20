#!/usr/bin/env python3

import numpy as np

from pathlib import Path
from yaml import dump as ydump
from typing import Optional, List, Tuple
from micasense.capture import Capture
from micasense.imageutils import aligned_capture, load_warp_matrices
from rglib.sungc_backend import hedley2005_sungc
from rglib.visualise import ComparisonSpectrumPlotter


def uav_hedley_wrapper(
    uav_yaml_file: Path,
    warp_npy_file: Path,
    hedley_oyaml: Path,
    vcg_md: Optional[dict] = None,
    ed_md: Optional[dict] = None,
    use_darkpixels: bool = True,
    pixels: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Perform the Hedley et al. (2005) sunglint correction specific for
    the Micasense Dual Camera UAV sensor.

    Parameters
    ----------
    uav_yaml_file : Path
        UAV acquisition metadata yaml containing all the relevant
        parameters needed to process imagery
    warp_npy_file : Path
        filename of Homographic warping numpy binaries
    hedley_oyaml : Path
        filename of the output yaml file, that contains the
        spectral slopes, minimum NIR value and other metadata
        used to perform sunglint correction on other UAV
        acquisition collected in the same survey
    vcg_md : dict  [Optional]
        Vicarious calibration for each band of the UAV sensor,
        e.g.
        `vcg_md`: {
            "444": 1.03,
            "475": 1.11,
            ... etc ...
        }
    ed_md : Path  [Optional]
        A dictionary containing the following keys:
        {
            "which_ed": str,  # "dls2" or "dalec"
            "spectrum": List[float]  # E{d} spectrum for `uav_yaml_file`
        }
        If `ed_md` is None, then the DLS2 data automatically loaded.

    use_darkpixels : bool
        Whether to use the dark pixels

    pixels : List[int]  [Optional]
        Pixel index locations with the format of:
        [(row1, col1), (row2, col2), (row3, col3), ..., (rowN, colN)]
        used in a comparison of `initial` vs `deglinted` Rrs

    Returns
    -------
    deglint_rrs : np.ndarray
        The deglinted water-leaving remote sensing reflectance image

    rrs_glint_560 : np.ndarray
        The estimated sunglint remote sensing reflectance image at the
        band whose wavelength is closest to 560 nm.

    higlint_mask : np.ndarray (dtype="bool") or None
        Boolean mask where pixels = True have a glint contribution > glint_thr.
        This array is only returned if `glint_thr` was specified.

    hedley_md : dict
        Dictionary containing parameters from the Hedley et al. (2005)
        algorithm including relevant metadata
    """

    # 1) load warp_npy_file
    warp_matrices = load_warp_matrices(warp_file=warp_npy_file)

    # 2) align reflectance image
    img_capture = Capture.from_yaml(uav_yaml_file)
    wavel = np.array(img_capture.center_wavelengths())

    # Get the irradiance from `ed_md` if it was specified
    which_ed = "dls2"
    irrad = None
    if ed_md:
        which_ed = ed_md["which_ed"]
        irrad = [
            ed_md["spectrum"][f"{int(_.center_wavelength)}"] for _ in img_capture.images
        ]

    vc_g = None
    if vcg_md:
        vc_g = [vcg_md[f"{int(_.center_wavelength)}"] for _ in img_capture.images]

    refl_kw = {
        "img_type": "reflectance",
        "warp_mode": 3,  # cv2.MOTION_HOMOGRAPHY = 3
        "irradiance": irrad,
        "vc_g": vc_g,
        "use_darkpixels": use_darkpixels,
        "crop_edges": False,
    }

    aligned_refl = aligned_capture(
        ms_capture=img_capture, warp_matrices=warp_matrices, **refl_kw
    )  # [nr, nc, nb], where nb contains unordered wavelegnths
    refl = np.moveaxis(a=aligned_refl, source=2, destination=0)  # [nb, nr, nc]

    # 3) Perform sunglint correction on im_aligned:

    # Calculate `glint_thr`a
    parent_dir = Path(hedley_oyaml).parent / f"{uav_yaml_file.stem}"
    parent_dir.mkdir(exist_ok=True)

    sza, _ = img_capture.solar_geoms()
    glint_thr = 0.01 / np.cos(sza * np.pi / 180.0)

    deglint_r, rrs_glint_560, higlint_mask, hedley_md = hedley2005_sungc(
        refl=refl,
        sensor_wvl=wavel,
        nir_wvl=842,
        clip=False,
        mfactor=1.0 / np.pi,
        scale_offset=0.0,
        glint_thr=glint_thr,
        roi_mask=None,  # interactive plot will be generated to select ROI
        plot_dir=parent_dir,  # to save the NIR-VIS correlations
    )  # [nb, nr, nc]
    hedley_md["acq_yaml"] = str(uav_yaml_file)
    hedley_md["warp_npy_file"] = str(warp_npy_file)
    hedley_md["alignment_params"] = refl_kw
    hedley_md["which_ed"] = which_ed

    # 4) save `hedley_md` as a yaml

    # ensure that the hedley directory exists
    with open(hedley_oyaml, "w") as fid:
        ydump(hedley_md, fid, default_flow_style=False)
    print(f"created: {hedley_oyaml}")

    if pixels:
        # 5) compare spectra between `deglint_r` and `aligned_refl`
        # [[r1, c1], [r2, c2], ..., [rN, cN]]
        sort_ix = np.argsort(wavel)
        csp = ComparisonSpectrumPlotter(
            spectral_im1=refl[sort_ix, :, :] / np.pi,  # total at-sensor reflectance
            spectral_im2=deglint_r[sort_ix, :, :],
            wavelength=wavel[sort_ix],
            pixels=pixels,
            ylabel=r"Remote Sensing Reflectance ($sr^{-1}$)",
            xlabel="Wavelength (nm)",
            label1="initial",
            label2="deglinted",
        )
        csp.show()

    return deglint_r, rrs_glint_560, higlint_mask, hedley_md
