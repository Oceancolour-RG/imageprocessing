#!/usr/bin/env python3

import yaml
from pathlib import Path
from typing import Optional, Union


def load_yaml(
    yaml_file: Union[Path, str], key: Optional[str] = None
) -> Union[dict, None]:
    # open the yaml file
    val = None
    with open(yaml_file, "r") as fid:
        if key:
            for i, row in enumerate(fid):
                if key in row:
                    vdict = yaml.safe_load(row)
                    val = vdict[key]
                    break
        else:
            # read yaml contents to a dict
            val = yaml.safe_load(fid)

    return val


def add_ppk_to_yaml(
    yml_f: Union[Path, str],
    ppk_lat: Optional[float] = None,
    ppk_lon: Optional[float] = None,
    ppk_height: Optional[float] = None,
    ppk_lat_uncert: Optional[float] = None,
    ppk_lon_uncert: Optional[float] = None,
    ppk_alt_uncert: Optional[float] = None,
) -> None:
    """add ppk lat/lon/height to yaml document"""
    acq_dict = load_yaml(yaml_file=yml_f)

    acq_dict["ppk_lat"] = ppk_lat
    acq_dict["ppk_lon"] = ppk_lon
    acq_dict["ppk_height"] = ppk_height
    acq_dict["ppk_lat_uncert"] = ppk_lat_uncert
    acq_dict["ppk_lon_uncert"] = ppk_lon_uncert
    acq_dict["ppk_alt_uncert"] = ppk_alt_uncert

    with open(yml_f, "w") as fid:
        yaml.dump(acq_dict, fid, default_flow_style=False)
    return


def replace_original_params(
    original_md: dict,
    vig_param: Union[dict, None],
    dc_param: Union[dict, None],
    lens_param: Union[dict, None],
) -> dict:

    mod_md = original_md.copy()
    # Add new vignetting coefficients here
    if vig_param:  # `vig_param` has already been checked and verified
        if "vignette_xy_original" not in mod_md:
            # do not replace original value
            mod_md["vignette_xy_original"] = mod_md["vignette_xy"]

        if "vignette_poly_original" not in mod_md:
            # do not replace original value
            mod_md["vignette_poly_original"] = mod_md["vignette_poly"]

        mod_md["vignette_xy"] = vig_param["vignette_center"]
        mod_md["vignette_poly"] = vig_param["vignette_polynomial"]

    # add new blacklevel value here
    if dc_param:  # `dc_param` has already been checked and verified
        if "blacklevel_original" not in mod_md:
            # do not replace original value
            mod_md["blacklevel_original"] = mod_md["blacklevel"]

        mod_md["blacklevel"] = dc_param["blacklevel"]

    if lens_param:
        raise NotImplementedError(
            "Adding lens calibration parameters has yet to be implemented"
        )

    return mod_md


def add_new_params(
    yml_f: Union[Path, str],
    vig_params: Union[dict, None],
    dc_params: Union[dict, None],
    lens_params: Union[dict, None],
) -> None:
    """
    Add PPK lat/lon/height and new vignetting, dark current and
    lens parameters to the micasense metadata yamls

    Parameters
    ----------
    vig_params : dict or None
        User defined vignetting parameters that will overwrite the
        default parameters. This dictionary must have the following keys
        {
            1: {  # band number
                "vignette_center": [float, float],  # x, y
                "vignette_polynomial": List[float],  # vignetting polynomials
            },
            ...,
            X: {  # band number
            }
        }
        Where X is the band number (1-5 for RedEdge-MX or 1-10 for Dual Camera)

        The vignetting polynomials must have six values for the model as in,
        https://support.micasense.com/hc/en-us/articles/
           115000351194-Radiometric-Calibration-Model-for-MicaSense-Sensors

    dc_params : dict or None
        User defined dark current values that will overwrite the
        default 'blacklevel' values. This dictionary must have the following keys,
        {
            1: {  # band number
                "blacklevel": float or int
            },
            ...,
            X: {  # band number
                "blacklevel": float or int
            }
        }

        The values of 'blacklevel' must be greater than 0 and less than
        the maximum DN value

    lens_params : dict or None
        Lens calibration parameters (e.g. focal length, etc)
        This has yet to be implemented
    """

    def get_bandnum(tif: str) -> int:
        """Return the band number from the filename"""
        return int(tif.split(".")[0].split("_")[-1])

    all_empty = (not vig_params) and (not dc_params) and (not lens_params)
    if not all_empty:
        acq_dict = load_yaml(yaml_file=yml_f)

        for tif in acq_dict["image_data"]:
            bn = get_bandnum(tif)
            acq_dict["image_data"][tif] = replace_original_params(
                original_md=acq_dict["image_data"][tif],
                vig_param=vig_params[bn] if vig_params else None,
                dc_param=dc_params[bn] if dc_params else None,
                lens_param=lens_params[bn] if lens_params else None,
            )

        with open(yml_f, "w") as fid:
            yaml.dump(acq_dict, fid, default_flow_style=False)

    return
