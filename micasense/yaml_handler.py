#!/usr/bin/env python3

import yaml
from pathlib import Path
from typing import Optional, Union


def load_yaml(yaml_file: Union[Path, str], key: Optional[str] = None) -> Union[dict, None]:
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
