#!/usr/bin/env python3

import yaml
from pathlib import Path
from typing import Optional, Union


def load_all(yaml_file: Union[Path, str], key: Optional[str] = None) -> Union[dict, None]:
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
