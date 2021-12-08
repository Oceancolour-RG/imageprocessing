#!/usr/bin/env python3

import yaml
from pathlib2 import Path
from typing import Union


def load_all(yaml_file: Union[Path, str]) -> dict:
    with open(yaml_file, "r") as fid:
        d = yaml.safe_load(fid)  # dict
    return d
