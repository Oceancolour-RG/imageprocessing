from typing import Union, Tuple, Iterable


def __check_bands_in_dict(d: dict, dname: str, camera: str = "dualcamera") -> None:
    nb = 10 if camera.lower() == "dualcamera" else 5
    expected_bands = [i + 1 for i in range(0, nb)]  # [1, 2, ..., nb]

    avail_bands = [int(_) for _ in d.keys()]
    if expected_bands != avail_bands:
        raise ValueError(
            f"available bands ({avail_bands}) in `{dname}` does not match "
            f"expected bands ({expected_bands})"
        )
    return


def check_camera(camera: str) -> None:
    """
    Check the camera type ('DualCamera', 'RedEdge-MX', 'RedEdge-MX-Blue')
    This function is used in acquisition_yamls.create_img_acqi_yamls
    """
    cam_l = camera.lower()
    avail_cams = ["dualcamera", "rededge-mx", "rededge-mx-blue"]
    if cam_l not in avail_cams:
        raise ValueError(f"'{cam_l}' not in {avail_cams}")

    return


def check_vrange(vrange: Union[Iterable[float], None]) -> Tuple[Union[None, float]]:
    """
    Check the value range. This function is used in plotutils.subplotwithcolorbar

    Parameters
    ----------
    vrange : Iterable[float] or None
        the common data range [vmin, vmax] that the colour map covers

    Returns
    -------
    vmin, vmax : float or None
    """
    if isinstance(vrange, Iterable):
        if len(vrange) != 2:
            raise ValueError("`vrange` must be Iterable with two elements")
        if vrange[1] <= vrange[0]:
            raise ValueError("vrange[1] must be greater or equal to vrange[0]")

        vmin = vrange[0]
        vmax = vrange[1]
    else:
        vmin = None
        vmax = None

    return vmin, vmax


def check_dc(dark_current: Union[float, int, None], bits_per_pixel: int = 16) -> None:
    """
    Check the dark current value. This function is primarily
    used in the image.Image class


    Parameters
    ----------
    dark_current : float, int or None
        dark current value (must be greater than 0 and less
        than the maximum DN value)

    bits_per_pixel : int
        pixel bits (used to get the maximum DN value)
    """
    max_dn = (2**bits_per_pixel) - 1

    if dark_current is not None:
        if isinstance(dark_current, (float, int)):
            err = f"`dark_current` ({dark_current}) must range between 0 and {max_dn}"
            if (dark_current < 0) or (dark_current >= max_dn):
                raise ValueError(err)
        else:
            raise TypeError("`dark_current` must be float, int or None")

    return


def check_cam_dc(
    cam_dc_params: Union[dict, None], bits_per_pixel: int = 16, camera: str = "dualcamera"
) -> None:
    """
    Check the dark current parameters for each band in the camera. This function
    is used in acquisition_yamls.create_img_acqi_yamls

    Parameters
    ----------
    cam_dc_params : dict or None
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

    bits_per_pixel : int
        pixel bits (used to get the maximum DN value)

    camera : str
         Micasense camera (DualCamera, RedEdge-MX, RedEdge-MX-Blue).

    """

    if cam_dc_params is not None:
        if not isinstance(cam_dc_params, dict):
            raise TypeError("`cam_dc_params` must be a dictionary (or None)")

        __check_bands_in_dict(cam_dc_params, "cam_dc_params", camera)

        for b in cam_dc_params:
            if "blacklevel" not in cam_dc_params[b]:
                raise ValueError(f"`cam_dc_params`[{b}] does not contain 'blacklevel'")

            check_dc(
                dark_current=cam_dc_params[b]["blacklevel"], bits_per_pixel=bits_per_pixel
            )

    return


def check_vigparms(vig_params: Union[dict, None], order: int = 6) -> None:
    """
    check the user-specified vignetting parameters. This function is
    primarily used in the image.Image class

    Parameters
    ----------
    vig_params : dict or None
        vignetting parameter dictionary with the following keys
        {
            "vignette_center": [float, float],  # x, y
            "vignette_polynomial": List[float],  # vignetting polynomials
        }
        Here, the 'vignette_polynomial' must be an Iterable
        with n-order of elements

    order : int [default = 6]
        Polynomial order of vignetting model
    """

    def _checknum(v: float) -> bool:
        """Check whether the variable `v` is float or int"""
        return isinstance(v, (int, float))

    err = "{0} must be an iterable with {1} elements\n{0}: {2}"

    if vig_params is not None:
        if not isinstance(vig_params, dict):
            raise TypeError("`vig_params` must be a dictionary (or None)")

        # --- check "vignette_centre" --- #
        if "vignette_center" not in vig_params:
            raise ValueError("`vig_params` must contain 'vignette_center'")
        else:
            tmp1 = vig_params["vignette_center"]
            err1 = err.format(
                "vig_params['vignette_center']", "two positive numeric", tmp1
            )
            if isinstance(tmp1, Iterable):
                if (len(tmp1) != 2) or (not all([_checknum(v) and v > 0 for v in tmp1])):
                    # check if vignette_centre has two positive numeric elements
                    raise ValueError(err1)
            else:
                # vignette_center is not an Iterable
                raise ValueError(err1)

        # --- check "vignette_polynomial" --- #
        if "vignette_polynomial" not in vig_params:
            raise ValueError("vig_params must contain 'vignette_polynomial'")
        else:
            tmp2 = vig_params["vignette_polynomial"]
            err2 = err.format(
                "vig_params['vignette_polynomial']", f"{order} numeric", tmp2
            )
            if isinstance(tmp2, Iterable):
                if (len(tmp2) != order) or (not all([_checknum(v) for v in tmp2])):
                    # check if vignette_polynomial has {order} numeric elements
                    raise ValueError(err2)
            else:
                # vignette_polynomial is not an Iterable
                raise ValueError(err2)

    return


def check_cam_vg(cam_vig_params: Union[dict, None], camera: str = "dualcamera") -> None:
    """
    Check the vignetting parameters for each band in the camera. This function
    is used in acquisition_yamls.create_img_acqi_yamls

    Parameters
    ----------
    cam_vig_params : dict or None
        The user defined vignetting parameters for each band of the camera,
        which will overwrite the default parameters. This dictionary
        must have the following keys
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

    camera : str
         Micasense camera (DualCamera, RedEdge-MX, RedEdge-MX-Blue).

    """
    if cam_vig_params is not None:
        if not isinstance(cam_vig_params, dict):
            raise TypeError("cam_vig_params must be a dictionary (or None)")

        __check_bands_in_dict(cam_vig_params, "cam_vig_params", camera)

        for b in cam_vig_params:
            check_vigparms(vig_params=cam_vig_params[b], order=6)

    return
