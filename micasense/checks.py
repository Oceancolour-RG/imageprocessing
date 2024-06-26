from typing import Union, Tuple, Iterable


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


def check_dc(dark_current: Union[float, int, None], bits_per_pixel: int) -> None:
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
            raise TypeError("vig_params must be a dictionary (or None)")

        # --- check "vignette_centre" --- #
        if "vignette_center" not in vig_params:
            raise ValueError("vig_params must contain 'vignette_center'")
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
