#!/usr/bin/env python3
# coding: utf-8
"""
MicaSense Downwelling Light Sensor Utilities

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


import warnings
import numpy as np

from datetime import datetime
from typing import Tuple, List

# for DLS correction, we need the sun position at the time the image was taken
# this can be computed using the pysolar package (ver 0.6)
# https://pypi.python.org/pypi/Pysolar/0.6
# we import multiple times with checking here because the case of Pysolar is
# different depending on the python version :(

HAVE_PYSOLAR = False

try:
    import pysolar.solar as pysolar

    HAVE_PYSOLAR = True
except ImportError:
    try:
        import Pysolar.solar as pysolar  # noqa: N813

        HAVE_PYSOLAR = True
    except ImportError:
        import pysolar.solar as pysolar

        HAVE_PYSOLAR = True
finally:
    if not HAVE_PYSOLAR:
        print("Unable to import pysolar")


def fresnel(phi: float):
    return __multilayer_transmission(phi, n=[1.000277, 1.6, 1.38])


# define functions to compute the DLS-Sun angle:
def __fresnel_transmission(
    phi: float,
    n1: float = 1.000277,
    n2: float = 1.38,
    polarization: List[float] = [0.5, 0.5],
) -> float:
    """compute fresnel transmission between media with refractive indices n1 and n2"""
    # computes the reflection and transmittance
    # for incidence angles  phi for transition from medium
    # with refractive index n1 to n2
    # teflon e.g. n2=1.38
    # polycarbonate n2=1.6
    # polarization=[.5,.5] - unpolarized light
    # polarization=[1.,0] - s-polarized light - perpendicular to plane of incidence
    # polarization=[0,1.] - p-polarized light - parallel to plane of incidence
    f1 = np.cos(phi)
    f2 = np.sqrt(1 - (n1 / n2 * np.sin(phi)) ** 2)
    rs = ((n1 * f1 - n2 * f2) / (n1 * f1 + n2 * f2)) ** 2
    rp = ((n1 * f2 - n2 * f1) / (n1 * f2 + n2 * f1)) ** 2
    f_t = 1.0 - polarization[0] * rs - polarization[1] * rp

    if f_t > 1:
        f_t = 0.0
    if f_t < 0:
        f_t = 0.0
    if np.isnan(f_t):
        f_t = 0.0

    return f_t


def __multilayer_transmission(
    phi: float, n: List[float], polarization: List[float] = [0.5, 0.5]
):
    ml_t = 1.0
    phi_eff = np.copy(phi)
    for i in range(0, len(n) - 1):
        n1 = n[i]
        n2 = n[i + 1]
        phi_eff = np.arcsin(np.sin(phi_eff) / n1)
        ml_t *= __fresnel_transmission(phi_eff, n1, n2, polarization=polarization)
    return ml_t


# get the position of the sun in North-East-Down (NED) coordinate system
def ned_from_pysolar(sun_azimuth: float, sun_altitude: float) -> np.ndarray:
    """Convert pysolar coordinates to NED coordinates."""
    elements = (
        np.cos(sun_azimuth) * np.cos(sun_altitude),
        np.sin(sun_azimuth) * np.cos(sun_altitude),
        -np.sin(sun_altitude),
    )
    return np.array(elements).transpose()


# get the sensor orientation in North-East-Down coordinates
# pose is a yaw/pitch/roll tuple of angles measured for the DLS
# ori is the 3D orientation vector of the DLS in body coordinates (typically [0,0,-1])
def get_orientation(pose: List[float], ori: np.ndarray):
    """Generate an orientation vector from yaw/pitch/roll angles in radians."""
    yaw, pitch, roll = pose
    c1 = np.cos(-yaw)
    s1 = np.sin(-yaw)
    c2 = np.cos(-pitch)
    s2 = np.sin(-pitch)
    c3 = np.cos(-roll)
    s3 = np.sin(-roll)

    r_yaw = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    r_pitch = np.array([[c2, 0, -s2], [0, 1, 0], [s2, 0, c2]])
    r_roll = np.array([[1, 0, 0], [0, c3, s3], [0, -s3, c3]])

    r_mat = np.dot(r_yaw, np.dot(r_pitch, r_roll))

    return np.dot(r_mat, ori)


# from the current position (lat,lon,alt) tuple
# and time (UTC), as well as the sensor orientation (yaw,pitch,roll) tuple
# compute a sensor sun angle - this is needed as the actual sun irradiance
# (for clear skies) is related to the measured irradiance by:

# I_measured = I_direct * cos (sun_sensor_angle) + I_diffuse
# For clear sky, I_direct/I_diffuse ~ 6 and we can simplify this to
# I_measured = I_direct * (cos (sun_sensor_angle) + 1/6)


def compute_sun_angle(
    position: Tuple[float, float, float],
    pose: Tuple[float, float, float],
    utc_datetime: datetime,
    sensor_orientation: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """compute the sun angle using pysolar functions"""
    altitude = 0
    azimuth = 0

    with warnings.catch_warnings():
        # Ignore pysolar leap seconds offset warning
        warnings.simplefilter("ignore")
        try:
            altitude = pysolar.get_altitude(position[0], position[1], utc_datetime)
            azimuth = pysolar.get_azimuth(position[0], position[1], utc_datetime)

        except AttributeError:
            # catch 0.6 version of pysolar required for python 2.7 support
            altitude = pysolar.GetAltitude(position[0], position[1], utc_datetime)
            azimuth = 180 - pysolar.GetAzimuth(position[0], position[1], utc_datetime)

        sun_altitude = np.radians(np.array(altitude))
        sun_azimuth = np.radians(np.array(azimuth))
        sun_azimuth = sun_azimuth % (2 * np.pi)  # wrap range 0 to 2*pi
        n_sun = ned_from_pysolar(sun_azimuth, sun_altitude)
        n_sensor = np.array(get_orientation(pose, sensor_orientation))
        angle = np.arccos(np.dot(n_sun, n_sensor))

    return n_sun, n_sensor, angle, sun_altitude, sun_azimuth
