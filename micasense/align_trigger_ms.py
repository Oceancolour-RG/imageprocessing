#!/usr/bin/env python3

"""
This script contains functions used in the alignment between the 
micasense camera and an externall GNSS unit such as the reach M2.
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import find_peaks
from xarray import Dataset, DataArray
from typing import Optional, Union, List, Tuple
from datetime import datetime, timedelta, timezone

import micasense.load_yaml as ms_yaml


def plot_histograms_frate(
    trg_ts: np.ndarray,
    mse_ts: np.ndarray,
    opng: Optional[Union[Path, str]] = None,
) -> None:
    """
    Plot:
    1. Histograms of the micasense and trigger frame rate, and;
    2. Histogram of the time difference between micasense and
       trigger events.

    Parameters
    ----------
    trg_ts : np.ndarray
        Trigger events POSIX timestamps (UTC)
    mse_ts : np.ndarray
        Micasense acquisition POSIX timestamps (UTC)
    opng : Path, str [Optional]
        Output figure filename
    """

    trg_frate = np.diff(trg_ts)
    mse_frate = np.diff(mse_ts)

    start = 0.50
    stop = 1.50
    bwidth = 0.02

    bedges = np.arange(start=start, stop=stop + bwidth, step=bwidth)
    cbins = get_ave_vals(bedges)

    trg_freq, _ = np.histogram(trg_frate, bins=bedges, density=True)
    mse_freq, _ = np.histogram(mse_frate, bins=bedges, density=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

    axes[0].plot(cbins, trg_freq * np.diff(bedges), "r.-", label="trigger events")
    axes[0].plot(cbins, mse_freq * np.diff(bedges), "k.--", label="Micasense image-sets")
    axes[0].set_xlabel("frame rate (seconds)")
    axes[0].set_ylabel("PDF")
    axes[0].legend(loc=1)

    tdif = trg_ts - mse_ts
    tbins = np.histogram_bin_edges(
        tdif, bins=20, range=[np.nanmin(tdif), np.nanmax(tdif)], weights=None
    )
    tfreq, _ = np.histogram(tdif, bins=tbins, density=True)
    axes[1].plot(get_ave_vals(tbins), tfreq * np.diff(tbins), "k.:")
    axes[1].set_xlabel("time difference (trigger - micasense) (s)")
    axes[1].set_ylabel("PDF")

    if opng is not None:
        fig.savefig(opng, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"\n    created: {opng}")
    return


def plot_frate(
    trg_ts: np.ndarray,
    mse_ts: np.ndarray,
    trg_ix: Optional[np.ndarray] = None,
    mse_ix: Optional[np.ndarray] = None,
    opng: Optional[Union[Path, str]] = None,
) -> None:
    """
    Compare the frame rates between trigger `trg_ts` and micasense
    `mse_ts` events/acqusitions

    Parameters
    ----------
    trg_ts : np.ndarray
        trigger POSIX timestamps (UTC)
    mse_ts : np.ndarray
        micasense POSIX timestamps (UTC)
    trg_ix : np.ndarray [Optional]
        trigger alignment indices
    mse_ix : np.ndarray [Optional]
        micasense alignment indices
    """

    event_fr = np.diff(trg_ts)
    event_dt = get_ave_vals(trg_ts)

    cam_fr = np.diff(mse_ts)
    cam_dt = get_ave_vals(mse_ts)

    # find peaks in event_fr
    peak_ix, _ = find_peaks(event_fr, height=1.5)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7))
    axes[0].plot(cam_dt, cam_fr, "r.--", label="camera")
    axes[0].plot(mse_ts[0], cam_fr[0], "r*", ms=8, ls="None")  # first camera acquisition

    axes[0].plot(event_dt, event_fr, "b.--", label="reach")
    axes[0].plot(trg_ts[0], event_fr[0], "b*", ms=8, ls="None")  # first reach acquisition
    axes[0].plot(event_dt[peak_ix], event_fr[peak_ix], "kd")
    axes[0].set_xlabel("POSIX time, seconds, (UTC)")
    axes[0].set_ylabel("sequential time difference (s)")

    for pix in peak_ix:
        axes[0].annotate(
            f"{pix}",
            xy=(event_dt[pix], event_fr[pix]),
            xycoords="data",
            color="b",
            ha="left",
        )
        axes[0].annotate(
            f"{pix}",
            xy=(cam_dt[pix], cam_fr[pix]),
            xycoords="data",
            color="r",
            ha="right",
        )

    axes[0].legend(loc=1)

    if (trg_ix is not None) and (mse_ix is not None):
        axes[1].plot(trg_ts[trg_ix], trg_ts[trg_ix] - mse_ts[mse_ix], "k.--")
        axes[1].set_xlabel("Trigger event POSIX timestamps (UTC)")
        axes[1].set_ylabel("time difference (trigger - micasense) (s)")

    if opng is not None:
        fig.savefig(opng, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"\n    created: {opng}")

    return


def get_dt(ts: float) -> datetime:
    """get datetime from POSIX timestamp"""
    return datetime.fromtimestamp(ts)


def get_ave_vals(arr: np.ndarray) -> np.ndarray:
    """Return the average (in-between) values of `arr`"""
    return 0.5 * (arr[0:-1] + (np.roll(arr, -1))[0:-1])


def dt_from_gps(
    date: str,
    gps_seconds: float,
    time_md: dict,
    leap_sec: float = 18,
) -> datetime:
    """
    Return the UTC datetime from gps date and gps time
    """
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    week_dt = datetime.strptime(date + "+0000", time_md["date"]["fmt"] + "%z")
    gps_week = (week_dt - gps_epoch).days // 7

    return (
        gps_epoch
        + timedelta(weeks=gps_week, seconds=gps_seconds)
        - timedelta(seconds=leap_sec)
    )


def create_dataset(data: dict, coords: dict, attrs: Optional[dict] = None) -> Dataset:
    return Dataset(
        data_vars={k: DataArray(**data[k]) for k in data},
        coords=coords,
        attrs=attrs,
    )


def load_events_pos(
    ppk_fn: Path,
    delim: Optional[str] = None,
    leap_sec: float = 18,
) -> Dataset:
    """
    Load the Reach M2 *_events.pos file as an xarray.Dataset

    Parameters
    ----------
    ppk_fn : Path
        The *_events.pos filename

    delim : str (Optional)
        Delimiter; for csv files use `delim=","`

    leap_sec : float (default = 18)
        The leap seconds used to convert GPS time to UTC.
        see https://endruntechnologies.com/support/leap-seconds
        As of 13 Dec. 2016, the current GPS-UTC leap seconds is 18.
        `leap_sec` is ignored if time_md["epoch"] = "utc"

    Returns
    -------
    event_ds : xr.Dataset
        Dataset with the time ('ppk_time') as the coordinate and
        latitude, longitude, height and num_sats as variables

       <xarray.Dataset>
       Dimensions:      (ppk_time: 540)
       Coordinates:
         * ppk_time     (ppk_time) float64 1.683e+09 1.683e+09 ... 1.683e+09 1.683e+09
       Data variables: (12/13)
           ppk_alt      (ppk_time) float64 -14.49 -14.49 -14.49 ... -14.47 -14.47
           ppk_lat      (ppk_time) float64 -22.71 -22.71 -22.71 ... -22.71 -22.71
           ppk_lon      (ppk_time) float64 113.7 113.7 113.7 ... 113.7 113.7 113.7
           lat_std      (ppk_time) float64 0.006 0.006 0.006 ... 0.006 0.006 0.008065
           lon_std      (ppk_time) float64 0.006 0.006 0.006 ... 0.006 0.006 0.008065
           alt_std      (ppk_time) float64 0.017 0.017 0.017 ... 0.016 0.016 0.021
       Attributes:
           ppk event file:  /media/user_ra/alsi/uav_data/20230505_JanesBay01/reach/p...
           num_obs:         540
           leap_sec:        18

    """
    with open(ppk_fn, "r") as fid:
        contents = fid.readlines()

    dims = "ppk_time"

    data_md = {  # based on emlid studio processing
        "ppk_lat": {"ix": 2, "attrs": {"info": "latitude", "units": "degree"}},
        "ppk_lon": {"ix": 3, "attrs": {"info": "longitude", "units": "degree"}},
        "ppk_alt": {"ix": 4, "attrs": {"info": "height", "units": "metres"}},
        "lat_std": {"ix": 7, "attrs": {"info": "lat. stdev.", "units": "metres"}},
        "lon_std": {"ix": 8, "attrs": {"info": "lon. stdev.", "units": "metres"}},
        "alt_std": {"ix": 9, "attrs": {"info": "alt. stdev.", "units": "metres"}},
    }

    time_md = {  # based on emlid studio processing
        "date": {"ix": 0, "fmt": "%Y/%m/%d"},
        "time": {"ix": 1, "type": "datetime", "fmt": "%H:%M:%S.%f"},
        "epoch": "gps",  # "utc" or "gps"
    }

    # set `leap_sec` to 0.0 if epoch is "utc"
    leap_sec = 0.0 if time_md["epoch"] == "utc" else leap_sec

    data = {
        k: {"data": [], "dims": dims, "name": k, "attrs": data_md[k]["attrs"]}
        for k in data_md
    }
    coords = {dims: []}

    for i in range(len(contents)):
        if "%" in contents[i]:
            continue

        # Split the ascii row using the user-defined delimiter
        if delim is None:
            row = contents[i].strip().split()
        else:
            row = contents[i].strip().split(delim)

        # Get the datetime object using `time_md`
        if time_md["time"]["type"].lower() == "datetime":
            dt_format = f"{time_md['date']['fmt']} {time_md['time']['fmt']}%z"
            dt_str = f"{row[time_md['date']['ix']]} {row[time_md['time']['ix']]}+0000"
            dt = datetime.strptime(dt_str, dt_format) - timedelta(seconds=leap_sec)
        else:
            dt = dt_from_gps(
                date=row[time_md["date"]["ix"]],  # str
                gps_seconds=float(row[time_md["time"]["ix"]]),
                time_md=time_md,
                leap_sec=leap_sec,
            )

        coords[dims].append(dt.timestamp())

        for k in data_md:
            data[k]["data"].append(float(row[data_md[k]["ix"]]))

    global_attrs = {
        "ppk event file": str(ppk_fn),
        "num_obs": len(coords[dims]),
        "leap_sec": leap_sec,
    }

    return create_dataset(data=data, coords=coords, attrs=global_attrs)


def get_micasense_timestamps(tifs: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get micasense POSIX  timestamps from a list of metadata yaml
    files. Those yamls that do not have a 'dls_utctime' key  are
    considered invalid for trigger-to-micasense matching purposes
    """

    ms_dt = []
    valid_mse_ix = []

    for ms_ix, f in enumerate(tifs):
        ms_time = ms_yaml.load_all(yaml_file=f, key="dls_utctime")  # None or datetime
        if ms_time is None:
            valid_mse_ix.append(False)
            ms_dt.append(np.nan)
            continue

        valid_mse_ix.append(True)
        ts_ = datetime.timestamp(ms_time)  # convert to POSIX timestamp
        ms_dt.append(ts_)

    ms_dt = np.array(ms_dt, order="C", dtype="float64")
    valid_mse_ix = np.array(valid_mse_ix, order="C", dtype="bool")

    return ms_dt, valid_mse_ix


def get_alignment(
    dt1: np.ndarray,
    dt2: np.ndarray,
    nmiss1: int,
    info: str = "",
) -> Tuple[int, int, int, int]:
    """
    Perform simple 1:1 alignment. Here, we assume that the
    missing acquisitions/events occur either at the start
    (e.g. case 1) or at the end (case 2).

    --- case 1 ---
    The Reach M2 unit takes a few minutes to initialise and acquire
    a GPS fix. During this time the micasense camera has been capt-
    uring image data at rate of ~1 Hz. In this case, the Reach M2
    is missing events at the start of the survey

    --- case 2 ---
    An extra trigger at the end of the survey. Presumably during
    power-off when a trigger signal was sent to the Reach M2, but
    the camera didn't acquire image data.

    Parameters
    ----------
    dt1 : ndarray
        POSIX timestamps
    dt2 : ndarray
        POSIX timestamps
    nmiss1 : int
        number of missing acquisitions/events in `dt1`
    info : str
        information of `dt1`
    """
    n_d2 = len(dt2)

    # `dt2` alignment indices
    d2_six = 0
    d2_lix = n_d2 - 1

    # `dt1` alignment indicies
    d1_six = nmiss1  # assumes missing acquisitions/events at the start
    d1_lix = d1_six + n_d2 - 1

    # Check if you where to get smaller time difference by aligning from the end
    tdiff_start = dt1[d1_six : d1_lix + 1] - dt2[d2_six : d2_lix + 1]
    tdiff_end = dt1[0:n_d2] - dt2[d2_six : d2_lix + 1]

    mean_tds = np.nanmedian(tdiff_start)
    mean_tde = np.nanmedian(tdiff_end)

    if abs(mean_tde) < abs(mean_tds):
        print(
            f"    Best alignment found after removing extra {info} "
            "events at the end\n"
            f"        median(time diff by removing end)  : {mean_tde:0.5f} s\n"
            f"        median(time diff by removing start): {mean_tds:0.5f} s\n"
        )
        d1_six = 0
        d1_lix = n_d2 - 1

        best_mean_td = mean_tde
    else:
        best_mean_td = mean_tds

    return d1_six, d1_lix, d2_six, d2_lix, best_mean_td


def alignment_wrapper(
    mse_ts: np.ndarray,
    trigger_ts: np.ndarray,
    valid_mse_ix: np.ndarray,
    ms_frate: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Micasense - Trigger event 1:1 alignment wrapper. See
    ``align_trigger_ms.get_alignment()`` for more information

    Parameters
    ----------
    mse_ts : np.ndarray
        micasense acquisition POSIX timestamps (UTC)
    trigger_ts : np.ndarray
        trigger event POSIX timestamps (UTC)
    valid_mse_ix : np.ndarray
        valid micasense indices

    Returns
    -------
    mse_ix : np.ndarray
        micasense alignment indices
    trg_ix : np.ndarray
        trigger event alignment indices
    """
    n_mse = len(mse_ts)
    n_trg = len(trigger_ts)
    n_invalid_mse = len(np.where(~valid_mse_ix)[0])
    print(f"    Number of recorded trigger events: {n_trg}")
    print(f"    Number of Micasense  acquisitions: {n_mse}")
    print(f"    Number of invalid Micasense image sets: {n_invalid_mse}")

    if n_mse >= n_trg:
        n_missing = n_mse - n_trg  # number of missing trigger events
        print(f"    Number of missing trigger events: {n_missing}")
        mse_six, mse_lix, trg_six, trg_lix, best_median_td = get_alignment(
            dt1=mse_ts, dt2=trigger_ts, nmiss1=n_missing, info="micasense"
        )

    else:
        n_missing = n_trg - n_mse  # number of missing micasense acquisitions
        print(f"    Number of missing micasense acq.: {n_missing}")
        trg_six, trg_lix, mse_six, mse_lix, best_median_td = get_alignment(
            dt1=trigger_ts, dt2=mse_ts, nmiss1=n_missing, info="trigger"
        )

    print(f"    best median time diff (trigger vs micasense): {best_median_td:0.5f} s")

    # recheck alignment (we still could one second off)
    # note `ms_frate` is in Hz (1/s), while `best_mean_td` is in (s)
    if abs(best_median_td) > (0.7 / ms_frate):  # e.g. abs(mean_tds) = 1.0008 seconds
        est_off = int(round(best_median_td * ms_frate))  # positive or negative
        print(f"        * Estimated micasense index offset, est_off: {-est_off}")
        print("        * Positive `est_off` ==> subtract to align")
        print("        * Negative `est_off` ==> add to align")
        mse_six -= est_off
        mse_lix -= est_off
        if (mse_lix >= n_mse) or (mse_six < 0):
            # if `mse_lix` >= n_mse then we have to crop the ends of
            # both the micasense and trigger acquisitions
            print(
                "\n   **** FIX CODE ****\n"
                "    modified `mse_lix` >= `n_mse` or `mse_six` < 0\n"
                f"    n_mse = {n_mse}\n"
                f"    mse_six (adjusted) = {mse_six}\n"
                f"    mse_lix (adjusted) = {mse_lix}\n"
                f"    est_off = {est_off}\n"
                "    *****************"
            )
            return None, None

        best_median_td = np.nanmean(
            mse_ts[mse_six : mse_lix + 1] - trigger_ts[trg_six : trg_lix + 1]
        )
        print(f"        *Recalculated median time diff: {best_median_td:0.5f} s")

    # Check for invalid micasense acqusitions
    lastvalid_mse_ix = np.where(valid_mse_ix)[0][-1]
    n_excess_mse = mse_lix - lastvalid_mse_ix

    mse_lix -= n_excess_mse
    trg_lix -= n_excess_mse

    first_tdif = abs(trigger_ts[trg_six] - mse_ts[mse_six])
    last_tdif = abs(mse_ts[mse_lix] - trigger_ts[trg_lix])

    print("\n    ------------------------------------------")
    print("    --------------- ALIGNMENT ----------------")
    print(f"    Micasense alignment index: {mse_six}")
    print(f"    Trigger   alignment index: {trg_six}")

    print(f"\n    * Datetime of first aligned trigger  : {get_dt(trigger_ts[trg_six])}")
    print(f"      Datetime of first aligned micasense: {get_dt(mse_ts[mse_six])}")
    print(f"      Difference: {first_tdif:0.4f} seconds\n")

    print(f"    * Datetime of last trigger  event: {get_dt(trigger_ts[trg_lix])}")
    print(f"      Datetime of last micasense aqc.: {get_dt(mse_ts[mse_lix])}")
    print(f"      Difference: {last_tdif:0.4f} seconds")

    if last_tdif > 2:
        print("\n    ------------------------------------------")
        print("    ---------------- CAUTION -----------------")
        print("    Time difference between last recorded trigger event")
        print("    and last micasense acquisition is UNUSALLY LARGE:")
        print(f"    time difference: {last_tdif} seconds")

    match_thr = ms_frate / 10
    if (first_tdif < match_thr) and (last_tdif < match_thr):
        print("    ---- [EXCELLENT MATCH] ----\n")

    if (first_tdif > match_thr) and (last_tdif < match_thr):
        print("    ---- [OK MATCH] ----\n")

    if (first_tdif < match_thr) and (last_tdif > match_thr):
        print("    ---- [OK MATCH] ----\n")

    if (first_tdif > match_thr) and (last_tdif > match_thr):
        print("    ---- [CHECK MATCH] ----\n")

    mse_ix = np.arange(mse_six, mse_lix + 1)  # alignment indices
    trg_ix = np.arange(trg_six, trg_lix + 1)  # alignment indices

    return mse_ix, trg_ix


def append_reach2yaml(
    yaml_path: Union[Path, str],
    events_file: Union[Path, str, None],
    events_ds: Union[Dataset, None],
    leap_sec: float = 18,
    ms_frate: int = 1,
    debug: bool = False,
    events_slice: Optional[slice] = None,
) -> None:
    """
    Perform a simple 1:1 alignment between PPK trigger events and the
    micasense acquisition. This function adds the ppk latitude/longitude/
    altitude and the uncertainty into the metadata yamls created by the
    `create_img_acqi_yamls()` function.

    Parameters
    ----------
    yaml_path : Path or str
        The directory containing the yaml's for a given flight
    events_file : Path, str or None
        The events.pos file associated with the flight. This file can be gen-
        erated from PPK processing using rtklib or the Emlid studio's software
    events_ds : xarray.Dataset or None
        The events dataset (see Notes)
    leap_sec : float (default = 18)
        The leap seconds used to convert the Reach M2 GPS time to UTC.
        see https://endruntechnologies.com/support/leap-seconds
        As of 13 Dec. 2016, the current GPS-UTC leap seconds is 18
    ms_frate : int
        micasense frame rate (Hz)
    debug : bool
        whether to run debug mode
    events_slice : slice [Optional]
        Optional slice applied to the trigger events to aid in the
        alignment process. For example,
        >>> events_slice = numpy.s_[0:-15]   # excludes the last 15 trigger events

    Raises
    ------
    ** RuntimeError when `events_file` and `events_ds` are both None
    ** RuntimeError when `events_file` and `events_ds` are specified

    Notes
    -----
    ++ `events_ds` is an xarray.Dataset with the following coordinates
       and variables:

       <xarray.Dataset>
       Dimensions:      (ppk_time: 540)
       Coordinates:
         * ppk_time     (ppk_time) float64 1.683e+09 1.683e+09 ... 1.683e+09 1.683e+09
       Data variables: (12/13)
           ppk_alt      (ppk_time) float64 -14.49 -14.49 -14.49 ... -14.47 -14.47
           ppk_lat      (ppk_time) float64 -22.71 -22.71 -22.71 ... -22.71 -22.71
           ppk_lon      (ppk_time) float64 113.7 113.7 113.7 ... 113.7 113.7 113.7
           lat_std      (ppk_time) float64 0.006 0.006 0.006 ... 0.006 0.006 0.008065
           lon_std      (ppk_time) float64 0.006 0.006 0.006 ... 0.006 0.006 0.008065
           alt_std      (ppk_time) float64 0.017 0.017 0.017 ... 0.016 0.016 0.021
           event_id     (ppk_time) int64 1 2 3 4 5 6 7 ... 534 535 536 537 538 539 540
       Attributes:
           ppk event file:  /media/user_ra/alsi/uav_data/20230505_JanesBay01/reach/p...
           num_obs:         540
           leap_sec:        18

    ++ `events_file` or `events_ds` must be specified

    ++ The number of trigger events in the  *_events.pos file may not
       match the number tiff's acquired during the flight. The reason
       for this is unclear at the moment, but may be caused by captu-
       ring image data as the reach  m2 is turning on or in the proc-
       ess of acquiring a satellite fix.

       Histograms of the Reach M2 trigger event frame rate and the
       Micasense frame rate are nearly identical, where a mode exists
       at 1.0 seconds with ~90% of the frame rates existing between
       0.90 to 1.10 seconds.

    ++ There is no correlation between the Micasense frame rate and
       the max. exposure time for an acquisition.

    Usage
    -----
    Example when using an events.pos file generated by an external software:
    >>> append_reach2yaml(
    >>>    yaml_path=Path("/path/to/micasense/metadata/SYNC0009SET"),
    >>>    events_file=Path("/path/to/reach_m2/ppk/reach_raw_202111260212_events.pos"),
    >>>    events_ds=None,
    >>>    leap_sec=18.0,
    >>> )

    Example when using the events xarray.Dataset
    >>> from pydem.camera.events import interp_events_from_epochs
    >>> events_ds = interp_events_from_epochs(
    >>>    epoch_fn=Path("/path/to/reach_m2/ppk/ppk_202111260212_epochs.pos"),
    >>>    obs_fn=Path("/path/to/reach_m2/raw/reach_raw_202111260212.23O"),
    >>>    leap_sec=18.0,
    >>> )
    >>>
    >>> append_reach2yaml(
    >>>    yaml_path=Path("/path/to/micasense/metadata/SYNC0009SET"),
    >>>    events_file=None,
    >>>    events_ds=events_ds,
    >>>    leap_sec=18.0,
    >>> )

    """

    input_check = []
    input_check.append(True if isinstance(events_file, (Path, str)) else False)
    input_check.append(True if isinstance(events_ds, Dataset) else False)

    if not any(input_check):
        raise RuntimeError("`events_file` and `events_ds` are both None")

    if all(input_check):
        raise RuntimeError("`events_file` and `events_ds` have both been specified")

    print(f"\nAppending Reach M2 GPS data to yamls in {yaml_path}")

    # Get trigger events
    if input_check[0]:  # `events_file` was specified
        events_ds = load_events_pos(ppk_fn=events_file, delim=None, leap_sec=leap_sec)

    if events_slice is None:
        events_slice = np.s_[:]  # this will include all the trigger data

    lat = events_ds.ppk_lat.data[events_slice]
    lon = events_ds.ppk_lon.data[events_slice]
    alt = events_ds.ppk_alt.data[events_slice]
    lat_std = events_ds.lat_std.data[events_slice]
    lon_std = events_ds.lon_std.data[events_slice]
    alt_std = events_ds.alt_std.data[events_slice]
    trigger_ts = events_ds.ppk_time.data[events_slice]  # POSIX timestamp

    # Get micasense acquisitions
    tifs = np.array(sorted(yaml_path.glob("**/IMG_*.yaml")))
    mse_ts, valid_mse_ix = get_micasense_timestamps(tifs)

    # ---------------------- #
    #        ALIGNMENT       #
    # ---------------------- #
    mse_ix, trg_ix = alignment_wrapper(
        mse_ts=mse_ts, trigger_ts=trigger_ts, valid_mse_ix=valid_mse_ix, ms_frate=ms_frate
    )

    # ------------------------- #
    #  APPEND REACH M2 TO YAML  #
    # ------------------------- #
    if (mse_ix is not None) and (trg_ix is not None):
        plot_histograms_frate(
            trg_ts=trigger_ts[trg_ix],
            mse_ts=mse_ts[mse_ix],
            opng=yaml_path / f"{yaml_path.parts[-1]}_alignment_plot1.png",
        )

    plot_frate(
            trg_ts=trigger_ts,
            mse_ts=mse_ts,
            trg_ix=trg_ix,
            mse_ix=mse_ix,
            opng=yaml_path / f"{yaml_path.parts[-1]}_alignment_plot2.png",
    )

    if debug:
        plt.show()
        return

    for m_, r_ in zip(mse_ix, trg_ix):
        ms_yaml.add_ppk_to_yaml(
            yml_f=tifs[m_],
            ppk_lat=float(lat[r_]),
            ppk_lon=float(lon[r_]),
            ppk_height=float(alt[r_]),
            ppk_lat_uncert=float(lat_std[r_]),
            ppk_lon_uncert=float(lon_std[r_]),
            ppk_alt_uncert=float(alt_std[r_]),
        )

    return
