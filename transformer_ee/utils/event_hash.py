"""Deterministic per-event uniform numbers, used to assign and subsample dataset splits.

An event's (run, subrun, cycle, evt) is packed into one 64-bit key and mixed with SplitMix64, so the
same event always gets the same number for a given seed, independent of file order.
"""
import numpy as np


def splitmix64(x):
    """SplitMix64 finaliser on a uint64 array (wraps modulo 2^64)."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def packed_event_key(run, subrun, cycle, evt):
    """One uint64 per event: ((run * 1000 + subrun) * 1000 + cycle) * 10^6 + evt."""
    return ((run * np.uint64(1000) + subrun) * np.uint64(1000) + cycle) * np.uint64(10**6) + evt


def event_uniform(run, subrun, cycle, evt, seed):
    """A number in [0, 1) per event, from its packed key and the seed."""
    key = packed_event_key(run, subrun, cycle, evt)
    with np.errstate(over="ignore"):
        return splitmix64(key ^ np.uint64(seed)).astype(np.float64) / 2.0**64
