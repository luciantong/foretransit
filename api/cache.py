import time

# ─── Simple In-Memory Cache ───────────────────
# Stores station forecast results so repeated clicks
# on the same stop don't re-run MAGI every time.
# TTL = 60 seconds (one data refresh cycle)

CACHE_TTL = 60  # seconds

_store: dict = {}

def get(key: str):
    """
    Returns cached value if it exists and hasn't expired.
    Returns None otherwise.
    """
    entry = _store.get(key)
    if entry is None:
        return None
    if time.time() - entry["ts"] > CACHE_TTL:
        del _store[key]
        return None
    return entry["value"]

def set(key: str, value):
    """
    Stores a value with the current timestamp.
    """
    _store[key] = {
        "value": value,
        "ts":    time.time()
    }

def invalidate(key: str):
    """
    Manually evict a single key (e.g. after a data refresh).
    """
    _store.pop(key, None)

def clear():
    """
    Wipe the entire cache (e.g. on server restart or new GTFS pull).
    """
    _store.clear()

def stats():
    """
    Returns number of cached entries and their keys.
    Useful for debugging.
    """
    return {
        "count": len(_store),
        "keys":  list(_store.keys())
    }