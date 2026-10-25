"""Import a sensor-log CSV into one master DataFrame, grouped by sensor.

Expects a CSV with a header row:
    id,sensorid,devicetype,jsondata,status,processing_errors,createdate,processdate

`jsondata` is a JSON array; each element looks like:
    {"values": [{"time": 1702273965, "value": 603}],
     "variable": {"id": 67, "name": "Carbon dioxide", "unit": "ppm", ...}}

Pipeline:
    1. Read the CSV once and optionally filter by sensor id.
    2. Split rows by sensor and parse each sensor on its own: JSON -> one float
       column per variable, drop undated rows, sort by time, remove sporadic runs.
    3. Recombine into one table `data` with columns id, sensorid, timestamp,
       created_at, processed_at and one float column per variable.

`data` is grouped by sensor (one contiguous, time-sorted block per sensor), so it
is not sorted by time across sensors. Rows with no sensor id form a last block.
There is no deduplication: exact repeats are kept unless they form a sporadic run.

Helpers:
    SENSOR_SLICES: {sensorid: slice into the master table}.
    get_sensor / iter_sensors / sensor_summary / save_grouped / list_sensors.
    VARIABLES: {column name: (display name, unit)}; metadata only.

Sensor filter (applied before any JSON is parsed; ids are matched as text):
    SENSOR_IDS_INCLUDE: Only load these sensors, e.g. ["A12", "A13"]; None = all.
    SENSOR_IDS_EXCLUDE: Skip these sensors, e.g. ["A14"].
"""

import json

import numpy as np
import pandas as pd

PATH = "5EnvSensor_MayToDec2024_180kRows.csv"  # Path to the CSV file; change as needed.
SENSOR_IDS_INCLUDE = None   # only load these sensor ids, e.g. ["A12", "A13"]; None = all sensors
SENSOR_IDS_EXCLUDE = None   # skip these sensor ids, e.g. ["A14"]
LONE_GAP_DAYS = 1       # readings further apart than this (days) start a new "run"
MIN_RUN_DISTINCT = 20   # runs with fewer DISTINCT readings than this are sporadic and removed
                        # (exact repeats of time AND values count once, so bursts of repeats are caught)

# (column name in `data`, exact JSON "name" field, unit)
VARIABLE_SPEC = (
    ("battery",      "Battery",                          "V"),
    ("co2",          "Carbon dioxide",                   "ppm"),
    ("humidity",     "Humidity",                         "HR%"),
    ("formaldehyde", "Formaldehyde",                     "\u00b5g/m3"),
    ("pressure",     "Pressure",                         "mb"),
    ("temperature",  "Temperature",                      "\u00b0C"),
    ("voc",          "Light Volatile Organic Compounds", "ppb"),
    ("pm1",          "Particulate matter 1",             "\u00b5g/m3"),
    ("pm4",          "Particulate matter 4",             "\u00b5g/m3"),
    ("pm25",         "Particulate matter 2.5",           "\u00b5g/m3"),
    ("pm10",         "Particulate matter 10",            "\u00b5g/m3"),
)

VALUE_COLUMNS = [key for key, _, _ in VARIABLE_SPEC]
_NAME_TO_KEY = {name: key for key, name, _ in VARIABLE_SPEC}   # JSON name -> column name

# Metadata only: column name -> (display name, unit). Holds no readings.
VARIABLES = {key: (name, unit) for key, name, unit in VARIABLE_SPEC}


def _parse_row(text):
    """Parses one `jsondata` string into a flat dict of readings.

    Variables not listed in VARIABLE_SPEC are ignored.

    Args:
        text: The raw JSON string from the CSV.

    Returns:
        None if the JSON is unreadable (the caller drops the row); otherwise a dict
        like {"co2": 603, ..., "_time": 1702273965}, where `_time` is Unix seconds
        (NaN if absent).
    """
    try:
        readings = json.loads(text)
    except (json.JSONDecodeError, TypeError):  # malformed text, or NaN input
        return None
    if not isinstance(readings, list):
        return {"_time": np.nan}

    record = {}
    row_time = np.nan
    for entry in readings:
        values = entry.get("values", [])
        if not values:
            continue
        point = values[0]  # first/only reading per variable
        if "time" in point:
            row_time = point["time"]  # shared timestamp for the row
        key = _NAME_TO_KEY.get(entry.get("variable", {}).get("name"))
        if key is not None:
            record[key] = point.get("value")
    record["_time"] = row_time
    return record


def _as_id_set(ids):
    """Normalises sensor ids to a set of stripped text ids.

    Args:
        ids: None, a single id, or an iterable of ids.

    Returns:
        A set of text ids (e.g. 5.0 -> "5"), or None if `ids` is None.
    """
    if ids is None:
        return None
    if isinstance(ids, (str, int, np.integer, float, np.floating)):
        ids = [ids]
    cleaned = set()
    for i in ids:
        if isinstance(i, (float, np.floating)) and float(i).is_integer():
            i = int(i)                                   # 5.0 -> "5"
        cleaned.add(str(i).strip())
    return cleaned


def list_sensors(path=PATH):
    """Counts rows per sensor id in the CSV (reads only the sensorid column, so it is fast).

    Args:
        path: Path to the CSV file.

    Returns:
        Series of row counts indexed by sensor id.
    """
    return pd.read_csv(path, dtype=str, usecols=["sensorid"])["sensorid"].value_counts(dropna=False)


def _filter_sensors(raw, include, exclude):
    """Keeps only the rows of the selected sensors; runs before any JSON is parsed.

    Args:
        raw: Raw CSV rows.
        include: Sensor ids to keep, or None for all.
        exclude: Sensor ids to drop, or None for none.

    Returns:
        The filtered rows, with a fresh index.

    Raises:
        ValueError: If the filter matches no rows.
    """
    include, exclude = _as_id_set(include), _as_id_set(exclude)
    if include is None and exclude is None:
        return raw

    # Build the keep mask: in `include` (if given) and not in `exclude`
    text = raw["sensorid"].fillna("")
    keep = np.ones(len(raw), dtype=bool)
    if include is not None:
        keep &= text.isin(include).to_numpy()
    if exclude is not None:
        keep &= ~text.isin(exclude).to_numpy()

    # Warn about requested ids that are not in the file
    present = set(text.unique())
    missing = sorted(((include or set()) | (exclude or set())) - present)
    if missing:
        print(f"[warning] sensor id(s) not found in the file: {missing}")
    if not keep.any():
        raise ValueError("The sensor filter matched no rows. Sensors in the file:\n"
                         + text.value_counts().to_string())
    return raw.loc[keep].reset_index(drop=True)


def _warn_if_constant(label, block):
    """Warns if a sensor repeats one value for a whole variable (possibly stuck or faulty).

    A variable counts as constant if it has at least 2 readings, all equal.

    Args:
        label: Sensor id, used in the message.
        block: The sensor's parsed rows.
    """
    values = block[VALUE_COLUMNS]
    count = values.count()                                   # readings per variable (NaN excluded)
    constant = (count >= 2) & (values.min() == values.max())
    if not constant.any():
        return
    detail = ", ".join(f"{VARIABLES[c][0]} = {values[c].iloc[values[c].first_valid_index()]:g} "
                       f"({count[c]} readings)" for c in constant.index[constant])
    if constant[count > 0].all():
        print(f"[warning] sensor {label}: ALL readings are identical. "
              f"The sensor may be stuck or faulty.")
    else:
        print(f"[warning] sensor {label}: the same value in some readings. {detail}"
              f"The sensor may be stuck or faulty.")


def _parse_sensor(sensor_id, rows):
    """Parses and cleans one sensor's raw CSV rows into a time-sorted block.

    Removes unreadable rows, undated rows and sporadic data: readings are split into
    runs wherever the gap between consecutive readings exceeds LONE_GAP_DAYS, and runs
    with fewer than MIN_RUN_DISTINCT distinct readings are dropped. Exact repeats (same
    time and values) count once; repeats in dense runs are kept.

    Args:
        sensor_id: The sensor's id (NaN for rows with no id).
        rows: That sensor's raw CSV rows.

    Returns:
        DataFrame with one float column per variable plus `_time` (Unix seconds),
        `id`, `sensorid`, `createdate` and `processdate`; None if no rows remain.
    """
    label = "(missing)" if pd.isna(sensor_id) else sensor_id
    rows = rows.reset_index(drop=True)

    # JSON -> flat dicts, one parse per row; unreadable rows dropped.
    parsed = rows["jsondata"].map(_parse_row)
    bad = parsed.isna().to_numpy()
    if bad.any():
        print(f"[skip] sensor {label}: {bad.sum()} row(s) with unreadable jsondata: "
              f"ids {rows.loc[bad, 'id'].tolist()}")
        rows = rows.loc[~bad].reset_index(drop=True)
        parsed = parsed.loc[~bad].reset_index(drop=True)

    # One wide table, one float column per variable (missing -> all-NaN).
    wide = pd.DataFrame(parsed.tolist(), columns=[*VALUE_COLUMNS, "_time"]).astype(float)

    # Which rows survive, and in what (time) order, decided on a bare numpy array.
    t = wide["_time"].to_numpy()                       # Unix seconds, NaN if undated
    undated = np.isnan(t)
    if undated.any():
        print(f"[warning] sensor {label}: {undated.sum()} reading(s) had no timestamp and were removed")

    candidates = np.flatnonzero(~undated)
    order = candidates[np.argsort(t[candidates], kind="stable")]   # time-sorted row positions
    t_sorted = t[order]

    # Split the time-sorted readings into runs at every gap larger than LONE_GAP_DAYS.
    breaks = np.flatnonzero(np.diff(t_sorted) > LONE_GAP_DAYS * 86400.0) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(t_sorted)]))
    sizes = stops - starts                              # readings per run
    if len(t_sorted):
        # Distinct readings per run: exact repeats of time AND every value count only once.
        is_new = ~wide.iloc[order].duplicated().to_numpy()
        distinct = np.add.reduceat(is_new.astype(int), starts)
    else:
        distinct = sizes
    # Runs with too few distinct readings are sporadic; mark every reading in them for removal.
    sporadic = distinct < MIN_RUN_DISTINCT
    lone = np.repeat(sporadic, sizes)
    for a, b, n, d in zip(starts[sporadic], stops[sporadic], sizes[sporadic], distinct[sporadic]):
        if n == 0:
            continue
        first, last = pd.to_datetime([t_sorted[a], t_sorted[b - 1]], unit="s", utc=True)
        print(f"[warning] sensor {label}: removing sporadic run of {n} reading(s), {d} distinct "
              f"({first} to {last}): fewer than {MIN_RUN_DISTINCT}.")
    order = order[~lone]

    if len(order) == 0:
        print(f"[warning] sensor {label}: no rows left after cleaning -- sensor dropped")
        return None

    # Build the final block in time order and attach the identifying columns
    block = wide.iloc[order].reset_index(drop=True)
    _warn_if_constant(label, block)
    block["id"] = rows["id"].to_numpy()[order]
    block["sensorid"] = sensor_id
    block["createdate"] = rows["createdate"].to_numpy()[order]
    block["processdate"] = rows["processdate"].to_numpy()[order]
    return block


def load_sensor_data(path=PATH, sensor_ids_include=SENSOR_IDS_INCLUDE,
                     sensor_ids_exclude=SENSOR_IDS_EXCLUDE):
    """Loads the CSV into the master DataFrame, grouped by sensor.

    Args:
        path: Path to the CSV file.
        sensor_ids_include: Only load these sensor ids; None = all.
        sensor_ids_exclude: Skip these sensor ids; None = none.

    Returns:
        DataFrame with columns id, sensorid, timestamp, created_at, processed_at and one
        float column per variable. Sensors are in id order, time-sorted within each.

    Raises:
        ValueError: If no rows are left after parsing and cleaning.
    """
    # 1. Read only the columns we need (vectorised C-level CSV parsing).
    raw = pd.read_csv(path, dtype=str,
                      usecols=["id", "sensorid", "jsondata", "createdate", "processdate"])
    raw["sensorid"] = raw["sensorid"].str.strip()
    raw = _filter_sensors(raw, sensor_ids_include, sensor_ids_exclude)   # before any JSON parsing

    # 2. Split by sensor id and parse each sensor on its own (rows with no sensor id form a last group).
    blocks = []
    for sensor_id, rows in raw.groupby("sensorid", sort=True, dropna=False):
        block = _parse_sensor(sensor_id, rows)
        if block is not None:
            blocks.append(block)
    del raw
    if not blocks:
        raise ValueError("No rows left after parsing and cleaning. Check the CSV and the sensor filter.")

    # 3. Recombine into ONE master table, grouped by sensor.
    data = pd.concat(blocks, ignore_index=True)
    del blocks

    # 4. Final column layout; dates converted once per column over the whole table.
    data.insert(0, "id", data.pop("id"))
    data.insert(1, "sensorid", data.pop("sensorid").astype("category"))   # few distinct values: compact
    data.insert(2, "timestamp", pd.to_datetime(data.pop("_time"), unit="s", utc=True))
    data.insert(3, "created_at", pd.to_datetime(data.pop("createdate"), utc=True))
    data.insert(4, "processed_at", pd.to_datetime(data.pop("processdate"), utc=True))
    return data


def group_by_sensor(df):
    """Maps each sensor to its block of rows.

    Args:
        df: Table whose sensors form contiguous blocks, as built by load_sensor_data.

    Returns:
        Dict of sensor id -> slice of row positions (views, not copies). Rows with no
        sensor id are keyed "(missing)".

    Raises:
        ValueError: If a sensor's rows are not contiguous.
    """
    codes = df["sensorid"].cat.codes.to_numpy()             # -1 for a missing sensor id
    # A new block starts wherever the sensor code changes
    cuts = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [len(codes)]))
    if len(starts) != len(np.unique(codes)):
        raise ValueError("Sensors are not contiguous in the table; it is not grouped by sensor.")
    categories = df["sensorid"].cat.categories
    return {(categories[codes[a]] if codes[a] >= 0 else "(missing)"): slice(int(a), int(b))
            for a, b in zip(starts, stops)}


def get_sensor(sensor_id):
    """Returns one sensor's time-sorted rows as a view of the master table (no copy).

    Args:
        sensor_id: The sensor id (matched as text).

    Returns:
        DataFrame view of that sensor's rows.

    Raises:
        KeyError: If the sensor id is unknown.
    """
    key = str(sensor_id).strip()
    if key not in SENSOR_SLICES:
        raise KeyError(f"Unknown sensor id {sensor_id!r}. Available: {list(SENSOR_SLICES)}")
    return data.iloc[SENSOR_SLICES[key]]


def iter_sensors():
    """Yields (sensor id, view of that sensor's rows) in table order."""
    for sensor_id, sl in SENSOR_SLICES.items():
        yield sensor_id, data.iloc[sl]


def sensor_summary():
    """Summarises each sensor.

    Returns:
        DataFrame with one row per sensor: row count, first and last timestamp, and the
        number of non-missing readings per variable.
    """
    summary = data.groupby("sensorid", observed=True, dropna=False)["timestamp"].agg(
        rows="size", first="min", last="max")
    counts = data[VALUE_COLUMNS].notna().groupby(data["sensorid"], observed=True, dropna=False).sum()
    summary = summary.join(counts)
    summary.index = summary.index.astype(object).where(summary.index.notna(), "(missing)")
    return summary.rename_axis("sensorid")


# Module-level master table. Everything else should reference this object.
full_data = load_sensor_data()
SENSOR_SLICES = group_by_sensor(full_data)   # {sensorid: slice into `data`}; views, not copies
data = full_data[full_data["sensorid"]=="6012002000326"]   # keep one sensor for modelling

TIME_COL = "timestamp"
VARIABLE_TO_MODEL = "co2"   # column name in `data`; look up metadata in VARIABLES


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30)
