import csv
import io

import pandas as pd


def _detect_separator(sample: bytes) -> str:
    """Return the most likely CSV separator."""
    sniffer = csv.Sniffer()
    text = sample.decode("utf-8", errors="ignore")
    try:
        dialect = sniffer.sniff(text)
        return dialect.delimiter
    except Exception:
        for sep in [",", ";", "\t", "|"]:
            if text.count(sep) > 0:
                return sep
        return ","


def _unique_columns(cols):
    """Return a list of column names that are guaranteed to be unique."""
    seen = {}
    new_cols = []
    for col in cols:
        count = seen.get(col, 0)
        if count == 0:
            new_cols.append(col)
        else:
            new_cols.append(f"{col}_{count}")
        seen[col] = count + 1
    return new_cols


def load_file(file_obj):
    """Read a CSV or Excel file uploaded via Streamlit."""
    raw_bytes = file_obj.read()

    if file_obj.name.lower().endswith(".csv"):
        sample = raw_bytes[:1024]
        sep = _detect_separator(sample)
        content = raw_bytes.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(content), delimiter=sep, header=0)
    else:  # Excel
        df = pd.read_excel(io.BytesIO(raw_bytes))
        sep = None

    # ---- make column names unique ------------------------------------
    df.columns = _unique_columns(df.columns)

    return df, sep
