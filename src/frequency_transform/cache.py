#!/usr/bin/env python3
"""
cache.py

Disk-based caching for frequency response data.

Stores computed frequency response DataFrames as CSV files keyed by a
deterministic hash of the input configuration (file list, frequency
parameters, estimation method).  This avoids redundant recomputation
when re-running the same analysis pipeline.
"""

import hashlib
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional


class FrequencyDataCache:
    """Disk-based cache for frequency response data.

    The cache directory defaults to .cache/frequency_data relative to
    the current working directory.  Each entry is a CSV file whose name
    is a truncated SHA-256 hash of the configuration that produced it.
    """

    # Default cache directory (can be overridden at construction time)
    DEFAULT_CACHE_DIR = Path(".cache/frequency_data")

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Key generation                                                     #
    # ------------------------------------------------------------------ #

    def get_cache_key(
        self,
        mat_files: List[str],
        n_files: int,
        time_duration: Optional[float],
        nd: int,
        freq_method: str,
    ) -> str:
        """Generate a deterministic cache key from the analysis configuration.

        Args:
            mat_files: List of .mat file paths used as input.
            n_files: Number of files actually processed (first N).
            time_duration: Time duration limit in seconds (None if unused).
            nd: Number of frequency grid points.
            freq_method: Estimation method name ('frf' or 'fourier').

        Returns:
            A 16-character hex string derived from SHA-256.
        """
        # Resolve and sort paths for determinism
        sorted_files = sorted([str(Path(f).resolve()) for f in mat_files[:n_files]])
        config = json.dumps(
            {
                "mat_files": sorted_files,
                "n_files": n_files,
                "time_duration": time_duration,
                "nd": nd,
                "freq_method": freq_method,
            },
            sort_keys=True,
        )
        return hashlib.sha256(config.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    #  Read / write / invalidate                                          #
    # ------------------------------------------------------------------ #

    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Return cached DataFrame, or None if not found.

        Args:
            cache_key: Key previously returned by get_cache_key().

        Returns:
            pandas DataFrame read from CSV, or None.
        """
        cache_file = self.cache_dir / f"{cache_key}.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file)
        return None

    def put(self, cache_key: str, data: pd.DataFrame) -> Path:
        """Save a DataFrame to the cache.

        Args:
            cache_key: Key previously returned by get_cache_key().
            data: DataFrame to persist.

        Returns:
            Path to the written CSV file.
        """
        cache_file = self.cache_dir / f"{cache_key}.csv"
        data.to_csv(cache_file, index=False)
        return cache_file

    def invalidate(self, cache_key: Optional[str] = None) -> None:
        """Clear a specific cache entry or the entire cache.

        Args:
            cache_key: If provided, remove only that entry.
                       If None, remove all cached CSV files.
        """
        if cache_key is not None:
            f = self.cache_dir / f"{cache_key}.csv"
            if f.exists():
                f.unlink()
        else:
            for f in self.cache_dir.glob("*.csv"):
                f.unlink()
