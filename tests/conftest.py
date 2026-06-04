"""
Shared fixtures for v03 regression tests.

Each test takes a `master` fixture that points to the catalog
under test.  Tests skip cleanly if the artefact isn't built yet.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest

CSV = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star')

# Catalog under test for v03.  When v03 lands, point V03_MASTER at it
# and the same tests run unchanged.
V03_MASTER = CSV / 'v03' / 'master_stars_4way.parquet'

# v01/v02 master, for measuring the bug magnitudes BEFORE v03 fixes.
V01_MASTER = CSV / 'master_stars_4way_with_desi.parquet'


@pytest.fixture(scope='session')
def v03_master():
    """The v03 master catalog under test. Skip if not yet built."""
    if not V03_MASTER.exists():
        pytest.skip(f'v03 master not built yet: {V03_MASTER}')
    return pd.read_parquet(V03_MASTER)


@pytest.fixture(scope='session')
def v01_master():
    """The v01/v02 master, used as the 'before' measurement."""
    if not V01_MASTER.exists():
        pytest.skip(f'v01 master not found: {V01_MASTER}')
    return pd.read_parquet(V01_MASTER)
