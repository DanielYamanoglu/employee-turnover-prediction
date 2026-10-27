import pytest

from turnover.config import DATA_PATH
from turnover.preprocess import encode, load_data, split_xy


@pytest.fixture(scope="session")
def raw():
    return load_data(DATA_PATH)


@pytest.fixture(scope="session")
def small(raw):
    """Stratified-ish small sample, no network."""
    return raw.groupby("Attrition", group_keys=False).sample(frac=0.3, random_state=0)
