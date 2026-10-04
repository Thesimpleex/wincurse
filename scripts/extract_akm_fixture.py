"""Extract the JobStart inputs of the AKM replication package as a small test fixture.

Andrews, Kitagawa and McCloskey (2024) publish their replication package
(Harvard Dataverse, doi:10.7910/DVN/9GAYB5, CC0). Their empirical application
(``Empirical Application/empirical.m``) starts from a 13-vector of estimated effects
``mu_n`` and its covariance ``Sigma_n``. Only these two small arrays are committed
as ``tests/data/akm_jobstart.npz``; the tests never read ``data/raw``.

Usage: ``python scripts/extract_akm_fixture.py`` (downloads about 267 MB on first use).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "akm_replication.zip"
URL = (
    "https://dataverse.harvard.edu/api/access/dataset/:persistentId/"
    "?persistentId=doi:10.7910/DVN/9GAYB5"
)
FOLDER = "JobStart Replication Files/Empirical Application/"
OUT = ROOT / "tests" / "data" / "akm_jobstart.npz"


def main() -> None:
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as zf:
        mu = loadmat(io.BytesIO(zf.read(FOLDER + "mu_js.mat")))["mu_n"].ravel().astype(float)
        sigma = loadmat(io.BytesIO(zf.read(FOLDER + "Sigma_js.mat")))["Sigma_n"].astype(float)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        mu=mu,
        sigma=sigma,
        oos_scale=np.array(1315.0),
        source=np.array("Harvard Dataverse doi:10.7910/DVN/9GAYB5, JobStart Empirical Application"),
    )
    print(f"wrote {OUT} (mu {mu.shape}, sigma {sigma.shape})")


if __name__ == "__main__":
    main()
