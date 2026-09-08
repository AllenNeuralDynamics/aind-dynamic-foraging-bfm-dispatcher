"""Pinned source records and checksum-verifying downloads."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    dataset_id: str
    species: str
    title: str
    repository: str
    doi: str
    version: str
    license: str
    url: str
    filename: str
    digest_algorithm: str
    digest: str
    archive_member: str | None = None
    collection_provider: str | None = None


SOURCES: dict[str, Source] = {
    "grossman": Source(
        dataset_id="grossman-bari-cohen-2021",
        species="mouse",
        title="Serotonin neurons modulate learning rate through uncertainty",
        repository="Dryad",
        doi="10.5061/dryad.cz8w9gj4s",
        version="4 (2021-12-27; Dryad resource 156295)",
        license="CC0-1.0",
        url="https://datadryad.org/api/v2/versions/156295/download",
        filename="grossmanBariCohenData.zip",
        digest_algorithm="sha256",
        digest="43a19b171f88430d524557a5c2e13518d6d37ffcfa1dcddc4159b420b1f0485a",
        archive_member="grossmanBariCohenData.zip",
    ),
    "chen": Source(
        dataset_id="chen-et-al-2021",
        species="mouse",
        title="Sex differences in learning from exploration",
        repository="Dryad",
        doi="10.5061/dryad.z612jm6c0",
        version="5 (2022-02-07; Dryad resource 162666)",
        license="CC0-1.0",
        url="https://datadryad.org/api/v2/versions/162666/download",
        filename="cleaned_up_restless_final_data.zip",
        digest_algorithm="sha256",
        digest="90f0f9fa843a16788d0dcd7b857f81db068e8d18b8dd4eabf20ccaee3b67db04",
        archive_member="cleaned_up_restless_final_data.zip",
    ),
    "zid": Source(
        dataset_id="zid-et-al-2026-experiment-1",
        species="human",
        title="Foraging models explain human exploration in uncertain tasks",
        repository="Figshare",
        doi="10.6084/m9.figshare.32193990.v5",
        version="5 (Figshare file 64312005)",
        license="MIT",
        url="https://ndownloader.figshare.com/files/64312005",
        filename="singleiti_202203011023_lightweight.mat",
        digest_algorithm="sha256",
        digest="57ca9c8b7389d28f097c18b683b8ee65eaf34ae128e6825105e20956f5b51a73",
    ),
    "lebedeva": Source(
        dataset_id="lebedeva-et-al-2026",
        species="mouse",
        title="Dorsal prefrontal cortex drives perseverative behavior in mice",
        repository="Figshare",
        doi="10.6084/m9.figshare.31231741.v8",
        version="8 (2026-06-27; Figshare file 65844573)",
        license="CC-BY-4.0",
        url="https://ndownloader.figshare.com/files/65844573",
        filename="figure1.zip",
        digest_algorithm="md5",
        digest="002010655bf2a57fe472426e6e507974",
    ),
    "beron": Source(
        dataset_id="beron-et-al-2022",
        species="mouse",
        title=(
            "Mice exhibit stochastic and efficient action switching during "
            "probabilistic decision making"
        ),
        repository="Harvard Dataverse",
        doi="10.7910/DVN/7E0NM5",
        version="1.1 (Dataverse file 5677011; original format)",
        license="CC0-1.0",
        url=(
            "https://dataverse.harvard.edu/api/access/datafile/5677011"
            "?format=original"
        ),
        filename="bandit_data.csv",
        digest_algorithm="md5",
        digest="d4dae8253c952d2d4f82680da4a9dbf1",
    ),
    "kwak": Source(
        dataset_id="kwak-jung-2019-tab",
        species="mouse",
        title=(
            "Distinct roles of striatal direct and indirect pathways in "
            "value-based decision making"
        ),
        repository="Dryad",
        doi="10.5061/dryad.4c80mn5",
        version="1 (2019-02-27; Dryad resource 28143)",
        license="CC0-1.0",
        url="https://datadryad.org/api/v2/versions/28143/download",
        filename="data_all.zip",
        digest_algorithm="md5",
        digest="00cefa79b2e1b5dbc84e8ba5980fe8e2",
        archive_member="data_all.zip",
    ),
    "miller": Source(
        dataset_id="miller-et-al-2022-tab",
        species="rat",
        title=(
            "From predictive models to cognitive models: Separable behavioral "
            "processes underlying reward learning in the rat"
        ),
        repository="Figshare",
        doi="10.6084/m9.figshare.20449356.v2",
        version="2 (Figshare file 40442660)",
        license="CC-BY-4.0",
        url="https://ndownloader.figshare.com/files/40442660",
        filename="tab_dataset.json",
        digest_algorithm="md5",
        digest="fec60bb91e2e7c297c7cf87508b83065",
    ),
    "findling": Source(
        dataset_id="findling-et-al-volnoise",
        species="human",
        title=(
            "Neural variability in the medial prefrontal cortex contributes "
            "to efficient adaptive behavior"
        ),
        repository="GitHub",
        doi="10.1038/s41467-025-66444-x",
        version="commit ee688535b569a8af8c0531350ec06f34cb989f8e",
        license="MIT",
        url=(
            "https://codeload.github.com/csmfindling/Volnoise/tar.gz/"
            "ee688535b569a8af8c0531350ec06f34cb989f8e"
        ),
        filename="volnoise-ee688535.tar.gz",
        digest_algorithm="sha256",
        digest="ac6d8acb50cdb7b3877f40f57b5e17182d73b4228c78a4ed01b8fa38b19ca296",
    ),
    "tang": Source(
        dataset_id="tang-bartolo-averbeck-2021",
        species="macaque",
        title=(
            "Reward-related choices determine information timing and flow "
            "across macaque lateral prefrontal cortex"
        ),
        repository="Mendeley Data",
        doi="10.17632/m4f38w49fb.1",
        version="1",
        license="CC-BY-4.0",
        url="https://data.mendeley.com/public-api/zip/m4f38w49fb/download/1",
        filename="tang-mendeley-v1.zip",
        digest_algorithm="sha256",
        digest="bd9c9aab489b36518cd1f5c3fa26182f10d20d61bb01bcc5e84fe573eb3b5de5",
    ),
    "alsio": Source(
        dataset_id="alsio-et-al-2019-vpvd-tsvr",
        species="rat",
        title=(
            "Dopamine D2-like receptor stimulation blocks negative feedback "
            "in visual and spatial reversal learning in the rat"
        ),
        repository="Apollo - University of Cambridge Repository",
        doi="10.17863/CAM.80290",
        version="item f86454cf-8f6c-4f5f-8278-d0481a5c5b14",
        license="CC-BY-4.0",
        url=(
            "https://api.repository.cam.ac.uk/server/api/core/bitstreams/"
            "403a40a0-7c20-449c-9440-aae020e96f3b/content"
        ),
        filename="alsio-2019-psychopharm.zip",
        digest_algorithm="sha256",
        digest="25a2788e4580cfa9cce28184c23ecda15c107e7e12f34a562b082acf68a69a57",
    ),
    "eckstein": Source(
        dataset_id="eckstein-et-al-2022",
        species="human",
        title=(
            "Reinforcement learning and Bayesian inference provide complementary "
            "models for the unique advantage of adolescents in stochastic reversal"
        ),
        repository="OSF",
        doi="10.1016/j.dcn.2022.101106",
        version="OSF 7wuh4 (modified 2022-10-11; 306 files)",
        license="not specified on OSF",
        url=(
            "https://api.osf.io/v2/nodes/7wuh4/files/osfstorage/" "?page%5Bsize%5D=100"
        ),
        filename="eckstein-osf-7wuh4.zip",
        digest_algorithm="sha256",
        digest="bf1276533eee0ddb62f2ff0d7a7cc460be3f8e2f1e81940cbb2285bd7543bce4",
        collection_provider="osf",
    ),
    "costa": Source(
        dataset_id="costa-averbeck-2016-stochastic",
        species="macaque",
        title="Amygdala and ventral striatum make distinct contributions to reinforcement learning",
        repository="Zenodo",
        doi="10.5281/zenodo.20086410",
        version="record 20086410 (published 2016-10-06)",
        license="CC-BY-4.0",
        url=(
            "https://zenodo.org/api/records/20086410/files/"
            "stochasticRL_CostaAverbeck2016.mat/content"
        ),
        filename="stochasticRL_CostaAverbeck2016.mat",
        digest_algorithm="md5",
        digest="b388fe98ad9197c696ed87882b8fdbc5",
    ),
    "lopez_mouse": Source(
        dataset_id="lopez-yepez-et-al-2021-mouse",
        species="mouse",
        title=(
            "Choice history effects in mice and humans improve reward harvesting "
            "efficiency"
        ),
        repository="Figshare",
        doi="10.6084/m9.figshare.14540283.v1",
        version="1 (218 files; Figshare article 14540283)",
        license="CC-BY-4.0",
        url="https://api.figshare.com/v2/articles/14540283",
        filename="lopez-mouse-figshare-v1.zip",
        digest_algorithm="sha256",
        digest="f8ec0ce1da0f98f53d170d257ba5267932a9a4e2b1001e9499d5a3c39210ad1b",
        collection_provider="figshare",
    ),
    "hattori": Source(
        dataset_id="hattori-et-al-2023-imaging-mature",
        species="mouse",
        title="Meta-reinforcement learning via orbitofrontal cortex",
        repository="Zenodo",
        doi="10.5281/zenodo.10969434",
        version="v3 (2024-04-13; Zenodo record 10969434)",
        license="not specified on Zenodo",
        url=(
            "https://zenodo.org/records/10969434/files/"
            "Hattori_NatureNeuroscience_Data.zip?download=1"
        ),
        filename="Hattori_NatureNeuroscience_Data.zip",
        digest_algorithm="md5",
        digest="b326372c694d9ec45ab90bfbf3e05846",
    ),
}


def file_digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_source_file(path: Path, source: Source) -> None:
    actual = file_digest(path, source.digest_algorithm)
    if actual != source.digest:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {source.digest}, got {actual}."
        )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/zip, application/octet-stream",
            "User-Agent": "study09-audit/1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)


def _read_with_backoff(url: str) -> bytes:
    for attempt in range(9):
        request = urllib.request.Request(url, headers={"User-Agent": "study09-audit/1"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 8:
                raise
            time.sleep(min(60, 2**attempt))
    raise AssertionError("unreachable")


def _download_osf_collection(url: str, destination: Path) -> None:
    files: list[dict[str, object]] = []
    while url:
        page = json.loads(_read_with_backoff(url))
        for item in page["data"]:
            attributes = item["attributes"]
            files.append(
                {
                    "name": attributes["name"],
                    "size": attributes["size"],
                    "sha256": attributes["extra"]["hashes"]["sha256"],
                    "url": item["links"]["download"],
                }
            )
        url = page["links"].get("next")

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(files, key=lambda row: str(row["name"])):
            payload = _read_with_backoff(str(item["url"]))
            if (
                len(payload) != item["size"]
                or hashlib.sha256(payload).hexdigest() != item["sha256"]
            ):
                raise ValueError(
                    f"Published OSF hash/size mismatch for {item['name']!r}."
                )
            member = zipfile.ZipInfo(str(item["name"]), date_time=(1980, 1, 1, 0, 0, 0))
            member.external_attr = 0o644 << 16
            archive.writestr(member, payload, compress_type=zipfile.ZIP_DEFLATED)


def _download_figshare_collection(url: str, destination: Path) -> None:
    metadata = json.loads(_read_with_backoff(url))
    files = sorted(metadata["files"], key=lambda row: row["name"])
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for item in files:
            payload = _read_with_backoff(item["download_url"])
            if (
                len(payload) != item["size"]
                or hashlib.md5(payload).hexdigest() != item["supplied_md5"]
            ):
                raise ValueError(
                    f"Published Figshare hash/size mismatch for {item['name']!r}."
                )
            member = zipfile.ZipInfo(item["name"], date_time=(1980, 1, 1, 0, 0, 0))
            member.external_attr = 0o644 << 16
            archive.writestr(member, payload, compress_type=zipfile.ZIP_STORED)


def download_source(name: str, raw_root: str | Path, *, force: bool = False) -> Path:
    """Download one pinned source and return the checksum-verified payload path."""
    source = SOURCES[name]
    source_dir = Path(raw_root) / name
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / source.filename
    if destination.exists() and not force:
        verify_source_file(destination, source)
        return destination

    temporary = source_dir / f".{source.filename}.download"
    extracted = source_dir / f".{source.filename}.candidate"
    if temporary.exists():
        temporary.unlink()
    if extracted.exists():
        extracted.unlink()
    candidate = temporary
    try:
        if source.collection_provider == "osf":
            _download_osf_collection(source.url, temporary)
        elif source.collection_provider == "figshare":
            _download_figshare_collection(source.url, temporary)
        else:
            _download(source.url, temporary)
        if source.archive_member is not None:
            candidate = extracted
            with zipfile.ZipFile(temporary) as archive:
                member = archive.getinfo(source.archive_member)
                if Path(member.filename).name != member.filename:
                    raise ValueError(
                        f"Unsafe archive member name: {member.filename!r}."
                    )
                with archive.open(member) as input_stream, extracted.open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(input_stream, output)
        verify_source_file(candidate, source)
        candidate.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        if extracted.exists():
            extracted.unlink()
    return destination
