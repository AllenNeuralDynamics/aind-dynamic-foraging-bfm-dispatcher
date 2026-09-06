"""Source-specific adapters into the shared trial-level bandit schema."""

from __future__ import annotations

import io
import json
import pickle
import re
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    CANONICAL_REQUIRED_COLUMNS,
    interleaved_session_manifest,
    prefix_trial_manifest,
    validate_canonical_table,
)
from .sources import SOURCES, verify_source_file


AdapterResult = tuple[pd.DataFrame, dict[str, object], dict[str, object]]

EXPECTED_AUDITS: dict[str, dict[str, int]] = {
    "grossman": {
        "num_subjects": 48,
        "num_sessions": 754,
        "num_trials": 210159,
        "excluded_trials": 11456,
    },
    "chen": {
        "num_subjects": 32,
        "num_sessions": 256,
        "num_trials": 70778,
        "excluded_trials": 0,
    },
    "zid": {
        "num_subjects": 258,
        "num_sessions": 258,
        "num_trials": 77400,
        "excluded_trials": 6450,
    },
    "lebedeva": {
        "num_subjects": 10,
        "num_sessions": 254,
        "num_trials": 132494,
        "excluded_trials": 0,
    },
    "beron": {
        "num_subjects": 6,
        "num_sessions": 525,
        "num_trials": 378351,
        "excluded_trials": 0,
    },
    "kwak": {
        "num_subjects": 39,
        "num_sessions": 780,
        "num_trials": 121100,
        "excluded_trials": 0,
    },
    "miller": {
        "num_subjects": 20,
        "num_sessions": 1857,
        "num_trials": 1040731,
        "excluded_trials": 6061,
    },
    "findling": {
        "num_subjects": 22,
        "num_sessions": 132,
        "num_trials": 23275,
        "excluded_trials": 485,
    },
    "tang": {
        "num_subjects": 2,
        "num_sessions": 8,
        "num_trials": 15375,
        "excluded_trials": 0,
    },
    "alsio": {
        "num_subjects": 95,
        "num_sessions": 2334,
        "num_trials": 457007,
        "excluded_trials": 16075,
    },
    "eckstein": {
        "num_subjects": 306,
        "num_sessions": 306,
        "num_trials": 40229,
        "excluded_trials": 0,
    },
    "costa": {
        "num_subjects": 11,
        "num_sessions": 245,
        "num_trials": 329840,
        "excluded_trials": 0,
    },
    "lopez_mouse": {
        "num_subjects": 8,
        "num_sessions": 218,
        "num_trials": 147726,
        "excluded_trials": 13125,
    },
}


def _finish(
    rows: list[dict[str, object]],
    *,
    name: str,
    excluded_trials: int,
    split: str,
) -> AdapterResult:
    source = SOURCES[name]
    df = pd.DataFrame.from_records(rows)
    validate_canonical_table(df)
    extra_columns = [
        column for column in df.columns if column not in CANONICAL_REQUIRED_COLUMNS
    ]
    df = df[
        [
            *CANONICAL_REQUIRED_COLUMNS,
            *extra_columns,
        ]
    ]
    if split == "sessions":
        manifest = interleaved_session_manifest(
            df,
            dataset_id=source.dataset_id,
            species=source.species,
        )
    elif split == "trials":
        manifest = prefix_trial_manifest(
            df,
            dataset_id=source.dataset_id,
            species=source.species,
        )
    else:
        raise ValueError(f"Unknown split mode {split!r}.")
    session_counts = df.groupby("subject_id", sort=False)["ses_idx"].nunique()
    trial_counts = df.groupby(["subject_id", "ses_idx"], sort=False).size()
    audit = {
        "dataset_id": source.dataset_id,
        "species": source.species,
        "source_doi": source.doi,
        "source_version": source.version,
        "source_license": source.license,
        "num_subjects": int(df["subject_id"].nunique()),
        "num_sessions": int(df.groupby(["subject_id", "ses_idx"]).ngroups),
        "num_trials": int(len(df)),
        "excluded_trials": int(excluded_trials),
        "min_sessions_per_subject": int(session_counts.min()),
        "max_sessions_per_subject": int(session_counts.max()),
        "min_trials_per_session": int(trial_counts.min()),
        "max_trials_per_session": int(trial_counts.max()),
        "choice_counts": {
            str(key): int(value)
            for key, value in df["animal_response"].value_counts().sort_index().items()
        },
        "reward_counts": {
            str(key): int(value)
            for key, value in df["rewarded"].value_counts().sort_index().items()
        },
        "split_strategy": manifest["split_strategy"],
    }
    mismatches = {
        field: {"expected": expected, "actual": audit[field]}
        for field, expected in EXPECTED_AUDITS[name].items()
        if audit[field] != expected
    }
    if mismatches:
        raise ValueError(
            f"{name} source audit did not match the pinned release: {mismatches}."
        )
    return df, manifest, audit


def adapt_grossman(path: str | Path) -> AdapterResult:
    """Adapt the 48-mouse dynamic-foraging behavior cohort from MATLAB."""
    source = SOURCES["grossman"]
    path = Path(path)
    verify_source_file(path, source)
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The Grossman adapter requires scipy.") from exc

    with zipfile.ZipFile(path) as archive:
        mat_bytes = archive.read("data/data.mat")
    behavior = loadmat(io.BytesIO(mat_bytes), simplify_cells=True)["data"][
        "dynamicForaging"
    ]["behavior"]

    rows: list[dict[str, object]] = []
    excluded_trials = 0
    for subject_id in sorted(behavior):
        for session_id in sorted(behavior[subject_id]):
            canonical_trial = 0
            for source_trial, record in enumerate(behavior[subject_id][session_id]):
                if record.get("trialType") != "CSplus":
                    excluded_trials += 1
                    continue
                left_observed = pd.notna(record.get("rewardL"))
                right_observed = pd.notna(record.get("rewardR"))
                if left_observed == right_observed:
                    excluded_trials += 1
                    continue
                choice = 0 if left_observed else 1
                reward = record["rewardL"] if left_observed else record["rewardR"]
                if reward not in (0, 1):
                    excluded_trials += 1
                    continue
                row: dict[str, object] = {
                    "subject_id": subject_id,
                    "ses_idx": session_id,
                    "trial": canonical_trial,
                    "animal_response": choice,
                    "rewarded": int(reward),
                    "earned_reward": int(reward),
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "source_trial": source_trial,
                    "source_trial_type": record["trialType"],
                }
                if "rewardProbL" in record and "rewardProbR" in record:
                    row["reward_probability_arm_0"] = (
                        float(record["rewardProbL"]) / 100.0
                    )
                    row["reward_probability_arm_1"] = (
                        float(record["rewardProbR"]) / 100.0
                    )
                rows.append(row)
                canonical_trial += 1
    return _finish(
        rows,
        name="grossman",
        excluded_trials=excluded_trials,
        split="sessions",
    )


_CHEN_FILE = re.compile(
    r"^cleaned up restless final data/session(?P<session>\d+)/(?P<subject>\d+)\.csv$"
)


def adapt_chen(path: str | Path) -> AdapterResult:
    """Adapt the 32-mouse, eight-session restless-bandit cohort."""
    source = SOURCES["chen"]
    path = Path(path)
    verify_source_file(path, source)
    chunks: dict[tuple[int, int], pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            match = _CHEN_FILE.match(member.filename)
            if match is None:
                continue
            key = (int(match.group("subject")), int(match.group("session")))
            with archive.open(member) as stream:
                chunks[key] = pd.read_csv(stream)

    rows: list[dict[str, object]] = []
    for (subject, session), source_df in sorted(chunks.items()):
        for trial, record in source_df.reset_index(drop=True).iterrows():
            choice = int(record["choice"]) - 1
            reward = int(record["reward"])
            rows.append(
                {
                    "subject_id": f"mouse-{subject:02d}",
                    "ses_idx": f"session-{session:02d}",
                    "trial": trial,
                    "animal_response": choice,
                    "rewarded": reward,
                    "earned_reward": reward,
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "sex": "male" if subject <= 16 else "female",
                    "source_subject": subject,
                    "source_session": session,
                    "source_trial": int(record.iloc[0]),
                    "reward_probability_arm_0": float(record["left"]),
                    "reward_probability_arm_1": float(record["right"]),
                    "source_hmm_state": int(record["state"]),
                    "response_time_s": float(record["RT"]),
                }
            )
    return _finish(rows, name="chen", excluded_trials=0, split="sessions")


def adapt_zid(path: str | Path) -> AdapterResult:
    """Adapt Experiment 1 from MATLAB, removing its 25-trial practice block."""
    source = SOURCES["zid"]
    path = Path(path)
    verify_source_file(path, source)
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The Zid adapter requires scipy.") from exc

    source_subjects = loadmat(path, simplify_cells=True).get("trials")
    if not isinstance(source_subjects, list):
        raise ValueError("Expected the Zid MATLAB file to contain a trials list.")

    rows: list[dict[str, object]] = []
    practice_trials = 25
    for subject, source_record in enumerate(source_subjects):
        source_trials = source_record.get("trials", [])
        if len(source_trials) != 325:
            raise ValueError(
                f"Expected 325 Zid trials for subject {subject}; got {len(source_trials)}."
            )
        main_trials = [
            record for record in source_trials if int(record["practice"]) == 0
        ]
        if len(main_trials) != 300:
            raise ValueError(
                f"Expected 300 non-practice Zid trials for subject {subject}; "
                f"got {len(main_trials)}."
            )
        for trial, record in enumerate(main_trials):
            choice = int(record["choice"])
            reward = int(record["reward"])
            reward_seed = record["reward_seed"]
            rows.append(
                {
                    "subject_id": f"human-{int(subject):03d}",
                    "ses_idx": "main",
                    "trial": trial,
                    "animal_response": choice,
                    "rewarded": reward,
                    "earned_reward": reward,
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "source_subject": int(subject),
                    "source_trial": int(record["trial_index"]),
                    "reward_probability_arm_0": float(reward_seed[0]),
                    "reward_probability_arm_1": float(reward_seed[1]),
                }
            )
    return _finish(
        rows,
        name="zid",
        excluded_trials=len(source_subjects) * practice_trials,
        split="trials",
    )


_LEBEDEVA_FILE = re.compile(
    r"^figure1/(?P<subject>[^/]+)/(?P<session>\d{4}-\d{2}-\d{2})/"
    r"trials\.choices\.npy$"
)


def adapt_lebedeva(path: str | Path) -> AdapterResult:
    """Adapt the 10-mouse binary reversal-learning behavior archive."""
    source = SOURCES["lebedeva"]
    path = Path(path)
    verify_source_file(path, source)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        sessions: list[tuple[str, str, str]] = []
        for member in archive.infolist():
            match = _LEBEDEVA_FILE.match(member.filename)
            if match is not None:
                sessions.append(
                    (match.group("subject"), match.group("session"), member.filename)
                )

        for subject_id, session_id, choice_member in sorted(sessions):
            prefix = choice_member.removesuffix("trials.choices.npy")
            with archive.open(choice_member) as stream:
                choices = np.load(io.BytesIO(stream.read())).reshape(-1)
            with archive.open(f"{prefix}trials.feedback.npy") as stream:
                feedback = np.load(io.BytesIO(stream.read())).reshape(-1)
            with archive.open(f"{prefix}trials.correct_answers.npy") as stream:
                correct_answers = np.load(io.BytesIO(stream.read())).reshape(-1)
            if not (len(choices) == len(feedback) == len(correct_answers)):
                raise ValueError(
                    f"Lebedeva session lengths disagree for {subject_id}/{session_id}."
                )
            if not set(np.unique(choices)).issubset({-1, 1}):
                raise ValueError(f"Unexpected choices in {subject_id}/{session_id}.")
            if not set(np.unique(feedback)).issubset({0, 1}):
                raise ValueError(f"Unexpected feedback in {subject_id}/{session_id}.")
            if not set(np.unique(correct_answers)).issubset({-1, 1}):
                raise ValueError(
                    f"Unexpected correct sides in {subject_id}/{session_id}."
                )

            for trial, (choice, reward, correct) in enumerate(
                zip(choices, feedback, correct_answers, strict=True)
            ):
                choice_binary = int(choice == 1)
                reward_binary = int(reward)
                correct_binary = int(correct == 1)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "ses_idx": session_id,
                        "trial": trial,
                        "animal_response": choice_binary,
                        "rewarded": reward_binary,
                        "earned_reward": reward_binary,
                        "dataset_id": source.dataset_id,
                        "species": source.species,
                        "source_choice": int(choice),
                        "source_feedback": int(reward),
                        "source_correct_answer": int(correct),
                        "correct_arm": correct_binary,
                        "reward_probability_arm_0": 0.8 if correct_binary == 0 else 0.2,
                        "reward_probability_arm_1": 0.8 if correct_binary == 1 else 0.2,
                    }
                )
    return _finish(rows, name="lebedeva", excluded_trials=0, split="sessions")


def adapt_beron(path: str | Path) -> AdapterResult:
    """Adapt the clean six-mouse probabilistic reversal-learning table."""
    source = SOURCES["beron"]
    path = Path(path)
    verify_source_file(path, source)
    source_df = pd.read_csv(path)
    required = {
        "Trial",
        "blockTrial",
        "Decision",
        "Reward",
        "Condition",
        "Target",
        "blockLength",
        "Session",
        "Mouse",
    }
    missing = sorted(required - set(source_df.columns))
    if missing:
        raise ValueError(f"Beron source is missing columns: {missing}.")

    rows: list[dict[str, object]] = []
    for (mouse, source_session), session_df in source_df.groupby(
        ["Mouse", "Session"], sort=False
    ):
        session_match = re.fullmatch(r"m\d+_(\d+)", str(source_session))
        if session_match is None:
            raise ValueError(f"Unexpected Beron session ID {source_session!r}.")
        session_id = f"session-{int(session_match.group(1)):03d}"
        for trial, (_, record) in enumerate(session_df.iterrows()):
            choice = int(record["Decision"])
            reward = int(record["Reward"])
            target = int(record["Target"])
            high, low = (float(value) / 100 for value in record["Condition"].split("-"))
            rows.append(
                {
                    "subject_id": str(mouse),
                    "ses_idx": session_id,
                    "trial": trial,
                    "animal_response": choice,
                    "rewarded": reward,
                    "earned_reward": reward,
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "source_session": str(source_session),
                    "source_trial": int(record["Trial"]),
                    "block_trial": int(record["blockTrial"]),
                    "block_length": int(record["blockLength"]),
                    "task_condition": str(record["Condition"]),
                    "correct_arm": target,
                    "reward_probability_arm_0": high if target == 0 else low,
                    "reward_probability_arm_1": high if target == 1 else low,
                }
            )
    return _finish(rows, name="beron", excluded_trials=0, split="sessions")


def adapt_kwak(path: str | Path) -> AdapterResult:
    """Adapt all D1/D2 CNO/DMSO dynamic two-arm sessions from Kwak."""
    source = SOURCES["kwak"]
    path = Path(path)
    verify_source_file(path, source)
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The Kwak adapter requires scipy.") from exc

    with zipfile.ZipFile(path) as archive:
        source_data = loadmat(
            io.BytesIO(archive.read("TAB_data.mat")), simplify_cells=True
        )
    behavior_groups = source_data["behavior"]
    condition_groups = source_data["condition"]
    time_groups = source_data["time"]
    group_metadata = (
        ("d1", "cno"),
        ("d1", "dmso"),
        ("d2", "cno"),
        ("d2", "dmso"),
    )

    rows: list[dict[str, object]] = []
    for group_index, (receptor, treatment) in enumerate(group_metadata):
        behavior = np.asarray(behavior_groups[group_index], dtype=object)
        conditions = np.asarray(condition_groups[group_index], dtype=object)
        times = np.asarray(time_groups[group_index], dtype=object)
        if not (behavior.shape == conditions.shape == times.shape):
            raise ValueError(f"Kwak group {group_index} arrays disagree in shape.")
        for session_index in range(behavior.shape[0]):
            for subject_index in range(behavior.shape[1]):
                trials = np.asarray(behavior[session_index, subject_index])
                probabilities = np.asarray(conditions[session_index, subject_index])
                timestamps = np.asarray(times[session_index, subject_index])
                if not (
                    trials.shape == probabilities.shape == timestamps.shape
                    and trials.ndim == 2
                    and trials.shape[1] == 2
                ):
                    raise ValueError(
                        f"Kwak session arrays disagree for group {group_index}, "
                        f"session {session_index}, subject {subject_index}."
                    )
                subject_id = f"{receptor}-mouse-{subject_index + 1:02d}"
                session_id = f"session-{session_index + 1:02d}-{treatment}"
                for trial, (record, probability, timestamp) in enumerate(
                    zip(trials, probabilities, timestamps, strict=True)
                ):
                    source_choice, reward = (int(value) for value in record)
                    choice = 1 - source_choice
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "ses_idx": session_id,
                            "trial": trial,
                            "animal_response": choice,
                            "rewarded": reward,
                            "earned_reward": reward,
                            "dataset_id": source.dataset_id,
                            "species": source.species,
                            "receptor_group": receptor,
                            "treatment": treatment,
                            "source_group": group_index + 1,
                            "source_session": session_index + 1,
                            "source_subject": subject_index + 1,
                            "source_trial": trial,
                            "source_choice": source_choice,
                            "reward_probability_arm_0": float(probability[0]),
                            "reward_probability_arm_1": float(probability[1]),
                            "center_poke_time_ms": int(timestamp[0]),
                            "choice_poke_time_ms": int(timestamp[1]),
                        }
                    )
    return _finish(rows, name="kwak", excluded_trials=0, split="sessions")


def adapt_miller(path: str | Path) -> AdapterResult:
    """Adapt the complete 20-rat JSON cohort, excluding explicit invalid choices."""
    source = SOURCES["miller"]
    path = Path(path)
    verify_source_file(path, source)
    source_subjects = json.loads(path.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    excluded_trials = 0
    for subject in source_subjects:
        subject_id = str(subject["ratname"])
        sides = subject["sides"]
        fields = (
            subject["rewards"],
            subject["left_prob1"],
            subject["right_prob1"],
            subject["new_sess"],
        )
        if any(len(field) != len(sides) for field in fields):
            raise ValueError(f"Miller source fields disagree for {subject_id}.")
        source_session = -1
        canonical_trial = 0
        for source_trial, (side, reward, p_left, p_right, new_session) in enumerate(
            zip(sides, *fields, strict=True)
        ):
            if int(new_session) == 1:
                source_session += 1
                canonical_trial = 0
            if source_session < 0:
                raise ValueError(f"Miller {subject_id} does not start with new_sess=1.")
            if side == "v":
                excluded_trials += 1
                continue
            if side not in {"l", "r"}:
                raise ValueError(f"Unexpected Miller side {side!r} for {subject_id}.")
            reward_binary = int(reward)
            rows.append(
                {
                    "subject_id": subject_id,
                    "ses_idx": f"session-{source_session + 1:03d}",
                    "trial": canonical_trial,
                    "animal_response": int(side == "r"),
                    "rewarded": reward_binary,
                    "earned_reward": reward_binary,
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "source_trial": source_trial,
                    "source_side": side,
                    "reward_probability_arm_0": float(p_left),
                    "reward_probability_arm_1": float(p_right),
                }
            )
            canonical_trial += 1
    return _finish(
        rows,
        name="miller",
        excluded_trials=excluded_trials,
        split="sessions",
    )


_FINDLING_FILE = re.compile(
    r"^[^/]+/main/data/python/td_volnoise_subj_(?P<subject>\d+)_"
    r"run_(?P<run>\d+)_session_(?P<session>\d+)\.pkl$"
)
_FINDLING_RETAINED_SUBJECTS = {
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
}


def adapt_findling(path: str | Path) -> AdapterResult:
    """Adapt every released Volnoise subject using its checksum-pinned pickles."""
    source = SOURCES["findling"]
    path = Path(path)
    verify_source_file(path, source)

    rows: list[dict[str, object]] = []
    excluded_trials = 0
    with tarfile.open(path, "r:gz") as archive:
        members: list[tuple[int, int, int, tarfile.TarInfo]] = []
        for member in archive.getmembers():
            match = _FINDLING_FILE.match(member.name)
            if match is not None and member.isfile():
                members.append(
                    (
                        int(match.group("subject")),
                        int(match.group("run")),
                        int(match.group("session")),
                        member,
                    )
                )
        for subject, run, session, member in sorted(members):
            if subject not in _FINDLING_RETAINED_SUBJECTS:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Could not read Findling member {member.name!r}.")
            # The official release is pickle-only. The immutable archive checksum above
            # is verified before deserializing this trusted publication artifact.
            record = pickle.loads(stream.read(), encoding="latin1")
            choices = np.asarray(record["A_chosen"]).reshape(-1)
            rewards = np.asarray(record["reward"]).reshape(-1)
            states = np.asarray(record["Z"]).reshape(-1)
            volatility = np.asarray(record["tau"]).reshape(-1)
            if not (len(choices) == len(rewards) == len(states) == len(volatility)):
                raise ValueError(
                    f"Findling arrays disagree for subject {subject}, run {run}, "
                    f"session {session}."
                )
            excluded_trials += int(record.get("nb_timeout", 0))
            for trial, (choice, reward, state, tau) in enumerate(
                zip(choices, rewards, states, volatility, strict=True)
            ):
                reward_binary = int(reward)
                rows.append(
                    {
                        "subject_id": f"human-{subject:02d}",
                        "ses_idx": f"session-{session + 1:02d}-run-{run + 1:02d}",
                        "trial": trial,
                        "animal_response": int(choice),
                        "rewarded": reward_binary,
                        "earned_reward": reward_binary,
                        "dataset_id": source.dataset_id,
                        "species": source.species,
                        "source_subject": subject,
                        "source_run": run,
                        "source_session": session,
                        "source_trial": trial,
                        "latent_state": int(state),
                        "volatility": float(tau),
                    }
                )
    result = _finish(
        rows,
        name="findling",
        excluded_trials=excluded_trials,
        split="sessions",
    )
    result[2].update(
        {
            "paper_analytic_subject_ids": sorted(_FINDLING_RETAINED_SUBJECTS),
            "excluded_subject_ids": [6, 7, 12],
            "inclusion_rule": (
                "The 22-subject analytic cohort is verified by exact agreement "
                "between per-subject performance in the 2025 Source Data workbook "
                "and released trial files; subjects 6 and 12 have incomplete runs, "
                "and subject 7 is the paper's below-criterion exclusion."
            ),
        }
    )
    return result


_TANG_FILE = re.compile(
    r"^Neurophysiology/(?P<subject>[Vw])(?P<date>\d{8})_neurons\.mat$"
)


def _load_tang_behavior(payload: bytes) -> dict[str, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The Tang adapter requires scipy.") from exc

    try:
        return loadmat(io.BytesIO(payload), simplify_cells=True)["beh"]
    except NotImplementedError:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("The Tang MATLAB 7.3 sessions require h5py.") from exc
        with h5py.File(io.BytesIO(payload), "r") as source:
            return {
                key: np.asarray(source["beh"][key]).reshape(-1) for key in source["beh"]
            }


def adapt_tang(path: str | Path) -> AdapterResult:
    """Adapt all eight released two-arm sessions from the two macaques."""
    source = SOURCES["tang"]
    path = Path(path)
    verify_source_file(path, source)

    rows: list[dict[str, object]] = []
    subject_names = {"V": "voltaire", "w": "waldo"}
    with zipfile.ZipFile(path) as archive:
        sessions = []
        for member in archive.infolist():
            match = _TANG_FILE.match(member.filename)
            if match is not None:
                sessions.append((match.group("subject"), match.group("date"), member))
        for subject_code, session_date, member in sorted(sessions):
            behavior = _load_tang_behavior(archive.read(member))
            fields = {
                name: np.asarray(behavior[name]).reshape(-1)
                for name in (
                    "trialDirection",
                    "reward",
                    "blockType",
                    "blockIndex",
                    "trialObject",
                    "trialValue",
                    "optimal",
                )
            }
            lengths = {len(values) for values in fields.values()}
            if len(lengths) != 1:
                raise ValueError(f"Tang fields disagree for {member.filename!r}.")
            for name in ("trialDirection", "reward"):
                if not set(np.unique(fields[name])).issubset({0, 1}):
                    raise ValueError(
                        f"Tang {name} is not binary in {member.filename!r}."
                    )
            for trial, values in enumerate(zip(*fields.values(), strict=True)):
                choice, reward, block_type, block_index, obj, value, optimal = values
                reward_binary = int(reward)
                rows.append(
                    {
                        "subject_id": subject_names[subject_code],
                        "ses_idx": session_date,
                        "trial": trial,
                        "animal_response": int(choice),
                        "rewarded": reward_binary,
                        "earned_reward": reward_binary,
                        "dataset_id": source.dataset_id,
                        "species": source.species,
                        "source_trial": trial,
                        "block_type": int(block_type),
                        "block_index": int(block_index),
                        "trial_object": int(obj),
                        "trial_value": float(value),
                        "source_optimal": int(optimal),
                    }
                )
    return _finish(rows, name="tang", excluded_trials=0, split="sessions")


_ALSIO_COHORTS = (
    (
        "skf-vpvd",
        "Alsio_2019_SKF81297_VPVD_trials.csv",
        "Subject",
        "StimChosenPosition",
    ),
    (
        "quin-vpvd",
        "Alsio_2019_quinpirole_VPVD_trials.csv",
        "Subject",
        "StimChosenPosition",
    ),
    (
        "raclo-tsvr",
        "Alsio_2019_Raclo_SCH39166_TSVR_trials.csv",
        "ID",
        "selected_location",
    ),
    (
        "raclo-vpvd",
        "Alsio_2019_raclopride_SCH39166_VPVD_trials.csv",
        "Subject",
        "StimChosenPosition",
    ),
)


def adapt_alsio(path: str | Path) -> AdapterResult:
    """Adapt the complete VPVD/TSVR cohorts with chronological session IDs."""
    source = SOURCES["alsio"]
    path = Path(path)
    verify_source_file(path, source)

    rows: list[dict[str, object]] = []
    excluded_trials = 0
    with zipfile.ZipFile(path) as archive:
        for task, filename, subject_column, choice_column in _ALSIO_COHORTS:
            source_df = pd.read_csv(archive.open(filename))
            for (subject, session_code), session_df in source_df.groupby(
                [subject_column, "session_code"], sort=False
            ):
                canonical_trial = 0
                for _, record in session_df.iterrows():
                    source_choice = record[choice_column]
                    if pd.isna(source_choice):
                        excluded_trials += 1
                        continue
                    if choice_column == "StimChosenPosition":
                        if source_choice not in {"left", "right"}:
                            raise ValueError(
                                f"Unexpected Alsiö choice {source_choice!r}."
                            )
                        choice = int(source_choice == "right")
                    else:
                        if int(source_choice) not in {-1, 1}:
                            raise ValueError(
                                f"Unexpected Alsiö choice {source_choice!r}."
                            )
                        choice = int(int(source_choice) == 1)
                    reward = int(record["outcome"])
                    if reward not in {0, 1}:
                        raise ValueError(f"Unexpected Alsiö outcome {reward!r}.")
                    row: dict[str, object] = {
                        "subject_id": str(subject),
                        "ses_idx": f"session-{int(session_code):03d}-{task}",
                        "trial": canonical_trial,
                        "animal_response": choice,
                        "rewarded": reward,
                        "earned_reward": reward,
                        "dataset_id": source.dataset_id,
                        "species": source.species,
                        "task_cohort": task,
                        "source_session": int(session_code),
                        "source_trial": int(record["trial_within_session"]),
                    }
                    if "IsProbe" in record:
                        row["is_probe"] = int(record["IsProbe"])
                        row["chosen_stimulus"] = str(record["StimChosen"])
                    else:
                        row["chosen_stimulus"] = str(record["selected_stimulus"])
                        row["drug_condition"] = str(record["condition"])
                    rows.append(row)
                    canonical_trial += 1
    result = _finish(
        rows,
        name="alsio",
        excluded_trials=excluded_trials,
        split="sessions",
    )
    result[2].update(
        {
            "included_cohorts": "II-V: complete VPVD and TSVR trial releases",
            "excluded_cohort": (
                "Cohort VI PRL: the release gives dose and reversal labels but "
                "does not identify chronological real sessions, so schema v1 "
                "cannot be constructed without inventing an order"
            ),
            "inclusion_rule": (
                "Retain every subject and every valid left/right response in the "
                "complete VPVD and TSVR cohorts; exclude only explicit no-response rows"
            ),
        }
    )
    return result


_ECKSTEIN_FILE = re.compile(r"^PS_(?P<subject>\d+)\.csv$")


def adapt_eckstein(path: str | Path) -> AdapterResult:
    """Adapt every released one-session stochastic-reversal participant."""
    source = SOURCES["eckstein"]
    path = Path(path)
    verify_source_file(path, source)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        members = []
        for member in archive.infolist():
            match = _ECKSTEIN_FILE.match(member.filename)
            if match is not None:
                members.append((int(match.group("subject")), member))
        for subject, member in sorted(members):
            source_df = pd.read_csv(archive.open(member))
            if (
                source_df["sID"].nunique() != 1
                or int(source_df["sID"].iloc[0]) != subject
            ):
                raise ValueError(
                    f"Eckstein subject ID mismatch in {member.filename!r}."
                )
            for trial, (_, record) in enumerate(source_df.iterrows()):
                choice = int(record["selected_box"])
                reward = int(record["reward"])
                rows.append(
                    {
                        "subject_id": f"human-{subject:03d}",
                        "ses_idx": "main",
                        "trial": trial,
                        "animal_response": choice,
                        "rewarded": reward,
                        "earned_reward": reward,
                        "dataset_id": source.dataset_id,
                        "species": source.species,
                        "source_trial": int(record["TrialID"]),
                        "correct_arm": int(record["correct_box"]),
                        "reward_version": int(record["rewardversion"]),
                        "block": int(record["block"]),
                        "trials_since_switch": float(record["trialsinceswitch"]),
                        "response_time_ms": float(record["RT"]),
                    }
                )
    result = _finish(rows, name="eckstein", excluded_trials=0, split="trials")
    result[2].update(
        {
            "inclusion_rule": (
                "Retain all 306 public participant files: every file contains one "
                "binary-choice, binary-reward session with at least 101 trials"
            ),
            "paper_reported_analytic_subjects": 291,
        }
    )
    return result


def _costa_session_id(source_date: int) -> str:
    if source_date < 1_000_000:
        value = f"{source_date:06d}"
        year = 2000 + int(value[4:])
    else:
        value = f"{source_date:08d}"
        year = int(value[4:])
    timestamp = pd.Timestamp(year=year, month=int(value[:2]), day=int(value[2:4]))
    return timestamp.strftime("%Y-%m-%d")


def adapt_costa(path: str | Path) -> AdapterResult:
    """Adapt the complete stochastic two-shape macaque cohort."""
    source = SOURCES["costa"]
    path = Path(path)
    verify_source_file(path, source)
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The Costa adapter requires scipy.") from exc

    source_data = np.asarray(loadmat(path)["stochasticRL"])
    rows: list[dict[str, object]] = []
    for (subject, source_date), session_rows in pd.DataFrame(source_data).groupby(
        [0, 1], sort=False
    ):
        for trial, (_, record) in enumerate(session_rows.iterrows()):
            source_choice = int(record.iloc[13])
            if source_choice not in {10, 20}:
                raise ValueError(f"Unexpected Costa chosen shape {source_choice!r}.")
            reward = int(record.iloc[8])
            rows.append(
                {
                    "subject_id": f"macaque-{int(subject):02d}",
                    "ses_idx": _costa_session_id(int(source_date)),
                    "trial": trial,
                    "animal_response": int(source_choice == 20),
                    "rewarded": reward,
                    "earned_reward": reward,
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "source_trial": int(record.iloc[2]),
                    "source_block": int(record.iloc[3]),
                    "reward_schedule_code": int(record.iloc[11]),
                    "response_time_ms": int(record.iloc[12]),
                    "source_chosen_shape": source_choice,
                    "source_chosen_color": int(record.iloc[14]),
                    "lesion_group": int(record.iloc[15]),
                }
            )
    return _finish(rows, name="costa", excluded_trials=0, split="sessions")


_LOPEZ_MOUSE_FILE = re.compile(
    r"^(?P<file_index>\d+)_(?P<subject>.+?)_FreeChoice.*_(?P<month>[A-Z][a-z]{2})"
    r"(?P<day>\d{2})_(?P<year>\d{4})_Session(?P<session>\d+)\.mat$"
)
_LOPEZ_LEFT_STATES = (
    "LeftReward",
    "LeftRewardDelay",
    "NoRewardLeft",
    "LeftInactivated",
)
_LOPEZ_RIGHT_STATES = (
    "RightReward",
    "RightRewardDelay",
    "NoRewardRight",
    "RightInactivated",
)


def _state_happened(states: dict[str, object], name: str) -> bool:
    values = np.asarray(states.get(name, np.nan), dtype=float).reshape(-1)
    return bool(np.isfinite(values).any())


def adapt_lopez_mouse(path: str | Path) -> AdapterResult:
    """Adapt all released mouse sessions, excluding timeout/no-choice trials."""
    source = SOURCES["lopez_mouse"]
    path = Path(path)
    verify_source_file(path, source)
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The López-Yépez mouse adapter requires scipy.") from exc

    rows: list[dict[str, object]] = []
    excluded_trials = 0
    with zipfile.ZipFile(path) as archive:
        sessions = []
        for member in archive.infolist():
            match = _LOPEZ_MOUSE_FILE.match(member.filename)
            if match is not None:
                session_date = pd.to_datetime(
                    f"{match.group('month')}{match.group('day')}{match.group('year')}",
                    format="%b%d%Y",
                ).strftime("%Y-%m-%d")
                sessions.append(
                    (
                        match.group("subject"),
                        int(match.group("file_index")),
                        session_date,
                        int(match.group("session")),
                        member,
                    )
                )
        for subject, file_index, session_date, source_session, member in sorted(
            sessions
        ):
            session_data = loadmat(
                io.BytesIO(archive.read(member)), simplify_cells=True
            )["SessionData"]
            source_trials = session_data["RawEvents"]["Trial"]
            if not isinstance(source_trials, list):
                source_trials = [source_trials]
            canonical_trial = 0
            for source_trial, record in enumerate(source_trials):
                states = record["States"]
                chose_left = any(
                    _state_happened(states, name) for name in _LOPEZ_LEFT_STATES
                )
                chose_right = any(
                    _state_happened(states, name) for name in _LOPEZ_RIGHT_STATES
                )
                if chose_left == chose_right:
                    excluded_trials += 1
                    continue
                reward = int(
                    _state_happened(states, "LeftReward")
                    or _state_happened(states, "RightReward")
                )
                rows.append(
                    {
                        "subject_id": subject,
                        "ses_idx": (
                            f"session-{file_index:03d}-{session_date}-"
                            f"run-{source_session:02d}"
                        ),
                        "trial": canonical_trial,
                        "animal_response": int(chose_right),
                        "rewarded": reward,
                        "earned_reward": reward,
                        "dataset_id": source.dataset_id,
                        "species": source.species,
                        "source_trial": source_trial,
                    }
                )
                canonical_trial += 1
    return _finish(
        rows,
        name="lopez_mouse",
        excluded_trials=excluded_trials,
        split="sessions",
    )


ADAPTERS: dict[str, Callable[[str | Path], AdapterResult]] = {
    "grossman": adapt_grossman,
    "chen": adapt_chen,
    "zid": adapt_zid,
    "lebedeva": adapt_lebedeva,
    "beron": adapt_beron,
    "kwak": adapt_kwak,
    "miller": adapt_miller,
    "findling": adapt_findling,
    "tang": adapt_tang,
    "alsio": adapt_alsio,
    "eckstein": adapt_eckstein,
    "costa": adapt_costa,
    "lopez_mouse": adapt_lopez_mouse,
}


def build_dataset(name: str, source_path: str | Path) -> AdapterResult:
    return ADAPTERS[name](source_path)
