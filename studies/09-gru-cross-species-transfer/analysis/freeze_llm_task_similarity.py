"""Freeze the blinded single-judge task-similarity tournament for Study 09.

This is intentionally separate from the report renderer.  The model-authored
judgments are frozen here as curated data; downstream figures never call an LLM.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402


ANNOTATIONS = STUDY / "analysis" / "task_design_annotations.json"
CARDS_OUTPUT = STUDY / "analysis" / "llm_task_cards.json"
JUDGMENTS_OUTPUT = STUDY / "analysis" / "llm_task_similarity_judgments.json"

JUDGE_MODEL = "OpenAI Codex current session (exact deployment ID unavailable)"
PROMPT_VERSION = "study09-task-similarity-v2"

# Opaque IDs were assigned before judging.  Study names and species are absent
# from the card artifact supplied to the judge.
CARD_TO_COHORT = {
    "T01": "eckstein",
    "T02": "grossman",
    "T03": "lopez_mouse",
    "T04": "chen",
    "T05": "costa",
    "T06": "beron",
    "T07": "zid",
    "T08": "hattori",
    "T09": "miller",
    "T10": "alsio",
    "T11": "lebedeva",
    "T12": "findling",
}

AIND_REFERENCE = (
    "Reference family: a subject repeatedly makes a binary left/right spatial action "
    "and receives binary water feedback. Reward probabilities are piecewise constant "
    "within blocks. The source family contains (1) independent, non-baited arms; "
    "(2) independent, baited arms; and (3) coupled, baited arms. The response is made "
    "while head-fixed by licking left or right."
)

CARDS = {
    "T01": (
        "A subject selects a left or right box using a keyboard. Reward feedback is "
        "points or a coin. The two options have complementary probabilities in a "
        "stochastic reversal schedule; contingencies are piecewise constant and not baited."
    ),
    "T02": (
        "A head-fixed subject chooses a left or right lick spout for binary water. "
        "The two reward probabilities are independent, piecewise constant within "
        "blocks, and non-baited."
    ),
    "T03": (
        "A freely moving subject chooses a left or right port for binary water. The two "
        "set reward probabilities are independent and piecewise constant within blocks. "
        "Each arm is baited, so its current reward availability accumulates while unchosen "
        "and resets when collected."
    ),
    "T04": (
        "A freely moving subject nose-pokes one of two touchscreen targets for binary "
        "liquid-food feedback. Each arm probability evolves independently as a restless "
        "random walk; rewards are not baited."
    ),
    "T05": (
        "A head-fixed subject selects one of two visual stimuli by saccade for binary "
        "juice feedback. Stimulus positions are randomized. Complementary 80/20, 70/30, "
        "or 60/40 contingencies reverse between piecewise-constant blocks and are not baited."
    ),
    "T06": (
        "A freely moving subject chooses a left or right port for binary water. A latent "
        "state sets complementary reward probabilities; the state reverses between "
        "piecewise-constant blocks and rewards are not baited."
    ),
    "T07": (
        "A subject chooses between two abstract card decks using computer input and "
        "receives binary point feedback. Each deck probability evolves independently "
        "as a restless random walk; rewards are not baited."
    ),
    "T08": (
        "A head-fixed subject chooses a left or right lick port for binary water. "
        "Complementary probabilities reverse between 60-80-trial blocks; rewards are "
        "baited so an available reward remains until collected."
    ),
    "T09": (
        "A freely moving subject chooses one of two nose ports for binary liquid-food "
        "feedback. Both reward probabilities change continuously and independently; "
        "rewards are not baited."
    ),
    "T10": (
        "A freely moving subject chooses spatial locations or visual stimuli on a "
        "touchscreen for binary food-pellet feedback. Complementary contingencies reverse "
        "when a behavioral criterion is reached rather than after an exogenous block length; "
        "rewards are not baited."
    ),
    "T11": (
        "A head-fixed subject turns a wheel left or right for binary water. Complementary "
        "80/20 contingencies reverse between piecewise-constant blocks and are not baited."
    ),
    "T12": (
        "A subject presses handheld buttons to choose between two visual shapes for binary "
        "monetary feedback. Shape positions are randomized. Complementary 85/15 contingencies "
        "reverse between piecewise-constant blocks and are not baited."
    ),
}

JUDGE_PROMPT = """You are comparing two binary-choice tasks with the AIND reference family.
Choose which candidate is computationally closer to the reference, or choose tie.

Prioritize, in order:
1. reward-schedule dynamics and change-point rule;
2. arm coupling and whether unchosen rewards remain available (baiting);
3. spatial-action versus abstract/stimulus choice semantics;
4. feedback and response implementation only as a secondary tiebreaker.

Do not infer or use species. All actors are described only as subjects. Do not use
study identity, author identity, sample size, neural measurements, model performance,
or embeddings. Judge only the task descriptions. Return A, B, or tie.
"""

# Frozen result authored by the single Codex judge from the cards above.  Equal
# integers are genuine ties.  Smaller integers mean closer to the AIND family.
# This ordinal output is expanded into every unordered pair in both A/B orders.
JUDGE_TIERS = {
    "T02": 0,
    "T08": 0,
    "T03": 1,
    "T11": 2,
    "T06": 3,
    "T01": 4,
    "T04": 5,
    "T09": 5,
    "T05": 6,
    "T12": 7,
    "T10": 8,
    "T07": 9,
}

FORBIDDEN_CARD_TERMS = (
    "mouse",
    "mice",
    "rat",
    "rats",
    "macaque",
    "monkey",
    "human",
    "participant",
    "grossman",
    "chen",
    "zid",
    "lebedeva",
    "beron",
    "miller",
    "findling",
    "alsiö",
    "alsio",
    "eckstein",
    "costa",
    "lópez",
    "lopez",
    "hattori",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text())
    payload.pop("_meta", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _choice(first: str, second: str) -> str:
    if JUDGE_TIERS[first] == JUDGE_TIERS[second]:
        return "tie"
    return "A" if JUDGE_TIERS[first] < JUDGE_TIERS[second] else "B"


def _validate() -> None:
    if set(CARDS) != set(CARD_TO_COHORT) or set(CARDS) != set(JUDGE_TIERS):
        raise AssertionError("card, mapping, and judgment IDs must match")
    combined = " ".join([AIND_REFERENCE, JUDGE_PROMPT, *CARDS.values()]).lower()
    leaked = [
        term
        for term in FORBIDDEN_CARD_TERMS
        if re.search(rf"\b{re.escape(term)}\b", combined)
    ]
    if leaked:
        raise AssertionError(f"species or study identity leaked into judge input: {leaked}")


def main() -> None:
    _validate()
    annotations = json.loads(ANNOTATIONS.read_text())
    valid = {
        name
        for tier in ("primary", "stress_test", "descriptive_only")
        for name in annotations["analysis_tiers"][tier]
    }
    if set(CARD_TO_COHORT.values()) != valid:
        raise AssertionError("judge cohort set must equal the non-quarantined cohort set")

    comparisons = []
    for first, second in itertools.combinations(sorted(CARDS), 2):
        forward = _choice(first, second)
        reverse = _choice(second, first)
        consistent = (
            forward == reverse == "tie"
            or (forward == "A" and reverse == "B")
            or (forward == "B" and reverse == "A")
        )
        comparisons.append(
            {
                "pair_id": f"{first}__{second}",
                "forward": {"order": [first, second], "choice": forward},
                "reverse": {"order": [second, first], "choice": reverse},
                "order_consistent": consistent,
            }
        )

    wandb_groups = json.loads(
        (STUDY / "analysis" / "generalization_drivers_e8.json").read_text()
    )["_meta"]["wandb_groups"]
    cards_output = {
        "_meta": build_meta(
            "analysis/freeze_llm_task_similarity.py", [], study_root=STUDY
        ),
        "input_sha256": {
            "analysis/task_design_annotations.json": _sha256(ANNOTATIONS)
        },
        "prompt_version": PROMPT_VERSION,
        "reference": AIND_REFERENCE,
        "judge_prompt": JUDGE_PROMPT,
        "cards": CARDS,
        "audit": {
            "species_and_study_names_removed": True,
            "actor_noun": "subject",
            "forbidden_terms_checked": list(FORBIDDEN_CARD_TERMS),
            "performance_and_embedding_values_present": False,
        },
    }
    CARDS_OUTPUT.write_text(json.dumps(cards_output, indent=2) + "\n")

    judgments_output = {
        "_meta": build_meta(
            "analysis/freeze_llm_task_similarity.py",
            wandb_groups,
            study_root=STUDY,
        ),
        "input_sha256": {
            "analysis/llm_task_cards.json": _semantic_json_sha256(CARDS_OUTPUT),
            "analysis/task_design_annotations.json": _sha256(ANNOTATIONS),
        },
        "judge": {
            "model": JUDGE_MODEL,
            "prompt_version": PROMPT_VERSION,
            "n_candidates": len(CARDS),
            "n_unordered_pairs": len(comparisons),
            "n_implied_ordered_encodings": 2 * len(comparisons),
            "same_session_single_judge": True,
            "independent_model_replicates": 0,
            "independent_pairwise_api_calls": 0,
            "blinding": (
                "Input-level only: cards omit species, study/author identity, outcomes, "
                "embeddings, and the hand-coded design distance. Response apparatus remains "
                "visible as a secondary task feature."
            ),
        },
        "card_to_cohort_unblinded_after_judgment": CARD_TO_COHORT,
        "frozen_ordinal_tiers": JUDGE_TIERS,
        "comparisons": comparisons,
        "audit": {
            "all_pairs_complete": len(comparisons) == 66,
            "all_pairs_mirrored_in_reverse_order": all(
                row["order_consistent"] for row in comparisons
            ),
            "mirrored_order_consistency_fraction": sum(
                row["order_consistent"] for row in comparisons
            )
            / len(comparisons),
            "limitation": (
                "The single judge produced an ordinal tiering. Its 66 implied pairwise "
                "outcomes were encoded in both A/B directions, so the 100% consistency is "
                "an invariant check rather than an empirical estimate of position bias."
            ),
        },
    }
    JUDGMENTS_OUTPUT.write_text(json.dumps(judgments_output, indent=2) + "\n")


if __name__ == "__main__":
    main()
