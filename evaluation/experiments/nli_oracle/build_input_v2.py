"""
Builds the v2 NLI input set - same model, redesigned input construction.

Three changes from v1:
1. "quiet" (old single hypothesis, 10 examples, catastrophic 10%
   accuracy in v1) is REPLACED by three narrower atomic hypotheses
   (Q1/Q2/Q3), and all 10 existing quiet-related evidence chunks are
   relabeled against each of them from scratch, from the meaning of
   the text (not from v1's NLI output, not from retrieval).
2. The 4 long "child policy" chunks (family_friendly/FC1) are split
   into atomic sentences (via the same split_into_sentences() helper
   production code already uses for evidence extraction) and each
   sentence is relabeled individually, keeping source_chunk_id linking
   back to the original chunk.
3. FC2, FC3, RW1, RW2a are carried over unchanged from
   retrieval_checkpoint_v2.jsonl (loaded programmatically, not
   retyped, to avoid transcription drift).

Does not touch retrieval_checkpoint_v2.jsonl or any retrieval code -
this lives entirely inside nli_oracle/ as its own artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logic.listing_signals import split_into_sentences
from evaluation.core.io import load_jsonl, save_json

RETRIEVAL_CHECKPOINT = (
    PROJECT_ROOT
    / "evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "nli_input_v2.jsonl"

# ---------------------------------------------------------------------------
# 1. Quiet -> Q1/Q2/Q3 atomic hypotheses, evidence relabeled from scratch.
# ---------------------------------------------------------------------------

QUIET_HYPOTHESES = {
    "Q1": "The room/property has soundproofing.",
    "Q2": "The immediate surroundings are explicitly described as quiet.",
    "Q3": "Guests report little or no significant noise disturbance.",
}

# (property_slug, chunk_id, text) for every chunk that was evidence for
# the old "quiet" hypothesis in v1 - reused verbatim, relation reassigned.
QUIET_CHUNKS = [
    ("apt_city_center", "12895996::9",
     "This property will not accommodate hen, stag or similar parties.Quiet hours are between 22:00:00 and 07:00:00."),
    ("apt_city_center", "12895996::118", "Quiet hours"),
    ("apt_city_center", "12895996::119", "Guests must be quiet between 22:00 and 07:00."),
    ("fermaart_hotel", "16084035::108", "Soundproofing"),
    ("fermaart_hotel", "16084035::113", "Soundproof rooms"),
    ("fermaart_hotel", "16084035::133", "Soundproofing"),
    ("fermaart_hotel", "16084035::213", "Soundproofing"),
    ("fermaart_hotel", "16084035::283", "Soundproofing"),
    ("fermaart_hotel", "16084035::352", "Quiet street view"),
    ("modern_apt_nizami", "16578874::160", "Quiet street view"),
]

# relation per (chunk_id, subclaim_key) - decided purely from text
# meaning against each narrower hypothesis, not from any model output.
QUIET_RELABEL: dict[tuple[str, str], str] = {
    # Q1: soundproofing
    ("12895996::9", "Q1"): "RELEVANT_NEUTRAL",
    ("12895996::118", "Q1"): "RELEVANT_NEUTRAL",
    ("12895996::119", "Q1"): "RELEVANT_NEUTRAL",
    ("16084035::108", "Q1"): "SUPPORT",
    ("16084035::113", "Q1"): "SUPPORT",
    ("16084035::133", "Q1"): "SUPPORT",
    ("16084035::213", "Q1"): "SUPPORT",
    ("16084035::283", "Q1"): "SUPPORT",
    ("16084035::352", "Q1"): "RELEVANT_NEUTRAL",
    ("16578874::160", "Q1"): "RELEVANT_NEUTRAL",
    # Q2: surroundings explicitly described as quiet
    ("12895996::9", "Q2"): "RELEVANT_NEUTRAL",
    ("12895996::118", "Q2"): "RELEVANT_NEUTRAL",
    ("12895996::119", "Q2"): "RELEVANT_NEUTRAL",
    ("16084035::108", "Q2"): "RELEVANT_NEUTRAL",
    ("16084035::113", "Q2"): "RELEVANT_NEUTRAL",
    ("16084035::133", "Q2"): "RELEVANT_NEUTRAL",
    ("16084035::213", "Q2"): "RELEVANT_NEUTRAL",
    ("16084035::283", "Q2"): "RELEVANT_NEUTRAL",
    ("16084035::352", "Q2"): "SUPPORT",
    ("16578874::160", "Q2"): "SUPPORT",
    # Q3: guests report little/no noise disturbance - nothing in the
    # current pipeline is an actual guest report (no reviews exist),
    # so every chunk here is RELEVANT_NEUTRAL by construction. This is
    # an intentional, honest finding, not an oversight.
    ("12895996::9", "Q3"): "RELEVANT_NEUTRAL",
    ("12895996::118", "Q3"): "RELEVANT_NEUTRAL",
    ("12895996::119", "Q3"): "RELEVANT_NEUTRAL",
    ("16084035::108", "Q3"): "RELEVANT_NEUTRAL",
    ("16084035::113", "Q3"): "RELEVANT_NEUTRAL",
    ("16084035::133", "Q3"): "RELEVANT_NEUTRAL",
    ("16084035::213", "Q3"): "RELEVANT_NEUTRAL",
    ("16084035::283", "Q3"): "RELEVANT_NEUTRAL",
    ("16084035::352", "Q3"): "RELEVANT_NEUTRAL",
    ("16578874::160", "Q3"): "RELEVANT_NEUTRAL",
}


def build_quiet_examples() -> list[dict[str, Any]]:
    examples = []
    property_by_chunk = {chunk_id: slug for slug, chunk_id, _ in QUIET_CHUNKS}
    text_by_chunk = {chunk_id: text for _, chunk_id, text in QUIET_CHUNKS}

    for chunk_id in text_by_chunk:
        for subclaim_key, hypothesis in QUIET_HYPOTHESES.items():
            examples.append(
                {
                    "pair_id": f"{property_by_chunk[chunk_id]}__quiet",
                    "property_slug": property_by_chunk[chunk_id],
                    "constraint_text": "quiet",
                    "subclaim": subclaim_key,
                    "source_chunk_id": chunk_id,
                    "chunk_id": chunk_id,  # premise itself is unsplit here
                    "premise": text_by_chunk[chunk_id],
                    "hypothesis": hypothesis,
                    "gold_relation": QUIET_RELABEL[(chunk_id, subclaim_key)],
                    "input_change": "quiet_atomic_hypothesis",
                }
            )
    return examples


# ---------------------------------------------------------------------------
# 2. Long child-policy chunks -> atomic sentences.
# ---------------------------------------------------------------------------

FC1_HYPOTHESIS = "The property's policies formally accommodate children."

FC1_CHUNKS = [
    ("apt_city_center", "12895996::111",
     "Child policies Children of any age are welcome. To see correct prices and occupancy information, "
     "please add the number of children in your group and their ages to your search. Cot and extra bed "
     "policies 0 - 2 years Cot upon request Prices for cots are not included in the total price, and will "
     "have to be paid for separately during your stay. The number of cots allowed is dependent on the option "
     "you choose. Please check your selected option for more information. There are no extra beds available "
     "at this property. All cots are subject to availability."),
    ("nizami_studio", "16406264::97",
     "Child policies Children of any age are welcome. Children 18 years and above will be charged as adults "
     "at this property. To see correct prices and occupancy information, please add the number of children "
     "in your group and their ages to your search. Cot and extra bed policies 0 - 3 years Cot upon request "
     "The number of cots allowed is dependent on the option you choose. Please check your selected option "
     "for more information. There are no extra beds available at this property. All cots are subject to "
     "availability."),
    ("fermaart_hotel", "16084035::364",
     "Child policies Children of any age are welcome. Children 18 years and above will be charged as adults "
     "at this property. To see correct prices and occupancy information, please add the number of children "
     "in your group and their ages to your search. Cot and extra bed policies 0 - 5 years Cot always "
     "available 6 years Cot always available Extra bed upon request 7+ years Extra bed upon request Prices "
     "for cots and extra beds are not included in the total price, and will have to be paid for separately "
     "during your stay. The number of extra beds and cots allowed is dependent on the option you choose. "
     "Please check your selected option for more information. All extra beds are subject to availability."),
    ("modern_apt_nizami", "16578874::169",
     "Child policies Children of any age are welcome. Children 18 years and above will be charged as adults "
     "at this property. To see correct prices and occupancy information, please add the number of children "
     "in your group and their ages to your search. Cot and extra bed policies 0 - 3 years Cot upon request "
     "The number of cots allowed is dependent on the option you choose. Please check your selected option "
     "for more information. There are no extra beds available at this property. All cots are subject to "
     "availability."),
]

# sentence index -> relation, per chunk. Determined purely from what
# each sentence itself says relative to FC1_HYPOTHESIS.
FC1_SENTENCE_RELATIONS = {
    "12895996::111": [
        "SUPPORT",          # s0: Children of any age are welcome.
        "RELEVANT_NEUTRAL",  # s1: procedural search instruction
        "SUPPORT",           # s2: cot policy (0-2y, upon request) - accommodation mechanism
        "RELEVANT_NEUTRAL",  # s3: procedural (depends on option)
        "RELEVANT_NEUTRAL",  # s4: procedural
        "RELEVANT_NEUTRAL",  # s5: no extra beds - a capacity limitation, not a policy contradiction
        "RELEVANT_NEUTRAL",  # s6: availability caveat
    ],
    "16406264::97": [
        "SUPPORT",           # s0
        "RELEVANT_NEUTRAL",  # s1: 18+ charged as adults - pricing rule, not accommodation signal either way
        "RELEVANT_NEUTRAL",  # s2: procedural
        "SUPPORT",           # s3: cot policy (0-3y, upon request + depends-on-option, merged by sentence split)
        "RELEVANT_NEUTRAL",  # s4: procedural
        "RELEVANT_NEUTRAL",  # s5: no extra beds
        "RELEVANT_NEUTRAL",  # s6: availability caveat
    ],
    "16084035::364": [
        "SUPPORT",           # s0
        "RELEVANT_NEUTRAL",  # s1: 18+ charged as adults
        "RELEVANT_NEUTRAL",  # s2: procedural
        "SUPPORT",           # s3: cot ALWAYS available (0-6y) + extra bed upon request (7+y) - strongest accommodation mechanism in the set
        "RELEVANT_NEUTRAL",  # s4: procedural
        "RELEVANT_NEUTRAL",  # s5: procedural
        "RELEVANT_NEUTRAL",  # s6: availability caveat (note: no "no extra beds" sentence in this property's text)
    ],
    "16578874::169": [
        "SUPPORT",
        "RELEVANT_NEUTRAL",
        "RELEVANT_NEUTRAL",
        "SUPPORT",
        "RELEVANT_NEUTRAL",
        "RELEVANT_NEUTRAL",
        "RELEVANT_NEUTRAL",
    ],
}


def build_fc1_atomic_examples() -> list[dict[str, Any]]:
    examples = []
    for slug, chunk_id, text in FC1_CHUNKS:
        sentences = split_into_sentences(text)
        relations = FC1_SENTENCE_RELATIONS[chunk_id]
        if len(sentences) != len(relations):
            raise RuntimeError(
                f"Sentence count mismatch for {chunk_id}: "
                f"split produced {len(sentences)}, but {len(relations)} relations were hand-labeled. "
                f"Sentences: {sentences}"
            )
        for i, (sentence, relation) in enumerate(zip(sentences, relations)):
            examples.append(
                {
                    "pair_id": f"{slug}__family_friendly",
                    "property_slug": slug,
                    "constraint_text": "family-friendly",
                    "subclaim": "FC1",
                    "source_chunk_id": chunk_id,
                    "chunk_id": f"{chunk_id}#s{i}",
                    "premise": sentence,
                    "hypothesis": FC1_HYPOTHESIS,
                    "gold_relation": relation,
                    "input_change": "fc1_atomic_sentence_split",
                }
            )
    return examples


# ---------------------------------------------------------------------------
# 3. Carry over FC2, FC3, RW1, RW2a unchanged from the retrieval checkpoint.
# ---------------------------------------------------------------------------

CARRIED_OVER_SUBCLAIMS = {"FC2", "FC3", "RW1", "RW2a"}


def build_carried_over_examples() -> list[dict[str, Any]]:
    pairs = load_jsonl(RETRIEVAL_CHECKPOINT)
    examples = []
    for pair in pairs:
        for subclaim_key, subclaim in pair["subclaims"].items():
            if subclaim_key not in CARRIED_OVER_SUBCLAIMS:
                continue
            for evidence in subclaim["evidence"]:
                examples.append(
                    {
                        "pair_id": pair["pair_id"],
                        "property_slug": pair["property_slug"],
                        "constraint_text": pair["constraint_text"],
                        "subclaim": subclaim_key,
                        "source_chunk_id": evidence["chunk_id"],
                        "chunk_id": evidence["chunk_id"],
                        "premise": evidence["text"],
                        "hypothesis": subclaim["hypothesis"],
                        "gold_relation": evidence["relation"],
                        "input_change": "carried_over_unchanged_from_v1",
                    }
                )
    return examples


# ---------------------------------------------------------------------------
# Structured vs free-text tagging.
# ---------------------------------------------------------------------------


def is_structured_attribute(premise: str) -> bool:
    """
    Structured attribute = a bare Booking-style tag phrase (no verb,
    no clause structure) - e.g. "Desk", "Free WiFi", "Soundproofing",
    "Family rooms", "Quiet hours". Free text = anything with sentence/
    clause structure, including short ones with a verb.

    Heuristic: word_count <= 3. Verified against the actual v1/v2
    example set - every genuine bare tag in this dataset is <=3 words,
    and every sentence (however short, e.g. "Guests must be quiet
    between 22:00 and 07:00.") has more than 3 words. "Quiet street
    view" and "Quiet hours" are policy/highlight text, not literal
    facility-list entries, but are grammatically indistinguishable
    from a tag (no verb) - documented explicitly as a rule edge case
    rather than special-cased away.
    """
    return len(premise.split()) <= 3


def main() -> None:
    examples = build_quiet_examples() + build_fc1_atomic_examples() + build_carried_over_examples()

    for ex in examples:
        ex["is_structured"] = is_structured_attribute(ex["premise"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(__import__("json").dumps(ex, ensure_ascii=False) + "\n")

    n_structured = sum(1 for ex in examples if ex["is_structured"])
    print(f"Built {len(examples)} v2 examples -> {OUTPUT_PATH}")
    print(f"  quiet (Q1/Q2/Q3 relabel): {len(build_quiet_examples())}")
    print(f"  FC1 atomic sentences:     {len(build_fc1_atomic_examples())}")
    print(f"  carried over (FC2/FC3/RW1/RW2a): {len(build_carried_over_examples())}")
    print(f"  structured: {n_structured}, free_text: {len(examples) - n_structured}")


if __name__ == "__main__":
    main()
