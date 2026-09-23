"""
Builds the held-out semantic challenge set (~80 (evidence, hypothesis)
pairs) for semantic verifier comparison.

17 held-out properties (Baku + Barcelona), NONE overlapping with the
frozen-architecture DEV set (apt_city_center, nizami_studio,
fermaart_hotel, modern_apt_nizami). Properties were selected for
semantic richness (real text mentioning our target families), so this
is NOT a representative production sample - a challenge set.

Gold relation follows the approved v2 annotation guideline:
judge ONLY evidence_text vs hypothesis, semantically, ignoring source
provenance/trust (tracked separately as metadata). Manual annotation,
no model was used to produce these labels.

Once finalized (after user confirmation), this file is FROZEN:
- no relabeling after seeing model results
- no rebalancing of the distribution
- no dropping of inconvenient examples

Run: python -m evaluation.experiments.verifier_comparison.build_benchmark
(no model inference here - pure data construction)
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent / "benchmark_v1.jsonl"

HYPOTHESES = {
    "Q1": "The room/property has soundproofing.",
    "Q2": "The immediate surroundings are explicitly described as quiet.",
    "Q3": "Guests report little or no significant noise disturbance.",
    "FC1": "The property's policies formally accommodate children.",
    "FAM1": "The property is suitable for families.",
    "RW1": "The room has a desk or dedicated workspace.",
    "RW2A": "Wi-Fi / internet access is available.",
    "RW2B": "Internet connection is reliable enough for remote work.",
    "BQ1": "The breakfast is good in quality.",
    "CL1": "The property is clean.",
    "NL1": "Nightlife is available nearby.",
    "BC1": "The bed is comfortable.",
}

# Each row: (property_slug, source_type, source_path, evidence_text,
#            hypothesis_id, gold_relation, semantic_case_type,
#            annotation_confidence, annotation_note)
ROWS = [
    # ---------------- quiet: Q1 (soundproofing feature) ----------------
    ("perron_hotel", "review_summary", "reviewSummary.summary",
     "Many guests noted poor soundproofing with noise from nearby bars",
     "Q1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "high",
     "Describes noise OUTCOME, not the presence/absence of a soundproofing feature. 'Poor' implies some measure exists but underperforms - not a direct feature-presence claim."),
    ("nest_hotel", "review_summary", "reviewSummary.cons[]",
     "Room Soundproofing: Disturbance from street traffic, loud music, and internal ventilation noise.",
     "Q1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "high",
     "Same reasoning as perron_hotel case - outcome-level complaint, not a feature-presence statement."),
    ("city_center_hotel", "review_summary", "reviewSummary.cons[]",
     "Soundproofing: Thin walls let in street music and noise from neighboring rooms.",
     "Q1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "medium",
     "Borderline: 'thin walls' arguably describes a structural absence of sound insulation more directly than the other two - kept medium confidence to flag the ambiguity."),
    ("alcam_plaza", "policies", "policies[N].content",
     "Guests must be quiet between 21:00 and 09:00.",
     "Q1", "NOT_ENOUGH_EVIDENCE", "policy_vs_experience", "high",
     "Behavioral policy, not a statement about the property's structural soundproofing."),
    ("walker_apartment", "highlights", "highlights[N].contents",
     "Quiet street view",
     "Q1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "medium",
     "Lexical overlap with 'quiet' but topically about a view, not a soundproofing feature - deliberate adversarial lexical-overlap case."),

    # ---------------- quiet: Q2 (surroundings described as quiet) ----------------
    ("seven_eleven", "review_summary", "reviewSummary.pros[]",
     "A quiet setting within walking distance of the Old City and main attractions.",
     "Q2", "SUPPORT", "explicit_support", "high",
     "Explicitly describes the setting/surroundings as quiet."),
    ("walker_apartment", "highlights", "highlights[N].contents",
     "Quiet street view",
     "Q2", "SUPPORT", "explicit_support", "high",
     "Directly describes the street (surroundings) as quiet - correct target hypothesis for this phrase."),
    ("city_center_hotel", "highlights", "highlights[N].contents",
     "Quiet street view",
     "Q2", "SUPPORT", "explicit_support", "high",
     "Second property, same phrase - included once more for cross-property consistency check, not further duplicated."),
    ("rent_top_rambla", "other", "fine_print",
     "This property is located in a residential area and guests are asked to refrain from excessive noise.",
     "Q2", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "medium",
     "Zoning context + guest-behavior request, not a direct 'surroundings are quiet' claim."),
    ("alcam_plaza", "policies", "policies[N].title",
     "Quiet hours",
     "Q2", "NOT_ENOUGH_EVIDENCE", "policy_vs_experience", "high",
     "Policy label, not a description of the surroundings."),

    # ---------------- quiet: Q3 (guest reports of little/no noise) ----------------
    ("perron_hotel", "review_summary", "reviewSummary.summary",
     "Many guests noted poor soundproofing with noise from nearby bars",
     "Q3", "CONTRADICT", "explicit_contradiction", "high",
     "Direct guest-report of noise disturbance - explicitly incompatible with 'little or no noise disturbance'."),
    ("nest_hotel", "review_summary", "reviewSummary.cons[]",
     "Room Soundproofing: Disturbance from street traffic, loud music, and internal ventilation noise.",
     "Q3", "CONTRADICT", "explicit_contradiction", "high",
     "Explicit disturbance report."),
    ("city_center_hotel", "review_summary", "reviewSummary.cons[]",
     "Soundproofing: Thin walls let in street music and noise from neighboring rooms.",
     "Q3", "CONTRADICT", "explicit_contradiction", "high",
     "Explicit noise-from-neighboring-rooms report."),
    ("alcam_plaza", "policies", "policies[N].content",
     "Guests must be quiet between 21:00 and 09:00.",
     "Q3", "NOT_ENOUGH_EVIDENCE", "policy_vs_experience", "high",
     "A behavioral request is not itself a guest report of actual noise experience."),

    # ---------------- family: FC1 (policy accommodates children) ----------------
    ("hotel_ambit", "policies", "policies[N].content",
     "Child policies Children are not allowed.",
     "FC1", "CONTRADICT", "explicit_contradiction", "high",
     "Unambiguous explicit policy negation."),
    ("nemi_hotel", "policies", "policies[N].content",
     "Child policies Children of any age are welcome. Children 3 years and above will be charged as adults at this property.",
     "FC1", "SUPPORT", "explicit_support", "high",
     "Direct policy statement of accommodation."),
    ("lantern_central", "policies", "policies[N].content",
     "Child policies Children of any age are welcome. Children 10 years and above will be charged as adults at this property.",
     "FC1", "SUPPORT", "explicit_support", "high",
     "Second held-out instance of the same policy pattern, different property."),

    # ---------------- family: FAM1 (broader family suitability) ----------------
    ("alcam_plaza", "description", "description",
     "Guest Favorites: Guests appreciate the convenient location, family-friendly environment, and suitability for city trips.",
     "FAM1", "SUPPORT", "explicit_support", "high",
     "Explicit 'family-friendly environment' phrase in Booking-aggregated description summary."),
    ("alcam_plaza", "policies", "policies[N].content",
     "The minimum age for check-in is 25",
     "FAM1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "medium",
     "Near-miss control: minimum check-in age constrains the lead booker, not a ban on children in the party (same property separately confirms 'Children of any age are welcome'). Tests whether verifiers over-react to 'age restriction' phrasing."),
    ("city_center_hotel", "description", "description",
     "Dining Experience: The hotel features a family-friendly restaurant serving Greek, Italian, and Middle Eastern cuisines.",
     "FAM1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "high",
     "Corrected: 'family-friendly restaurant' supports only a dining-level aspect (closer to a narrower FC3-style claim), not the broad property-level FAM1 claim - evidence supporting one aspect of a broader claim is NEE per guideline."),
    ("lantern_central", "description", "description",
     "Dining Experience: Guests can enjoy a buffet breakfast with halal options and a family-friendly restaurant serving Turkish cuisine in a traditional and modern ambience.",
     "FAM1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "high",
     "Corrected: same dining-aspect-only reasoning as the city_center_hotel instance."),
    ("nemi_hotel", "description", "description",
     "About this propertyComfortable Accommodations: Nemi Hotel Baku in Baku offers family rooms with air-conditioning, private bathrooms, and free WiFi.",
     "FAM1", "SUPPORT", "explicit_support", "high",
     "'Family rooms' stated in free-text description prose (not a structured facility.name tag)."),
    ("hotel_ambit", "policies", "policies[N].content",
     "Child policies Children are not allowed.",
     "FAM1", "CONTRADICT", "explicit_contradiction", "high",
     "Same evidence as the FC1 pair above, paired against the broader FAM1 hypothesis too. Unlike the dining-aspect cases above, a blanket booking prohibition is property-wide, not aspect-limited - legitimately CONTRADICT for the broad claim too."),
    ("azcot_hotel", "policies", "policies[N].content",
     "Child policies Children are not allowed.",
     "FC1", "CONTRADICT", "explicit_contradiction", "high",
     "Second held-out property with the identical explicit policy negation - cross-property CONTRADICT robustness check."),
    ("azcot_hotel", "policies", "policies[N].content",
     "Child policies Children are not allowed.",
     "FAM1", "CONTRADICT", "explicit_contradiction", "high",
     "Same reasoning as hotel_ambit's FAM1 pair - blanket prohibition, property-wide."),
    ("apartment_3bedrooms", "policies", "policies[N].content",
     "Child policies Children are not allowed.",
     "FAM1", "CONTRADICT", "explicit_contradiction", "high",
     "Third distinct property with the same explicit policy pattern; only FAM1 included here (not FC1 too) to avoid over-duplicating the identical template across every property."),

    # ---------------- workspace: RW1 (desk) ----------------
    ("walker_apartment", "description", "description",
     "Additional amenities include a work desk, microwave, and dining table.",
     "RW1", "SUPPORT", "explicit_support", "high",
     "Free-text description prose mentioning a work desk."),
    ("nemi_hotel", "description", "description",
     "Each room includes a work desk, sofa bed, and TV.",
     "RW1", "SUPPORT", "explicit_support", "high",
     "Second property, description prose."),
    ("nest_hotel", "description", "description",
     "Each room includes a work desk, minibar, and free WiFi.",
     "RW1", "SUPPORT", "explicit_support", "high",
     "Third property, description prose - also doubles as an RW2A example below (same sentence, two hypotheses)."),
    ("alcam_plaza", "description", "description",
     "Additional amenities include a washing machine, work desk, and parquet floors.",
     "RW1", "SUPPORT", "explicit_support", "high",
     "Fourth property, Barcelona - city diversity check."),

    # ---------------- workspace: RW2A (wifi availability, free text) ----------------
    ("alfa_apartman", "description", "description",
     "About this propertyModern Comforts: Alfa Apartman in Baku offers a recently renovated apartment with free WiFi, air-conditioning, and a fully equipped kitchenette.",
     "RW2A", "SUPPORT", "explicit_support", "high",
     "Free-text WiFi-availability claim in description."),
    ("rent_top_rambla", "description", "description",
     "Featuring free WiFi and air conditioning, Rent Top Apartments Rambla Catalunya 8 is located in Barcelona, 200 metres from Plaça Catalunya.",
     "RW2A", "SUPPORT", "explicit_support", "high",
     "Second instance, Barcelona."),
    ("nest_hotel", "description", "description",
     "Each room includes a work desk, minibar, and free WiFi.",
     "RW2A", "SUPPORT", "explicit_support", "high",
     "Same sentence as the RW1 example above - one evidence text legitimately supports two distinct atomic hypotheses at once."),

    # ---------------- workspace: RW2B (wifi reliability) ----------------
    ("walker_apartment", "description", "description",
     "Modern Amenities: Guests enjoy free WiFi, air-conditioning, and a balcony with city views.",
     "RW2B", "NOT_ENOUGH_EVIDENCE", "availability_vs_reliability", "high",
     "Presence-only claim; says nothing about connection speed/reliability."),
    ("alfa_apartman", "description", "description",
     "About this propertyModern Comforts: Alfa Apartman in Baku offers a recently renovated apartment with free WiFi, air-conditioning, and a fully equipped kitchenette.",
     "RW2B", "NOT_ENOUGH_EVIDENCE", "availability_vs_reliability", "high",
     "Same presence-only text as the RW2A SUPPORT pair above - illustrates presence vs reliability as genuinely different hypotheses for the identical evidence."),

    # ---------------- breakfast: BQ1 ----------------
    ("lantern_central", "review_summary", "reviewSummary.pros[]",
     "Breakfast: Tasty morning meals with a comforting home-cooked feel.",
     "BQ1", "SUPPORT", "explicit_support", "high",
     "Direct positive quality claim."),
    ("nizami_boutique", "review_summary", "reviewSummary.cons[]",
     "Breakfast: Fresh and tasty options for some, though others found the quality poor or lacking.",
     "BQ1", "NOT_ENOUGH_EVIDENCE", "aggregated_guest_sentiment_mixed", "medium",
     "Both explicit positive and explicit negative quality claims in the same text - does not resolve to one side."),
    ("nemi_hotel", "review_summary", "reviewSummary.cons[]",
     "Breakfast: Tasty options available, though selection is often described as limited, repetitive, or meager.",
     "BQ1", "NOT_ENOUGH_EVIDENCE", "aggregated_guest_sentiment_mixed", "medium",
     "Mixed: 'tasty' (support-leaning) vs 'limited, repetitive, meager' (variety/quality criticism)."),
    ("perron_hotel", "review_summary", "reviewSummary.summary",
     "Breakfast was generally criticised for limited variety and quality",
     "BQ1", "CONTRADICT", "explicit_contradiction", "high",
     "Direct, one-sided quality criticism, not hedged by a counterbalancing positive claim."),
    ("four_seasson_apt", "description", "description",
     "Delicious Breakfast: A variety of breakfast options are available, including continental, American, Italian, full English/Irish, vegetarian, vegan, halal, gluten-free, kosher, and Asian.",
     "BQ1", "SUPPORT", "explicit_support", "high",
     "Text literally contains 'Delicious' - a direct quality claim. Source is a marketing template header, tracked as metadata only, not used to downgrade the label."),
    ("nest_hotel", "highlights", "highlights[N].contents",
     "Continental, Vegan, Halal, Buffet",
     "BQ1", "NOT_ENOUGH_EVIDENCE", "presence_vs_quality", "high",
     "Pure format/diet list, zero quality language - clean contrast pair to the 'Delicious Breakfast' case."),
    ("city_center_hotel", "highlights", "highlights[N].contents",
     "Halal, Asian, Buffet",
     "BQ1", "NOT_ENOUGH_EVIDENCE", "presence_vs_quality", "high",
     "Same pattern, second property."),
    ("h10_catalunya", "highlights", "highlights[N].contents",
     "Full English/Irish, Vegetarian, Vegan, Gluten-free, American, Buffet",
     "BQ1", "NOT_ENOUGH_EVIDENCE", "presence_vs_quality", "high",
     "Same pattern, Barcelona property."),

    # ---------------- cleanliness: CL1 ----------------
    ("city_center_hotel", "review_summary", "reviewSummary.pros[]",
     "Cleanliness: Neat and spotless interiors maintained by daily housekeeping.",
     "CL1", "SUPPORT", "explicit_support", "high",
     "Direct positive cleanliness claim."),
    ("nemi_hotel", "review_summary", "reviewSummary.pros[]",
     "Room Cleanliness: Consistently described as spotless, tidy, and well-maintained.",
     "CL1", "SUPPORT", "explicit_support", "high",
     "No hedging language - clean positive case."),
    ("seven_eleven", "review_summary", "reviewSummary.pros[]",
     "Cleanliness: Spotless and well-maintained living spaces.",
     "CL1", "SUPPORT", "explicit_support", "high",
     "Third property, unhedged."),
    ("nizami_boutique", "review_summary", "reviewSummary.pros[]",
     "Room Cleanliness: Well-maintained and tidy living spaces.",
     "CL1", "SUPPORT", "explicit_support", "high",
     "Fourth property."),
    ("lantern_central", "review_summary", "reviewSummary.pros[]",
     "Room Cleanliness: Tidy spaces featuring spotless sheets, well-kept carpets, and daily housekeeping.",
     "CL1", "SUPPORT", "explicit_support", "high",
     "Fifth property."),
    ("nest_hotel", "review_summary", "reviewSummary.cons[]",
     "Room Cleanliness: Generally tidy, though some guests report floor stains, dust, and ants.",
     "CL1", "NOT_ENOUGH_EVIDENCE", "aggregated_guest_sentiment_mixed", "medium",
     "Contains both a supporting claim ('generally tidy') and explicit negative facts (stains, dust, ants) - genuinely mixed."),
    ("perron_hotel", "review_summary", "reviewSummary.summary",
     "Room cleanliness received mixed feedback, some found it lacking",
     "CL1", "NOT_ENOUGH_EVIDENCE", "aggregated_guest_sentiment_mixed", "medium",
     "Explicitly hedged ('some', not 'most'/'guests generally') - kept consistent with the nest_hotel mixed case, not escalated to CONTRADICT."),

    # ---------------- nightlife: NL1 ----------------
    ("perron_hotel", "review_summary", "reviewSummary.summary",
     "Many guests noted poor soundproofing with noise from nearby bars",
     "NL1", "SUPPORT", "explicit_support", "high",
     "Explicitly names bars nearby - sufficient per the consistent bars/clubs-mention rule, independent of the noise complaint also present in the same sentence."),
    ("rent_top_plaza_catalunya", "description", "description",
     "Rent Top Apartments Passeig de Gràcia is situated in central Barcelona, surrounded by shops, restaurants and bars.",
     "NL1", "SUPPORT", "explicit_support", "high",
     "Explicit 'bars' mention."),
    ("letsgo_paseo", "description", "description",
     "Centrally located, you will find many bars, restaurants and supermarkets in the surrounding streets.",
     "NL1", "SUPPORT", "explicit_support", "high",
     "Second Barcelona property, explicit 'bars'."),
    ("city_center_hotel", "description", "description",
     "Additional amenities include a bar, coffee shop, and live music, providing entertainment options for all guests.",
     "NL1", "SUPPORT", "explicit_support", "high",
     "On-site bar + live music - even more direct than a 'nearby' claim."),
    ("h10_catalunya", "description", "description",
     "There is also a lobby-bar and a library.",
     "NL1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "medium",
     "An in-hotel lobby-bar is a quiet amenity, not evidence of external nightlife scene - deliberately testing whether 'bar' alone (without 'nearby'/'surrounding') is over-generalized to SUPPORT."),
    ("nizami_boutique", "review_summary", "reviewSummary.pros[]",
     "Location: Situated on Nizami Street near shops and restaurants, within walking distance of the Old City.",
     "NL1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "high",
     "Only 'shops and restaurants' named, no bars/nightlife - contrast pair to the SUPPORT cases above."),

    # ---------------- bed comfort: BC1 ----------------
    ("hotel_ambit", "highlights", "highlights[N].contents",
     "Want a great night's sleep? This hotel was highly rated for its very comfy beds.",
     "BC1", "SUPPORT", "explicit_support", "high",
     "Direct, explicit bed-comfort claim - the only genuine free-text bed-comfort evidence found across all 17 held-out properties."),
    ("four_seasson_apt", "description", "description",
     "The living room includes a sofa bed and a dining area, providing ample space for relaxation and meals.",
     "BC1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "high",
     "Mentions a sofa bed's presence, says nothing about its comfort - presence vs quality, same family of case as breakfast."),
    ("walker_apartment", "description", "description",
     "The living room includes a sofa bed and a dining area, ensuring comfort for all guests.",
     "BC1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "medium",
     "Contains the word 'comfort' but modifying the overall living-room amenity claim generically, not a specific statement about bed/mattress quality - kept medium given the lexical proximity to 'comfortable'."),

    # ---------------- additional cleanliness (service presence, not outcome) ----------------
    ("nemi_hotel", "description", "description",
     "Convenient Facilities: Guests benefit from a paid shuttle service, lift, 24-hour front desk, minimarket, daily housekeeping, hairdresser, tour desk, and luggage storage.",
     "CL1", "NOT_ENOUGH_EVIDENCE", "presence_vs_quality", "high",
     "States a housekeeping SERVICE exists, not an outcome claim about actual cleanliness - same presence-vs-quality logic as breakfast-format lists."),

    # ---------------- additional family (distinct phrasing + second near-miss control) ----------------
    ("lantern_central", "description", "description",
     "Convenient Facilities: The hotel provides a shared kitchen, minimarket, coffee shop, and child-friendly buffet.",
     "FAM1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "high",
     "Corrected: 'child-friendly buffet' is a single dining-aspect claim, not sufficient for the broad property-suitability claim."),
    ("nest_hotel", "other", "fine_print",
     "Guests under the age of 18 can only check in with a parent or official guardian.",
     "FAM1", "NOT_ENOUGH_EVIDENCE", "related_but_insufficient", "medium",
     "Second near-miss variant (different phrasing/threshold than the age-25 case): constrains who may be the lead booker, not whether children may stay."),

    # ---------------- additional breakfast presence-only (distinct short phrasings) ----------------
    ("nizami_boutique", "highlights", "highlights[N].contents",
     "Continental",
     "BQ1", "NOT_ENOUGH_EVIDENCE", "presence_vs_quality", "high",
     "Single diet/format word, zero quality language - shortest presence-only instance in the set."),
    ("alfa_apartman", "highlights", "highlights[N].contents",
     "Breakfast to go",
     "BQ1", "NOT_ENOUGH_EVIDENCE", "presence_vs_quality", "high",
     "Format claim (to-go), not a quality claim."),

    # ---------------- additional workspace/wifi nuance ----------------
    ("h10_catalunya", "description", "description",
     "There is also a 24-hour internet terminal.",
     "RW2A", "SUPPORT", "explicit_support", "medium",
     "States internet access exists via a shared terminal, not confirmed in-room WiFi - genuinely borderline for a hypothesis phrased broadly as 'Wi-Fi / internet access', kept medium rather than high."),
]


def _compute_template_group_ids(rows: list[dict]) -> None:
    """
    Groups examples that share the EXACT SAME (evidence_text,
    hypothesis_id) across DIFFERENT properties - i.e. the identical
    Booking-generated template text tested against the identical
    hypothesis, just copy-pasted onto another property. These are not
    independent semantic trials: the model sees byte-identical input
    each time, so getting one right/wrong is not N independent
    successes/failures.

    Deliberately conservative: only exact text matches are grouped.
    Structurally-similar-but-textually-different template instances
    (e.g. "family-friendly restaurant serving X cuisine" with
    different X) are NOT grouped - the model is processing genuinely
    different content there, even if generated by a similar
    underlying template mechanism.

    Same evidence_text tested against a DIFFERENT hypothesis (e.g.
    "Children are not allowed." against both FC1 and FAM1) is also
    NOT grouped together - that is a genuinely different verification
    task per hypothesis, not a duplicate.

    Examples that don't repeat across properties get
    template_group_id = None (singleton, not part of any group).
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["evidence_text"], row["hypothesis_id"])
        by_key.setdefault(key, []).append(row)

    group_counter = 0
    for (evidence_text, hyp_id), group in by_key.items():
        slugs = {r["property_slug"] for r in group}
        if len(slugs) <= 1:
            for r in group:
                r["template_group_id"] = None
            continue
        group_counter += 1
        gid = f"tg_{group_counter:03d}_{hyp_id.lower()}"
        for r in group:
            r["template_group_id"] = gid


def main() -> None:
    rows = []
    seen_ids: set[str] = set()
    counters: dict[str, int] = {}

    for property_slug, source_type, source_path, evidence_text, hyp_id, gold, case_type, confidence, note in ROWS:
        counters[hyp_id] = counters.get(hyp_id, 0) + 1
        example_id = f"vb_{hyp_id.lower()}_{counters[hyp_id]:02d}"
        if example_id in seen_ids:
            raise RuntimeError(f"duplicate example_id {example_id}")
        seen_ids.add(example_id)

        rows.append(
            {
                "example_id": example_id,
                "property_slug": property_slug,
                "source_type": source_type,
                "source_path": source_path,
                "evidence_text": evidence_text,
                "hypothesis_id": hyp_id,
                "hypothesis": HYPOTHESES[hyp_id],
                "gold_relation": gold,
                "annotation_note": note,
                "semantic_case_type": case_type,
                "annotation_confidence": confidence,
            }
        )

    _compute_template_group_ids(rows)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_grouped = sum(1 for r in rows if r["template_group_id"] is not None)
    n_groups = len({r["template_group_id"] for r in rows if r["template_group_id"] is not None})
    print(f"Wrote {len(rows)} examples to {OUTPUT_PATH}")
    print(f"template_group_id: {n_grouped} examples belong to {n_groups} template groups (rest are singletons)")


if __name__ == "__main__":
    main()
