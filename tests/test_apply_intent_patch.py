from datetime import date

from app.logic.apply_intent_patch import apply_intent_patch
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.filters import PriceConstraint, SearchFilters
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.query import SearchRequest


def test_set_nights_updates_checkout_from_existing_checkin():
    state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 21),
    )

    patch = SearchIntentPatch(set_nights=6)
    new_state = apply_intent_patch(state, patch)

    assert new_state.check_in == date(2026, 4, 20)
    assert new_state.check_out == date(2026, 4, 26)


def test_filter_merge_preserves_existing_fields():
    state = SearchRequest(
        city="Baku",
        filters=SearchFilters(bedrooms_min=2, area_sqm_min=80),
    )

    patch = SearchIntentPatch(
        set_filters=SearchFilters(bedrooms_min=3),
    )
    new_state = apply_intent_patch(state, patch)

    assert new_state.filters.bedrooms_min == 3
    assert new_state.filters.area_sqm_min == 80


def test_price_merge_preserves_other_price_fields():
    state = SearchRequest(
        city="Baku",
        filters=SearchFilters(
            price=PriceConstraint(max_amount=120, currency="USD", scope="per_night")
        ),
    )

    patch = SearchIntentPatch(
        set_filters=SearchFilters(
            price=PriceConstraint(max_amount=150)
        )
    )
    new_state = apply_intent_patch(state, patch)

    assert new_state.filters.price.max_amount == 150
    assert new_state.filters.price.currency == "USD"
    assert new_state.filters.price.scope == "per_night"





def test_clear_city_sets_city_to_none():
    state = SearchRequest(city="Baku")
    patch = SearchIntentPatch(clear_city=True)

    new_state = apply_intent_patch(state, patch)

    assert new_state.city is None
    
    


def test_apply_patch_updates_adults_children_and_rooms():
    state = SearchRequest(
        city="Baku",
        adults=2,
        children=0,
        rooms=1,
    )

    patch = SearchIntentPatch(
        set_adults=3,
        set_children=1,
        set_rooms=2,
    )

    new_state = apply_intent_patch(state, patch)

    assert new_state.adults == 3
    assert new_state.children == 1
    assert new_state.rooms == 2
    
    
def test_apply_patch_updates_only_adults_without_touching_other_fields():
    state = SearchRequest(
        city="Baku",
        adults=2,
        children=1,
        rooms=1,
    )

    patch = SearchIntentPatch(set_adults=4)

    new_state = apply_intent_patch(state, patch)

    assert new_state.adults == 4
    assert new_state.children == 1
    assert new_state.rooms == 1
    
    


    
    
def test_set_check_in_only_recomputes_checkout_to_one_night():
    state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 26),
    )

    patch = SearchIntentPatch(set_check_in="2026-04-08")
    new_state = apply_intent_patch(state, patch)

    assert new_state.check_in == date(2026, 4, 8)
    assert new_state.check_out == date(2026, 4, 9)
    
    
def test_set_check_in_with_nights_recomputes_checkout():
    state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 26),
    )

    patch = SearchIntentPatch(set_check_in="2026-04-08", set_nights=4)
    new_state = apply_intent_patch(state, patch)

    assert new_state.check_in == date(2026, 4, 8)
    assert new_state.check_out == date(2026, 4, 12)


def test_apply_patch_merges_duplicate_constraints_and_promotes_priority_to_must():
    state = SearchRequest(
        city="Baku",
        constraints=[
            UserConstraint(
                id="2",
                raw_text="bright rooms",
                normalized_text="bright rooms",
                priority=ConstraintPriority.NICE,
                category=ConstraintCategory.LAYOUT,
                mapping_status=ConstraintMappingStatus.UNRESOLVED,
                mapped_fields=[],
                evidence_strategy=EvidenceStrategy.TEXTUAL,
            )
        ],
    )

    patch = SearchIntentPatch(
        add_constraints=[
            UserConstraint(
                id="2",
                raw_text="bright rooms",
                normalized_text="bright rooms",
                priority=ConstraintPriority.MUST,
                category=ConstraintCategory.LAYOUT,
                mapping_status=ConstraintMappingStatus.KNOWN,
                mapped_fields=[],
                evidence_strategy=EvidenceStrategy.TEXTUAL,
            )
        ]
    )

    new_state = apply_intent_patch(state, patch)

    bright_constraints = [
        c for c in new_state.constraints
        if c.normalized_text == "bright rooms"
    ]

    assert len(bright_constraints) == 1

    bright = bright_constraints[0]

    assert bright.priority == ConstraintPriority.MUST
    assert bright.mapping_status == ConstraintMappingStatus.UNRESOLVED
    assert bright.mapped_fields == []
    assert bright.evidence_strategy == EvidenceStrategy.TEXTUAL

# --- explicit priority changes: the latest wording replaces the old priority --


def _parking(priority: ConstraintPriority, raw_text: str = "parking") -> UserConstraint:
    return UserConstraint(
        raw_text=raw_text,
        normalized_text="parking",
        priority=priority,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=["parking"],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _wifi() -> UserConstraint:
    return UserConstraint(
        raw_text="wifi",
        normalized_text="wifi",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=["wifi"],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _state(*constraints: UserConstraint) -> SearchRequest:
    return SearchRequest(
        city="Barcelona", check_in=date(2026, 12, 10), check_out=date(2026, 12, 13), constraints=list(constraints)
    )


def _priorities(state: SearchRequest) -> list[tuple[str, str]]:
    return [(c.normalized_text, c.priority.value) for c in state.constraints]


def test_an_explicit_downgrade_replaces_must_with_nice():
    # "Parking is no longer essential, but I'd still prefer it"
    state = _state(_wifi(), _parking(ConstraintPriority.MUST))

    new_state = apply_intent_patch(state, SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.NICE)]))

    assert _priorities(new_state) == [("wifi", "must"), ("parking", "nice")]


def test_an_explicit_upgrade_replaces_nice_with_must():
    # "Parking is essential"
    state = _state(_wifi(), _parking(ConstraintPriority.NICE))

    new_state = apply_intent_patch(state, SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.MUST)]))

    assert _priorities(new_state) == [("wifi", "must"), ("parking", "must")]


def test_restating_must_keeps_must_and_changes_nothing():
    state = _state(_wifi(), _parking(ConstraintPriority.MUST))

    new_state = apply_intent_patch(state, SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.MUST)]))

    assert _priorities(new_state) == [("wifi", "must"), ("parking", "must")]
    assert new_state == state  # the no-op guard sees an unchanged search


def test_an_exclusion_can_be_lifted_explicitly():
    # "Actually, I do need parking" after "no parking"
    state = _state(_parking(ConstraintPriority.FORBIDDEN))

    new_state = apply_intent_patch(state, SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.MUST)]))

    assert _priorities(new_state) == [("parking", "must")]


def test_a_new_soft_preference_is_added_as_nice():
    # "I'd prefer parking"
    state = _state(_wifi())

    new_state = apply_intent_patch(state, SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.NICE)]))

    assert _priorities(new_state) == [("wifi", "must"), ("parking", "nice")]


def test_forgetting_a_constraint_removes_it_and_keeps_the_rest():
    # "Forget about parking"
    state = _state(_wifi(), _parking(ConstraintPriority.MUST))

    new_state = apply_intent_patch(state, SearchIntentPatch(remove_constraint_texts=["parking"]))

    assert _priorities(new_state) == [("wifi", "must")]
    assert (new_state.city, new_state.check_in, new_state.check_out) == (state.city, state.check_in, state.check_out)


def test_duplicates_within_one_patch_still_merge_to_the_stronger_priority():
    # One message saying both is not a downgrade: within one list stronger wins.
    state = _state(_wifi())

    new_state = apply_intent_patch(
        state,
        SearchIntentPatch(add_constraints=[_parking(ConstraintPriority.NICE), _parking(ConstraintPriority.MUST)]),
    )

    assert _priorities(new_state) == [("wifi", "must"), ("parking", "must")]
