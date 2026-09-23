from app.schemas.filters import SearchFilters, PriceConstraint


def test_empty_price_constraint_becomes_none():
    filters = SearchFilters(
        price=PriceConstraint()
    )

    assert filters.price is None


def test_price_constraint_with_max_is_preserved():
    filters = SearchFilters(
        price=PriceConstraint(
            max_amount=100
        )
    )

    assert filters.price is not None
    assert filters.price.max_amount == 100


def test_price_constraint_with_min_is_preserved():
    filters = SearchFilters(
        price=PriceConstraint(
            min_amount=50
        )
    )

    assert filters.price is not None
    assert filters.price.min_amount == 50