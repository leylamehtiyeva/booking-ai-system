"""
Publicly displayable part of one provider offer (app.retrieval.liteapi).

Only what may be shown to a user, sent to an LLM or written to telemetry
lives here. Internal rates (retail/net totals, initial prices, commissions,
fee and penalty amounts) are never read into this model, and extra="forbid"
keeps them from being added by accident.

public_total is the provider's suggested selling price (SSP) for the whole
stay, all rooms of the offer. It is None whenever that price is missing or
ambiguous - no other amount is ever substituted or calculated in its place.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PublicOffer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    room_name: str | None = None
    board_type: str | None = None
    board_name: str | None = None

    # The provider's refundability tag for the offer: True when every rate is
    # refundable (RFN), False when every rate is non-refundable (NRFN), None
    # when it is missing or mixed. Refundable is NOT free cancellation: a
    # refundable rate may still carry cancellation charges, so nothing here
    # states that cancelling is free.
    refundable: bool | None = None
    # Provider cancellation-policy times exactly as given ("2026-11-20
    # 10:00:00 GMT"), kept uninterpreted for a later booking step: their
    # meaning (free until / charged from) is not confirmed, so they are never
    # turned into a deadline or into listing evidence. Amounts are not kept.
    cancel_policy_times: list[str] = Field(default_factory=list)
    max_occupancy: int | None = None

    public_total: float | None = Field(default=None, gt=0)
    currency: str | None = None
    price_basis: Literal["total_stay"] = "total_stay"
    price_source: Literal["suggested_selling_price"] = "suggested_selling_price"

    # Taxes and fees the provider marks as not included in the price (paid
    # at the property, e.g. city tax) - descriptions only, no amounts.
    excluded_taxes_and_fees: list[str] = Field(default_factory=list)

    @property
    def refundability_label(self) -> str:
        if self.refundable is True:
            return "Refundable rate"
        if self.refundable is False:
            return "Non-refundable rate"
        return "Unknown"

    @model_validator(mode="after")
    def _price_needs_currency(self) -> PublicOffer:
        if self.public_total is not None and not self.currency:
            raise ValueError("public_total requires a currency")
        return self
