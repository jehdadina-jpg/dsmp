"""Fare estimation and environmental-impact modeling for a computed route.

Two independent, cheap-to-compute enrichments layered on top of a route's
total distance -- neither needs its own graph or state, so they live as
pure functions the routing layer (and, via the embedded JSON payload, the
dashboard's client-side route planner) can both call.
"""

from __future__ import annotations

# Indian Railways Mumbai suburban 2nd-class single-journey fare slabs (INR).
# These are the standard distance-slab fares published for suburban EMU
# travel; ordered (max_km_inclusive, fare_inr) so the first slab a distance
# falls under wins. Real fares vary slightly by exact zone boundary and are
# revised periodically -- treat this as a "documented estimate", the same
# tolerance this project already applies to footfall (see data_loader.py).
_FARE_SLABS_2ND_CLASS: list[tuple[float, int]] = [
    (5, 5), (10, 10), (15, 15), (20, 15), (30, 20),
    (40, 25), (50, 30), (60, 35), (80, 40), (120, 50),
]

# Approximate lifecycle-average CO2 intensity, grams per passenger-km.
# Electric suburban EMU: India's grid-average generation mix (~700 gCO2/kWh)
# spread over a full 12/15-car rake at typical suburban loading, roughly
# consistent with published Indian Railways electric-traction estimates.
# Private car: a single-occupant petrol hatchback/sedan, the common baseline
# used in modal-shift comparisons. Both are order-of-magnitude estimates for
# an illustrative comparison, not a certified emissions audit.
_CO2_G_PER_KM_TRAIN = 41.0
_CO2_G_PER_KM_CAR = 121.0


def estimate_fare_inr(distance_km: float) -> int:
    """2nd-class single-journey suburban fare (INR) for a given distance."""
    for max_km, fare in _FARE_SLABS_2ND_CLASS:
        if distance_km <= max_km:
            return fare
    return _FARE_SLABS_2ND_CLASS[-1][1]


def estimate_co2_saved_kg(distance_km: float) -> float:
    """CO2 (kg) saved by taking the train instead of driving alone, one-way."""
    saved_g = (_CO2_G_PER_KM_CAR - _CO2_G_PER_KM_TRAIN) * distance_km
    return round(saved_g / 1000.0, 2)
