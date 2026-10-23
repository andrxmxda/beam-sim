from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    elastic_modulus: float  # Pa
    strength: float  # Pa
    # Wood has no clear yield point, so its strength is the bending
    # strength (modulus of rupture) rather than a yield strength.
    strength_kind: str = "yield strength"


MATERIALS = {
    m.name: m
    for m in (
        Material("Structural steel (A36)", 200e9, 250e6),
        Material("Aluminum 6061-T6", 69e9, 276e6),
        Material("Douglas fir", 12.4e9, 40e6, "bending strength"),
    )
}
