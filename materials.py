from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    elastic_modulus: float  # Pa
    # Yield strength for metals. Wood has no clear yield point, so its
    # entry uses bending strength (modulus of rupture) instead.
    strength: float  # Pa


MATERIALS = {
    m.name: m
    for m in (
        Material("Structural steel (A36)", 200e9, 250e6),
        Material("Aluminum 6061-T6", 69e9, 276e6),
        Material("Douglas fir", 12.4e9, 40e6),
    )
}
