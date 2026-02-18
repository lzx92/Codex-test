# Particle Design: 2D Lock-and-Key Magnetic Squares

This project now supports **auto-search + scoring** of lock/key edge shapes to balance:

1. **Final-bond anti-sliding confinement** (suppress secondary barcode registry), and
2. **Approach pathway width** (leave enough capture corridor so particles can still find each other).

## Physical model used for fast iteration

For each edge template, the script samples a protrusion profile `p(x)` and evaluates:

- **lock strength**: minimum extra normal gap required to avoid overlap when tangentially shifted by 0.5–2.5 µm,
- **approach window**: tangential shift range still admissible at a partial approach gap (0.95×tab_depth).

Higher lock strength + moderate/large approach window is preferred.

## Edge templates included

- `single_tooth`: simple centered key/notch with entry ramps.
- `dovetail`: necked lock with tapered sidewalls.
- `double_tooth_asym`: asymmetric dual keys (strong anti-sliding due to non-periodic constraint).

## Run auto-search (recommended)

```bash
python3 generate_particles.py --mode search --output-dir output --top-k 5
```

Outputs:

- `output/candidate_report.md` and `output/candidate_report.json`
- `output/candidate_01_*.svg ... candidate_05_*.svg`
  - `*_pair4.svg`: complementary A/B all-edge version (for 2D assemblies)
  - `*_selfY.svg`: single-shape Y-lock version (for 1D-like chains)

## Run baseline exports only

```bash
python3 generate_particles.py --mode baseline --output-dir output
```

## Parameter notes for 20 µm particles

- `tab_depth`: 1.0–1.8 µm (larger => stronger lock, less approach tolerance)
- `tab_width`: 3.0–5.0 µm (wider => better guidance but may reduce uniqueness)
- `ramp`: 0.5–0.9 µm (larger => easier approach and self-centering)
- `corner_chamfer`: ~1.1 µm default

## Suggested starting strategy

- For **1D chains**: use `candidate_*_selfY.svg` from top-ranked `dovetail` or `single_tooth` designs.
- For **2D structures**: use `candidate_*_pair4.svg`, typically `double_tooth_asym` candidates for stronger anti-slide locking.
- Keep your magnetic barcode polarity mapped consistently with edge polarity (A vs B) so geometric and magnetic selectivity reinforce each other.
