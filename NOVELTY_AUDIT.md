# Novelty Audit — RG-SCSO (SOP N1 / Q1)

> Full-text / abstract audit of the negative novelty claim: *"across the
> SCSO-family literature we screened, we found none that makes the
> binarization operator itself per-feature and relevance-aware."*
> Method: each reference's binarization/transfer mechanism checked against two
> questions — (a) does it binarize via a transfer function? (b) is that
> binarization **per-feature relevance-modulated** (a per-feature
> importance/MI signal biases each bit's flip), or feature-agnostic?
> Access: PMC open-access full text where available; publisher
> abstract/metadata + CrossRef otherwise (MDPI full text blocks automated
> fetch — marked accordingly). Date 2026-10-03.

## Directly-comparable binary SCSO feature selectors

| Ref | Paper | Verified via | Binarization mechanism | Per-feature relevance-aware? | Differs from RG-SCSO |
|---|---|---|---|---|---|
| bscso | Seyyedabbasi 2023, *Binary SCSO for Wrapper FS on Biological Data* (Biomimetics 8(3):310) | **PMC10807367 full text** | **V-shaped** transfer, Eq. 7 `V(x)=2/π·arctan(π/2·x)`, applied identically to all features | **NO** — same transfer for every feature, no MI/importance bias | RG-SCSO modulates each feature's flip by an MI prior |
| scsofs2 | Pashaei 2023, *Efficient Binary SCSO (PILC-BSCSO) for FS in High-Dim Biomedical* (Bioengineering 10(10):1123) | **PMC10604175 full text** | **tansig** transfer + threshold; PIOBL + crossover act on binary solutions; DE pre-filter at **initialization** | **NO** at binarization — relevance (DE) is injected **upstream** (init), transfer stays feature-agnostic | RG-SCSO injects at the binarization operator, not upstream — exactly the distinction this paper tests (RQ3) |
| scsofs3 | Liu 2024, *Novel Adaptive SCSO for FS and Global Optimization* (Biomimetics 9(11):701) | abstract + CrossRef (MDPI full text 403) | **adaptive** transfer function (time/convergence-varying) | **NO** — adaptive in time but still applied identically across features | RG-SCSO's modulation is per-feature by relevance, orthogonal to an adaptive-in-time transfer |

## Continuous-space SCSO variants (not feature selectors — no binarization operator)

| Ref | Paper | Verified via | What it modifies | Binarization for FS? |
|---|---|---|---|---|
| imscso2024 | Zhang 2024, Improved Multi-Strategy SCSO (Biomimetics 9(5):280) | CrossRef + title/abstract | continuous search (multi-strategy) | **No** — global optimization only |
| mescso2025 | Lin 2025, Multi-Strategy Enhanced SCSO (Sci. Rep. 15:34391) | CrossRef + title/abstract | continuous search + engineering apps | **No** |
| scsolensobl2024 | Cai 2024, SCSO + Lens-OBL + SSA (Sci. Rep. 14:20690) | CrossRef + title/abstract | continuous init/search (lens OBL) | **No** |
| improvedscso2024 | Jia 2024, Improved SCSO for Global Optimum (Artif. Intell. Rev. 58:5) | CrossRef + title/abstract | continuous search dynamics | **No** |

## Verdict

- **Directly-comparable (binary SCSO FS):** all three use a feature-agnostic transfer (V-shaped / tansig / adaptive-in-time); none makes the binarization **per-feature relevance-modulated**. Pashaei injects relevance (DE) at *initialization*, which is the upstream-injection alternative RG-SCSO explicitly compares against (signal-position experiment, RQ3).
- **Continuous SCSO variants:** not feature selectors; no binarization operator to make relevance-aware.
- **Claim status:** the negative novelty claim is **SUPPORTED for the screened SCSO-family literature** (2 full-text-verified, 1 abstract-verified binary FS; 4 continuous non-FS). It is correctly stated as "literature we screened," not an exhaustive first-ever claim.
- **Residual limitation:** scsofs3 full text not machine-accessible (MDPI 403); verified from abstract/metadata only → that single row is NOT_VERIFIABLE at full-text depth, verified at abstract depth. Recorded honestly rather than overstated.
