# Multi-Ratio ConceptMoE — Project Website

Project page for **One Model, Multiple Compute Budgets: Multi-Ratio ConceptMoE for Flexible Compute Across Turns**.

- Website: https://dagroup-pku.github.io/Multi-Ratio-ConceptMoE/
- Demo repository: https://github.com/DAGroup-PKU/Multi-Ratio-ConceptMoE
- Paper: https://arxiv.org/abs/2610.14074

## Local preview

```bash
python3 -m http.server 8767 --bind 127.0.0.1
```

Open http://localhost:8767. The website is static; no build step is required.

## Publication

This website lives in the `gh-pages` branch of [DAGroup-PKU/Multi-Ratio-ConceptMoE](https://github.com/DAGroup-PKU/Multi-Ratio-ConceptMoE). The `main` branch contains the demo code and its README. GitHub Pages serves the root of `gh-pages`; push changes to that branch to update the website. `.nojekyll` preserves the static files without a Jekyll build.

Results and the four paper figures follow arXiv:2610.14074v1. The compression-mode explorer is an illustrative schematic. The downloadable demo uses random weights and placeholder attention/MoE layers.

## Acknowledgements

Adapted from [StableVLA](https://github.com/DAGroup-PKU/StableVLA/tree/page), with the original template attribution and third-party licenses retained in `licenses/`.
