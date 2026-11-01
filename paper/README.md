# Mathematical paper

The manuscript is [`orientation-convention-recovery.tex`](orientation-convention-recovery.tex).

The current draft includes the five-module local recovery engine and its product
boundary: prior-separated evidence packets, quotient-space compatibility, calibration
gates, atomic lineage checks, reference coalescence, and robust sharedness fusion for a
weak target. The executable packet and result contracts are maintained in the repository
schemas under `../schemas/`.

It is a mathematical and algorithmic description of the repository's convention
inference method, including its identifiability limits. A LaTeX distribution with
`amsmath`, `amssymb`, `amsthm`, `booktabs`, `enumitem`, and `hyperref` can compile it with:

```bash
tectonic orientation-convention-recovery.tex
```

`pdflatex` is also sufficient when a local TeX distribution is already configured.
