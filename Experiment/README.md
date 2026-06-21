# Experiment

The field experiment: groups of children aged 11–13 played schoolyard tag under two framings, *playing for fun* and *playing to win*. This folder holds the data-collection script, the analysis code, the rendered analyses, and the figures.

## Rendered analyses

Click to view each report (rendered through htmlpreview):

- [01 — H1: prior predictive checks](https://htmlpreview.github.io/?https://github.com/AddiH/Tag_thesis/blob/main/Experiment/html_output/01_H1_bayesian_priors.html)
- [02 — H1: Bayesian Poisson mixed model (+ encounter sensitivity)](https://htmlpreview.github.io/?https://github.com/AddiH/Tag_thesis/blob/main/Experiment/html_output/02_H1_bayes_poisson_model.html)
- [03 — H1: frequentist GLMM robustness check (lme4)](https://htmlpreview.github.io/?https://github.com/AddiH/Tag_thesis/blob/main/Experiment/html_output/03_H1_gee_poisson_model.html)
- [04 — H2: self-handicapping (Wilcoxon)](https://htmlpreview.github.io/?https://github.com/AddiH/Tag_thesis/blob/main/Experiment/html_output/04_H2_wilcoxon.html)
- [05 — Exploratory: transition entropy](https://htmlpreview.github.io/?https://github.com/AddiH/Tag_thesis/blob/main/Experiment/html_output/05_Exploratory_entropy.html)
- [06 — Main figures](https://htmlpreview.github.io/?https://github.com/AddiH/Tag_thesis/blob/main/Experiment/html_output/06_Plots.html)

## Structure

```
├── code            <- Analysis source (.Rmd for H1/H2, .ipynb for priors, entropy, plots)
├── html_output     <- Rendered analyses (linked above)
├── plots           <- Generated figures
└── run.sh          <- Renders the analyses in order
```

Raw participant data is not included, for privacy reasons; this folder holds code, rendered output, and figures only.
