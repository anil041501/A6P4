# A6P4 code - Explainable LinUCB next-best-action for utility energy programmes

## Run
    pip install -r requirements.txt      # plus Graphviz `dot` on PATH
    python3 run_all.py                   # ~80 s on one CPU core, fixed seeds

`run_all.py` reads `../data/A6P4_energy_program_offers.csv` and writes everything to `../results/` and `../figures/`.

## What it does
1. Preprocessing per training fold: median imputation, IQR (3x) winsorising of kWh/area/age, z-scoring; 26-dim context.
2. Eligibility rule engine (5 programme rules) used as a hard mask; agreement with the logged `eligible` flag is reported.
3. 5-fold customer-disjoint CV (GroupKFold). In each fold:
   - online replay learning (Li et al., WSDM 2011) on the chronological training stream for LinUCB, LinUCB-kWh and epsilon-greedy;
   - rule playbook and popularity baselines;
   - frozen-policy IPS and doubly-robust (DR) acceptance and kWh estimates, HR@1 and NDCG@3 on held-out customers;
   - alpha sensitivity, ablation, explanation tests (additive fidelity, top-3 deletion, 5-NN stability, sign agreement).
4. Paired bootstrap (2,000 resamples) of pooled out-of-fold DR/IPS terms: LinUCB vs best baseline.
`accepted` and `annual_kwh_savings` are used only as rewards, never as features.

## Outputs (../results)
| File | Content |
|---|---|
| table_main_comparison.csv / main_comparison_per_fold.csv | Table IV (mean ± std) / per-fold values |
| table_ablation.csv / ablation_per_fold.csv | Table V |
| table_sensitivity_alpha.csv / sensitivity_alpha_per_fold.csv | Fig. 4 data |
| table_dataset_summary.csv / table_dataset_per_offer.csv | Table II, per-offer statistics |
| bootstrap_linucb_vs_best_baseline.csv | paired bootstrap differences (da = DR acceptance, dk = DR kWh, ia = IPS acceptance) |
| explanation_metrics_per_fold.csv | fidelity / deletion / stability metrics |
| linucb_coefficients_mean.csv | Fig. 5 data (mean theta over folds) |
| offer_mix_by_season.csv | Fig. 7 data |
| metrics.json | all headline numbers incl. selected alpha, sign agreement, local example (Fig. 6) |
| fig1..fig7 *.jpg | figure copies (PNG 300 dpi in ../figures) |
