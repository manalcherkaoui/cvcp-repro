# cvcp-repro

Replication code for a scenario-based, context-aware recommender system (RecSys 2025).

## 🔍 Overview

This repository contains a complete implementation of the CVCPR method — a lightweight, interpretable recommender system based on contextual variable clustering and scenario mapping.

The method addresses key challenges in recommender systems:
- **Sparsity** by clustering contextual data
- **Explainability** through fuzzy rule-based labeling
- **Personalization** via scenario-based matching tailored to partial user context

## ✅ Pipeline Stages

1. **Clustering**: Each contextual variable `c_i` is clustered independently.
2. **Labeling**: Fuzzy membership functions assign interpretable labels to clusters.
3. **Scenario Construction**: We build high-quality contextual scenarios by correlating clusters across variables.
4. **Recommendation**: Given a partial user context, the system recommends controllable values based on the closest scenario.

## 📥 Input Format

To ensure compatibility, please respect the following input format:

- Each contextual variable `c_i` must be an `m × n` matrix:
  - `m` = number of entities (e.g., users, regions)
  - `n` = number of temporal or spatial measurements
- The variable names must follow the convention `c1`, `c2`, ..., `cN`
- The target variable `Y` must be provided for evaluation
- Input must be clean and numeric (missing values should be pre-imputed)

## ⚙️ Reproducibility Notes

This version uses placeholder data to demonstrate functionality. Due to dataset confidentiality, real data is not included.

## 📦 Running the Code

```bash
python cvcp_function.py
