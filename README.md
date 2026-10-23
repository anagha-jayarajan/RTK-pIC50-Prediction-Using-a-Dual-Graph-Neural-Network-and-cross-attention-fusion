# RTK pIC50 Prediction Using a Dual Graph Neural Network

A graph-based deep learning framework for predicting small-molecule binding affinity across the human receptor tyrosine kinase (RTK) family using dual Graph Attention Network v2 (GATv2) encoders and cross-attention fusion.

**Project Status:** Ongoing MSc Data Science & BioAI Project — October 2026

---

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Current Progress](#current-progress)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Evaluation Strategy](#evaluation-strategy)
- [Initial Results](#initial-results)
- [Interpretability and Visualization](#interpretability-and-visualization)
- [Key Findings and Limitations](#key-findings-and-limitations)
- [Next Steps](#next-steps)
- [Project Assets](#project-assets)
- [Reproducibility Notes](#reproducibility-notes)
- [Technology Stack](#technology-stack)
- [Future Scope](#future-scope)
- [Disclaimer](#disclaimer)

---

## 1. Overview

Predicting protein–ligand binding affinity is an important task in computational drug discovery. This project investigates whether graph neural networks can learn structural relationships between small molecules and receptor tyrosine kinases (RTKs) to predict binding affinity.

The project focuses on predicting **pIC50 values** using a dual-graph neural network architecture. Instead of representing molecules using only conventional molecular descriptors, both the ligand and protein binding pocket are represented as graphs.

The model consists of:

1. **Ligand graph:** Represents molecular atoms, bonds, stereochemistry, and pharmacophore features.
2. **Protein-pocket graph:** Represents binding-pocket residues and their spatial relationships.
3. **Dual GATv2 encoders:** Learn representations of ligand and protein graphs independently.
4. **Cross-attention fusion:** Learns relationships between ligand atoms and pocket residues.
5. **Regression head:** Predicts the pIC50 value from the fused representation.

The model is evaluated using random, scaffold-based, and target-held-out splits to investigate its ability to generalize to unseen compounds and proteins.

## 2. Research Objectives

The main objectives of this project are:

- Develop a curated protein–ligand dataset for the RTK family.
- Construct graph representations of ligands and protein binding pockets.
- Develop a dual GATv2 model with cross-attention fusion.
- Predict continuous pIC50 values from structural representations.
- Evaluate the model using multiple data-splitting strategies.
- Investigate generalization to unseen molecular scaffolds and protein targets.
- Visualize learned atom–residue attention patterns.
- Establish a framework that can potentially be extended to other protein families.

## 3. Current Progress

| Component | Status |
|---|---|
| RTK family selection and scope definition | Completed |
| Protein metadata curation | Completed |
| Ligand graph construction | Completed |
| Protein-pocket graph construction | Completed |
| Protein–ligand activity dataset | Completed |
| Dual GATv2 model development | Completed |
| Cross-attention fusion module | Implemented |
| Initial training on three splits | Completed |
| Ligand-only and simpler GNN baselines | Pending |
| Hyperparameter tuning | Pending |
| Error analysis | Pending |
| Final interpretation figures | Pending |
| Docking cross-check | Optional future work |
| Extension to other protein families | Future work |

## 4. Dataset Description

The dataset combines protein structural information and ligand activity measurements from the following sources:

- **Protein Data Bank (PDB):** Protein structural information.
- **BindingDB:** Protein–ligand binding affinity records.
- **ChEMBL:** Bioactivity and compound information.

### Dataset Summary

| Component | Current Statistics |
|---|---:|
| RTK structures catalogued | 36 |
| Protein-pocket graphs constructed | 35 |
| Ligand graphs constructed | 87,921 |
| Usable protein–ligand pairs | 87,241 |
| Records skipped due to unavailable protein structure | 680 |
| Ligand atom feature dimensions | 36 |
| Ligand bond feature dimensions | 10 |
| Protein residue feature dimensions | 27 |
| Pocket graph connectivity threshold | 8 Å |

### Data Preprocessing

The preprocessing pipeline includes:

- Cleaning and correcting protein metadata.
- Updating UniProt identifiers, kinase-domain ranges, and residue-numbering offsets.
- Extracting binding-pocket residues using ligand-contact or catalytic-site strategies.
- Constructing protein-pocket graphs using spatial residue relationships.
- Converting ligand SMILES into molecular graphs.
- Incorporating stereochemistry and pharmacophore information.
- Matching protein structures with ligand activity records.
- Removing duplicate or incomplete records.
- Converting IC50 measurements to pIC50 values.

For IC50 expressed in molar units:

\[
pIC_{50}=-\log_{10}(IC_{50})
\]

The activity values must be converted using consistent units before calculating pIC50.

A validation check confirmed that 40 out of 40 sampled ligand graph records matched their corresponding SMILES strings.

**Dataset note:** The 680 skipped records are reported as NTRK2 records for which a corresponding protein structure was unavailable.

## 5. Methodology

The overall workflow consists of seven stages.

### Stage 1: Protein Family Selection

The receptor tyrosine kinase family was selected as the initial target family.

### Stage 2: Data Collection

Protein structures and ligand activity measurements were collected from PDB, BindingDB, and ChEMBL.

### Stage 3: Data Preprocessing

Protein metadata was corrected, activity records were cleaned, protein–ligand pairs were matched, and IC50 measurements were converted to pIC50.

### Stage 4: Graph Construction

Two graph representations were constructed independently:

- Ligand molecular graphs.
- Protein binding-pocket graphs.

### Stage 5: Model Development

Dual GATv2 encoders were combined with a cross-attention fusion module and an MLP regression head.

### Stage 6: Model Evaluation

Performance was evaluated using random, scaffold-based, and target-held-out splits.

### Stage 7: Interpretation and Extension

Attention visualizations, error analysis, and possible extension to other protein families are planned.

## 6. Model Architecture

The proposed model uses two graph encoders connected through a cross-attention module.

### 6.1 Ligand Graph

The ligand is represented as a molecular graph generated from SMILES.

- **Node features:** 36-dimensional atom features.
- **Edge features:** 10-dimensional bond features.
- **Additional information:** Stereochemistry and pharmacophore features.
- **Encoder:** Three-layer GATv2 with four attention heads and 128-dimensional representations.

### 6.2 Protein-Pocket Graph

The protein binding pocket is represented as a residue-level graph.

- **Node features:** 27-dimensional residue features.
- **Features include:** Amino-acid identity and pharmacophore information.
- **Edges:** Connections between residues whose Cα atoms are within 8 Å.
- **Encoder:** Three-layer GATv2 with four attention heads and 128-dimensional representations.

### 6.3 Cross-Attention Fusion

Cross-attention allows ligand atoms to attend to protein-pocket residues and protein-pocket residues to attend to ligand atoms.

This module learns relationships between the two graph representations without merging them into a single graph before encoding.

The resulting atom-by-residue attention weights are retained for subsequent interpretation and visualization.

### 6.4 Regression Head

The encoded representations are mean-pooled and concatenated before being passed through a multilayer perceptron:

`256 → 128 → 1`

The final layer outputs a continuous predicted pIC50 value.

### Model Size

The current model contains approximately **325,889 trainable parameters**.

A ligand bond-feature flow issue was identified and fixed before the initial training runs.

## 7. Training Configuration

The initial experiments were conducted using a GPU runtime in Google Colab.

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Batch size | 256 |
| Maximum epochs | 100 per split |
| Learning-rate scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 5 |
| Early stopping monitor | Validation RMSE |
| Early stopping patience | 15 epochs |
| Loss function | Mean Squared Error (MSE) |
| Target preprocessing | Standardized pIC50 |
| Mixed precision | Enabled |
| Random seed | 42 |
| Validation allocation | 10% of training data |

The best validation checkpoint was retained during training.

Early stopping did not activate during the initial runs because the validation curves were still improving near epoch 100.

## 8. Evaluation Strategy

Three evaluation strategies were used to investigate different types of generalization.

### 8.1 Random Split

An 80/20 split across the available protein–ligand pairs.

Reported split sizes:

- Training: 62,814 pairs
- Validation: 6,979 pairs
- Testing: 17,448 pairs

This split evaluates performance on randomly held-out examples, where related proteins or chemistry may be represented in training.

### 8.2 Scaffold Split

Molecules are grouped according to their Bemis–Murcko scaffolds. Approximately 20% of pairs are held out by scaffold.

The purpose is to evaluate how well the model generalizes to new molecular frameworks rather than relying on structurally similar compounds encountered during training.

### 8.3 Target-Held-Out Split

Seven complete RTK targets were held out:

- EGFR
- KIT
- FGFR1
- FGFR4
- AXL
- DDR1
- VEGFR1

Reported counts:

- Training and validation combined: 55,108 pairs
- Test: 32,133 pairs

This is the most challenging evaluation because the model must predict binding affinity for proteins that were not represented in training.

### Evaluation Metrics

The following metrics are used:

- **RMSE:** Root Mean Squared Error.
- **MAE:** Mean Absolute Error.
- **R²:** Coefficient of determination.
- **Pearson r:** Correlation between predicted and measured values.

Lower RMSE and MAE indicate smaller prediction errors. Higher R² and absolute Pearson correlation generally indicate stronger predictive performance, although correlation alone does not guarantee accurate predictions.

## 9. Initial Results

The following metrics are from the first full training run reported in the October 2026 project progress update.

| Evaluation Split | RMSE ↓ | MAE ↓ | R² ↑ | Pearson r ↑ |
|---|---:|---:|---:|---:|
| Random | 0.669 | 0.476 | 0.746 | 0.866 |
| Scaffold | 0.934 | 0.680 | 0.526 | 0.742 |
| Target-held-out RTKs | 1.477 | 1.142 | -0.167 | 0.312 |

### Results Interpretation

**Random split**

The model achieved an R² of 0.746 and Pearson correlation of 0.866, indicating strong predictive performance on randomly held-out pairs.

**Scaffold split**

Performance decreased to an R² of 0.526, suggesting that generalization becomes more difficult when the test compounds have unseen molecular scaffolds.

**Target-held-out split**

Performance decreased substantially, with an R² of -0.167 and Pearson correlation of 0.312.

A negative R² indicates that predictions on this test set were worse than using the mean of the observed test targets as the prediction.

These results highlight the difficulty of generalizing binding-affinity predictions to unseen proteins.

### Validation RMSE

| Split | Best Validation RMSE | Approximate Best Epoch |
|---|---:|---:|
| Random | 0.686 | 97 |
| Scaffold | 0.615 | 96 |
| Target | 0.647 | 98 |

The validation curves were still improving near epoch 100. Longer training and further tuning are planned.

### Earlier XGBoost Reference Experiment

An earlier molecular-fingerprint XGBoost experiment provides additional context.

| Model | RMSE | MAE | R² | Pearson r |
|---|---:|---:|---:|---:|
| Ligand fingerprint only | 0.966 | 0.761 | 0.480 | 0.709 |
| Fingerprint + protein identity | 0.885 | 0.694 | 0.564 | 0.772 |

These are earlier reference results and are not a controlled comparison with the current GNN results. The GNN baselines, including ligand-only and no-cross-attention variants, still need to be completed.

## 10. Interpretability and Visualization

Four interpretation pipelines have been implemented and tested. Final figures from the selected trained model remain to be generated.

### 10.1 Cross-Attention Heatmaps

Visualize attention weights between ligand atoms and protein-pocket residues.

### 10.2 Bipartite Interaction Graphs

Represent ligand atoms and pocket residues as two node sets, with edges corresponding to selected high-attention relationships.

### 10.3 2D Ligand Visualization

Use RDKit to generate 2D ligand depictions with atoms coloured according to their attention values.

### 10.4 3D Binding-Pocket Visualization

Visualize protein binding pockets and annotate relevant residues.

**Interpretation note:** Attention visualizations show model-derived relationships. They should not automatically be treated as experimentally validated interactions or causal explanations of binding.

## 11. Key Findings and Limitations

The primary finding so far is that model performance declines as evaluation moves from familiar protein–ligand pairs to unseen scaffolds and unseen protein targets.

Potential factors requiring investigation include:

- Only 34 unique pockets were used by the model, limiting protein structural diversity.
- The project notes that 18 pocket structures are AlphaFold models; the impact of predicted versus experimental structures remains to be tested.
- The target-held-out test set represents a substantial share of all pairs and is imbalanced across targets.
- EGFR contributes a large number of test examples.
- Ligand-only and no-cross-attention GNN ablations are still pending.
- Hyperparameter tuning and detailed error analysis have not yet been completed.
- Validation RMSE was still improving near the end of the initial 100-epoch runs.

These are potential explanations and research questions, not established causes of the observed performance gap.

## 12. Next Steps

The planned next steps are:

1. Complete ligand-only and simpler baseline experiments on all three splits.
2. Tune hyperparameters and investigate longer training.
3. Conduct per-protein, per-ligand, and RTK-subfamily error analysis.
4. Run ablation studies to evaluate the contribution of cross-attention.
5. Evaluate alternative target splits, including subfamily-wise hold-outs.
6. Generate final attention heatmaps, interaction graphs, and 2D/3D visualizations.
7. Optionally perform docking-based cross-checks for selected predicted binders.
8. Finalize the research report and package a reproducible demo.
9. Investigate extensions to GPCRs, proteases, and ion channels.

## 13. Project Assets

The current project progress records identify the following assets.

| File or Directory | Description |
|---|---|
| `protein_master_table.csv` | Corrected metadata for 36 RTK structures |
| `pocket_graphs_with_pharmacophore/` | 35 pocket graph files with 27-dimensional node features |
| `ligand_graphs/` | 87,921 ligand graph files with 36-dimensional atom features |
| `rtk_pic50_final_v2.csv` | Master ligand–protein–pIC50 table |
| Model and training code | Dataset handling, graph encoders, cross-attention, regression head, and training loop |
| Visualization code | Attention heatmaps, interaction graphs, and 2D/3D figures |

The final repository structure should be checked against the actual files before publication. Large datasets, model checkpoints, credentials, and files that cannot legally be redistributed should not be committed without appropriate review.

## 14. Reproducibility Notes

- Use the corrected 36-row `protein_master_table.csv`; older versions may exist.
- Preserve the mapping between ligand graphs and the master activity table.
- Run the documented splits in order: random, scaffold, then target.
- Record split assignments, random seeds, package versions, and checkpoint-selection criteria.
- Use a GPU runtime where available.
- The project notes a PyTorch 2.6 graph-loading compatibility issue involving `add_safe_globals([Data])` and `weights_only=False`. Only load trusted files when using `weights_only=False`, because loading untrusted serialized objects can execute unsafe code.
- Re-run data validation checks whenever the master table or graph files change.
- Keep evaluation splits consistent when comparing different models.

Exact installation commands, package versions, notebook names, and execution scripts were not specified in the current progress materials. These should be added once the repository environment is finalized.

## 15. Technology Stack

The current workflow uses or references:

- **Python** — data processing and model development
- **PyTorch** — deep learning
- **PyTorch Geometric** — graph neural network implementation; confirm the installed version
- **GATv2** — graph attention encoders
- **RDKit** — molecular graph processing and ligand visualization
- **Google Colab** — GPU-based training environment
- **PDB, BindingDB, and ChEMBL** — structural and bioactivity data sources

Exact dependency versions should be documented in a `requirements.txt` or environment file after checking the project environment.

## 16. Future Scope

The current work focuses on the human receptor tyrosine kinase family. Non-receptor tyrosine kinases were catalogued during project scoping but are outside the current modelling scope.

Potential future extensions include:

- G protein-coupled receptors (GPCRs)
- Proteases
- Ion channels

These extensions are future directions and are not represented by the current RTK model results.

## 17. Disclaimer

This repository documents an ongoing computational research project. Predicted pIC50 values and attention visualizations are model outputs, not experimental measurements or proof of binding. Results should be interpreted in the context of the evaluation design, baseline comparisons, and limitations described above.

## Citation

Add the repository citation and relevant dataset and methodology references before public release. Once a stable public version or accompanying research report is available, include its citation here.

