# PlantNutriAi — Project Commands & Pipeline Reference

## 1. Git Sync & Push Commands (Version Control)

### Check Local vs. Remote Changes
```bash
git status
git fetch origin
git log HEAD..origin/main --oneline

What it does: Checks which local files were modified or untracked, fetches the latest metadata from GitHub without changing your files, and lists any new commits waiting on origin/main.

Clean Local Conflicts & Pull Latest Code
PowerShell
git checkout -- README.md
Remove-Item -Recurse -Force data/raw
Remove-Item -Force outputs/data_source.txt, outputs/split.csv, outputs/tables/skipped_files.csv
git pull origin main
What it does: Resets local changes in README.md, removes untracked local files that would collide with files tracked on GitHub, and merges the latest code (train.py, evaluate.py, predict.py, app/, tests/) from origin/main.

Commit & Push All Updates to GitHub
Bash
git add .
git add -f outputs/ data/processed/
git commit -m "Add trained models, evaluation figures, metrics, and updated pipeline outputs"
git push origin main
What it does: Stages all updated source files and generated outputs (-f includes folders listed in .gitignore), saves a commit snapshot, and pushes everything to GitHub.

2. Machine Learning Pipeline Commands (Training & Evaluation)
Train All 40 Pipelines (5 Selectors × 8 Classifiers)
PowerShell
python -m src.train
What it does: Runs 5-fold stratified cross-validation and test evaluation across 40 pipelines on your dataset.

Outputs generated:

outputs/models/best_model.joblib — Saved winning pipeline (ANOVA + Decision Tree)

outputs/tables/results_all_combinations.csv — Accuracy, precision, recall, F1, and CV scores for all 40 combinations

outputs/tables/selected_features.csv — Selected feature subset for the best model

outputs/tables/best_model_summary.csv — Summary metrics for the winning pipeline

Generate Evaluation Figures & Reports
PowerShell
python -m src.evaluate
What it does: Evaluates the saved best model on the test set and generates visual plots and per-class metrics for your presentation slides.

Outputs generated:

outputs/figures/accuracy_heatmap.png — 5 × 8 pipeline accuracy heatmap (Slide 14)

outputs/figures/confusion_matrix_best.png — Confusion matrix for the winning pipeline (Slide 15)

outputs/tables/per_class_metrics.csv — Per-class precision, recall, and F1 (Slide 15)

outputs/tables/classification_report_best.txt — Detailed text classification report

Export Formatted Review Metrics (Optional)
PowerShell
python -m scripts.export_review_content
