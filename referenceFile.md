# PlantNutriAi — Project Commands & Pipeline Reference

## 1. Git Sync & Push Commands (Version Control)

### Check Local vs. Remote Changes
```bash
git status
git fetch origin
git log HEAD..origin/main --oneline
```
* **What it does:** Checks which local files were modified or untracked, fetches the latest metadata from GitHub without changing your files, and lists any new commits waiting on `origin/main`.

### Clean Local Conflicts & Pull Latest Code
```powershell
git checkout -- README.md
Remove-Item -Recurse -Force data/raw
Remove-Item -Force outputs/data_source.txt, outputs/split.csv, outputs/tables/skipped_files.csv
git pull origin main
```
* **What it does:** Resets local changes in `README.md`, removes untracked local files that would collide with files tracked on GitHub, and merges the latest code (`train.py`, `evaluate.py`, `predict.py`, `app/`, `tests/`) from `origin/main`.

### Commit & Push All Updates to GitHub
```bash
git add .
git add -f outputs/ data/processed/
git commit -m "Add trained models, evaluation figures, metrics, and updated pipeline outputs"
git push origin main
```
* **What it does:** Stages all updated source files and generated outputs (`-f` includes folders listed in `.gitignore`), saves a commit snapshot, and pushes everything to GitHub.

---

## 2. Machine Learning Pipeline Commands (Training & Evaluation)

### Train All 40 Pipelines (5 Selectors × 8 Classifiers)
```powershell
python -m src.train
```
* **What it does:** Runs 5-fold stratified cross-validation and test evaluation across 40 pipelines on your dataset.
* **Outputs generated:**
  * `outputs/models/best_model.joblib` — Saved winning pipeline (`ANOVA + Decision Tree`)
  * `outputs/tables/results_all_combinations.csv` — Accuracy, precision, recall, F1, and CV scores for all 40 combinations
  * `outputs/tables/selected_features.csv` — Selected feature subset for the best model
  * `outputs/tables/best_model_summary.csv` — Summary metrics for the winning pipeline

### Generate Evaluation Figures & Reports
```powershell
python -m src.evaluate
```
* **What it does:** Evaluates the saved best model on the test set and generates visual plots and per-class metrics for your presentation slides.
* **Outputs generated:**
  * `outputs/figures/accuracy_heatmap.png` — 5 × 8 pipeline accuracy heatmap (**Slide 14**)
  * `outputs/figures/confusion_matrix_best.png` — Confusion matrix for the winning pipeline (**Slide 15**)
  * `outputs/tables/per_class_metrics.csv` — Per-class precision, recall, and F1 (**Slide 15**)
  * `outputs/tables/classification_report_best.txt` — Detailed text classification report

### Export Formatted Review Metrics (Optional)
```powershell
python -m scripts.export_review_content
```
* **What it does:** Exports all exact evaluation metrics formatted specifically for your review presentation into `outputs/review_content.md`.

---

## 3. Metric Extraction One-Liners (For Presentation Slides)

### Extract Overall Test Metrics (Slide 18 — Stage 4)
```powershell
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/results_all_combinations.csv'); b = df.iloc[0]; print(f'Accuracy: {b[\"test_accuracy\"]*100:.2f}%, Macro Precision: {b[\"macro_precision\"]*100:.2f}%, Recall: {b[\"macro_recall\"]*100:.2f}%, F1: {b[\"macro_f1\"]*100:.2f}%')"
```
* **What it does:** Reads the top-ranked row from `results_all_combinations.csv` and prints the exact percentages for **Accuracy**, **Macro Precision**, **Macro Recall**, and **Macro F1**.

### Extract Best Feature-Selection Setting per Classifier (Slide 19 Table)
```powershell
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/results_all_combinations.csv'); f1_col = [c for c in df.columns if 'f1' in c.lower()][0]; cv_col = [c for c in df.columns if 'cv' in c.lower()][0]; order = ['knn','svm','gaussian_nb','logreg','mlp','random_forest','decision_tree','xgboost']; df['rank'] = df['classifier'].map({c:i for i,c in enumerate(order)}); best = df.sort_values([f1_col, cv_col], ascending=False).groupby('classifier', as_index=False).first().sort_values('rank'); print(best.drop(columns=['rank']).to_string(index=False))"
```
* **What it does:** Groups all 40 experiment rows by classifier, selects the top feature-selection setting for each of the 8 classifiers, and prints them in the exact row order of Slide 19.

### Extract Specific KNN Row (`knn` + `rf_importance`)
```powershell
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/results_all_combinations.csv'); print(df[(df['classifier']=='knn') & (df['selector']=='rf_importance')].to_string(index=False))"
```
* **What it does:** Filters `results_all_combinations.csv` to print the exact metrics for the `KNN + RF Importance` combination.

---

## 4. Web Application Command (Slide 17 Screenshot)

### Launch the Streamlit UI
```powershell
streamlit run app/streamlit_app.py
```
* **What it does:** Starts the interactive **PlantNutriAi** web app (`http://localhost:8501`) where you can upload a test leaf image (e.g., `data/raw/banana/N_deficiency/banana_nitrogen_000.jpg`), inspect the predicted nutrient deficiency, HSV severity estimation, and corrective fertilizer recommendation, and capture a screenshot for Slide 17.
