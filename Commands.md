# Commands Used to Generate All Slide Outputs

## 1. Train Models & Generate All Experiment Metrics (Slides 14, 18, 19)
```powershell
python -m src.train
```
* **Generates:**
  * `outputs/models/best_model.joblib` (Saved winning model: `ANOVA + Decision Tree`)
  * `outputs/tables/results_all_combinations.csv` (All 40 selector × classifier results)
  * `outputs/tables/selected_features.csv`
  * `outputs/tables/best_model_summary.csv`

---

## 2. Generate Evaluation Figures & Per-Class Reports (Slides 14, 15)
```powershell
python -m src.evaluate
```
* **Generates:**
  * `outputs/figures/accuracy_heatmap.png` (**Slide 14** accuracy heatmap)
  * `outputs/figures/confusion_matrix_best.png` (**Slide 15** confusion matrix)
  * `outputs/tables/per_class_metrics.csv` (**Slide 15** per-class metrics)
  * `outputs/tables/classification_report_best.txt`

---

## 3. Launch Web App for UI Prediction & Recommendation Screenshot (Slide 17)
```powershell
streamlit run app/streamlit_app.py
```
* **Generates:** Local web interface (`http://localhost:8501`) to upload a test leaf image and screenshot the prediction, severity, and fertilizer recommendation output for **Slide 17**.

---

## 4. Extract Exact Numbers for Slide 18 (Stage 4 Overall Metrics)
```powershell
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/results_all_combinations.csv'); b = df.iloc[0]; print(f'Accuracy: {b[\"test_accuracy\"]*100:.2f}%, Macro Precision: {b[\"macro_precision\"]*100:.2f}%, Recall: {b[\"macro_recall\"]*100:.2f}%, F1: {b[\"macro_f1\"]*100:.2f}%')"
```
* **Generates:** Exact `Accuracy`, `Macro Precision`, `Macro Recall`, and `Macro F1` percentages for **Slide 18**[cite: 1].

---

## 5. Extract Exact Table Rows for Slide 19 (Model Comparison Table)
```powershell
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/results_all_combinations.csv'); f1_col = [c for c in df.columns if 'f1' in c.lower()][0]; cv_col = [c for c in df.columns if 'cv' in c.lower()][0]; order = ['knn','svm','gaussian_nb','logreg','mlp','random_forest','decision_tree','xgboost']; df['rank'] = df['classifier'].map({c:i for i,c in enumerate(order)}); best = df.sort_values([f1_col, cv_col], ascending=False).groupby('classifier', as_index=False).first().sort_values('rank'); print(best.drop(columns=['rank']).to_string(index=False))"
```
* **Generates:** Best feature-selection setting and metrics for all 8 classifiers in the exact order of **Slide 19**[cite: 2, 3].

### Extract Specific KNN Row (`knn` + `rf_importance`) for Slide 19
```powershell
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/results_all_combinations.csv'); print(df[(df['classifier']=='knn') & (df['selector']=='rf_importance')].to_string(index=False))"
```
* **Generates:** Exact `Macro Precision` and `Macro Recall` values for the `KNN` row in **Slide 19**[cite: 2, 3].
