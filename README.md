# Breast Cancer Prediction

A supervised machine-learning study that classifies breast tumors as benign or malignant from diagnostic measurements. The repository covers exploratory analysis, preprocessing, model comparison, cross-validation and Random Forest hyperparameter tuning.

## Dataset

The notebook uses the Wisconsin Diagnostic Breast Cancer dataset structure:

- 569 observations
- 33 raw columns
- 357 benign cases
- 212 malignant cases
- Binary target: `diagnosis`

The dataset itself is not included in this repository. The current Python script reads it from a local path, so update the `pd.read_csv(...)` path before running the project.

## Selected predictors

The model comparison uses six diagnostic features:

- `radius_mean`
- `perimeter_mean`
- `area_mean`
- `symmetry_mean`
- `compactness_mean`
- `concave points_mean`

The workflow label-encodes the diagnosis, standardizes the predictors and uses a 55% training / 45% testing split with `random_state=20`.

## Models compared

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Classifier

### Saved notebook results

| Model | Test accuracy |
|---|---:|
| Random Forest | 92.61% |
| Logistic Regression | 92.22% |
| Support Vector Classifier | 92.22% |
| Decision Tree | 90.66% |

The notebook also produces classification reports, confusion matrices and a five-fold comparison. A 10-fold `GridSearchCV` experiment is included for Random Forest tuning. These results reflect the saved run and may change with a different split, seed or software version.

## Repository contents

- `Breast_Cancer_Prediction.ipynb` — complete analysis with saved outputs
- `Breast_Cancer_Prediction.py` — exported Python workflow
- `LetSintroduce.ipynb` — introductory notebook
- `Breast cancer prediction.pptx` — project presentation
- `Techincal Report.pdf` — technical report

## Run the project

### Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
```

1. Download the Wisconsin Diagnostic Breast Cancer data as a CSV file.
2. Update the dataset path in the notebook or Python script.
3. Start Jupyter:

```bash
jupyter notebook Breast_Cancer_Prediction.ipynb
```

4. Run the cells in order.

## Technical workflow

1. Inspect dataset shape, data types and class balance.
2. Remove empty columns and encode the target.
3. Explore feature distributions and relationships.
4. Select and standardize six predictors.
5. Train and compare four classifiers.
6. Review accuracy, precision, recall, F1 score and confusion matrices.
7. Tune the Random Forest configuration with grid search.

## Limitations

- The current holdout test set is derived from the same source dataset.
- The analysis does not include external clinical validation.
- Reported accuracy should not be interpreted as medical diagnostic performance.
- A production study should use stratified/nested validation, calibration analysis and an independent test cohort.

## Purpose

This repository is an academic machine-learning project. It is not a medical device and should not be used for clinical decision-making.
