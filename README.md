
# Logistic Regression with Genetic Algorithm (GA) Feature Selection

This project is an **interactive Streamlit web app** that demonstrates how to build a logistic regression classifier, apply preprocessing, and use a **Genetic Algorithm (GA)** for feature selection.

It supports:

* Default dataset (`data/data1.csv`) or user-uploaded CSV.
* Data preview and preprocessing (imputation, scaling, encoding).
* Logistic regression baseline model.
* GA-based feature selection with customizable GA parameters.
* Multiple **fitness equations**: `accuracy`, `F1-score`, `ROC-AUC`.
* Visualization of model performance before and after GA.

---

## Features

### 1. Data Handling

* Upload your own dataset (CSV) or use the default one.
* Choose the target column interactively.
* Preview raw and preprocessed data.

### 2. Preprocessing

* Numerical columns: imputation (median), standardization.
* Categorical columns: imputation (most frequent), one-hot encoding.
* Combined with `ColumnTransformer` to generate a clean feature matrix.

### 3. Logistic Regression

* Baseline model trained and evaluated on all features.
* Metrics: accuracy, precision, recall, F1, ROC-AUC.
* Visualization: ROC curve, confusion matrix.

### 4. Genetic Algorithm for Feature Selection

* Binary mask encodes which features to keep.
* Population evolves by crossover, mutation, and selection.
* Elitism ensures best solution is preserved across generations.
* Configurable GA parameters: population size, generations, crossover probability, mutation probability, tournament size, elitism, max features.
* Selectable **fitness equations** (see below).

---

## Fitness Equations

The GA searches for the feature subset (S) that maximizes a chosen fitness equation.

### (1) **Accuracy**

<img width="403" height="89" alt="image" src="https://github.com/user-attachments/assets/34dca7e5-0aab-468a-bb5f-5735a78bd68e" />

* Fraction of correct predictions.
* Good for balanced datasets.

---

### (2) **F1-score**

<img width="377" height="93" alt="image" src="https://github.com/user-attachments/assets/db38ab0b-cf6d-404c-9a2c-4c206cd74d7a" />


where

<img width="488" height="99" alt="image" src="https://github.com/user-attachments/assets/5e2fc982-601c-471c-b8c8-8d856e06141d" />


* Balances precision and recall.
* Useful for imbalanced classes.
* Requires the user to select the **positive class** in the UI.

---

### (3) **ROC-AUC**

<img width="352" height="74" alt="image" src="https://github.com/user-attachments/assets/e135a902-7091-41ca-b05e-3fbd2f4279ef" />


* Measures the area under the ROC curve for the positive class.
* Values between 0.5 (random) and 1.0 (perfect separation).
* Also requires the **positive class**.

---

## Project Structure

```
ml-ga-logreg/
├── app.py                      # main Streamlit entry point
├── pages/                      # multipage UI
│   ├── 1_Data_&_Preview.py
│   ├── 2_Preprocess.py
│   ├── 3_LogReg_Baseline.py
│   └── 4_GA_Feature_Selection.py
├── src/
│   ├── io_utils.py             # load config, CSV
│   ├── preprocess.py           # pipelines for numeric + categorical
│   ├── model.py                # logistic regression wrapper
│   ├── metrics.py              # accuracy, precision, recall, F1, ROC-AUC, curves
│   ├── ga.py                   # genetic algorithm core
│   └── ui.py                   # watermark / UI helpers
├── data/
│   └── data1.csv               # default dataset
├── assets/
│   └── logo.svg                # custom logo (also used as watermark)
├── config.yaml                 # default settings
├── requirements.txt
└── README.md                   # this file
|-- tests/                      # unit tests 
    ├── test_preprocess.py
    ├── test_model.py
    |── test_ga.py
    |-- test_metrics.py
    |-- test_io_utils.py
    |-- test_ui.py
    |-- conftest.py
    |-- test_metrics.py
    |-- test_preprocess_more.py
    |-- test_schema.py
    |-- test_preprocess_fallback.py
    |-- test_model_mask.py
    |-- test_ga_edges.py
    |-- test_io_utils_errors.py
    |-- test_metrics_exceptions.py
    |-- test_ui_empty.py
    
    
```

## Installation

1. Clone the repository.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run Streamlit:

   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Start app** → visit [http://localhost:8501](http://localhost:8501).
2. **Data & Preview** → view default dataset or upload your own CSV.
3. **Preprocess** → review numeric/categorical features and transformations.
4. **LogReg Baseline** → train & evaluate logistic regression on all features.
5. **GA Feature Selection**:

   * Adjust GA parameters (population, generations, crossover, mutation, etc.).
   * Select **fitness metric** (`accuracy`, `f1`, `roc_auc`) and, if needed, the **positive class**.
   * Run GA → see best feature subset and performance.
6. Compare baseline vs GA-selected results.


---

## Notes

* Default dataset is **Breast Cancer Diagnosis** (`diagnosis` as target).
* Any CSV with a labeled target column is supported.
* Performance depends on dataset, GA parameters, and chosen fitness equation.
* For large datasets, GA can be computationally expensive.

---

## License

MIT License. Free to use, modify, and distribute.


