# Interpretable Machine Learning Screening of High Property Copper Alloy Systems via GAM-SHAP Framework

Open-source code for the research paper "Interpretable Machine Learning Screening of High Property Copper Alloy Systems via GAM-SHAP Framework". This project focuses specifically on precipitation-strengthened copper alloys, implementing an interpretable machine learning pipeline for predicting material properties using SHAP-based feature selection and analysis.

## Project Overview

This repository contains the complete implementation of an interpretable machine learning approach for screening high-performance precipitation-strengthened copper alloy systems. The project focuses on predicting two critical properties:
- **Ultimate Tensile Strength (UTS)**: Mechanical property indicating maximum stress a copper alloy can withstand
- **Electrical Conductivity (EC)**: Electrical property measuring a copper alloy's ability to conduct electric current

The methodology employs SHAP (SHapley Additive exPlanations) for interpretable feature analysis, Boruta algorithm for robust feature screening, and advanced hyperparameter tuning to achieve high prediction accuracy specifically for precipitation-strengthened copper alloys.

## Project Structure

```
├── Data preprocessing/
│   ├── Mean_feature_descriptor.py      # Mean-based feature descriptors
│   ├── Variance_feature_descriptor.py  # Variance-based feature descriptors
│   └── supplementary materials.xlsx    # Additional material data
├── Feature engineering/
│   ├── bsx-ec.ipynb                   # Feature selection for EC prediction
│   ├── bsx-uts.ipynb                  # Feature selection for UTS prediction
│   └── xgb_wrapper.py                 # XGBoost wrapper utilities
└── Model training/
    ├── Algorithm selection/
    │   ├── model.xlsx                 # Model comparison results
    │   ├── model_comparison.xlsx      # Detailed model comparisons
    │   ├── model_comparison2.xlsx     # Extended model analysis
    │   └── selection.ipynb            # Algorithm selection analysis
    ├── EC---xgboost/
    │   ├── EC-xgboost.ipynb          # XGBoost model for EC prediction
    │   ├── calculated_properties_EC.xlsx  # EC dataset
    │   ├── calculated_properties_S.xlsx   # Supplementary dataset
    │   ├── catboost_results.xlsx     # CatBoost comparison results
    │   └── xgb_wrapper.py            # XGBoost utilities
    └── UTS---catboost/
        ├── UTS-CATBOOST.ipynb        # CatBoost model for UTS prediction
        ├── calculated_properties_EC.xlsx  # Dataset for UTS modeling
        └── xgb_wrapper.py            # Model utilities
```

## Key Features

### 1. SHAP-based Interpretable Framework
- **Interpretable ML**: SHAP-based framework for transparent feature analysis and model interpretation
- **Boruta Feature Selection**: Automated feature selection specifically tuned for copper alloy properties
- **SHAP Integration**: Explainable AI for understanding feature contributions in copper alloy design
- **Multi-stage Selection**: Three-stage feature selection process optimized for precipitation-strengthened systems

### 2. Copper Alloy Specific Hyperparameter Tuning
- **Optuna Integration**: Bayesian hyperparameter optimization for copper alloy property prediction models
- **Cross-validation**: Stratified K-fold cross-validation for robust copper alloy model evaluation
- **Multi-metric Optimization**: MSE, RMSE, MAE, and R² score optimization for UTS and EC properties

### 3. Specialized Algorithms for Copper Alloys
- **XGBoost**: Gradient boosting optimized for copper alloy electrical conductivity prediction
- **CatBoost**: Categorical boosting fine-tuned for copper alloy ultimate tensile strength prediction
- **Algorithm Comparison**: Systematic comparison of ML algorithms for precipitation-strengthened copper alloys

### 4. Comprehensive Evaluation for Copper Systems
- **Performance Metrics**: MSE, RMSE, MAE, R² score specifically validated on copper alloy datasets
- **Feature Importance**: Visualization and analysis of metallurgical feature contributions
- **Cross-validation**: Robust model validation techniques for copper alloy property prediction

## Usage

### 1. Data Preprocessing
Run the preprocessing scripts to generate feature descriptors:
```python
python "Data preprocessing/Mean_feature_descriptor.py"
python "Data preprocessing/Variance_feature_descriptor.py"
```

### 2. Feature Engineering
Execute feature selection notebooks:
- For EC prediction: `Feature engineering/bsx-ec.ipynb`
- For UTS prediction: `Feature engineering/bsx-uts.ipynb`

### 3. Model Training
#### For EC Prediction (XGBoost):
```python
# Navigate to EC---xgboost folder and run:
jupyter notebook EC-xgboost.ipynb
```

#### For UTS Prediction (CatBoost):
```python
# Navigate to UTS---catboost folder and run:
jupyter notebook UTS-CATBOOST.ipynb
```

### 4. Algorithm Selection
Compare different algorithms using:
```python
jupyter notebook "Model training/Algorithm selection/selection.ipynb"
```


## Results

The SHAP-based framework achieves high prediction accuracy for precipitation-strengthened copper alloy properties:
- **EC Prediction**: R² > 0.98 using optimized XGBoost for copper alloy electrical conductivity
- **UTS Prediction**: R² > 0.96 using optimized CatBoost for copper alloy ultimate tensile strength

## Model Performance for Copper Alloys

### SHAP-based Feature Selection Results
- **EC Model**: 5 optimal metallurgical features selected from 200+ candidates for copper alloy EC prediction
- **UTS Model**: 5 optimal features selected from 200+ candidates for copper alloy UTS prediction
- **Feature Importance**: SHAP-based interpretable feature rankings specific to precipitation-strengthened copper systems

### Hyperparameter Tuning Results for Copper Alloy Systems
- **Hyperparameter Optimization**: 30-100 trials per model using Optuna for optimal model parameters
- **Cross-validation**: 5-fold stratified validation on precipitation-strengthened copper alloy samples
- **Performance Metrics**: Comprehensive evaluation using multiple metrics validated on copper alloy experimental data

## License

This project is licensed under the MIT License - see the LICENSE file for details.


For questions or collaborations, please contact [your-email@example.com]

## Acknowledgments

- Thanks to the open-source community for the excellent ML libraries
- Special thanks to the copper alloy research community for domain expertise in precipitation-strengthened systems
- Optuna team for the excellent hyperparameter optimization framework
- SHAP developers for enabling interpretable machine learning in metallurgy
