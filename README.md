**🚲 Bike Sharing Demand – End-to-End ML & A/B Testing System**

This repository contains an end-to-end machine learning pipeline built on the Bike Sharing Demand dataset.

It covers EDA → Feature Engineering → Offline Model Training → Offline A/B Testing → Evaluation → Monitoring, with a strong focus on production-style ML workflow.

**📌 Project Goals**

- Build regression models to predict bike rental demand
- Implement offline (simulated) A/B testing using historical data
- Compare models using A/B testing
- Log predictions for offline analysis and evaluation
- Compute performance metrics and improvements
- Perform statistical significance testing to support model promotion decisions
- Conduct data-drift monitoring
- Run inference through pre-trained models

**🛠️ Tech Stack**
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Random Forest
- ONNX Runtime
- SciPy (statistical tests)

**⚙️ Workflow Overview**

- **Data Preparation**
    - Load the data from `data/train.csv`.
    - Perform data cleaning and feature engineering.
    - Split the dataset into training, validation, and A/B testing (holdout) subsets.

- **Model Training**
    - Execute `python main_train.py` to train regression models (Random Forest and XGBoost) using the training dataset only.
    - Use the validation dataset for model evaluation.
    - Save trained models to the `models/` directory.
        
- **A/B Testing**
    - Execute `python main_ab_test.py` to conduct A/B testing on A/B testing holdout dataset.
    - Holdout data is routed deterministically into control and treatment.
    - Predictions are logged with:
        - request_id
        - variant
        - model
        - prediction
        - ground truth
        - latency

    - Logs are stored to
        `logs/<experiment_id>_predictions.jsonl`
    
- **Metrics Evaluation**
    - Metrics computed per variant:
        - R² score
        - MAE
        - RMSE
        - Sample size

    - Comparison includes:
        - % improvement of treatment over control
        - Direction-aware (lower-is-better handled correctly)

- **Statistical Significance Testing**
    - Uses Welch’s t-test on absolute error
    - Outputs:
        - p-value
        - statistical significance
        - winning variant
        - recommendation

**🚀 Future Enhancements**
- Real-time inference API
- Automated drift detection
- Dashboard (Streamlit / Evidently)
- Multi-variant (A/B/n) testing
- Power analysis for sample sizing
