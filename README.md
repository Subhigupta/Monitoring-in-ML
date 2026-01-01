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
    - Execute `python main_ab_test.py` to conduct offline A/B testing on the holdout dataset.
    - Holdout data is deterministically  routed into control and treatment.
    - For each data point in the holdout set, a prediction is made using either of the two variants:
        - Control: Random Forest model
        - Treatment: XGBoost model
    - Deterministic routing ensures that each data point is consistently assigned to the same A/B variant using a hash-based strategy, enabling reproducible and fair model comparison.
    - Predictions are logged with:
        - request_id
        - experiment_id
        - variant
        - model
        - features
        - prediction
        - ground truth
        - latency_ms
        - error

    - Logs are stored at
        `logs/<experiment_id>_predictions.jsonl`
    
- **Metrics Evaluation and Comparison**
    - Model performance is evaluated per variant on the A/B test holdout dataset using:
        - R² score
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
    - Computed metrics are stored at `evaluation/model_comparison_v1_metrics.json`
    - A comparative analysis is performed between control and treatment, including:
        - Percentage improvement of the treatment model over the control.
        - Direction-aware comparisons (e.g., lower-is-better metrics such as MAE and RMSE are handled correctly).

- **Statistical Significance Testing**
    - Hypotheses
        - H₀ (Null Hypothesis):The mean absolute prediction error of the control variant equals that of the treatment variant.
        - H₁ (Alternative Hypothesis):The mean absolute prediction errors of the control and treatment variants differ.

    - Methodology
        - Per-sample prediction errors are logged during A/B testing.
        - Absolute prediction errors are derived from logged predictions
        - Welch’s t-test (unequal variance) is applied to compare the distributions of absolute errors between control and treatment variants.

    - Outputs Generated
        - p-value
        - Statistical significance flag
        - Winning variant
        - Recommendation (promote / continue / stop experiment)

**🚀 Future Enhancements**
- Real-time inference API
    - Expose trained models via FastAPI/Flask for online predictions.
- Production A/B testing on live traffic
    - Route real user requests to control and treatment models.
- Interactive experiment dashboard
    - Visualize metrics, drift, and experiment outcomes (Streamlit / Evidently).
- Multi-variant experimentation (A/B/n)
    - Compare more than two models simultaneously.
