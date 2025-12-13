# Model Card — City Risk Prediction Model (Baseline)

## Use case
Predict next-interval (T+1) risk by geo area and time window for multiple incident domains: **traffic, drugs, public_order, health, civil_defense, security**.

## Input data sources (synthetic)
- Incidents: 911 + multiple responding departments
- Weather conditions
- Social media signals
- Trusted news signals
- Sports/major events signals

## Features
- Lag counts and rolling counts per domain (lags: [1, 2], rolling: [8])
- Calendar features (hour/day/month/weekend + synthetic holiday/Ramadan flags)
- Weather aggregates (temp/rain/wind)
- Signal aggregates (social/news/events/weather confidence sums)
- One-hot geo area

## Model type
Multi-label classifier using **RandomForest** with a **MultiOutput** wrapper (one classifier per domain).

## Evaluation
Time-based split; test window = last 30 days.
Thresholding mode: per_domain_tuned_on_validation

Metrics (per domain):
- traffic: ROC-AUC=0.633 | AP=0.553 | P/R/F1/Acc@0.45: 0.501/0.752/0.601/0.566 | positive_rate=0.435
- drugs: ROC-AUC=0.585 | AP=0.393 | P/R/F1/Acc@0.45: 0.357/0.745/0.482/0.483 | positive_rate=0.323
- public_order: ROC-AUC=0.701 | AP=0.803 | P/R/F1/Acc@0.35: 0.681/0.928/0.785/0.674 | positive_rate=0.642
- health: ROC-AUC=0.593 | AP=0.453 | P/R/F1/Acc@0.5: 0.428/0.577/0.491/0.558 | positive_rate=0.369
- civil_defense: ROC-AUC=0.530 | AP=0.147 | P/R/F1/Acc@0.5: 0.154/0.188/0.169/0.759 | positive_rate=0.131
- security: ROC-AUC=0.596 | AP=0.485 | P/R/F1/Acc@0.5: 0.462/0.508/0.484/0.578 | positive_rate=0.389

Overall mean ROC-AUC=0.6064180444891999 | mean AP=0.47267415507184435
Overall mean P/R/F1/Acc@None: 0.4303024250290648/0.6160827095729137/0.502137923524067/0.6031489186828992


## Limitations
- This model is trained on **synthetic data** and is for engineering integration only.
- Probability calibration and real-world validation are not performed.
- Spatial effects are simplified (geo areas are coarse).

## Privacy
No personal data is used. All text, locations, and IDs are synthetic.
