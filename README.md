# Airbnb Price Prediction

This project focuses on predicting Airbnb listing prices using machine learning and deep learning techniques. The dataset includes various features such as number of beds, bedrooms, location, and more. The project includes data preprocessing, feature engineering, model building (Random Forest, Gradient Boosting, Voting Regressor, and Neural Networks), and performance evaluation using K-Fold cross-validation.

## 📁 Dataset

The dataset used is a cleaned and encoded CSV file containing Airbnb listing data. Key features include:

- `beds`, `bedrooms`
- `host_name`, `checkin`, `checkout`
- `country`
- `price` (target variable)

## 🛠️ Features Engineering

We created additional informative features to boost model performance:

- `bed_to_bedroom_ratio`
- `total_bed_capacity`
- `bedroom_bed_interaction`
- `average_beds_per_bedroom`

## 🧼 Data Preprocessing

- Checked and handled missing values (`host_name`, `checkin`, `checkout`)
- Label encoding or one-hot encoding for categorical variables
- Standard scaling of numerical features for neural networks

## 🤖 Models Implemented

- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `VotingRegressor` (ensemble of above two)
- `Neural Network` using TensorFlow Keras (with 5-Fold Cross Validation)

## 🔁 Cross Validation

Applied **5-Fold K-Fold Cross Validation** to evaluate the robustness of the models. Metrics used:

- Mean Squared Error (MSE)
- R² Score

### Best Model Results (Voting Regressor - Ensemble Model):

- **Average CV MSE**: `0.2361`
- **Average CV R²**: `0.7152`

The ensemble model that combines Random Forest and Gradient Boosting performed the best, with a strong **R² score** indicating good model accuracy in predicting the price of Airbnb listings.

### Neural Network Example Results:

```

Fold 1 — MSE: 0.5758, R2: 0.3056
Fold 2 — MSE: 1.0585, R2: 0.2193
Fold 3 — MSE: 0.3682, R2: 0.3876
Fold 4 — MSE: 0.4864, R2: 0.2912
Fold 5 — MSE: 1.2248, R2: 0.1971

Average CV MSE: 0.7428 ± 0.3364
Average CV R² : 0.2802 ± 0.0677

```

## 📊 Evaluation Metrics

- **MSE (Mean Squared Error)** for loss measurement
- **R² Score** for model performance
- Final ensemble and neural models evaluated on cross-validation scores

## 🧪 Technologies Used

- Python 3
- Pandas, NumPy
- scikit-learn
- TensorFlow / Keras
- Google Colab / Jupyter Notebooks

## 📂 Project Structure

```

📁 airbnb-price-prediction/
├── data/                      # Raw and processed datasets
├── notebooks/                # Jupyter/Colab notebooks
├── models/                   # Trained models (optional)
├── results/                  # Plots and metrics
├── README.md                 # Project overview

```

## ✅ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/touhidhtonu/airbnb-price-prediction.git
   cd airbnb-price-prediction
   ```
2. Run notebooks or Python scripts inside `notebooks/`.
