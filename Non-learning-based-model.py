import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre, calc_rmse
from pypots.imputation import LOCF
from fancyimpute import KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_and_preprocess_data(file_path, nrows=None):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path, nrows=nrows)
    data = data.drop(['id', 'power_p'], axis=1)  # Drop unnecessary columns
    data.columns = data.columns.astype(str)  # Ensure column names are strings
    X = StandardScaler().fit_transform(data.to_numpy())  # Standardize the data
    return X

def introduce_missing_data(X, missing_rate):
    """Introduce missing data into the dataset using MCAR (Missing Completely at Random)."""
    Y = mcar(X, missing_rate)
    return Y

def evaluate_imputation(imputation, Y_ori, Y):
    """Evaluate the imputation results using MAE, MSE, MRE, and RMSE."""
    indicating_mask = np.isnan(Y) ^ np.isnan(Y_ori)  # Mask to identify imputed values
    mae = calc_mae(imputation, np.nan_to_num(Y_ori), indicating_mask)  # Mean Absolute Error
    mse = calc_mse(imputation, np.nan_to_num(Y_ori), indicating_mask)  # Mean Squared Error
    mre = calc_mre(imputation, np.nan_to_num(Y_ori), indicating_mask)  # Mean Relative Error
    rmse = calc_rmse(imputation, np.nan_to_num(Y_ori), indicating_mask)  # Root Mean Squared Error
    return mae, mse, mre, rmse

def locf_imputation(dataset):
    """Perform LOCF (Last Observation Carried Forward) imputation."""
    locf = LOCF()
    locf.fit(dataset)  # Train the model
    imputation = locf.impute(dataset)  # Perform imputation
    return imputation

def knn_imputation(Y, k=10):
    """Perform KNN (K-Nearest Neighbors) imputation."""
    knn_imputer = KNN(k=k)
    imputation = knn_imputer.fit_transform(Y)  # Perform imputation
    return imputation

def mean_imputation(Y):
    """Perform mean imputation."""
    Meandataset = pd.DataFrame(Y)
    imputation = Meandataset.fillna(Meandataset.mean()).to_numpy()  # Fill missing values with column means
    return imputation

def mice_imputation(Y, max_iter=10, random_state=0):
    """Perform MICE (Multiple Imputation by Chained Equations) imputation."""
    MICEImputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputation = MICEImputer.fit_transform(Y)  # Perform imputation
    return imputation

def run_experiment_for_missing_rate(X, missing_rate):
    """Run the experiment for a single missing rate."""
    Y = introduce_missing_data(X, missing_rate)  # Introduce missing data
    Y_ori = X  # Keep the original data for evaluation

    # LOCF Imputation
    dataset = {"X": Y.reshape(Y.shape[0], 1, Y.shape[1])}  # Reshape for LOCF
    locf_imp = locf_imputation(dataset)
    locf_metrics = evaluate_imputation(locf_imp, Y_ori, Y)

    # KNN Imputation
    knn_imp = knn_imputation(Y)
    knn_metrics = evaluate_imputation(knn_imp, Y_ori, Y)

    # Mean Imputation
    mean_imp = mean_imputation(Y)
    mean_metrics = evaluate_imputation(mean_imp, Y_ori, Y)

    # MICE Imputation
    mice_imp = mice_imputation(Y)
    mice_metrics = evaluate_imputation(mice_imp, Y_ori, Y)

    # Return results
    results = {
        "Missing Rate": missing_rate,
        "LOCF": locf_metrics,
        "KNN": knn_metrics,
        "Mean": mean_metrics,
        "MICE": mice_metrics,
    }
    return results

def run_experiment(file_path, missing_rates, nrows=None):
    """Run the experiment for multiple missing rates."""
    X = load_and_preprocess_data(file_path, nrows)  # Load and preprocess data
    all_results = []

    for rate in missing_rates:
        print(f"Running experiment for missing rate: {rate}")
        results = run_experiment_for_missing_rate(X, rate)  # Run experiment for each missing rate
        all_results.append(results)

    # Print results
    for result in all_results:
        print(f"\nMissing Rate: {result['Missing Rate']}")
        print("LOCF Metrics (MAE, MSE, MRE, RMSE):", result['LOCF'])
        print("KNN Metrics (MAE, MSE, MRE, RMSE):", result['KNN'])
        print("Mean Metrics (MAE, MSE, MRE, RMSE):", result['Mean'])
        print("MICE Metrics (MAE, MSE, MRE, RMSE):", result['MICE'])

# Example usage
file_path = 'test_data.csv'  # Path to the dataset
missing_rates = [0.1, 0.3, 0.5]  # List of missing rates to test
nrows = 10000  # Number of rows to load (optional)
run_experiment(file_path, missing_rates, nrows)