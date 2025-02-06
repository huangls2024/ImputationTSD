import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
import pandas as pd
data = pd.read_csv('test_data.csv')
data = data.drop(['id', 'power_p'], axis = 1)
X = StandardScaler().fit_transform(data.to_numpy())
X = X.reshape(2730, 48, -1)
X_ori = X  # keep X_ori for validation
X = mcar(X, 0.2)  # randomly hold out 10% observed values as ground truth
dataset = {"X": X}  # X for model input
print(X.shape)  # (2730, 48, 11), 11988 samples and each sample has 48 time steps, 37 features

# Model training. This is PyPOTS showtime.    iTransformer模型
from imputation import iTransformer
from utils.metrics import calc_mae
from utils.metrics import calc_mse
from utils.metrics import calc_mre
from utils.metrics import calc_rmse
it = iTransformer(n_steps=48, n_features=11, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64,d_ffn=128, dropout=0.1, epochs=200)
# Here I use the whole dataset as the training set because ground truth is not visible to the model, you can also split it into train/val/test sets
it.fit(dataset)  # train the model on the dataset
imputation = it.impute(dataset)  # impute the originally-missing values and artificially-missing values
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(mae)
mse = calc_mse(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(mse)
mre = calc_mre(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(mre)
rmse = calc_rmse(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(rmse)
it.save("save_it_here/iTransformer_hls2020.pypots")  # save the model for future use
it.load("save_it_here/iTransformer_hls2020.pypots")  # reload the serialized model file for following imputation or training