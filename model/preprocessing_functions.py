import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
	df = pd.read_csv(df_path)
	return df


def divide_train_test(df, target):
	# Function divides data set in train and test
	X_train, X_test, y_train, y_test = train_test_split(
		df.drop(target, axis=1),  # predictors
		df[target],  # target
		test_size=0.2,  # percentage of obs in test set
		random_state=0)  # seed to ensure reproducibility

	return X_train, X_test, y_train, y_test


def extract_cabin_letter(df, var):
	df[var] = df[var].str[0]  # captures the first letter
	return df


def add_missing_indicator(df, var):
	# function adds a binary missing value indicator
	df[var + '_NA'] = np.where(df[var].isnull(), 1, 0)

	return df


def impute_na(df, var, imputation_dict, method='categorical'):
	# function replaces NA by value entered by user
	# defaults to string "missing"
	if method == "numerical":
		for col in var:
			df = add_missing_indicator(df, col)

			# replace NaN by median
			if imputation_dict['numerical'] == 'median':
				median_val = df[col].median()
				df[col] = df[col].fillna(median_val)

	else:
		for col in var:
			df[col] = df[col].fillna(imputation_dict[method])

	return df


def remove_rare_labels(df, vars_cat, frequent_ls):
	# groups labels that are not in the frequent list into the umbrella
	# group Rare
	for var in vars_cat:
		# replace rare categories by the string "Rare"
		df[var] = np.where(df[var].isin(frequent_ls[var]), df[var], 'Rare')

	return df


def encode_categorical(df, vars_cat):
	# adds ohe variables and removes original categorical variable

	for var in vars_cat:

		# to create the binary variables, we use get_dummies from pandas

		df = pd.concat([df, pd.get_dummies(df[var],
										prefix=var, drop_first=True)], axis=1)

	df = df.drop(labels=vars_cat, axis=1)

	return df


def check_dummy_variables(df, dummy_list):
	# check that all missing variables where added when encoding, otherwise
	# add the ones that are missing
	columns = set(df.columns)

	for col in dummy_list:
		if col not in columns:
			df[col] = 0

	return df


def train_scaler(df, output_path):
	# train and save scaler
	scaler = StandardScaler()

	#  fit  the scaler to the train set
	scaler.fit(df)

	# save scaler
	joblib.dump(scaler, output_path)


def scale_features(df, output_path):
	# load scaler and transform data
	scaler = joblib.load(output_path)

	df = scaler.transform(df)

	return df


def train_model(X_train, y_train, target, output_path):
	# train and save model
	model = LogisticRegression(C=0.0005, random_state=0)

	model.fit(X_train, y_train)

	joblib.dump(model, output_path)

	return model


def predict(df, model):
	# load model and get predictions

	y_pred = model.predict(df)

	pred_prob = model.predict_proba(df)[:, 1]

	return y_pred, pred_prob
