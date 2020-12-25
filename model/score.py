import preprocessing_functions as pf
import config
import joblib

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):

	# extract first letter from cabin
	data = pf.extract_cabin_letter(data, 'cabin')

	# impute NA categorical
	data = pf.impute_na(data, config.CATEGORICAL_VARS,
						config.IMPUTATION_DICT, method='categorical')

	# impute NA numerical
	data = pf.impute_na(data, config.NUMERICAL_TO_IMPUTE,
						config.IMPUTATION_DICT, method='numerical')

	# Group rare labels
	data = pf.remove_rare_labels(
		data, config.CATEGORICAL_VARS, config.FREQUENT_LABELS)

	# encode variables
	data = pf.encode_categorical(data, config.CATEGORICAL_VARS)

	# check all dummies were added
	data = pf.check_dummy_variables(data, config.DUMMY_VARIABLES)

	# scale variables
	data = pf.scale_features(data, config.OUTPUT_SCALER_PATH)

	# make predictions
	model = joblib.load(config.OUTPUT_MODEL_PATH)
	predictions, pred_prob = pf.predict(data, model)

	return predictions

# ======================================

# small test that scripts are working ok

if __name__ == '__main__':

	from sklearn.metrics import accuracy_score
	import warnings
	warnings.simplefilter(action='ignore')

	# Load data
	data = pf.load_data(config.PATH_TO_DATASET)

	X_train, X_test, y_train, y_test = pf.divide_train_test(data,
															config.TARGET)

	pred = predict(X_test)

	# evaluate
	# if your code reprodues the notebook, your output should be:
	# test accuracy: 0.6832
	print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
	print()
