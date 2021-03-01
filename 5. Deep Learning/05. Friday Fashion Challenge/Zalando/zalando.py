# %%
%load_ext autotime
# %%
from sklearn.model_selection import train_test_split
from autoPyTorch import (AutoNetClassification)
import pandas as pd
import numpy as np
import os as os
import json
from sklearn.metrics import accuracy_score

train = pd.read_csv('D:/Data/fashion-mnist/data/archive/fashion-mnist_train.csv')
test = pd.read_csv('D:/Data/fashion-mnist/data/archive/fashion-mnist_test.csv')
X = train.drop(columns=['label'],axis=1)
y = train.label
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)
# %%
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                   log_level='debug',
                                    use_tensorboard_logger=True,
                                    #save_models=True !gives error!
                                    cross_validator='k_fold',
                                    cuda=True,
                                    full_eval_each_epoch = True,
                                    validation_split=0.2,
                                    budget_type='epochs',
                                    result_logger_dir='Experiment_logs/Zalando/',
                                    random_seed=17,
                                    max_runtime=300,
                                    min_budget=1,
                                    max_budget=10)
# %%
autoPyTorch.fit(X_train, y_train, cross_validator_args={"n_splits": 20}, num_iterations=10, max_runtime=7000, max_budget=200)

# %%

# %%
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

y_pred = autoPyTorch.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))


# %%
results_refit = autoPyTorch.refit(X,
                              y,
                              autonet_config=autoPyTorch.get_current_autonet_config(),
                              )

# %%

with open("Experiment_logs/Zalando/results_refit.json", "w") as file:
    json.dump(results_refit, file)

zalando_model = autoPyTorch.get_pytorch_model()
print(zalando_model)    

# %%
import pickle

with open('Experiment_logs/Zalando/zalando.pickle', 'wb') as f:
    pickle.dump(zalando_model, f)

# %%
from autoPyTorch import AutoNetClassification, AutoNetEnsemble
from autoPyTorch.data_management.data_manager import DataManager

# Note: You can write your own datamanager! Call fit with respective train, valid data (numpy matrices) 
dm = DataManager()
dm.generate_classification(num_classes=3, num_features=784, num_samples=60000)


#  %%
import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from autoPyTorch import AutoNetClassification,AutoNetEnsemble
from autoPyTorch.data_management.data_manager import DataManager

# Note: every parameter has a default value, you do not have to specify anything. The given parameters allow for a fast test.
autonet = AutoNetEnsemble(AutoNetClassification, budget_type='epochs',
                          min_budget=1,
                          max_budget=5, 
                          num_iterations=1,
                          use_tensorboard_logger=True, 
                          log_level='debug',
                          result_logger_dir='Experiment_logs/Zalando/',
                          random_seed=17,
                          max_runtime=300,
                         )


res = autonet.fit(X_train, y_train, cross_validator="k_fold", cross_validator_args={"n_splits": 1}, validation_split=0.2,
    ensemble_only_consider_n_best=3)

y_pred_ens = autonet.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred_ens))
# %%
