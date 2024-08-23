from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from tqdm.notebook import tqdm
from hyperopt import fmin, tpe, hp,STATUS_OK, SparkTrials, space_eval
import torch
import logging
from pyspark import SparkContext
import warnings
import copy
import time
from IPython.display import clear_output

def index_to_mask(rows, index_array):
    mask_array = np.zeros(rows, dtype=int)
    mask_array[index_array] = 1
    return torch.from_numpy(mask_array.astype(np.bool_))


class NestedCV:
    def __init__(self,k_outer, k_inner, train_fun, eval_fun, max_evals, parallelism, minimalize, k_fold_class):
        self.k_outer = k_outer
        self.kf_outer = k_fold_class(n_splits=self.k_outer)
        self.k_inner = k_inner
        self.kf_inner = k_fold_class(n_splits=self.k_inner)
        self.train_fun = train_fun
        self.evaluate_fun = eval_fun
        self.outer_scores = np.zeros(k_outer)
        self.inner_scores = np.zeros((k_outer, k_inner))
        self.minimalize = minimalize
        self.parallelism = parallelism
        self.max_evals = max_evals
        self.best_params_per_fold = []
        self.best_models = []
        self.train_times = []

    def add_params(self, best_params):
        self.best_params_per_fold.append(best_params)

class NestedInductiveCV(NestedCV):
    def __init__(self, data, k_outer, k_inner, train_fun, eval_fun,max_evals = 100, parallelism = 1, minimalize = True, k_fold_class = KFold):
        self.cv_args = [k_outer, k_inner, train_fun, eval_fun,max_evals, parallelism, minimalize, k_fold_class]
        super().__init__(*self.cv_args)
        self.data = data

    def outer_cv(self, space):        
        for outer_i, (train_index, test_index) in tqdm(enumerate(self.kf_outer.split(self.data))):
            inner_indcutive_cv = InnerInductiveCV(train_index, *[self.data, *self.cv_args])
            fitted_model, best_params, best_inner_scores, train_time = inner_indcutive_cv.hyperparam_tuning(space)
            # self.best_models.append(copy.deepcopy(fitted_model).cpu() if hasattr(fitted_model, "cpu") else fitted_model)
            self.add_params(best_params)
            self.inner_scores[outer_i, :] = best_inner_scores
            self.outer_scores[outer_i] = self.evaluate_fun(fitted_model, self.data[test_index.tolist()])
            self.train_times.append(train_time)
            del fitted_model
        return self.outer_scores

class InnerInductiveCV(NestedInductiveCV):
    def __init__(self, train_index, *args):
        super().__init__(*args)
        self.outer_train_index = train_index
        self.train_data = self.data[self.outer_train_index.tolist()]

    def inner_fold(self, hyperparameters):
        scores = np.zeros(self.k_inner)
        for inner_i, (inner_train_index, inner_test_index) in enumerate(self.kf_inner.split(self.train_data)): 
            fitted_model = self.train_fun(self.train_data[inner_train_index.tolist()], hyperparameters)
            scores[inner_i] = self.evaluate_fun(fitted_model, self.train_data[inner_test_index.tolist()])
        return scores

    def objective(self, hyperparameters):    
        scores = self.inner_fold(hyperparameters)
        score = scores.mean() if self.minimalize else -scores.mean()    
        return {'loss': score, 'status': STATUS_OK}

    def hyperparam_tuning(self, space):
        sc = SparkContext.getOrCreate()
        sc.setLogLevel("OFF")
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger('hyperopt').setLevel(logging.CRITICAL)
        logging.getLogger('py4j').setLevel(logging.CRITICAL)
        warnings.filterwarnings('ignore')
        
        spark_trials = SparkTrials(parallelism = self.parallelism)
        
        best_params = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.max_evals, trials=spark_trials, verbose = False)
        best_params = space_eval(space, best_params)
        self.add_params(best_params)
        best_inner_scores = self.inner_fold(best_params)
        train_start_time = time.time()
        fitted_model = self.train_fun(self.train_data, best_params)
        train_time = time.time() - train_start_time
        return fitted_model, best_params, best_inner_scores, train_time
        
    
class NestedTransductiveCV(NestedCV):
    def __init__(self, data, target_node_type, k_outer, k_inner, train_fun, eval_fun,max_evals = 100, parallelism = 1, minimalize = True, k_fold_class = StratifiedKFold):
        self.cv_args = [k_outer, k_inner, train_fun, eval_fun,max_evals, parallelism, minimalize, k_fold_class]
        super().__init__(*self.cv_args)
        self.data = data
        self.target_node_type = target_node_type
        
    def outer_cv(self, space):        
        for outer_i, (train_index, test_index) in tqdm(enumerate(self.kf_outer.split(self.data[self.target_node_type].x, self.data[self.target_node_type].y))):
            inner_transd_cv = InnerTransductiveCV(train_index, *[self.data, self.target_node_type, *self.cv_args])
            fitted_model, best_params, best_inner_scores, train_time = inner_transd_cv.hyperparam_tuning(space)
            # self.best_models.append(copy.deepcopy(fitted_model).cpu() if hasattr(fitted_model, "cpu") else fitted_model)
            self.train_times.append(train_time)
            self.add_params(best_params)
            self.inner_scores[outer_i, :] = best_inner_scores
            test_mask = index_to_mask(self.data[self.target_node_type].x.shape[0], test_index)
            self.outer_scores[outer_i] = self.evaluate_fun(fitted_model, self.data, test_mask)
        clear_output(wait=True)
        return self.outer_scores

    def __repr__(self):
        return f"""
        Using a {self.k_outer} x {self.k_inner} nested {self.kf_outer.__class__.__name__} Cross-Validation, we obtain:
        {self.outer_scores.mean():.4f} +- {self.outer_scores.std():.4f}.\n
        self.outer_scores: {self.outer_scores}\n
        self.best_params_per_fold: {self.best_params_per_fold}\n
        self.best_models: {self.best_models}\n
        self.train_times: {torch.tensor(self.train_times).mean()}[{self.train_times}]\n
        """

class InnerTransductiveCV(NestedTransductiveCV):
    def __init__(self, train_index, *args):
        super().__init__(*args)
        self.outer_train_index = train_index

    def inner_fold(self, hyperparameters):
        scores = np.zeros(self.k_inner)
        outer_train_mask = index_to_mask(self.data[self.target_node_type].x.shape[0], self.outer_train_index)
        for inner_i, (inner_train_index, inner_test_index) in enumerate(self.kf_inner.split(self.data[self.target_node_type].x[self.outer_train_index], self.data[self.target_node_type].y[self.outer_train_index])): 
            inner_train_mask = index_to_mask(self.data[self.target_node_type].x.shape[0], self.outer_train_index[inner_train_index])
            inner_test_mask = index_to_mask(self.data[self.target_node_type].x.shape[0], self.outer_train_index[inner_test_index])
            fitted_model = self.train_fun(self.data, inner_train_mask, hyperparameters)
            
            scores[inner_i] = self.evaluate_fun(fitted_model, self.data, inner_test_mask)
        return scores

    def objective(self, hyperparameters):    
        scores = self.inner_fold(hyperparameters)
        score = scores.mean() if self.minimalize else -scores.mean()    
        return {'loss': score, 'status': STATUS_OK}

    def hyperparam_tuning(self, space):
        sc = SparkContext.getOrCreate()
        sc.setLogLevel("OFF")
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger('hyperopt').setLevel(logging.CRITICAL)
        logging.getLogger('py4j').setLevel(logging.CRITICAL)
        warnings.filterwarnings('ignore')
        
        spark_trials = SparkTrials(parallelism = self.parallelism)
        
        best_params = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.max_evals, trials=spark_trials, verbose = False)
        best_params = space_eval(space, best_params)
        self.add_params(best_params)
        best_inner_scores = self.inner_fold(best_params)
        train_start_time = time.time()
        fitted_model = self.train_fun(self.data, index_to_mask(self.data[self.target_node_type].x.shape[0], self.outer_train_index), best_params)
        train_time = time.time() - train_start_time
        return fitted_model, best_params, best_inner_scores, train_time