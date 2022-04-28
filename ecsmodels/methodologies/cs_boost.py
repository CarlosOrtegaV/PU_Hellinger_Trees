import xgboost as xgb
from scipy.special import expit
import numpy as np

__all__ = ["CSBoost"]

# Todo: make wrapper?
class CSBoost:
    def __init__(self, obj, lambda1=0, lambda2=0, prior_y=None, prob_lab=None,
                 prob_unl_pos_train=None, prob_unl_pos_val=None, oversampling=None,
                 validation=False, random_state=None):
      
        self.obj = obj
        self.prior_y = prior_y
        self.prob_lab = prob_lab
        self.oversampling = oversampling
        self.prob_unl_pos_train = prob_unl_pos_train
        self.prob_unl_pos_val = prob_unl_pos_val
        self.random_state = random_state
        self.validation = validation
        
        # alpha is l1, lambda is l2
        params = {'random_state': random_state, 'tree_method': 'exact', 'verbosity': 0, 'reg_alpha': lambda1,
                  'reg_lambda': lambda2}
        
        if obj == 'ce' or obj == 'weightedce':
            params['objective'] = 'binary:logistic'
        elif obj == 'aec' or obj == 'puwce' or obj == 'pucano' or obj =='nnpu' or obj =='imbnnpu':
            params['disable_default_eval_metric'] = True

        self.params = params

    def fit(self, x_train, y_train, x_val=None, y_val=None, cost_matrix_train=None, cost_matrix_val=None):
      
        if self.obj == 'ce':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)

            xgboost = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=100, early_stopping_rounds=100,
                            evals=[(dval, 'eval')], verbose_eval=False)

        elif self.obj == 'weightedce':
            misclass_costs = np.zeros(len(y_train))
            misclass_costs[y_train == 0] = cost_matrix_train[:, 1, 0][y_train == 0]
            misclass_costs[y_train == 1] = cost_matrix_train[:, 0, 1][y_train == 1]

            misclass_costs_val = np.zeros(len(y_val))
            misclass_costs_val[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
            misclass_costs_val[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

            dtrain = xgb.DMatrix(x_train, label=y_train, weight=misclass_costs)
            dval = xgb.DMatrix(x_val, label=y_val, weight=misclass_costs_val)

            xgboost = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=100, early_stopping_rounds=10,
                                evals=[(dval, 'eval')], verbose_eval=False)
            
            
        elif self.obj == 'puwce':
          
            dtrain = xgb.DMatrix(x_train, label=y_train)
            
            if self.validation == True:
              dval = xgb.DMatrix(x_val, label=y_val)
              
              
            prior_y_ = self.prior_y 
            prob_lab_ = self.prob_lab
            c_inverse_ = prior_y_/prob_lab_
            eps = 1e-9  # small value to avoid log(0)

            def puwce_train(raw_scores, y_true):
                 y = y_true.get_label()
                 scores = expit(raw_scores) + eps
                 f = cost_matrix_train[:, 1, 0]  # cost of false positive
                 A = cost_matrix_train[:, 0, 1]  # cost of false negative
                 
                 # Gradient
                 grad = -y*c_inverse_*A*(1-scores) + y*f*(1-c_inverse_)*scores \
                         + (1-y)*f*scores
 
                 # Hessian
                 hess = (y*c_inverse_*A + y*f*(1-c_inverse_) + (1-y)*f)*scores*(1-scores)
 
                 return grad, hess
 
            def puwce_val(raw_scores, y_true):
                y = y_true.get_label()
                scores = expit(raw_scores) + eps
                f = cost_matrix_val[:, 1, 0]  # cost of false positive
                A = cost_matrix_val[:, 0, 1]  # cost of false negative
                
                loss = np.empty_like(scores)
                pos = y == 1
               
                loss[pos] = -1* ((c_inverse_*np.log(scores[pos])*A[pos] \
                                  + f[pos]*(1-c_inverse_)*np.log(1 - scores[pos])))
                  
                loss[~pos] = -1*(f[~pos] * np.log(1 - scores[~pos]))

                puwce = loss.mean()

                return 'PUWCE', puwce
              
            if self.validation == True:
              xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=puwce_train, 
                                  feval=puwce_val, num_boost_round=100,
                                  early_stopping_rounds=10, evals=[(dval, 'eval')], 
                                  verbose_eval=False)
            else:
              
              xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=puwce_train, 
                                  num_boost_round=100)
              
              
            
        elif self.obj == 'pucano':
          
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)
            eps = 1e-9  # small value to avoid log(0)

            def pucano_train(raw_scores, y_true):
                 y = y_true.get_label()
                 scores = expit(raw_scores) + eps
                 
                 # Gradient
                 grad = scores - self.prob_unl_pos_train + self.prob_unl_pos_train*y - y
 
                 # Hessian
                 hess = scores * (1 - scores)
 
                 return grad, hess
 
            def pucano_val(raw_scores, y_true):
                y = y_true.get_label()
                scores = expit(raw_scores) + eps
                
                loss = np.empty_like(scores)
                pos = y == 1
               
                loss[pos] = -1* (np.log(scores[pos]))
                  
                loss[~pos] = -1*(self.prob_unl_pos_val[~pos] * np.log(scores[~pos]) + (1 - self.prob_unl_pos_val[~pos]) *  np.log(1 - scores[~pos]) )

                pucano = loss.mean()

                return 'PUCANO', pucano
              

            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=pucano_train, 
                                feval=pucano_val, num_boost_round=100,
                                early_stopping_rounds=10, evals=[(dval, 'eval')], 
                                verbose_eval=False)


        elif self.obj == 'nnpu':
          
            dtrain = xgb.DMatrix(x_train, label=y_train)
            if self.validation == True:
              dval = xgb.DMatrix(x_val, label=y_val)
            prior_y_ = self.prior_y 
            prob_lab_ = self.prob_lab
            c_inverse_ = prior_y_/prob_lab_

            
            eps = 1e-9  # small value to avoid log(0)

            def nnpu_train(raw_scores, y_true):
                 y = y_true.get_label()
                 scores = expit(raw_scores) + eps
                 zero_col_train = np.zeros_like(y)
                 pu_col_train = 1 - y*c_inverse_
                 
                 
                 nnpu_col_train = np.amax(np.vstack((zero_col_train, pu_col_train)),0)
                                  
                 # Gradient
                 grad = -y*(1-scores)*c_inverse_ + nnpu_col_train*scores
                 # Hessian
                 hess = (y*c_inverse_ + nnpu_col_train) * scores * (1 - scores)
 
                 return grad, hess
 
            def nnpu_val(raw_scores, y_true):
                y = y_true.get_label()
                scores = expit(raw_scores) + eps
                
                loss = np.empty_like(scores)

                pos = y == 1
                
                zero_col = np.zeros_like(scores[pos])
                pos_col = -1*np.log(1-scores[pos])*(1-c_inverse_)
                max_col = np.amax(np.vstack((zero_col, pos_col)),0)
               
                loss[pos] = -1*np.log(scores[pos])*c_inverse_ + max_col
                
                
                loss[~pos] = -1*np.log(1-scores[~pos])

                nnpu = loss.mean()

                return 'nnPU', nnpu

            if self.validation == True:
              xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=nnpu_train, 
                                  feval=nnpu_val, num_boost_round=100,
                                  early_stopping_rounds=10, evals=[(dval, 'eval')], 
                                  verbose_eval=False)
            else:
              
              xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=nnpu_train, 
                                  num_boost_round=100)


        elif self.obj == 'imbnnpu':
          
            dtrain = xgb.DMatrix(x_train, label=y_train)
            if self.validation == True:
              dval = xgb.DMatrix(x_val, label=y_val)
            oversampling_ = self.oversampling
            prior_y_ = self.prior_y 
            prob_lab_ = self.prob_lab
            c_inverse_ = prior_y_/prob_lab_

            
            eps = 1e-9  # small value to avoid log(0)

            def imbnnpu_train(raw_scores, y_true):
                 y = y_true.get_label()
                 scores = expit(raw_scores) + eps
                 zero_col_train = np.zeros_like(y)
                 pu_col_train = (1 - oversampling_)*(1-y)/(1 - prior_y_) - (1 - oversampling_)*prior_y_*y/(1 - prior_y_)
                 
                 
                 nnpu_col_train = np.amax(np.vstack((zero_col_train, pu_col_train)),0)
                                  
                 # Gradient
                 grad = -y*oversampling_*(1-scores) + nnpu_col_train*scores
                                  
                 # Hessian
                 hess = (y*oversampling_ + nnpu_col_train) * scores * (1 - scores)
 
                 return grad, hess
 
            def imbnnpu_val(raw_scores, y_true):
                y = y_true.get_label()
                scores = expit(raw_scores) + eps
                
                loss = np.empty_like(scores)

                pos = y == 1
                
                zero_col = np.zeros_like(scores[pos])
                
                pos_col = (1 - oversampling_)*prior_y_/(1 - prior_y_) * np.log(1 - scores[pos])
                max_col = np.amax(np.vstack((zero_col, pos_col)),0)
               
                loss[pos] = -1*oversampling_*np.log(scores[pos]) + max_col
                
                
                loss[~pos] = -1* (1 - oversampling_)/(1 - prior_y_) * np.log(1-scores[~pos]) 

                imbnnpu = loss.mean()

                return 'imbnnPU', imbnnpu
              
            if self.validation == True:
              xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=imbnnpu_train, 
                                  feval=imbnnpu_val, num_boost_round=100,
                                  early_stopping_rounds=10, evals=[(dval, 'eval')], 
                                  verbose_eval=False)
            else:
              
              xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=imbnnpu_train, 
                                  num_boost_round=100)


        elif self.obj == 'aec':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)

            # Do constant computations here to avoid DMatrix error
            # diff_costs_train = fixed_cost - y_train * amounts_train

            train_constant = (y_train * (cost_matrix_train[:, 1, 1] - cost_matrix_train[:, 0, 1])
                             + (1 - y_train) * (cost_matrix_train[:, 1, 0] - cost_matrix_train[:, 0, 0]))

            def aec_train(raw_scores, y_true):
                scores = expit(raw_scores)

                # Average expected cost:
                # ec = np.multiply(np.multiply(y_true, (1 - scores)), amounts_train) + np.multiply(scores, fixed_cost)
                # ec = y_true * (
                #     scores * cost_matrix_train[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                #     + (1 - y_true) * (
                #     scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                # Gradient
                # Use diff_costs_train instead of (fixed_cost - y_true*amounts_train)
                # grad = scores * (1 - scores) * diff_costs_train
                grad = scores * (1 - scores) * train_constant

                # Hessian
                hess = np.abs((1 - 2 * scores) * grad)
                # hess = scores * (1 - scores) * (1 - 2 * scores) * train_constant

                return grad, hess

            def aec_val(raw_scores, y_true):
                scores = expit(raw_scores)

                # Return AEC (not grad/hess)
                # ec = (1 - scores) * y_val * amounts_val + scores * fixed_cost
                # ec = y_true * (
                #     scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                #     + (1 - y_true) * (
                #     scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                # Avoid computations with y_true (DMatrix)
                # Todo: what happens if y_true is a vector?
                if y_true:
                    ec = scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]
                else:
                    ec = scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0]

                aec = ec.mean()

                return 'AEC', aec

            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=aec_train, feval=aec_val, num_boost_round=500,
                                early_stopping_rounds=50, evals=[(dval, 'eval')], verbose_eval=False)

        # print('\tBest number of trees = %i' % xgboost.best_ntree_limit)

        return xgboost

    def tune(self, l1, lambda1_list, l2, lambda2_list, x_train, y_train, cost_matrix_train, x_val, y_val,
             cost_matrix_val):
        if l1:
            self.params['reg_lambda'] = 0
            losses_list = []
            for lambda1 in lambda1_list:
                xgboost = CSBoost(obj=self.obj, lambda1=lambda1)
                xgboost = xgboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)
                scores = xgboost.inplace_predict(x_val)

                # Evaluate loss (without regularization term!)
                if self.obj == 'ce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))
                    val_loss = ce.mean()
                elif self.obj == 'weightedce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))

                    cost_misclass = np.zeros(len(y_val))
                    cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    weighted_ce = cost_misclass * ce
                    val_loss = weighted_ce.mean()
                elif self.obj == 'aec':
                    def aec_val(raw_scores, y_true):
                        scores = expit(raw_scores)

                        # Return AEC (not grad/hess)
                        # ec = (1 - scores) * y_val * amounts_val + scores * fixed_cost
                        ec = y_true * (
                            scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                            + (1 - y_true) * (
                            scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                        aec = ec.mean()

                        return 'AEC', aec  # Todo: do not return 'AEC'

                    aec = aec_val(scores, y_val)
                    val_loss = aec[1]
                print('\t\tLambda l1 = %.4f;\tLoss = %.5f' % (lambda1, val_loss))
                losses_list.append(val_loss)
            lambda1_opt = lambda1_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.4f' % lambda1_opt)
            self.params['reg_alpha'] = lambda1_opt
        elif l2:
            self.params['reg_alpha'] = 0
            losses_list = []
            for lambda2 in lambda2_list:
                xgboost = CSBoost(obj=self.obj, lambda2=lambda2)
                xgboost = xgboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)
                scores = xgboost.inplace_predict(x_val)

                # Evaluate loss (without regularization term!)
                if self.obj == 'ce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))
                    val_loss = ce.mean()
                elif self.obj == 'weightedce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))

                    cost_misclass = np.zeros(len(y_val))
                    cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    weighted_ce = cost_misclass * ce
                    val_loss = weighted_ce.mean()
                elif self.obj == 'aec':
                    def aec_val(raw_scores, y_true):
                        scores = expit(raw_scores)

                        # Return AEC (not grad/hess)
                        ec = y_true * (
                                scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                             + (1 - y_true) * (
                                     scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                        aec = ec.mean()

                        return 'AEC', aec

                    aec = aec_val(scores, y_val)
                    val_loss = aec[1]
                print('\t\tLambda l2 = %.4f;\tLoss = %.5f' % (lambda2, val_loss))
                losses_list.append(val_loss)
            lambda2_opt = lambda2_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.4f' % lambda2_opt)
            self.params['reg_alpha'] = lambda2_opt
        else:
            self.lambda1 = 0
            self.lambda2 = 0
