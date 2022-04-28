from timeit import default_timer as timer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import scipy.optimize
from scipy.special import expit, xlog1py

# Todo: xlog1py?

__all__ = ["CSLogit"]

class CSLogit:

    def __init__(self, initial_theta, lambda1=0, lambda2=0, theta_opt=None, 
                 obj='aec', prior_y=None, prob_lab=None, prob_unl_pos=None):

        self.initial_theta = initial_theta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.theta_opt = theta_opt
        self.obj = obj
        self.corrected_cost_matrix = None
        self.prior_y = prior_y
        self.prob_lab = prob_lab
        self.prob_unl_pos = prob_unl_pos
        
    def fitting(self, x, y, cost_matrix=None):
        if self.obj == 'weightedce':
            misclass_cost = np.zeros(len(y))
            misclass_cost[y == 0] = cost_matrix[:, 1, 0][y == 0]
            misclass_cost[y == 1] = cost_matrix[:, 0, 1][y == 1]

        def get_obj_func(theta):
            if self.obj == 'ce':
                return self.ce(theta, x, y)
            elif self.obj == 'weightedce':
                return self.weighted_ce(theta, x, y, misclass_cost)
            elif self.obj == 'puwce':
                return self.puwce(theta, x, y, cost_matrix)
            elif self.obj == 'puce':
                return self.puce(theta, x, y)
            elif self.obj == 'pucano':
                  return self.pucano(theta, x, y)
            elif self.obj == 'aec':
                return self.aec(theta, x, y, cost_matrix)
            elif self.obj == 'puaec':
                return self.puaec(theta, x, y, cost_matrix)

        self.theta, func_min, _ = self.optimization(get_obj_func, self.initial_theta)

        return func_min

    def optimization(self, obj_func, initial_theta):
        # Todo:
        #   Compare results for different solvers
        #   Solver has to be available for normal LogisticRegression
        #
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B',
                                          options={'ftol': 1e-6, 'disp': False})
        theta_opt, func_min, n_iter = opt_res.x, opt_res.fun, opt_res.nfev

        self.theta_opt = theta_opt

        return theta_opt, func_min, n_iter

    def ce(self, theta, x, y_true):
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        ce = - (y_true * np.log(scores + eps) + (1 - y_true) * np.log(1 - scores + eps))

        # Add regularization
        ce += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return ce.mean()

    def weighted_ce(self, theta, x, y_true, cost_misclass):
        # Weighted CE: Misclassification costs as weights
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        # Normalized cost of misclassification per instance
        ce = - (y_true * np.log(scores + eps) + (1 - y_true) * np.log(1 - scores + eps))
        weightedce = cost_misclass * ce

        # Add regularization
        weightedce += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return weightedce.mean()
      
    def puwce(self, theta, x, y_true, cost_matrix):
      
        prior_y_ = self.prior_y
        prob_lab_ = self.prob_lab
        c_inverse_ = prior_y_/prob_lab_
        
        # Weighted CE: Misclassification costs as weights
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        # Normalized cost of misclassification per instance
        puwce = -1* (y_true*[c_inverse_*np.log(scores + eps)*cost_matrix[:, 0, 1] \
                           + cost_matrix[:, 1, 0]*(1-c_inverse_)*np.log(1 - scores + eps)] \
                   + (1 - y_true) * cost_matrix[:, 1, 0] * np.log(1 - scores + eps))
        
        # Add regularization
        puwce += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)
        
        return puwce.mean()
      
    def puce(self, theta, x, y_true):
      
        prior_y_ = self.prior_y
        prob_lab_ = self.prob_lab
        c_inverse_ = prior_y_/prob_lab_
        
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        # Normalized cost of misclassification per instance
        puce = -1* (y_true*[c_inverse_*np.log(scores + eps) \
                           + (1-c_inverse_)*np.log(1 - scores + eps)] \
                   + (1 - y_true)*np.log(1 - scores + eps))
        
        # Add regularization
        puce += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)
        
        return puce.mean()
      
    def pucano(self, theta, x, y_true):
        prob_unl_pos_ = self.prob_unl_pos
        
        # Weighted CE: Misclassification costs as weights
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        # Normalized cost of misclassification per instance
        pucano = -1* (y_true * np.log(scores + eps) + (1 - y_true)*(prob_unl_pos_ * np.log(scores + eps)  + (1 - prob_unl_pos_) * np.log(1 - scores + eps) ))

        # Add regularization
        pucano += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return pucano.mean()

    def aec(self, theta, x, y, cost_matrix):  # Average Expected Cost

        scores = expit(theta[0] + x.dot(theta[1:]))  # eq 9

        # objective value = average expected cost + lambd * L1 norm of betas (without intercept)
        # eq 7 and 13
        aec = y * (scores * cost_matrix[:, 1, 1] + (1 - scores) * cost_matrix[:, 0, 1]) \
            + (1 - y) * (scores * cost_matrix[:, 1, 0] + (1 - scores) * cost_matrix[:, 0, 0])

        # Add regularization
        aec += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return aec.mean()
      
    def puaec(self, theta, x, y, cost_matrix):  # Average Expected Cost
        prob_unl_pos_ = self.prob_unl_pos
        scores = expit(theta[0] + x.dot(theta[1:]))  # eq 9

        # objective value = average expected cost + lambd * L1 norm of betas (without intercept)
        # eq 7 and 13
        cost_matrix_ = cost_matrix
        cost_matrix_[:, 0, 0] = cost_matrix_[:, 0, 0]*(1-prob_unl_pos_) + cost_matrix_[:, 0, 1]*prob_unl_pos_
        cost_matrix_[:, 1, 0] = cost_matrix_[:, 1, 0]*(1-prob_unl_pos_) + cost_matrix_[:, 1, 1]*prob_unl_pos_
        self.corrected_cost_matrix = cost_matrix_
        
        aec = y * (scores * cost_matrix_[:, 1, 1] + (1 - scores) * cost_matrix_[:, 0, 1]) \
            + (1 - y) * (scores * cost_matrix_[:, 1, 0] + (1 - scores) * cost_matrix_[:, 0, 0])

        # Add regularization
        aec += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return aec.mean()

    def predict(self, x_predict):

        # scores = 1 / (1 + np.exp(-self.theta[0] - x_predict.dot(self.theta[1:])))    #eq 9
        scores = expit(self.theta[0] + x_predict.dot(self.theta[1:]))

        return scores

    def tune(self, l1, lambda1_list, l2, lambda2_list, x_train, y_train, cost_matrix_train, x_val, y_val,
             cost_matrix_val):
        if l1:
            self.lambda2 = 0
            losses_list = []
            for lambda1 in lambda1_list:
                logit = CSLogit(self.initial_theta, lambda1=lambda1, lambda2=0, obj=self.obj)
                logit.fitting(x_train, y_train, cost_matrix_train)

                # Evaluate loss (without regularization term!)
                logit.lambda1 = 0
                if self.obj == 'ce':
                    val_loss = self.ce(logit.theta_opt, x_val, y_val)
                elif self.obj == 'weightedce':
                    misclass_cost_val = np.zeros(len(y_val))
                    misclass_cost_val[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    misclass_cost_val[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    val_loss = self.weighted_ce(logit.theta_opt, x_val, y_val, misclass_cost_val)
                elif self.obj == 'aec':
                    val_loss = self.aec(logit.theta_opt, x_val, y_val, cost_matrix_val)
                print('\t\tLambda l1 = %.4f;\tLoss = %.5f' % (lambda1, val_loss))
                losses_list.append(val_loss)
            lambda1_opt = lambda1_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.4f' % lambda1_opt)
            self.lambda1 = lambda1_opt

        elif l2:
            self.lambda1 = 0
            losses_list = []
            for lambda2 in lambda2_list:
                logit = CSLogit(self.initial_theta, lambda1=0, lambda2=lambda2, obj=self.obj)
                logit.fitting(x_train, y_train, cost_matrix_train)

                # Evaluate loss (without regularization term!)
                logit.lambda2 = 0
                if self.obj == 'ce':
                    val_loss = self.ce(logit.theta_opt, x_val, y_val)
                elif self.obj == 'weightedce':
                    misclass_cost_val = np.zeros(len(y_val))
                    misclass_cost_val[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    misclass_cost_val[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    val_loss = self.weighted_ce(logit.theta_opt, x_val, y_val, misclass_cost_val)
                elif self.obj == 'aec':
                    val_loss = self.aec(logit.theta_opt, x_val, y_val, cost_matrix_val)
                print('\t\tLambda l1 = %.4f;\tLoss %.5f' % (lambda2, val_loss))
                losses_list.append(val_loss)
            lambda2_opt = lambda2_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.4f' % lambda2_opt)
            self.lambda2 = lambda2_opt
        else:
            self.lambda1 = 0
            self.lambda2 = 0
