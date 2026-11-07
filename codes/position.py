"""Firth logistic fitting for the exploratory answer-position analysis."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def firth_logistic(x,y):
    """Maximize the Jeffreys-penalized logistic likelihood; verify its score."""
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError('Design matrix is not full rank')
    beta=np.zeros(x.shape[1])

    def evaluate(beta):
        eta=x@beta
        probability=expit(eta)
        weight=probability*(1-probability)
        information=x.T@(weight[:,None]*x)
        sign,logdet=np.linalg.slogdet(information)
        if sign<=0:
            return -np.inf,None,None
        inverse=np.linalg.inv(information)
        leverage=weight*np.einsum('ij,jk,ik->i',x,inverse,x)
        score=x.T@(y-probability+leverage*(.5-probability))
        likelihood=np.sum(y*eta-np.logaddexp(0,eta))+.5*logdet
        return likelihood,score,inverse

    def objective(value):
        likelihood,score,_=evaluate(value)
        return -likelihood,-score

    fit=minimize(objective,beta,jac=True,method='BFGS',
                 options={'gtol':1e-8,'maxiter':2000})
    likelihood,score,inverse=evaluate(fit.x)
    error=float(np.max(np.abs(score)))
    if error>1e-6 or not np.isfinite(likelihood):
        raise RuntimeError(f'Firth optimization failed: {fit.message}; score={error}')
    return fit.x,inverse,int(fit.nit),error
