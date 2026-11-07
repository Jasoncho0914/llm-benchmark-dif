"""Family-by-score logistic fits with lineage-cluster sandwich covariance.

Scores are standardized internally and Newton steps use per-item backtracking.
Fits without a finite, converged solution do not receive a Wald test.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import norm


def fit_many_logistic(y, score, focal, clusters, max_iter=100, return_diagnostics=False):
    center, scale = np.mean(score), np.std(score)
    if scale <= 0:
        raise ValueError('Matching scores have no variation')
    zscore = (score-center)/scale
    x = np.column_stack([np.ones(len(score)), zscore, focal, zscore*focal])
    j = y.shape[1]
    beta = np.zeros((4,j))
    mean = np.clip(y.mean(axis=0),1e-6,1-1e-6)
    beta[0] = np.log(mean/(1-mean))
    regularizer = np.eye(4)[None,:,:]*1e-10
    converged = np.zeros(j,bool)
    stopped = np.zeros(j,bool)
    last_step = np.full(j,np.inf)
    for iteration in range(max_iter):
        eta=x@beta;mu=expit(eta)
        w=np.maximum(mu*(1-mu),1e-12)
        gradient=x.T@(y-mu)
        hessian=np.einsum('ni,nj,nq->qij',x,x,w,optimize=True)
        step=np.linalg.solve(hessian+regularizer,gradient.T[...,None])[...,0].T
        step[:,converged|stopped]=0
        ll=np.sum(y*eta-np.logaddexp(0,eta),axis=0)
        factor=np.ones(j)
        for _ in range(30):
            trial=beta+step*factor
            et=x@trial
            proposed=np.sum(y*et-np.logaddexp(0,et),axis=0)
            worse=(proposed<ll-1e-10)&~converged
            if not worse.any():break
            factor[worse]*=.5
        beta=trial
        last_step=np.max(np.abs(step*factor),axis=0)
        residual=np.max(np.abs(x.T@(y-expit(x@beta))),axis=0)
        converged |= (last_step<1e-6)&(residual<1e-6)
        # Effectively separated fits continue toward an infinite coefficient;
        # stop spending iterations on these, but retain them as untestable.
        stopped |= np.max(np.abs(beta),axis=0)>50
        if np.all(converged|stopped):break
    mu=expit(x@beta);w=np.maximum(mu*(1-mu),1e-12)
    information=np.einsum('ni,nj,nq->qij',x,x,w,optimize=True)
    min_eigenvalue=np.linalg.eigvalsh(information)[:,0]
    usable=(converged & (np.max(np.abs(beta),axis=0)<50)
            & (min_eigenvalue>1e-8) & (y.sum(axis=0)>0) & (y.sum(axis=0)<len(y)))
    bread=np.linalg.inv(information+regularizer)
    residual=y-mu
    meat=np.zeros((j,4,4))
    for cluster in np.unique(clusters):
        take=clusters==cluster
        sc=x[take].T@residual[take]
        meat+=np.einsum('iq,jq->qij',sc,sc,optimize=True)
    covariance=bread@meat@bread
    se=np.sqrt(np.maximum(covariance[:,3,3],0))
    z=np.divide(beta[3],se,out=np.full(j,np.nan),where=usable & (se>0))
    p=2*norm.sf(np.abs(z))
    q25,q75=np.quantile(score,[.25,.75])
    def gap(s):
        t=(s-center)/scale
        return expit(beta[0]+beta[1]*t+beta[2]+beta[3]*t)-expit(beta[0]+beta[1]*t)
    low,high=gap(q25),gap(q75)
    low[~usable]=np.nan;high[~usable]=np.nan
    # Return coefficients and interaction SE on the original score scale.
    raw=beta.copy()
    raw[0]=beta[0]-center/scale*beta[1]
    raw[1]=beta[1]/scale
    raw[2]=beta[2]-center/scale*beta[3]
    raw[3]=beta[3]/scale
    raw[:,~usable]=np.nan
    se=se/scale;se[~usable]=np.nan
    result=(raw,se,p,low,high)
    if return_diagnostics:
        return result+({'converged':converged,'testable':usable,
            'last_step':last_step,'max_score_residual':np.max(np.abs(x.T@(y-mu)),axis=0),
            'minimum_information_eigenvalue':min_eigenvalue,'iterations':iteration+1},)
    return result
