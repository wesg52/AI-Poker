"""
P(y|x) \propto P(x|y)P(y)
With Laplace smoothing
"""

def PY(x,y):
    y = np.concatenate([y, [-1,1]])
    n = len(y)
    s = np.sum(y)
    pos = ((n + s)/2.0 + 1.0)/(n + 2)
    neg = ((n - s)/2.0 + 1.0)/(n + 2)
    return pos, neg

def PXY(x,y):
    n, d = x.shape
    x = np.concatenate([x, np.ones((2,d))])
    y = np.concatenate([y, [-1,1]])
    n, d = x.shape
    
    pos = x[np.where(y > 0)[0]]
    neg = x[np.where(y < 0)[0]]
    posN = np.sum(pos)
    negN = np.sum(neg)
    posSum = np.sum(pos, 0)
    negSum = np.sum(neg, 0)
    posprob = np.apply_along_axis(lambda i: i/posN, 0, posSum)
    negprob = np.apply_along_axis(lambda i: i/negN, 0, negSum)
    return posprob, negprob

def logratio(x,y,xtest):
    """
    log (P(Y = 1|X=xtest)/P(Y=-1|X=xtest))
    """
    
    PYpos, PYneg = PY(x, y)
    PXYpos, PXYneg = PXY(x, y)
    featInd = np.where(xtest == 1)[0]
    posProd = np.prod(PXYpos[featInd], 0)
    negProd = np.prod(PXYneg[featInd], 0)
    log = np.log(PYpos*posProd) - np.log(PYneg*negProd)
    return log

def classifier(x,y):
    n, d = x.shape
    PYpos, PYneg = PY(x, y)
    PXYpos, PXYneg = PXY(x, y)
    w = np.log(PXYpos) - np.log(PXYneg)
    b = np.log(PYpos) - np.log(PYneg)
    return w, b

def sign(w, xi):
    if np.inner(w, xi) > 0:
        output = 1
    else:
        output = -1
    return output

def pred(x,w,b=0):
    """
    Returns predictions for the test data.
    """
    w = w.reshape(-1)
    if b != 0:
        x = np.column_stack((x, np.ones(x.shape[0])))
        w = np.append(w, [b])
    if x.ndim == 1:
        preds = sign(w, x)
    else:
        preds = np.apply_along_axis(lambda xi: sign(w, xi), 1, x)
    return preds
