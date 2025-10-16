from Predictors.Predictors import Gaussian, ARIMApred, Naive, LinearExtrapolation
from Predictors.runner import *

#Directly using TSF methods on given series:
l = Naive(10)
print(l.predict([0.1, 0.3, 0.2, 0.6, 0.3, 0.5]))
le = LinearExtrapolation(10)
print(le.predict([0.1, 0.3, 0.2, 0.6, 0.3, 0.5]))
ar = ARIMApred(10, dist=True)
print(ar.predict([0.1, 0.3, 0.2, 0.6, 0.3, 0.5]))
g = Gaussian(10)
print(g.predict([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
print(g.predict([0.9, 0.8, 0.4, 0.2, 0.3, 0.5]))


listOpp = [0.2, 0.2, 0.2, 0.3, 0.2, 0.6, 0.3, 0.5, 0.52, 0.6]
listOwn = [1, 0.9, 0.8, 0.8, 0.75, 0.8, 0.7, 0.7, 0.69, 0.65]

values = [
    1.0000000000000002,
    1.0000000000000002,
    0.9824561422899372,
    0.9824561422899372,
    0.9649122477377363,
    0.9649122477377363,
    0.9473683900276733,
    0.9473683900276733,
    0.9298244954754724,
    0.9298244954754724,
    0.9122806377654096,
    0.9122806377654096,
    0.8947367432132085,
    0.8947367432132085,
    0.8771929223452837,
    0.8771929591874217,
    0.8596491014773587,
    0.8596491014773587,
    0.8421052069251579,
    0.8421052069251579
]


#One can also use the function "predict outcome", that can do the whole process: From applying Time series forecasts to
#sampling possible intersections points, predicting the agreement probability, and visually portraying the results.

predict_outcome(listOwn, listOpp, 50, "Gaussian", verbose=True)
#predict_outcome(listOwn, listOpp, 50, "ARIMA", verbose=True)