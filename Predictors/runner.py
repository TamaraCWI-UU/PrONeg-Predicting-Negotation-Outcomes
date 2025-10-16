from sklearn.neighbors import KernelDensity

from Predictors.Predictors import Naive, LinearExtrapolation, ARIMApred, ExponentialSmoothing, LinearRegression, Gaussian
import random
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy import stats
import numpy as np


__all__ = [
    "predict_series",
    "find_outcome_utility_distr",
    "sample_monte_carlo",
    "find_histogram",
    "find_outcome_from_prob_distr",
    "intersect_series",
    "predict_outcome"
]

def inbetween(own_utils, opponent_utils, deadline):
    #breaks if opponent_utils is greater than own-utils
    end_est = (own_utils[0] + opponent_utils[0]) / 2
    intersected_points = [(deadline, end_est)]
    if end_est>0:
        agreement_prob = 1
    else:
        agreement_prob = 0
    return intersected_points, agreement_prob


#main method
def predict_outcome(own_utils: list|None, opponent_utils:list|None, limit: int = None, method:str=None, to_predict_own = True, to_predict_opponent=True, samplesize = 100, ufun = None, verbose = False, cutoff = -1):
    """This function should do everything.
    You have the options:
    - Supply the arguments to predict your own and the opponent's utilities. (to_predict_own = True, to_predict_opponent = True)
    - Supply the prediction series for the own and opponent's utilities.
        - Note, your own agents prediction series can also be your planned sequence.
    -For now, we assume that all observations of the previous opponent utils are supplied.
    Parameters:
    own_utils: list of (observed) own utilities
    opponent_utils: list of (observed) opponent utilities
    limit: the deadline of the negotiation
    method: the method used for prediction
    to_predict_own: whether to predict the own utilities
    to_predict_opponent: whether to predict the opponent's utilities
    samplesize: the number of samples to take for the Monte Carlo method

    Watch out: this method is mostly tested with ARIMA, Gaussian and inbetween.
    """
    ## only use cutoff if the prediction should not use the full data, e.g. in simulations
    if cutoff == -1:
        cutoff = len(own_utils)
    else:
        own_utils = own_utils[:cutoff]
        opponent_utils = opponent_utils[:cutoff]

    ## Check if the input is valid
    if limit < len(own_utils) & verbose:
        print("Warning! The limit is smaller than the length of the data.")
        return None
    if len(own_utils) != len(opponent_utils):
        print("Warning! The length of the data is not equal.")
        if len(own_utils)>len(opponent_utils):
            own_utils.pop()
        else:
            opponent_utils.pop()
        if len(own_utils) != len(opponent_utils):
            print("Warning! The length of the data is still not equal.")
            return None
        cutoff = len(opponent_utils)

    if method == "inbetween":
        intersected_points, agreement_probability = inbetween(own_utils, opponent_utils, limit)
        return None

    if to_predict_own:
        params, predicted_own_utils = predict_series(own_utils, limit, method, verbose) #own = True
    else:
        predicted_own_utils = own_utils
    if to_predict_opponent:
        params, predicted_opp_utils = predict_series(opponent_utils, limit, method, verbose)
    else:
        predicted_opp_utils = opponent_utils

    cutoff = len(opponent_utils)
    intersected_points, agreement_probability = intersect_series(predicted_own_utils, predicted_opp_utils, cutoff, samplesize, verbose=verbose)

    if len(intersected_points) != 0:
        if verbose:
            find_histogram(intersected_points, method=method)
            prob_distr = find_outcome_utility_distr(intersected_points, method=method, verbose=verbose)


    if ufun is not None and (verbose == True):
        find_outcome_from_prob_distr(prob_distr, ufun)
    kde, logprob = find_outcome_utility_distr(intersected_points, method=method, verbose=verbose)
    return None#results



def find_outcome_utility_distr(intersected_points, method = "None", verbose=False, true=0, band=0.05):
    if len(intersected_points) == 0:
        #print("No intersection points found.")
        return None, None
    points = getDim(intersected_points, 1)
    kde = KernelDensity(kernel='epanechnikov', bandwidth=band)
    kde.fit(points[:, None])
    x_d = np.linspace(0, 1, 1000)

    logprob = kde.score_samples(x_d[:, None])
    if verbose:
        plt.plot(points, np.full_like(points, -0.01), '|k', markeredgewidth=1)
        plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
        plt.plot(true, 0, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
        plt.xlim(0, 1)
        plt.xlabel('Utility of outcome')
        plt.ylabel('Density')
        plt.title(f"Utility density distribution of outcome by {method}")
        plt.show()
    return kde, logprob

def predict_series(utils: list, limit: int = 10, method: str = "Linear", verbose = False): #
    """ This method can be used by agents to predict the uilty of the outcome of a negotiation. One can use "Naive", "Linear" or "ARIMA" as the method parameter.
    The given limit parameter is treated as the deadline of the negotiation, so the prediction will extend the data until the deadline.

    Example: If the length of the data is 8, and the limit is 10, the 2 predictions will be made.

    Args:
        utils (list): The list of utilities to predict.
        limit (int): The deadline of the negotiation.
        method (str): The method used for prediction. Can be "Naive", "Linear" or "ARIMA".
        verbose (bool): If True, the method will print the data and the prediction.
    """
    if limit is None:
        limit = 10
    if limit < len(utils):
        if verbose:
            print("Warning! The limit is smaller than the length of the data.")
    if method is None:
        if verbose:
            method = "Linear"

    if verbose:
        # Print the method used for prediction, the limit and the data input.
        print(f"Predicting with {method} and limit {limit}. The data is: {utils}")


    params = (-1, -1, -1)
    if len(utils) == 0:
            print("Warning! No data to predict with.")
            return None
    if len(utils) > limit:
        print("Warning! The data is longer than the limit, so no prediction will be made.")
        return None
    predictVars=0
    if method == "ARIMA":
        params, predictionA, predictVars = ARIMApred(limit, dist=True).predict(utils)
        predictiongraph = [predictionA]
        prediction = ARIMAMonteCarlo(predictionA, predictVars, 100)
    elif method == "auto-ARIMA":
        params, predictionA, predictVars = ARIMApred(limit, dist=True, auto=True).predict(utils)
        predictiongraph = [predictionA]
        prediction = ARIMAMonteCarlo(predictionA, predictVars, 100)
    elif "ARIMA" in method:
        windows = int(method.split('-')[1])
        params, predictionA, predictVars = ARIMApred(limit, dist=True).predict(utils, windows=windows)
        predictiongraph = [predictionA]
        prediction = ARIMAMonteCarlo(predictionA, predictVars, 100)
    elif method == "Gaussian":
        predictionA, predictVars = Gaussian(limit).predict(utils, windows = 5)
        predictVars = predictVars.tolist()
        predictiongraph = [predictionA]
        prediction = ARIMAMonteCarlo(predictionA, predictVars, 100)
    elif "Gaussian" in method:
        windows = int(method.split('-')[1])
        predictionA, predictVars = Gaussian(limit).predict(utils, windows=windows)
        predictVars = predictVars.tolist()
        predictiongraph = [predictionA]
        prediction = ARIMAMonteCarlo(predictionA, predictVars, 100)
    elif method == "Naive":
        prediction = [Naive(limit).predict(utils)]
        predictiongraph = prediction
    elif method == "Linear":
        prediction = [LinearExtrapolation(limit).predict(utils)]
        predictiongraph = prediction
    elif method == "ExponentialSmoothing":
        prediction = [ExponentialSmoothing(limit).predict(utils)]
        predictiongraph = prediction
    elif method == "LinearRegression":
        prediction = [LinearRegression(limit).predict(utils)]
        predictiongraph = prediction
    else:
        raise Exception("Method not yet implemented")
    if verbose:
        #this map/float is to prevent the output to be np.float(0.5) for example
        print(f"Prediction: {list(map(float, prediction[0]))}")
        if predictVars:
            print(f"Prediction: {list(map(float, predictVars))}")
        series_plot(utils, predictiongraph[0], len(utils), method, predictVars)
    return params, prediction #TODO: return params

def series_plot(utils, predicted_series, cutOff, method, predictV=[-1]): #display_plot
    def calcInterval(means, vars, perc):
        up = []
        down = []
        upFrac = 0.5 + perc / 2
        downFrac = 0.5 - perc / 2
        for m, v in zip(means, vars):
            up.append(stats.norm.ppf(upFrac, loc=m, scale=v))
            down.append(stats.norm.ppf(downFrac, loc=m, scale=v))

        return up, down

    t = utils[:cutOff]
    if (predictV != [-1]) & (predictV!=0):
        fiftyUp, fiftyDown = calcInterval(predicted_series, predictV, 0.5)
        ninetyUp, ninetyDown = calcInterval(predicted_series, predictV, 0.9)
        r = list(range(len(t + list(ninetyUp))))
        plt.fill_between(r, t + list(ninetyUp), t + list(ninetyDown), color='pink', alpha=0.2)
        plt.fill_between(r, t + list(fiftyUp), t + list(fiftyDown), color='purple', alpha=0.2)
    plt.plot(t + list(predicted_series), color='red')
    plt.plot(utils[:cutOff])
    plt.ylim(0, 1)
    plt.ylabel('Utility received')
    plt.xlabel('Time')
    plt.title("Time series forecast by "+ method)
    plt.show()

def intersect_series(predicted_own_utils, predicted_opp_utils, cutoff = 10, samplez = 100, method ="Monte carlo", verbose=False):
    """ Given two (predicted) series, this method returns the expected potential agreements. Now, the method used is Monte Carlo sampling."""
    if method == "Monte carlo":
        points = sample_monte_carlo(predicted_own_utils, predicted_opp_utils, cutoff, samplez)

        if verbose:
            if len(points) == 0:
                print("No intersection found.")
            else:
               print("One sample agreement utility is ", float(points[0][1]), "at expected time ", points[0][0])
        agreement_rate = len(points) / samplez
        return points, agreement_rate
    else:
        #throw error:
        raise Exception("Method not yet implemented")


def ARIMAMonteCarlo(predictM, predictV, samples):
    total = []
    for i in range(samples):
        s = []
        for j in range(len(predictM)):
            s.append(max(min(np.random.normal(predictM[j], predictV[j]),1),0)) #cap prediction at 0 and 1.
        total.append(s)
    return total

def sample_monte_carlo(predOwn, predOppo, cutoff, samplez): #called checkcross
    """
    parameters:
    predOppo: list of series of predicted opponent utilities
    predOwn: list of series of (predicted) own utilities
    cutoff: the cutoff point is the point in the negotiation where the prediction starts, often the current time
    samplez: the number of samples to take
    """
    points = []
    crossed = False
    for i in range(samplez):
        oppo = random.choice(predOppo)
        own = random.choice(predOwn)
        for j in range(len(oppo)):
            # print(pickOppo[j],pickOwn[j])
            if oppo[j] >= own[j]:

                if j != 0:
                    A = (j - 1, oppo[j - 1])
                    B = (j, oppo[j])
                    C = (j - 1, own[j - 1])
                    D = (j, own[j])
                    line1 = LineString([A, B])
                    line2 = LineString([C, D])
                    int_pt = line1.intersection(line2)
                    s = cutoff + round(int_pt.x)
                    points.append((s, int_pt.y))
                else:
                    s = cutoff
                    points.append((s, (oppo[j] + own[j]) / 2))
                # print(points[-1])
                crossed=True
                break
        if not crossed:
            a =2
    return points

def find_histogram(points,plot = True, method="None"):
    """Using the given points, find a histogram for both the timing of the agreement and the utility of the agreement."""
    hist = []
    hist2 = []
    for p in points:
      hist.append(p[1])
      hist2.append(p[0])
    print(f'len points: {len(points)}')
    plt.hist(hist,bins=20)
    plt.xlim(0,1)
    plt.ylabel('Frequency')
    plt.xlabel('Outcome utility')
    plt.title("Histogram outcome utility by " + method)
    plt.show()
    plt.ylabel('Frequency')
    plt.xlabel('Time of outcome agreement (rounds)')
    plt.title(f"Histogram timing of agreement by {method}")
    plt.hist(hist2,bins=20)
    plt.show()


def find_outcome_from_prob_distr(prob_distr, ufun):
    mean_utility = prob_distr.mean
    range = (mean_utility - 0.05, mean_utility + 0.05)
    print(
        f"Sampling: {ufun.sample_outcome_with_utility(rng=range, outcome_space=ufun.outcome_space, n_trials=5)}")

    pass







def getDim(cross, i=1):
    c = []
    for p in cross:
        c.append(p[i])
    return np.asarray(c)


