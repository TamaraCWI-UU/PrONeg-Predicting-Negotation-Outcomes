# example of a predictor class that can be used

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import Holt
import numpy as np
import sklearn.linear_model as skl
from typing import List, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import math
#from pmdarima.arima import auto_arima

class Naive:

    def __init__(self,limi):
        self.limit = limi

    def predict(self, utils, own = False):
        rem = self.limit-len(utils)

        return [utils[-1]]*rem

class LinearExtrapolation:

    def __init__(self,limi):
        self.limit = limi

    def predict(self,utils, own=False):
        first = utils[0]
        final = utils[-1]
        slope = (final - first) / len(utils)
        prediction = []

        rem = self.limit - len(utils)
        for i in range(rem):
            prediction.append(slope * (i + len(utils)) + first)
        prediction = [min(x, 1) for x in prediction]

        return prediction


class ARIMApred:

    def __init__(self,limi,d=2,p=2,q=2,dist = False, auto=False):
        self.limit = limi
        self.d=d
        self.p=p
        self.q=q
        self.dist=dist
        self.auto = auto

    def predict(self, utils, windows=0, own=False):

    #uncommented

    #    if self.auto:
    #        values = auto_arima(utils)
    #        self.d, self.p, self.q = values.order
    #        #print("Auto Arima gave", d, p, q)

        rem = self.limit - len(utils)
        if windows != 0:
            if len(utils) > windows:
                utils_bids = window_max_min(utils, self.limit, windows, own)
                utils = [bid.utility for bid in utils_bids]  # Extract utilities from Bid objects

        model = ARIMA(utils, order=(self.d, self.p, self.q))
        try:
            model_fit = model.fit()#(method_kwargs={'maxiter':150})
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            model = ARIMA(utils[:-1], order=(self.d, self.p, self.q))
            model_fit = model.fit()
        #print(model_fit.mle_retvals)
        converged = model_fit.mle_retvals['converged']
        iterations = model_fit.mle_retvals['iterations']





        prediction = model_fit.get_forecast(rem)



        means = prediction.predicted_mean
        means = [min(x, 1) for x in means]

        if self.dist:
           vars = prediction.var_pred_mean
           vars = [max(min(x, 1), 0) for x in vars]
           return (converged, iterations, -1), means,vars #let op, removed [] #TODO: return order
        else:
            return means


class Bid:
    def __init__(self, utility, index, limit):
        self.utility = utility
        self.index = index
        self.time = index/limit
        self.max = False

def window_max_min(utils, limit, nr_of_windows=5, own = False):
    windowed_utils = []



    window_size = len(utils) // nr_of_windows  # Round down
    remainder = len(utils) % nr_of_windows

    if len(utils) <= nr_of_windows:
        for i, u in enumerate(utils):
            windowed_utils.append(Bid(u, i, limit))
        return windowed_utils

    # take the max from each window of size window_size and add it it to the list

    # you cut the number of utils in 5 pieces. If the number is not dividable by 5, the 6th window is
    # added to the 5th window.
    for w in range(nr_of_windows):
        max_util = -1
        max_index = -1
        min_util = 100
        min_index = -1

        start_index = w * window_size
        end_index = start_index + window_size

        if w == nr_of_windows - 1:
            end_index += remainder  # Include the remainder in the last window

        for i in range(start_index, min(end_index, len(utils))):
            if utils[i] > max_util:
                max_util = utils[i]
                max_index = i
            if utils[i] < min_util:
                min_util = utils[i]
                min_index = i
        # Append the maximum Bid for the current window
        if max_index != -1:  # Ensure we found a valid index
            if not own:
                windowed_utils.append(Bid(max_util, max_index, limit))
        if min_index != -1:
            if own:
                windowed_utils.append(Bid(min_util, min_index, limit))

    return windowed_utils

class Gaussian:
    def __init__(self,limi):
        self.limit = limi
        # initial values
        self.kernel = kernels.DotProduct(sigma_0_bounds=(1e-8, 1e6)) * kernels.Matern(length_scale_bounds=(1e-8, 1e6))#nu = 3/2 is equivalent of matern3
        # Model
        self.regressor = GaussianProcessRegressor(self.kernel)




    def predict(self, utils, own=False, windows = 5):
        rem = self.limit - len(utils)


        # Training and prediction
        intercept = utils[0]#self.firstBidFromOpponent.utility

        if windows == 0:
            max_min_utils = window_max_min(utils, self.limit, len(utils) + 1000, own)  # this returns all utils as bids.
        else:
            max_min_utils = window_max_min(utils, self.limit, windows, own)

        x, y = self.get_data(max_min_utils, intercept)#self.history.get_data(self.firstBidFromOpponent)
        # Fit the model
        # self.regressor = GaussianProcessRegressor(self.kernel) don't think again is necessary (?)
        self.model_fit = self.regressor.fit(x, y)

        # Predict mean and variance
        # what samples to take:
        #start at len(utils)/self.limit. rem = self.limit- len(utils)
        self.time_samples_list = []
        for i in range(len(utils), self.limit):
            self.time_samples_list.append(i/self.limit)
        #self.time_samples_list = [i * 0.01 for i in range(101)]
        self.time_samples = np.reshape(np.array(self.time_samples_list, dtype=np.float32), (-1, 1))
        #
        self.predict_mean_variance(intercept)

        #top off the means predictions at zero and 1.
        self.means = [max(min(x, 1),0) for x in self.means]
        return self.means, self.var

    def get_data(self, utils: List[Bid], intercept):
        """
            This method generates the corresponding X and Y numpy arrays for the training.
        :param first_bid: First bid of the opponent for adjusting.
        :return: X and Y numpy arrays
        """
        # Declare X and Y list
        x = []
        y = []
        # Inputs will be adjusted for a better result
        x_adjust = []
        #intercept = utils[0]#first_bid.utility
        gradient = 0.9 - intercept  #

        for index, opponent_bid in enumerate(utils):#self.history:
            # X: Negotiation Time, Y: Utility
            time = opponent_bid.time
            if time > 1 or time < 0:
                print("time out of bounds")
            x.append(time)#x.append(opponent_bid.time) #do the index/deadline here.
            y.append(opponent_bid.utility)#y.append(opponent_bid.bid.utility) #utility
            if False:
                print(f"Time: {time}, Utility: {opponent_bid.utility}, Adjusted: {intercept + (gradient * time)}")
            # x_adjust = first_utility + gradient * t
            x_adjust.append(intercept + (gradient * time))

        # Convert into numpy array
        x = np.reshape(np.array(x, dtype=np.float32), (-1, 1))
        y = np.reshape(np.array(y, dtype=np.float32) - x_adjust, (-1, 1))
        #y = np.reshape(np.array(y, dtype=np.float32), (-1, 1))

        if False:
            print(f"X: {x}, Y: {y}")

        return x, y

    def predict_mean_variance(self, intercept: float):
        """
            This method predicts the mean and variance
        :param intercept: First utility of the opponent
        :return: Nothing
        """

        # Prediction by the model
        self.means, self.sigma = self.model_fit.predict(self.time_samples, return_std=True)
        self.var = self.sigma ** 2

        # adjust = first_utility + gradient * t
        adjust = intercept + self.time_samples * (0.9 - intercept)
        adjust = adjust.reshape((adjust.shape[0],))

        self.means += adjust  # Adjusted mean


class ExponentialSmoothing:

    def __init__(self,limi):
        self.limit = limi

    def predict(self, utils, own=False):
        model = Holt(utils)
        model_fit = model.fit()
        rem = self.limit - len(utils)
        forecast = model_fit.forecast(rem)
        forecast = [min(x, 1) for x in forecast]
        return forecast

class LinearRegression:

    def __init__(self,limi):
        self.limit = limi

    def predict(self, utils, own=False):
        serie = np.array(utils).reshape(-1, 1)
        x = np.arange(len(utils)).reshape(-1, 1)
        model = skl.LinearRegression()
        model.fit(x, serie)
        predictLen = np.arange(len(utils), self.limit).reshape(-1, 1)
        predict = model.predict(predictLen)
        predict = list(predict.reshape(1, -1).tolist()[0])
        predict = [min(x, 1) for x in predict] #let op, later toegevoegd
        return predict




