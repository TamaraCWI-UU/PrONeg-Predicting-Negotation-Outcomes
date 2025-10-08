"""
Example agent based on template from ANAC 2024

"""
import random
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from negmas.models.future import FutureUtilityRegressor
from Predictors.runner import predict_series, predict_outcome

class OutcomeNegotiator(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """


    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return
        self.last_outcome_prediction = self.ufun.reserved_value

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of times at which the agent offers
        self.agent_times: list[float] = []
        # keeps track of agent utilities of its offers
        self.agent_utilities: list[float] = []

    def __call__(self, state: SAOState, dest:str|None=None) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer

        self.update_outcome_prediction(state.current_offer, state.relative_time)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)



        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def update_outcome_prediction(self, offer: Outcome, relative_time: float, bidding = False) -> None:
        """
        This function updates the prediction of the opponent's utility based on the offer and the time of the offer.

        Args:
            offer: The offer from the opponent.
            relative_time: The relative time of the offer.

        Returns:
            None
        """
        if offer is None:
            print("Offer is None, first round")
            return
        if bidding:
            self.agent_times.append(relative_time)
            self.agent_utilities.append(float(self.ufun(offer)))
        else:
            self.opponent_times.append(relative_time)
            self.opponent_utilities.append(float(self.ufun(offer)))
        deadline = self.nmi.n_steps

        predictionsOpp = None

        #Only use outcome prediction when the number of data points is large enough:
        if len(self.agent_utilities) > 3 and len(self.agent_utilities) < deadline:
            predictionsOpp = predict_series(self.opponent_utilities, limit = deadline, verbose = False, method="ARIMA")[0]
        #predictionsOwn = predict_series(self.agent_utilities, limit = deadline, verbose = True, method="ARIMA")[0

        #If the predictions for opponent utilities were made, and if it is not the last round, perform outcome prediction
        if predictionsOpp != None and len(self.agent_utilities) >3 and len(self.agent_utilities) < deadline:
            predict_outcome(self.agent_utilities, self.opponent_utilities, limit=deadline, method="ARIMA",
                            verbose=False)
            last_prediction = predictionsOpp[-1]
            if last_prediction == None:
                self.last_outcome_prediction = self.ufun.reserved_value
            elif last_prediction > self.ufun.reserved_value:
                self.last_outcome_prediction = last_prediction
            range = (last_prediction-0.05, last_prediction+0.05)
            print(f"Sampling: {self.ufun.sample_outcome_with_utility(rng=range, outcome_space=self.ufun.outcome_space, n_trials=5)}")




    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun
        #self.nmi.state.relative_time
        #self.nmi.time_limit
        #deadline = self.nmi.n_steps

        received_offer = state.current_offer
        #if the current offer is higher than what we expect the negotiation to end at, accept the offer.
        if self.ufun(received_offer) > self.last_outcome_prediction:
            return True
        return False



        #if self.ufun(offer) > (3 * self.ufun.reserved_value):
        #    return True
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used.
        if self.last_outcome_prediction > self.reserved_value:
            bid = self.ufun.sample_outcome_with_utility((self.reserved_value,self.last_outcome_prediction), self.ufun.outcome_space)
        else:
            bid = self.ufun.sample_outcome_with_utility((self.reserved_value,1), self.ufun.outcome_space)
        return bid

# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(OutcomeNegotiator, small=True, nologs=False)
