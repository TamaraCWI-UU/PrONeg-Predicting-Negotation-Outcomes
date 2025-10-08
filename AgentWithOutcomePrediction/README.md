This example agent is based on the ANL 2024 template agent. It is adapted with the module outcome prediction to showcase a very simple use.

Where is it used?
- As a start, in preference_changed, the outcome_prediction is set as the reservation value.
- When the agent is called to react, the outcome prediction module is called to update it's last outcome prediction. 
- In the acceptance_strategy function, the agent accepts when the offer is higher than what the current outcome prediction is.
- In the bidding strategy function, it only proposes bids with a higher utility than the current outcome prediction.

Note that this agent is not meant to perform well; instead, it shows how the outcome module can be called. 