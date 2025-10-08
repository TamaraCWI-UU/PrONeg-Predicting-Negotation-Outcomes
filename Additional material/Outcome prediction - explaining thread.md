In this document, we discuss an example of the pipeline and use of PrONeg within an ongoing negotiation. This document is supplementary material for the paper ``Predicting the Outcome of Ongoing Automated Negotiations`` accepted as full paper at PRIMA 2025.

Assume our perspective is Agent A, that bilaterally negotiates with Agent B. We inspect one moment in an exemplary negotiation. Suppose our current round is 10, the deadline is 15. The history of bids is as follows.



Note that this list of utility is the utility of the received bids as seen from our agent. So the opponent might value their first bid as 0.9, while in our agent's perspective, it is only 0.3 as included in the table.

In step 1 of the pipeline, we predict the bidding curve using Time Series Forecast methods. We can choose any TSF method as we prefer. For now, we choose Gaussian processes as that showed good performances in the associated paper.

First we predict the bidding curve of the opponent, as seen from the agent's utility perspective.

``listOpp = [...]
``g = Gaussian(15)
``g.predict(listOpp)

Dependent on the type of TSF method, we receive back a predictive distribution or a line. In this case, we receive a distribution in the form of a sequence of averages with associated standard deviation, e.g. 
`` [...]

Visually, this would look like this:
...

We perform the same steps for the history of bids of our own agent.
``listOwn - [...]
``g.predict(listOwn)

Giving this picture:
...

Next, we combined these two predictions for our outcome prediction step, step 2. From each distribution, we sample 100 times. Each sample consists of in this case a sequence of 5 utilities (15 -10, deadline - current round). Each pair is checked for a possible intersection.  E.g.
``...
``...

If there is an intersection like in this case, the outcome is logged as ``agreement`` ,  with corresponding utility. In this case, they meet between the 3rd and 4th bid. We then calculate the point of intersection as if they were lines, i.e., 0.35.

For all these 100 samples, we log whether the classification is agreement or failure, and log the corresponding outcome if applicable. For example, 70% of the samples resulted in agreement, giving an agreement prbobability of 70%.