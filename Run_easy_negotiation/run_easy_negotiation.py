"""
This is the code that is part of Tutorial 1 for the ANL 2024 competition.
This can give run the outcome prediction agent, and show visual results.

"""

from negmas import (
    make_issue,
    SAOMechanism,
)
from AgentWithOutcomePrediction.myagent import OutcomeNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as UFun
#from negmas.preferences.value_fun import IdentityFun, AffineFun
from negmas.preferences import UtilityFunction #b
import matplotlib.pyplot as plt
#from matplotlib import colormaps
import negmas.inout as io#b
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter
import pathlib
from anl.anl2024 import pie_scenarios, anl2024_tournament
from pathlib import Path
import pandas as pd
TRACE_COLS = (
    "time",
    "relative_time",
    "step",
    "negotiator",
    "offer",
    "responses",
    "state",
)
def holidaydomain():

    # create negotiation agenda (issues)
    issues = [
        make_issue(name="Location", values=['Antalya', 'Barcelona', 'Milan']),
        make_issue(name="Duration", values=['1 week', '2 weeks']),
        make_issue(name='Hotel Quality', values=['Hostel', '3star hotel', '5 star hotel']),
    ]

    # create the mechanism
    session = SAOMechanism(issues=issues, n_steps=30)

    A_utility = UFun(
        {'Location': {'Antalya': 4, 'Barcelona': 10, 'Milan': 2},
        'Duration': {'1 week': 3, '2 weeks': 10},
        'Hotel Quality': {'Hostel': 10, '3star hotel': 2, '5 star hotel': 3}},
        weights={'Location': 0.5, 'Duration': 0.2, 'Hotel Quality': 0.3},
        issues=issues
    )

    B_utility = UFun(
        {'Location': {'Antalya': 3, 'Barcelona': 2, 'Milan': 10},
        'Duration': {'1 week': 4, '2 weeks': 10},
        'Hotel Quality': {'Hostel': 3, '3star hotel': 3, '5 star hotel': 10}},
        weights={'Location': 0.5, 'Duration': 0.4, 'Hotel Quality': 0.1},
        issues=issues
    )
    print(list(map(str, issues)))
    A_utility = A_utility.normalize(to=(0.0,1.0))
    B_utility = B_utility.normalize()

    return (session, A_utility, B_utility)

def partydomain():
     path_to_folder = pathlib.Path(__file__).parent/'Party'
     domain = io.load_genius_domain_from_folder(path_to_folder) #Put the 'Party' folder in the same folder as this file. If file not found, paste the full directory path here.
     A_utility, _ = UtilityFunction.from_genius(file_name=path_to_folder/'Party-prof1.xml', issues=domain.issues)
     B_utility, _ = UtilityFunction.from_genius(file_name=path_to_folder/'Party-prof2.xml',issues=domain.issues)
     A_utility = A_utility.normalize()
     B_utility = B_utility.normalize()

     session = SAOMechanism(issues=domain.issues, n_steps=30)
     return (session, A_utility, B_utility)

def laptopdomain():
    path_to_folder = pathlib.Path(__file__).parent / 'laptop'
    domain = io.load_genius_domain_from_folder(path_to_folder)
    # I turned the profiles around.
    B_utility, _ = UFun.from_genius(file_name=path_to_folder / 'laptop-prof1.xml', issues=domain.issues)
    A_utility, _ = UFun.from_genius(file_name=path_to_folder / 'laptop-prof2.xml', issues=domain.issues)

    A_utility = A_utility.scale_max(1)
    B_utility = B_utility.scale_max(1)

    session = SAOMechanism(issues=domain.issues, n_steps=5)
    visualize((session, A_utility, B_utility))
    return (session, A_utility, B_utility)

def generate_domain():
    scenario = pie_scenarios(n_scenarios=1, n_outcomes=200)[0]
    mech = SAOMechanism(issues=scenario.issues, n_steps=15)
    A_utility = scenario.ufuns[0]
    B_utility = scenario.ufuns[1]
    return (mech, A_utility, B_utility)



def visualize(negotiation_setup: (SAOMechanism, UFun, UFun)):
    (mech, A_utility, B_utility) = negotiation_setup

    # create and add selller and buyer to the session
    AgentA= OutcomeNegotiator(name="A")
    AgentB = Conceder(name="B")

    mech.add(AgentA, ufun=A_utility)
    mech.add(AgentB, ufun=B_utility)

    #plot_scenario(mech) (uncomment to show only the scenario, not the negotiation)

    # run the negotiation and show the results
    results = mech.run()
    mech.plot(ylimits=(0.0, 1.01), mark_max_welfare_points=False, mark_kalai_points=False, mark_nash_points=False)
    plt.show()
    print("Negotiation results:", mech.extended_trace)
    print("Negotiation results:", mech.full_trace)

    print("Final agreement is:", results.agreement)
    #plot_result(1, mech)
    #results.mechanism.full_trace
    return mech


def plot_result(i, m, base=Path.home() / "negmas" / "outcome_prediction" / "sessions"):
    assert base is not None
    df = pd.DataFrame(data=m.full_trace, columns=TRACE_COLS)  # type: ignore
    df.to_csv(base / "log" / f"{m.id}.csv", index_label="index")
    m.plot(save_fig=False, path=str(base / "plots"), fig_name=f"n{i}.png")
    plt.close()

def plot_scenario(m):
    m.plot(ylimits=(0.0, 1.01), mark_max_welfare_points=False, mark_kalai_points=False, mark_nash_points=False, mark_ks_points=False, show_total_time=False, show_n_steps=False, show_agreement=False, show_last_negotiator=False)
    plt.show()


if __name__ == "__main__":
    #negotiation_setup = holidaydomain()
    negotiation_setup = generate_domain()
    #negotiation_setup = laptopdomain()
    m = visualize(negotiation_setup)

