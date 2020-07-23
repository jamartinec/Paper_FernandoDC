#!/usr/bin/env python3.7



import argparse
import numpy as np

import pandas as pd

import time



def split_slack(kp):
    """Computes the expected value of the splitting item r.v. and the expected value of the slack r.v.

    Computes E(S), the expected value of the splitting item r.v. S using the expression (26b) in theorem 8.
    Computes E(K), the expected value of the slack r.v. K, using the expression (29c) in Theorem 11.

    Args:
        kp: Int. knapsack capacity

    Returns:
        s: Int. the expected value of the splitting item r.v. S
        k: float. the expected value of the slack r.v. K

    """
    s = (1 + 1 / kp) ** kp
    base = 1 + 1 / kp
    k = - (base ** (kp) - 1) * base
    k = k + (base ** (kp + 1) - 1 - base) * (kp + 3)
    k = k - 2 * kp * (base ** (kp + 2) - (5 * kp ** 2 + 7 * kp + 2) / (2 * kp ** 2))
    return (s, k)

def func_split_item(k):
    """ Computes the expected value and variance of the splitting item random variable S.

    Computes the expression (26b) and (26c) in Theorem 8. Remember that r.v. S is the value of index s
    such that $\sum_{i=1}^{s-1} w(i) \leq k$ and $\sum_{i=1}^s w(i) > k$.

    Args:
        k: Int. The capacity of the Knapsack Problem instance.

    Returns:
        s: float. The expected value of the splitting item random variable.
        var_split: float. The variance of the splitting item random variable.

    """
    b = 1 + 1 / k  # Defining a cumbersome base
    s = (1 + 1 / k) ** k  # Computing the split item
    var_split = (3 + 1 / k) * b ** (k - 1) - b ** (2 * k)  # Computing the variance of the split item
    return s, var_split

def func_slack(k):
    """Computes the expected value of the slack random variable K.

    Computes the expressions (29c) in Theorem 11. Remember that r.v. $K:= k-\sum_{j=1}^{S-1}W_j$

    Args:
        k: Int. The capacity of the Knapsack Problem instance.

    Returns:
        slack_closed: float. The expected value of the slack random variable K.

    """
    b = 1 + 1 / k  # Defining a cumbersome base
    slack_closed = - (b ** (k) - 1) * b
    slack_closed = slack_closed + (b ** (k + 1) - 1 - b) * (k + 3)
    slack_closed = slack_closed - 2 * k * (b ** (k + 2)
                                           - (5 * k ** 2 + 7 * k + 2) / (2 * k ** 2))  #closed formula of the Slack
    rel_slack_closed = 100 * slack_closed / k  # relative slack
    return slack_closed, rel_slack_closed

def func_greedy(k):
    """ Computes the expected value of the objective function reached by the greedy algorithm.

    Computes the expression (28b) in Theorem 10. Remember that Z^G := \sum_{i=1}^{S-1} P(i).

    Args:
        k: Int. The capacity of the Knapsack Problem instance.

    Returns:
        greedy_closed: float. The expected value of the objective function reached by greedy algorithm

    """
    b = 1 + 1 / k  # Defining a cumbersome base
    greedy_closed = - (k + 1) ** 2 * b ** (k - 1) / (4 * k)
    greedy_closed = greedy_closed + (2 * k + 3) * (k + 2) * (k + 1) * (b ** k - 1) / (4 * k)
    greedy_closed = greedy_closed - (k + 2) ** 2 * (b ** (k + 1) - 1 - b)
    greedy_closed = greedy_closed + k * (2 * k + 5) * (
            b ** (k + 2) - 1 - (k + 2) / k - (k + 2) * (k + 1) / (2 * k ** 2)) / 2

    return greedy_closed

def func_slack_profit(k, s, greedy_closed, slack_closed):
    """Computes the extra profit from slack, and the linear program relaxation.

    The extra profit is the second summand in expression (30) Theorem 12.
    Then computes expression (30) and (33d).


    Args:
        k: Int. The capacity of the Knapsack Problem instance.
        s: float. The expected value of the splitting item random variable.
        greedy_closed: float. The expected value of the objective function reached by greedy algorithm
        slack_closed: float. The expected value of the slack random variable K.

    Returns:
        lp_closed: float. The expected value of the objective function reached by Linear Programming solution. Is the
                    greedy solution plus slack solution. Expression (30) Theorem 12.
        lp_approx: float. Approximation of the expected value of the Linear Programming o.f. solution.
                   See expression (33d) Theorem 13 (Asymptotic Relations).

        error_approximation: float. Error % when using the expected value approximation of the solution to the linear
                            programming problem
        radio_greedy_linear: ratio Greedy vs Linear Programming Relaxation

    """
    b = 1 + 1 / k  # Defining a cumbersome base

    slack_profit = (k + 1) * b ** (k - 1) / (2 * k)
    slack_profit = slack_profit - (k + 2) * (k + 1) * (b ** k - 1) / (k)
    slack_profit = slack_profit + (k + 2) * (k + 5) * (b ** (k + 1) - 1 - b) / 2
    slack_profit = slack_profit - k * (k + 3) * (b ** (k + 2) - 1 - (k + 2) / k - (k + 2) * (k + 1) / (
            2 * k ** 2))  # Extra Slack for Linear Programming Relaxation

    lp_closed = greedy_closed + slack_profit  # Computing the expected Linear Programming Solution

    lp_approx = greedy_closed + slack_closed * (k - s + 1) / 2 # Approximating Linear Programming Assuming Independence

    error_approximation = 100 * (lp_closed - lp_approx) / lp_closed

    radio_greedy_linear = 100 * greedy_closed / lp_closed # Computing ratio Greedy vs Linear Programming Relaxation

    return lp_closed, lp_approx, error_approximation, radio_greedy_linear

def func_eligible_first(k_idx,slack_closed,greedy_closed):
    """ Computes an approximation to the expected value of the o.f. value reached by the elegible first algorithm.

        Computes the expression (37) en Corollary 16.

    Args:
        k_idx: Int. Knapsack capacity.
        slack_closed: float. The expected value of the slack random variable K.
        greedy_closed: float. The expected value of the objective function reached by greedy algorithm

    Returns:
        ef_closed: float. An approximation (closed formula) to the expected value of the elegible-first algorithm o.f. value.

    """
    kp = k_idx # Knapsack Capacity

    k_0 = slack_closed  # Known average slack capacity
    s_0 = (1 + 1 / kp) ** kp  # Average split item index
    base_ef = 1 - k_0 / kp  # Cumbersom term, probability of success eligible-first post-greedy
    ef_closed = ((1 - base_ef ** (kp - s_0 + 1)) * (kp - s_0 + 1) * k_0 / 4)
    ef_closed = ef_closed - ((1 -
                              (1 + (kp - s_0) * k_0 / kp) * base_ef ** (kp - s_0)
                              ) * base_ef * kp / (4 * k_0))
    ef_closed = ef_closed + greedy_closed

    return ef_closed

def func_lef_right_capacities(k_idx):
    """Computes the expected value of the random variable capacitires of the left and right problems.

    Computes E(C_lef) and E(C_right) according to expressions (43a) and (43b) in Theorem 19.

    Args:
        k_idx: Int. Knapsack capacity.

    Returns:
        cap_left_closed: Int. Expected capacity of left subproblem.
        cap_right_closed: Int. Expected capacity of right subproblem
    """

    kp = k_idx  # Knapsack Capacity
    mu = kp + 1  # Quantity of Items
    deviation = mu * ((1 + 1 / kp) ** mu + (1 - 1 / kp) ** mu) / 4 - kp * (
            (1 + 1 / kp) ** (mu + 1) - (1 - 1 / kp) ** (mu + 1)) / 4  # Deviation of the Knapsack capacities
    # PILAS, CREO QUE ARRIBA LOS DENOMINADORES SON 8
    sum_deviation = 0
    for k in range(0, mu, 2):
        sum_deviation = sum_deviation + k * (1 + 1 / kp) ** (k - 1)  # Common sum for both terms

    cap_left_closed = kp / 2 + sum_deviation / (2 * kp ** 2) + deviation
    cap_right_closed = kp / 2 - sum_deviation / (2 * kp ** 2) - deviation

    return cap_left_closed, cap_right_closed



def func_greedy_left_solution(k_idx, cap_left_closed):
    """ Computes an approximation for E(Z_{left}^G) on left subproblem

    Computes an approximation for the expectation of the o.f. value reached by the greedy algorithm on left
    subproblem, using the expression (50a) in Corollary 23.

    Args:
        k_idx: Int. Knapsack capacity.
        cap_left_closed: Int. Expected capacity of left subproblem.

    Returns:
        closed_ZG_left: float. The expected value of Z^G (o.f. value of greedy algorithm)  for the left subproblem.

    """
    kp = k_idx  # Knapsack Capacity
    mu = kp + 1  # Quantity of Items
    closed_ZG_left = 0  # Computation of Greedy Solution Left
    c = cap_left_closed
    b = 1 + 1 / c
    closed_ZG_left = closed_ZG_left - c * b ** (c + 1) / 2
    closed_ZG_left = closed_ZG_left - (mu + 3) * (c + 2) * (b ** (c + 1) - b) / 2
    closed_ZG_left = closed_ZG_left + (2 * mu * c + 6 * c + 3 * mu + 10) / 2
    closed_ZG_left = closed_ZG_left + c * (mu + 3) * (
            b ** (c + 2) - 1 - (c + 2) / c - (c + 1) * (c + 2) / (2 * c ** 2))

    return closed_ZG_left

def func_greedy_right_solution(k_idx, cap_right_closed):
    """ Computes an approximation for E(Z_{right}^G) on left subproblem

    Computes an approximation for the expectation of the o.f. value reached by the greedy algorithm on right
    subproblem, using the expression (50b) in Corollary 23.

    Args:
        k_idx: Knapsack capacity.
        cap_right_closed: Int. Expected capacity of right subproblem.

    Returns:
        closed_ZG_right: float. The expected value of Z^G (o.f. value of greedy algorithm)  for the right subproblem.

    """
    kp = k_idx  # Knapsack Capacity
    mu = kp + 1  # Quantity of Items
    closed_ZG_right = 0  # Computation of Greedy Solution Right
    c = cap_right_closed
    b = 1 + 1 / c
    closed_ZG_right = closed_ZG_right - c * b ** (c + 1) / 2
    closed_ZG_right = closed_ZG_right - (mu + 2) * (c + 2) * (b ** (c + 1) - b) / 2
    closed_ZG_right = closed_ZG_right + (2 * mu * c + 4 * c + 3 * mu + 7) / 2
    closed_ZG_right = closed_ZG_right + c * (mu + 2) * (
            b ** (c + 2) - 1 - (c + 2) / c - (c + 1) * (c + 2) / (2 * c ** 2))

    return closed_ZG_right

def func_left_right_split_slack(cap_left_closed, cap_right_closed):
    """

    Args:
        cap_left_closed: Int. Expected capacity of left subproblem.
        cap_right_closed: Int. Expected capacity of right subproblem.

    Returns:
        s_left: Int.  Expected value of the splitting item for the left subproblem.
        k_left: float. The expected value of the slack for the left subproblem.
        s_right: Int.  Expected value of the splitting item for the right subproblem.
        k_right: float. The expected value of the slack for the right subproblem.

    """
    (s_left, k_left) = split_slack(cap_left_closed)  # Computing through subroutine "split_slack"
    (s_right, k_right) = split_slack(cap_right_closed)

    return (s_left, k_left), (s_right, k_right)

def func_left_linear_relaxation(k_idx, closed_ZG_left, k_left, s_left):
    """ Returns the expected value for the o.f. of the linear programming relaxation of left subproblem.

    Args:
        k_idx: Int. Knapsack capacity.
        closed_ZG_left: float. The expected value of Z^G (o.f. value of greedy algorithm)  for the left subproblem.
        k_left: float. The expected value of the slack for the left subproblem.
        s_left: Int.  Expected value of the splitting item for the left subproblem.

    Returns:
        lp_left: float. The linear relaxation solution for the left subproblem.

    """
    kp = k_idx  # Knapsack Capacity
    mu = kp + 1  # Quantity of Items
    lp_left = closed_ZG_left + k_left * (mu - s_left + 1) / 2

    return lp_left

def func_right_linear_relaxation(k_idx, closed_ZG_right, k_right, s_right):
    """Returns the expected value for the o.f. of the linear programming relaxation of right subproblem.

    Args:
        k_idx: Int. Knapsack capacity.
        closed_ZG_right: float. The expected value of Z^G (o.f. value of greedy algorithm)  for the right subproblem.
        k_right:  float. The expected value of the slack for the right subproblem.
        s_right: Int.  Expected value of the splitting item for the right subproblem.

    Returns:
        lp_right: float. The linear relaxation solution for the right subproblem.

    """
    kp = k_idx  # Knapsack Capacity
    mu = kp + 1  # Quantity of Items
    lp_right = closed_ZG_right + k_right * (mu - s_right + 1) / 2

    return lp_right


def func_left_elegible_first(k_idx,k_left,s_left, cap_left_closed, closed_ZG_left):
    """ Computes and approximation for the expected value of $Z_{left}^eF$ (left elegible first).

    Computes and approximation for the expected value of the o.f. value reached by the elegible first
    algorithm for the left subproblem using the expression (53a) in Corollary 26.

    Args:
        k_idx: Int. Knapsack capacity.
        k_left: float. The expected value of the slack for the left subproblem.
        s_left: Int.  Expected value of the splitting item for the left subproblem.
        cap_left_closed: Int. Expected capacity of left subproblem.
        closed_ZG_left: float. The expected value of Z^G (o.f. value of greedy algorithm)  for the left subproblem.

    Returns:
        closed_EF_left: float. An approximation for the elegible first solution on left subproblem

    """

    mu = k_idx + 1  # Quantity of Items
    lamb = mu / 2  # Number of Items in each subproblem
    k_0 = k_left  # Updating the split and slack values for the left problem
    s_0 = s_left
    kp = cap_left_closed
    closed_EF_left = k_0 * (kp - 2 * s_0) * (1 - (1 - k_0 / kp) ** (lamb - s_0)) / 4
    closed_EF_left = closed_EF_left - kp * (1 - k_0 / kp) * (
            1 - (1 + (kp - s_0 - 1) * k_0 / kp) * (1 - k_0 / kp) ** (lamb - s_0 + 1)) / (4 * k_0)
    ## REVISAR EL RENGLÓN ANTERIOR: CREO QUE ES LAMB-S0-1.
    closed_EF_left = closed_EF_left + closed_ZG_left  # Computing the eligible-first solution for the left subproblem

    return closed_EF_left


def func_right_elegible_first(k_idx, k_right, s_right, cap_right_closed, closed_ZG_right):
    """Computes and approximation for the expected value of $Z_{right}^eF$ (right elegible first).

    Computes and approximation for the expected value of the o.f. value reached by the elegible first
    algorithm for the right subproblem using the expression (53b) in Corollary 26.

    Args:
        k_idx:  Int. Knapsack capacity.
        k_right: float. The expected value of the slack for the right subproblem.
        s_right: Expected value of the splitting item for the right subproblem.
        cap_right_closed: Int. Expected capacity of right subproblem.
        closed_ZG_right: float. The expected value of Z^G (o.f. value of greedy algorithm)  for the right subproblem.

    Returns:
        closed_EF_right: float. An approximation for the elegible first solution on right subproblem

    """
    mu = k_idx + 1  # Quantity of Items
    lamb = mu / 2  # Number of Items in each subproblem
    k_0 = k_right  # Updating the split and slack values for the right problem
    s_0 = s_right
    kp = cap_right_closed
    closed_EF_right = k_0 * (kp - 2 * s_0 - 1) * (1 - (1 - k_0 / kp) ** (lamb - s_0)) / 4
    closed_EF_right = closed_EF_right - kp * (1 - k_0 / kp) * (
            1 - (1 + (kp - s_0 - 1) * k_0 / kp) * (1 - k_0 / kp) ** (lamb - s_0 + 1)) / (4 * k_0)
    closed_EF_right = closed_EF_right + closed_ZG_right  # Computing the eligible-first solution for the right subproblem

    return closed_EF_right

if __name__ == '__main__':
    '''CODE FOR CHECKING THE EVOLUTION OF CLOSED FORMULAS

    Args:
         Upper limit introduced by the uses in variable -l

    Returns:
        List of computed performance parameters

    '''
    parser = argparse.ArgumentParser(description='Closed Formulas Runner')
    parser.add_argument('-l', '--l', help='Knapsack Capacity', type=int, default=21)
    args = parser.parse_args()
    limit = args.l

    ##############################################################################
    ##    CREATING THE COLLECTING DATA LISTS
    ##############################################################################
    ##

    split_item = []      # Split Item index
    variance_split = []  # Variance of Split Item Index
    std_split = []       # Standard Deviation of the Split Item Index
    slack = []           # Slack avaiable in the Knapscak
    rel_slack = []       # Relavite Slack with respect to full capacity
    greedy_solution = [] # Greedy Solutions
    lp_list = []         # Linear Programming List
    lp_solution = []     # Linear Programming Solutions
    approx_error = []    # Approximation Error
    greedy_2_lp = []     # Greedy to Linear Programming Ratio
    approx_fF = []
    approx_eF = []       # Approximate Eligible-First Solution
    closed_fF = []
    closed_eF = []        # Closed Eligible-First Solution
    error_fF = []
    error_eF = []         # Error Eligible-First Solution
    capacity = []         # Knapsack Capacity
    capacity_left = []    # Knapsack Capacity on the left subproblem
    capacity_right = []   # Knapsack Capacity on the right subproblem
    greedy_left = []      # Greedy Algorithm on the left subproblem
    greedy_right = []     # Greedy Allgorithm on the right subproblem
    EF_left = []          # Eligible-First left subproblem
    EF_right = []         # Eligible-First right subproblem
    LP_left = []          # Linear Programming left subproblem
    LP_right = []         # Linear Programming right subproblem

    # Main Loop starting from 49 to the limit in increments of 20
    for k_idx in range(49, limit, 20):
        capacity.append(k_idx)
        ##############################################################################
        ##    SPLIT ITEM RANDOM VARIABLE
        ##############################################################################
        s, var_split = func_split_item(k_idx)
        split_item.append(s) # Storing the split item.
        std_split.append(np.sqrt(var_split)) # Storing the standard deviation of the split item
        ##############################################################################
        ##    SLACK RANDOM VARIABLE
        ##############################################################################
        slack_closed, rel_slack_closed = func_slack(k_idx)
        slack.append(slack_closed) # Storing the closed formula of the Slack
        rel_slack.append(rel_slack_closed) # Storing the relative slack
        ##############################################################################
        ##    GREEDY SOLUTION
        ##############################################################################
        greedy_closed = func_greedy(k_idx)
        greedy_solution.append(greedy_closed)  # Storing the closed formula of the Greedy Solution
        ##############################################################################
        ##    EXTRA PROFIT FROM SLACK AND LINEAR PROGRAM RELAXATION
        ##############################################################################
        lp_closed, lp_approx, error_approximation, radio_greedy_linear=\
            func_slack_profit(k_idx, s, greedy_closed, slack_closed)
        lp_solution.append(lp_closed)  # Storing the Linear Programming Solution
        lp_list.append(lp_approx) # Storing the Linear Programming approximation
        approx_error.append(error_approximation) # Storing the error approximation
        greedy_2_lp.append(radio_greedy_linear) # Storing the ratio Greedy vs Linear Programming Relaxation
        ##############################################################################
        ##    ELIGIBLE-FIRST ANALYSIS
        ##############################################################################
        ef_closed = func_eligible_first(k_idx, slack_closed, greedy_closed)
        closed_eF.append(ef_closed)  # Storing the eligible-first result
        ##############################################################################
        ##    LEFT AND RIGHT CAPACITIES
        ##############################################################################
        cap_left_closed, cap_right_closed = func_lef_right_capacities(k_idx)
        capacity_left.append(cap_left_closed)  # Storing the Left Capacity
        capacity_right.append(cap_right_closed)  # Storing the Right Capacity
        ##############################################################################
        ##    GREEDY LEFT AND RIGHT SOLUTIONS
        ##############################################################################
        closed_ZG_left = func_greedy_left_solution(k_idx, cap_left_closed)
        greedy_left.append(closed_ZG_left)  # Storing of Greedy Solution Left
        closed_ZG_right = func_greedy_right_solution(k_idx, cap_right_closed)
        greedy_right.append(closed_ZG_right)  # Storing of Greedy Solution Right
        ##############################################################################
        ##    LEFT AND RIGHT SPLIT AND SLACK
        ##############################################################################
        (s_left, k_left),(s_right, k_right) = func_left_right_split_slack(cap_left_closed, cap_right_closed)
        ##############################################################################
        ##    LEFT AND RIGHT LINEAR RELAXATION
        ##############################################################################
        lp_left = func_left_linear_relaxation(k_idx, closed_ZG_left, k_left, s_left)
        LP_left.append(lp_left)  # Storing the Linear Relaxation solution for left subproblem
        lp_right = func_right_linear_relaxation(k_idx, closed_ZG_right, k_right, s_right)
        LP_right.append(lp_right)  # Storing the Linear Relaxation solution for right subproblem
        ##############################################################################
        ##    LEFT AND RIGHT ELIGIBLE-FIRST
        ##############################################################################
        closed_EF_left = func_left_elegible_first(k_idx, k_left, s_left, cap_left_closed, closed_ZG_left)
        EF_left.append(closed_EF_left)  # Storing the eligible-first solution for the left subproblem
        closed_EF_right = func_right_elegible_first(k_idx, k_right, s_right, cap_right_closed, closed_ZG_right)
        EF_right.append(closed_EF_right)  # Storing the eligible-first solution for the right subproblem
        ##############################################################################
        ##    RESULTS DISPLAY
        ##############################################################################
        ##

        Results = pd.DataFrame({  # Asembly of results in common Data Frame
            'Capacity': capacity,
            ##                          'Items': np.array(capacity) + 1,
            ##                              'Split_Item': split_item,
            ##                          'Variance_Split': variance_split,
            ##                          'STD_Split': std_split,
            ##                          'Slack': slack,
            ##                          'Rel_Slack': rel_slack,
            ##                          'Greedy': greedy_solution,
            ##                          'Greedy_Left': greedy_left,
            ##                          'Greedy_Right': greedy_right,
            ##                          'Greedy_Performance': 100*( np.array(greedy_left) + np.array(greedy_right) ) / np.array(greedy_solution),
            ##                          'LP': lp_solution,
            ##                          'LP_Left': LP_left,
            ##                          'LP_Right': LP_right,
            'LP_Eff': 100 * (np.array(LP_left) + np.array(LP_right)) / np.array(lp_list),
            'LP_Eff_L': 100 * (np.array(LP_left)) / np.array(lp_list),
            'LP_Eff_R': 100 * (np.array(LP_right)) / np.array(lp_list),
            ##                          'E-First': closed_eF,
            ##                          'EF_Left': EF_left,
            ##                          'EF_Right': EF_right,
            'EF_Eff': 100 * (np.array(EF_left) + np.array(EF_right)) / np.array(closed_eF),
            'EF_Eff_L': 100 * (np.array(EF_left)) / np.array(closed_eF),
            'EF_Eff_R': 100 * (np.array(EF_right)) / np.array(closed_eF),
            ##                          'Cap_Left': capacity_left,
            ##                          'Cap_Right': capacity_right,
            'lb^G': 100 * (np.array(greedy_left) + np.array(greedy_right)) / np.array(lp_solution),
            'lb^G_Left': 100 * (np.array(greedy_left)) / np.array(lp_solution),
            'lb^G_Right': 100 * (np.array(greedy_right)) / np.array(lp_solution),
            'lb^eF': 100 * (np.array(EF_left) + np.array(EF_right)) / np.array(lp_list),
            'lb^eF_Left': 100 * (np.array(EF_left)) / np.array(lp_list),
            'lb^eF_Right': 100 * (np.array(EF_right)) / np.array(lp_list)
        })

        print(Results.tail(20))
        ##  print(Results.mean().round(decimals = 2))
        ##  print(Results.var().round(decimals = 2))

        Table = Results.round(decimals=2)

        ##########################################################################################
        ####  GRAPHICS FOR THE EXPECTED PERFORMANCE
        ##########################################################################################
        ####
        ##  del Table[ 'Items' ]
        ##  Table.columns = ['Capacity', 'LP', 'LP_Left', 'LP_Right', 'eF', 'eF_Left', 'eF_Right']
        ##  Table.plot( x = 'Capacity')
        ##  plt.savefig('Graph_Performance.eps')
        ##  plt.savefig('Graph_Performance.pdf')
        ##  plt.savefig('Graph_Performance.png')

        ############################################################################################
        ######  GRAPHICS FOR THE EXPECTED LOWER BOUNDS
        ############################################################################################
        ######
        ##  del Table[ 'Items' ]
        ####  Table.columns = ['Capacity', 'LP', 'LP_Left', 'LP_Right', 'eF', 'eF_Left', 'eF_Right']
        ##  Table.plot( x = 'Capacity')
        ##  plt.savefig('Graph_Bounds.eps')
        ##  plt.savefig('Graph_Bounds.pdf')
        ##  plt.savefig('Graph_Bounds.png')

        Table.to_csv('Table_Approximation.csv', index=False)

        print('tiempo de procesado', time.process_time())





















