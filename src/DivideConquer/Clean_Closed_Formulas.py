#!/usr/bin/env python3.7

'''
CODE FOR CHECKING THE EVOLUTION OF CLOSED FORMULAS

Input:
- Upper limit introduced by the uses in variable -l

Output:
- List of computed performance parameters

'''

import os, argparse
import numpy as np
from numpy import linalg as LA
import math
import xlrd
import xlwt
import statistics
import scipy
from scipy import stats
from scipy.optimize import linprog
from numbers import Number
import sys
import csv
from subprocess import check_output, call, Popen, PIPE
import shlex
import pandas as pd
##import matplotlib.pyplot as plt
##from matplotlib import colors
import time
import itertools


def split_slack(kp):
    """ Computes the expected value and

    Args:
        kp:

    Returns:

    """
    s = (1 + 1 / kp) ** kp
    base = 1 + 1 / kp
    k = - (base ** (kp) - 1) * base
    k = k + (base ** (kp + 1) - 1 - base) * (kp + 3)
    k = k - 2 * kp * (base ** (kp + 2) - (5 * kp ** 2 + 7 * kp + 2) / (2 * kp ** 2))
    return (s, k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Closed Formulas Runner')
    parser.add_argument('-l', '--l', help='Knapsack Capacity', type=int, default=21)
    args = parser.parse_args()
    limit = args.l

    ##############################################################################
    ##    CREATING THE COLLECTING DATA LISTS
    ##############################################################################
    ##

    split_item = []  # Split Item index
    variance_split = []  # Variance of Split Item Index
    std_split = []  # Standard Deviation of the Split Item Index
    slack = []  # Slack avaiable in the Knapscak
    rel_slack = []  # Relavite Slack with respect to full capacity
    greedy_solution = []  # Greedy Solutions
    lp_list = []  # Linear Programming List
    lp_solution = []  # Linear Programming Solutions
    approx_error = []  # Approximation Error
    greedy_2_lp = []  # Greedy to Linear Programming Ratio

    approx_fF = []
    approx_eF = []  # Approximate Eligible-First Solution
    closed_fF = []
    closed_eF = []  # Closed Eligible-First Solution

    error_fF = []
    error_eF = []  # Error Eligible-First Solution

    capacity = []  # Knapsack Capacity
    capacity_left = []  # Knapsack Capacity on the left subproblem
    capacity_right = []  # Knapsack Capacity on the right subproblem

    greedy_left = []  # Greedy Algorithm on the left subproblem
    greedy_right = []  # Greedy Allgorithm on the right subproblem

    EF_left = []  # Eligible-First left subproblem
    EF_right = []  # Eligible-First right subproblem
    LP_left = []  # Linear Programming left subproblem
    LP_right = []  # Linear Programming right subproblem

    for k_idx in range(49, limit, 20):  # Main Loop starting from 49 to the limit in increments of 20
        k = k_idx  # Defining the Knapsack Capacity
        capacity.append(k)
        b = 1 + 1 / k  # Defining a cumbsersome base

        ##############################################################################
        ##    SPLIT ITEM RANDOM VARIABLE
        ##############################################################################
        ##
        s = (1 + 1 / k) ** k  # Computing the split item
        split_item.append(s)

        var_split = (3 + 1 / k) * b ** (k - 1) - b ** (2 * k)  # Computing the variance of the split item
        variance_split.append(var_split)
        std_split.append(np.sqrt(var_split))  # Storing the standard deviation of the split item

        ##############################################################################
        ##    SLACK RANDOM VARIABLE
        ##############################################################################
        ##
        """" Equation (29c)"""
        slack_closed = - (b ** (k) - 1) * b
        slack_closed = slack_closed + (b ** (k + 1) - 1 - b) * (k + 3)
        slack_closed = slack_closed - 2 * k * (b ** (k + 2)
                                               - (5 * k ** 2 + 7 * k + 2) / (2 * k ** 2))

        slack.append(slack_closed)  # Storing the closed formula of the Slack
        rel_slack.append(100 * slack_closed / k)  # Storing the relative slack

        ##############################################################################
        ##    GREEDY SOLUTION
        ##############################################################################
        ##
        greedy_closed = - (k + 1) ** 2 * b ** (k - 1) / (4 * k)
        greedy_closed = greedy_closed + (2 * k + 3) * (k + 2) * (k + 1) * (b ** k - 1) / (4 * k)
        greedy_closed = greedy_closed - (k + 2) ** 2 * (b ** (k + 1) - 1 - b)
        greedy_closed = greedy_closed + k * (2 * k + 5) * (
                    b ** (k + 2) - 1 - (k + 2) / k - (k + 2) * (k + 1) / (2 * k ** 2)) / 2

        greedy_solution.append(greedy_closed)  # Storing the closef formula of the Greedy Solution

        ##############################################################################
        ##    EXTRA PROFIT FROM SLACK AND LINEAR PROGRAM RELAXATION
        ##############################################################################
        ##
        slack_profit = (k + 1) * b ** (k - 1) / (2 * k)
        slack_profit = slack_profit - (k + 2) * (k + 1) * (b ** k - 1) / (k)
        slack_profit = slack_profit + (k + 2) * (k + 5) * (b ** (k + 1) - 1 - b) / 2
        slack_profit = slack_profit - k * (k + 3) * (b ** (k + 2) - 1 - (k + 2) / k - (k + 2) * (k + 1) / (
                    2 * k ** 2))  # Extra Slack for Linear Programming Relaxation

        lp_closed = greedy_closed + slack_profit  # Computing the epected Linear Programming Solution
        lp_solution.append(lp_closed)  # Storing the Linear Programming Solution

        lp_approx = greedy_closed + slack_closed * (k - s + 1) / 2
        lp_list.append(
            greedy_closed + slack_closed * (k - s + 1) / 2)  # Approximating Linear Programming Assuming Independence

        approx_error.append(100 * (lp_closed - lp_approx) / lp_closed)

        greedy_2_lp.append(100 * greedy_closed / lp_closed)  # Computing ratio Greedy vs Linear Programming Relaxation

        ##
        ##############################################################################
        ##    ELIGIBLE-FIRST ANALYSIS
        ##############################################################################
        ##
        kp = k_idx  # Knapsack Capacity
        mu = kp + 1  # Quantity of Items
        k_0 = slack_closed  # Known average slack capacity
        s_0 = (1 + 1 / kp) ** kp  # Average split item index
        base_ef = 1 - k_0 / kp  # Cumbersom term, probability of success eligible-first post-greedy
        ef_closed = ((1 - base_ef ** (kp - s_0 + 1)) * (kp - s_0 + 1) * k_0 / 4)
        ef_closed = ef_closed - ((1 -
                                  (1 + (kp - s_0) * k_0 / kp) * base_ef ** (kp - s_0)
                                  ) * base_ef * kp / (4 * k_0))
        ef_closed = ef_closed + greedy_closed

        closed_eF.append(ef_closed)  # Storing the eligible-first result

        ##
        ##############################################################################
        ##    LEFT AND RIGHT CAPACITIES
        ##############################################################################
        ##

        deviation = mu * ((1 + 1 / kp) ** mu + (1 - 1 / kp) ** mu) / 4 - kp * (
                    (1 + 1 / kp) ** (mu + 1) - (1 - 1 / kp) ** (mu + 1)) / 4  # Deviation of the Knapsack capacities

        sum_ceil = 0
        sum_floor = 0
        sum_deviation = 0
        for k in range(0, mu, 2):
            sum_deviation = sum_deviation + k * (1 + 1 / kp) ** (k - 1)  # Common sum for both terms

        cap_left_closed = kp / 2 + sum_deviation / (2 * kp ** 2) + deviation
        cap_right_closed = kp / 2 - sum_deviation / (2 * kp ** 2) - deviation

        capacity_left.append(cap_left_closed)  # Storing the Left Capacity
        capacity_right.append(cap_right_closed)  # Storing the Right Capacity

        ##
        ##############################################################################
        ##    GREEDY LEFT AND RIGHT SOLUTIONS
        ##############################################################################
        ##
        closed_ZG_left = 0  # Computation of Greedy Solution Left
        c = cap_left_closed
        b = 1 + 1 / c
        closed_ZG_left = closed_ZG_left - c * b ** (c + 1) / 2
        closed_ZG_left = closed_ZG_left - (mu + 3) * (c + 2) * (b ** (c + 1) - b) / 2
        closed_ZG_left = closed_ZG_left + (2 * mu * c + 6 * c + 3 * mu + 10) / 2
        closed_ZG_left = closed_ZG_left + c * (mu + 3) * (
                    b ** (c + 2) - 1 - (c + 2) / c - (c + 1) * (c + 2) / (2 * c ** 2))

        greedy_left.append(closed_ZG_left)  # Storing of Greedy Solution Left

        closed_ZG_right = 0  # Computation of Greedy Solution Right
        c = cap_right_closed
        b = 1 + 1 / c
        closed_ZG_right = closed_ZG_right - c * b ** (c + 1) / 2
        closed_ZG_right = closed_ZG_right - (mu + 2) * (c + 2) * (b ** (c + 1) - b) / 2
        closed_ZG_right = closed_ZG_right + (2 * mu * c + 4 * c + 3 * mu + 7) / 2
        closed_ZG_right = closed_ZG_right + c * (mu + 2) * (
                    b ** (c + 2) - 1 - (c + 2) / c - (c + 1) * (c + 2) / (2 * c ** 2))

        greedy_right.append(closed_ZG_right)  # Storing of Greedy Solution Right

        ##
        ##############################################################################
        ##    LEFT AND RIGHT SPLIT AND SLACK
        ##############################################################################
        ##
        (s_left, k_left) = split_slack(cap_left_closed)  # Computing through subroutine "split_slack"
        (s_right, k_right) = split_slack(cap_right_closed)

        ##
        ##############################################################################
        ##    LEFT AND RIGHT LINEAR RELAXATION
        ##############################################################################
        ##
        lp_left = closed_ZG_left + k_left * (mu - s_left + 1) / 2
        LP_left.append(lp_left)  # Storing the Linear Relaxation solution for left subproblem
        lp_right = closed_ZG_right + k_right * (mu - s_right + 1) / 2
        LP_right.append(lp_right)  # Storing the Linear Relaxation solution for right subproblem

        ##
        ##############################################################################
        ##    LEFT AND RIGHT ELIGIBLE-FIRST
        ##############################################################################
        ##
        lamb = mu / 2  # Number of Items in each subproblem
        k_0 = k_left  # Updating the split and slack values for the left problem
        s_0 = s_left
        kp = cap_left_closed
        closed_EF_left = k_0 * (kp - 2 * s_0) * (1 - (1 - k_0 / kp) ** (lamb - s_0)) / 4
        closed_EF_left = closed_EF_left - kp * (1 - k_0 / kp) * (
                    1 - (1 + (kp - s_0 - 1) * k_0 / kp) * (1 - k_0 / kp) ** (lamb - s_0 + 1)) / (4 * k_0)
        closed_EF_left = closed_EF_left + closed_ZG_left  # Computing the eligible-first solution for the left subproblem

        EF_left.append(closed_EF_left)  # Storing the eligible-first solution for the left subproblem

        k_0 = k_right  # Updating the split and slack values for the right problem
        s_0 = s_right
        kp = cap_right_closed
        closed_EF_right = k_0 * (kp - 2 * s_0 - 1) * (1 - (1 - k_0 / kp) ** (lamb - s_0)) / 4
        closed_EF_right = closed_EF_right - kp * (1 - k_0 / kp) * (
                    1 - (1 + (kp - s_0 - 1) * k_0 / kp) * (1 - k_0 / kp) ** (lamb - s_0 + 1)) / (4 * k_0)
        closed_EF_right = closed_EF_right + closed_ZG_right  # Computing the eligible-first solution for the right subproblem

        EF_right.append(closed_EF_right)  # Storing the eligible-first solution for the right subproblem
    ##
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









