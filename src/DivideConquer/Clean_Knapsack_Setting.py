#!/usr/bin/env python3.7

'''
CODE FOR THE GENERATION OF THE KNAPSACK PROBLEM INSTANCES

Input:
- Capacity of the Knapsack. Introduced manually by the user in -k
- Number of random realizations. Introduce manually by the user in -n

Output:
- Available_Weights.csv, spreadsheet containing a column per problem realization. Each column is a list of weights.
- Available_Profits.csv, spreadsheet containing a column per problem realization. Each column is a list of profits.

'''

import os, argparse
import numpy as np
from numpy import linalg as LA
import math
import xlrd
import xlwt
import statistics
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Exploration for Stats-Omega')
    parser.add_argument('-k', '--k', help='Knapsack capacity', type=int, default=12)
    parser.add_argument('-n', '--n', help='Number of instances to be generated', type=int, default=5)
    args = parser.parse_args()
    knapsack_capacity = int(args.k)
    instances = int(args.n)

    #############################################
    ## Dictionary of Probabilistic Distribution
    #############################################
    ##

    Available_Weighs = pd.DataFrame(index=range(knapsack_capacity + 1))
    Available_Profits = pd.DataFrame(index=range(knapsack_capacity + 1))
    Available_Efficiencies = pd.DataFrame(index=range(knapsack_capacity + 1))

    for trial_idx in range(instances):  # Loop a desired number of realizations
        item_weights = list(np.random.randint(low=1, high=knapsack_capacity + 1,
                                              size=knapsack_capacity + 1))  # Random Realization of Weights U ~ [0, V + 1]
        increments = np.random.rand(knapsack_capacity + 1)  # Rando Realization of Increments U ~ [0, 1]

        efficiencies = []  # Creating the list of efficiencies and the list of profits
        profits = []

        cum_sum = np.sum(increments)  # Total sum of increments
        for row_idx in range(len(increments)):
            efficiencies.append(cum_sum)  # Storing the efficiency coefficient
            profits.append(cum_sum * item_weights[row_idx])  # Computing and storing the profits
            cum_sum = cum_sum - increments[row_idx]  # Updating the value of the efficiency

        Available_Weighs = Available_Weighs.join(pd.Series(item_weights,  # Pasting the weights realization
                                                           name=str(trial_idx)))  # in the global Data Frame
        Available_Efficiencies = Available_Efficiencies.join(
            pd.Series(efficiencies,  # Pasting the efficiencies realization
                      name=str(trial_idx)))  # in the global Data Frame
        Available_Profits = Available_Profits.join(pd.Series(profits,  # Pasting the profits realization
                                                             name=str(trial_idx)))  # in the global Data Frame

    Available_Weighs['Item'] = Available_Weighs.index + 1
    Available_Weighs.to_csv('Available_Weights.csv', index=False)  # Exporting the Weights Data Frame

    Available_Profits['Item'] = Available_Profits.index + 1
    Available_Profits.to_csv('Available_Profits.csv', index=False)  # Exporting the Profits Data Frame

    #####################################################################################
    ##  DISPLAYING DATA GENERATION RESULTS FOR USER'S CONTROL
    #####################################################################################
    ##  print(Available_Weighs)
    ##  print(Available_Profits)
    ##  print(Available_Efficiencies)

    #####################################################################################
    ##  GENERATION OF GRAPHICS, MULTIPLE GRAPHS EXPORTED
    #####################################################################################
    ##
    #####################################################################################
    ##  WEIGHTS REALIZATIONS GRAPH
    ##
    ##  Available_Weighs['Item'] = Available_Weighs.index + 1
    ##  Available_Weighs.plot(x = 'Item', kind = 'bar', legend = False)
    ##  plt.savefig('Weights_Realization.pdf')
    ##  plt.savefig('Weights_Realization.png')
    ##  plt.savefig('Weights_Realization.eps')
    ##  plt.show()
    ##
    #####################################################################################
    ##  EFFICIENCIES REALIZATIONS GRAPH
    ##
    ##  Available_Efficiencies['Item'] = Available_Efficiencies.index + 1
    ##  Available_Efficiencies.plot(x = 'Item', kind = 'bar', legend = False)
    ##  plt.savefig('Efficiencies_Realization.pdf')
    ##  plt.savefig('Efficiencies_Realization.png')
    ##  plt.savefig('Efficiencies_Realization.eps')
    ##  plt.show()
    ##
    #####################################################################################
    ##  PROFITS REALIZATIONS GRAPH
    ##
    ##  Available_Profits['Item'] = Available_Profits.index + 1
    ##  Available_Profits.plot(x = 'Item', kind = 'bar', legend = False)
    ##  plt.savefig('Profits_Realization.pdf')
    ##  plt.savefig('Profits_Realization.png')
    ##  plt.savefig('Profits_Realization.eps')
    ##  plt.show()
    ##
    ##  Realizations_Table = Available_Weighs.merge(Available_Profits, on = 'Item' )
    ##  Realizations_Table = Realizations_Table.merge(Available_Efficiencies, on = 'Item')
    ##  print(Realizations_Table)
    ##  Realizations_Table.round(decimals = 2).to_csv('Realizations_Table.csv', index = False)

    print('tiempo de procesado', time.process_time())
