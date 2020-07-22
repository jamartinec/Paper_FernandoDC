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
        slack_closed: float.

    """
    b = 1 + 1 / k  # Defining a cumbersome base
    slack_closed = - (b ** (k) - 1) * b
    slack_closed = slack_closed + (b ** (k + 1) - 1 - b) * (k + 3)
    slack_closed = slack_closed - 2 * k * (b ** (k + 2)
                                           - (5 * k ** 2 + 7 * k + 2) / (2 * k ** 2))  #closed formula of the Slack
    rel_slack_closed = 100 * slack_closed / k  # relative slack
    return slack_closed, rel_slack_closed

def func_greedy(k):
    """

    Args:
        k:

    Returns:

    """
    b = 1 + 1 / k  # Defining a cumbersome base
    greedy_closed = - (k + 1) ** 2 * b ** (k - 1) / (4 * k)
    greedy_closed = greedy_closed + (2 * k + 3) * (k + 2) * (k + 1) * (b ** k - 1) / (4 * k)
    greedy_closed = greedy_closed - (k + 2) ** 2 * (b ** (k + 1) - 1 - b)
    greedy_closed = greedy_closed + k * (2 * k + 5) * (
            b ** (k + 2) - 1 - (k + 2) / k - (k + 2) * (k + 1) / (2 * k ** 2)) / 2

    return greedy_closed




