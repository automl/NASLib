# this script calculates the amount of budget needed to create nice plots for Successhive Halving :D

from math import log
from math import ceil
from math import floor

max_budget = 128
eta = 2
number_archs = 128

def calc_sh_budget():
    i = 1
    budget = 0.0
    while i <= max_budget:
        print("fidelity: {}".format(i))
        budget = budget + (i / max_budget) * (number_archs / i)
        i = i * eta
    print(budget/2)

def calc_hb_budget():
    i = 1
    s = floor(log(max_budget, eta))

    n = ceil(int(b / max_budget / (s + 1)) * eta ** s)
    r = max_budget * eta ** (-s)
    # in a loop

def main():
    calc_hb_budget()

if __name__ == '__main__':
    main()
