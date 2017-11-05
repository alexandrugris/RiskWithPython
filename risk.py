import math
import random
import statistics as stats
import matplotlib.pyplot as plt


def B(alpha, beta):
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)


def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


def center(alpha, beta):
    return alpha / (alpha + beta)


def beta_pdf_array(alpha, beta, count):
    x_ = [x for x in range(0, count)]
    return x_, [beta_pdf(x / count, alpha, beta) for x in x_]


# x, y = beta_pdf_array(1, 1, 100)
# plt.plot(x, y, color="red")

# x, y = beta_pdf_array(1.2, 1.2, 100)
# plt.plot(x, y, color="blue")

# x, y = beta_pdf_array(10, 10, 100)
# plt.plot(x, y, color="magenta")

# x, y = beta_pdf_array(7, 3, 100)
# plt.plot(x, y, color="green")

# plt.show()

# 1x2 market -> start with a probability of equal opportunity,

# payout = 1 / (1/away + 1/draw + 1/away)
# normalizing_factor = 1 / payout
# normalized_odds = [away * norm_factor, ...]
# probabilities = [1/normalized_odds, ...]

def payout(odds):
    ret = 1 / sum([1.0 / o for o in odds])
    assert (ret <= 1)
    return ret


def normalize_odds(odds) -> list:
    norm = 1 / payout(odds)
    return [o * norm for o in odds]


def probabilities_from_normalzed_odds(odds) -> list:
    return [1 / o for o in odds]


def prob_1x2(_home_odds, _draw_odds, _away_odds) -> list:
    norm_odds = normalize_odds([_home_odds, _draw_odds, _away_odds])
    return probabilities_from_normalzed_odds(norm_odds)


def normalize(lst) -> list:
    s = sum(lst)
    r = [l * 1 / s for l in lst]
    return r


def probabilities_to_odds(probs: list, _set_payout: float):
    probs = normalize([x if x > 0.1 else 0.1 for x in probs])
    return [_set_payout / p for p in probs]


def fmt_f(x: float) -> str:
    return "{:.2f}".format(x)


def fmt_list_f(l: list) -> str:
    return "[" + ", ".join([fmt_f(x) for x in l]) + "]"


# place bet functions
def place_bet_random(max_bet):
    """Places a random bet, with probabilities of placing the bet for each outcome
    hardcoded in the first line of the function"""

    bet_probabilities = normalize([0.5, 0.3, 0.3])

    for i in range(0, number_of_bets):
        money = random.random() * max_bet()

        # bet according to local preferences
        f = random.random()

        if f < bet_probabilities[0]:
            yield [money, 0, 0]
        elif f < bet_probabilities[0] + bet_probabilities[1]:
            yield [0, money, 0]
        else:
            yield [0, 0, money]


def place_bet_max_stake(max_bet):
    """Places a bet on the highest stake"""

    for i in range(0, number_of_bets):
        _bet = [0, 0, 0]
        ix = odds_evolution[-1].index(max(odds_evolution[-1]))
        _bet[ix] = max_bet()
        yield _bet


def place_bet_diverse(max_bet=lambda: 100.0):
    """ A combination of the strategies above, with a probability of 1/20 to place bet on the max stake"""

    pbms = place_bet_max_stake(max_bet)
    pbr = place_bet_random(max_bet)

    for i in range(0, number_of_bets):

        if random.randint(0, 20) == 0:
            yield pbms.__next__()
        else:
            yield pbr.__next__()


def plot_odds_evolution(_odds_evolution):
    h = [x[0] for x in _odds_evolution]
    d = [x[1] for x in _odds_evolution]
    a = [x[2] for x in _odds_evolution]
    plt.plot(range(0, len(h)), h, color="red")
    plt.plot(range(0, len(h)), d, color="green")
    plt.plot(range(0, len(h)), a, color="blue")
    plt.show()


# SIMULATION; repeat the experiment X times

table_results = []  # for analysis at the end

for _ in range(0, 100):

    # initial odds
    # place bets
    # next results

    # parameters
    total_market_risk = 1000  # monetary units
    total_deposits = 0

    home_odds = 2.5
    draw_odds = 3.5
    away_odds = 2.8

    confidence_factor = 1

    set_payout = payout([home_odds, draw_odds, away_odds])
    initial_probabilities = prob_1x2(home_odds, draw_odds, away_odds)

    # simulation
    payments_per_outcome = [0, 0, 0]
    number_of_bets = 10000

    probabilities = initial_probabilities
    alpha_beta = [(p, (total_market_risk * confidence_factor - p))
                  for p in [i * total_market_risk * confidence_factor for i in initial_probabilities]]

    odds_evolution = []

    # print initial offering:
    print(probabilities_to_odds(probabilities, set_payout))


    def accept_bet_risk(_bet: list, _probabilities: list):
        global payments_per_outcome
        global total_deposits
        global alpha_beta

        assert (0.999 < sum(_probabilities) < 1.001)
        assert (sum(_bet) == max(_bet) and min(_bet) == 0)  # just one is > 0

        total_deposits += max(_bet)

        payment_per_bet = [pto * b for pto, b in zip(probabilities_to_odds(_probabilities, set_payout), _bet)]
        total_payment_per_outcome = [ppo + ppb for ppo, ppb in zip(payments_per_outcome, payment_per_bet)]

        # because we talk about mutually exclusive events:
        exposure = total_deposits - max(total_payment_per_outcome)

        if exposure < -total_market_risk:
            print("Bet Not Accepted" + str(_bet))
            return  # do not update the payments
        elif exposure > 0:
            print("ooops, we are in minus")

        payments_per_outcome = total_payment_per_outcome
        return


    # main service
    odds_evolution.append(probabilities_to_odds(probabilities, set_payout))
    # for bet in place_bet_diverse(lambda: min((total_market_risk + total_deposits) * 1 / 100, 100)):
    for bet in place_bet_diverse():
        accept_bet_risk(bet, probabilities)

        # 0. do not adjust odds

        # 1. bayesian learning -> assumes players are smart and place bet according to best win * prob result
        alpha_beta = [(alpha + r, beta + max(bet) - r) for (alpha, beta), r in zip(alpha_beta, bet)]
        probabilities = normalize([center(alpha, beta) for alpha, beta in alpha_beta])

        # 2. look only at the final pool
        # probabilities = normalize(payments_per_outcome)

        # 3. increase artificially the odds on lower probability outcome to draw money there
        # diff = max(probabilities) - min(probabilities)
        # probabilities = normalize([p - diff if p - diff > 0.01 else 0.01 for p in probabilities])

        odds_evolution.append(probabilities_to_odds(probabilities, set_payout))

    ######

    result = lambda: None
    result.worst_case_profit = total_deposits - max(payments_per_outcome)
    result.profit_per_each_outcome = [total_deposits - x for x in payments_per_outcome]
    result.total_deposits = total_deposits
    result.percentage_profit = result.worst_case_profit * 100 / result.total_deposits
    result.final_odds = odds_evolution[-1]

    table_results.append(result)

    print("Worst case scenario profit: " + fmt_f(result.worst_case_profit))
    print("Profit per each outcome: " + fmt_list_f(result.profit_per_each_outcome))
    print("Total deposits: {:.1f} and profit: {:.2f}%".format(
        result.total_deposits,
        result.percentage_profit))

    print("Odds: " + fmt_list_f(result.final_odds))

    plot_odds_evolution(odds_evolution)


# end statistics

def print_stats(string: str, lst: list):
    print(string + ": mean: {:.2f}, stddev: {:.2f}".format(stats.mean(lst), stats.stdev(lst)))


print_stats("Worst case profit: ", [x.worst_case_profit for x in table_results])
print("Min profit % of total stakes {:.2f}: ".format(
    sum([x.worst_case_profit for x in table_results]) * 100 / sum(x.total_deposits for x in table_results)))

plt.hist([x.worst_case_profit for x in table_results], 11)
plt.show()
