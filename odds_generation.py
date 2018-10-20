""" Simple generation odds in Python, based on weighted median of various bookies odds with various weights
Features: 
  - ability to configure through inheritance by changing the dictionaries with settings if an interface is provided
  - Offer expiration time
  - Bookie weights and minimum weights
  - Decrease the number of offer updates based on odds threshold

TODO:
 - Add interface for customization settings
 - Store customizations in redis and cache locally customizations for data handled in this process
 - Read data from an input stream (or message queue)
 - Post data to an output stream (or message queue)
 - Create a process in front for sharding in order to use several python processes and compute odds in parallel
 - Optimize with numba
 - use -1 instead of None in offers together with vectorized numpy / numba operations
"""

import statistics as stats
import datetime as dt

def now():
    """ shortcut function """
    return dt.datetime.now()

# for memory optimizations
class Offer:
    """ Memory optimized abstraction for an offer """
    __slots__ = ['value', 'timestamp']
    def __init__(self, value, timestamp=now()):
        self.value = value
        self.timestamp = timestamp

##### GLOBAL SETTINGS: these can be modified at runtime through a configuration interface

bookies = {
    'bet365' : 0,
    'williamhill' : 1,
    'digibet' : 2,
    'betano' : 3,
    'maxbet' : 4,
    'jetbull' : 5
}

# by default, all bookies have the same id
default_weights = [1] * len(bookies)

# the following group of options could be customized further
# but I kept the code simple with only globals
global_min_bookie_weight = 4 
global_offer_expiry_time = dt.timedelta(seconds=5)
global_offer_send_threshold = 0.1
global_payout = 0.99

# per-event bookie weights
event_settings = {
    'real-vs-liverpool' : [0, 0, 4, 6, 2, 0]
}

# per-market bookie weights
market_settings = {
    '1x2' : [0, 0, 4, 6, 2, 0]
}

# per discipline bookie weights
discipline_settings = {
    'football' : [0, 0, 4, 6, 2, 0]
}

# Map of values for each event / market
# This is the main object, which stores all offers from all providers 
discipline_odds = {
    'football' : {
        'real-vs-liverpool' : {
            '1x2': [ 
                    [Offer(2), Offer(2.1), Offer(2.4), None, Offer(2.4), None],  # 1 
                    [Offer(2), Offer(2.1), Offer(2.4), None, Offer(2.4), None],  # x 
                    [Offer(2), Offer(2.1), Offer(2.4), None, Offer(2.4), None]   # 2 
                ]
        }
    }
}

# The last offer that was sent for a specific market / discipline / event
# This is used to decrease the number of odds changes sent downstream
last_sent_offer = {}

# the function used to send the changes; in our case, it is simple print
send_func = print

def get_odds(bookie_weights, bookie_values):
    """ 
    Returns our own odds for an outcome; weighted median with bookie weights. 
    None if the bookie weights do not go over the minimum bookie weight.
    """

    lst = []
    total_weights = 0

    for i in range(0, len(bookie_weights)):
        if bookie_values[i] is not None:
            total_weights = total_weights + bookie_weights[i]
            lst.extend([bookie_values[i]] * bookie_weights[i])

    return stats.median(sorted(lst)) if lst and total_weights >= global_min_bookie_weight else None

def get_weights(event_id, market_id, discipline_id):
    """ 
    Returns the bookie weights for a specific event, market and discipline.
    Uses inheritance in case specific values are not set.

    This function should be optimized as such:
      - use redis as store for all settings
      - use local process as cache for settings of events processed by this process
      - instead of keeping as hierarchy, use a single key for all "event_id-market_id-discipline_id" so less lookups
      - check periodically redis for updates
    """
    
    weights = event_settings.get(event_id, 
        market_settings.get(market_id, 
        discipline_settings.get(discipline_id, 
        default_weights)))

    if len(weights) != len(bookies):
        raise Exception("Invalid setting")
    
    return weights
    
def get_market_odds(market_id, event_id, discipline_id):
    """ Returns the odds for a specific market in an event"""
    
    event_odds = discipline_odds[discipline_id]
    event = event_odds.get(event_id)
    
    if event is None:
        return [None] * len(bookies)
    
    market = event.get(market_id)
    
    if market is None:
        return [None] * len(bookies)
    
    return [ [odds.value if odds is not None and now() - odds.timestamp  < global_offer_expiry_time else None for odds in offers] for offers in market]
    

def normalize(offer, payout):
    """ 
    Normalizes an offer given a payout. 
    If the offer contains a None, returns the initial offer untouched. 

    TODO:
     - use -1 instead of None and use numpy.array and/or numba for vectorized operations
    """

    # if one is None, return the original offer

    for elem in offer:
        if elem is None:
            return offer

    # normalize
    probabilities=[1 / o for o in offer]
    sp=sum(probabilities)

    return [(sp * payout) / p for p in probabilities]


def get_offer(market_id, event_id, discipline_id):
    """Returns an offer for a specific market / event / discipline """

    bookie_weights = get_weights(event_id, market_id, discipline_id)

    return normalize([get_odds(bookie_weights, odds) for odds in get_market_odds(market_id, event_id, discipline_id)], global_payout)

def create_market(market_id, event_id, discipline_id, no_of_outcomes):
    """ Creates a new market initialized to None """

    event_odds = discipline_odds.setdefault(discipline_id, {})
    event_odds.setdefault(event_id, dict()).setdefault(market_id, [ [None] * len(bookies) for i in range(0, no_of_outcomes)] ) 


def offer_out_of_threshhold(offer_1, offer_2):
    """ Checks if a specific offer is out of threshold """

    for i in range(0, len(offer_1)):

        if offer_1[i] is None and offer_2[i] is None:
            continue

        if offer_1[i] is None and offer_2[i] is not None:
            return True

        if offer_2[i] is None and offer_1[i] is not None:
            return True

        if abs(offer_1[i] - offer_2[i]) >= global_offer_send_threshold:
            return True

    return False

def send_offer(market_id, event_id, discipline_id):
    """ Decides whether to send an offer and sends it """

    key = '-'.join([market_id, event_id, discipline_id])
    offer = get_offer(market_id, event_id, discipline_id)

    last_offer = last_sent_offer.get(key)

    if last_offer is None and not [l for l in offer if l is not None]:
        return
    
    if last_offer is None or offer_out_of_threshhold(last_offer, offer):
        last_sent_offer[key] = offer
        send_func(offer) # actual send


def set_outcome(bookie_name, discipline_id, event_id, market_id, outcome_id, odds):
    """ sets an outcome to a specifc value """

    event_odds = discipline_odds[discipline_id]

    try:
        bookie_id = bookies[bookie_name]
        event_odds[event_id][market_id][outcome_id][bookie_id] = Offer(odds, now())
        send_offer(market_id, event_id, discipline_id)
    except Exception as exx:
        print("Error setting outcome {}".format(exx))


def expire():
    """ parses the offer tree and expires offers which are out of threshold """

    for discipline_id, event_odds in discipline_odds.items():
        for event_id, market in event_odds.items():
            for market_id, outcomes in market.items():
                changed = False
                for outcome in outcomes:
                    for i in range(0, len(outcome)):
                        if outcome[i] is not None and now() - outcome[i].timestamp  > global_offer_expiry_time:
                            outcome[i] = None
                            changed = True

                if changed:
                    send_offer(market_id, event_id, discipline_id)


############## TEST-BED ##############

create_market('1-2', 'hello-world', 'football', 2)

set_outcome('bet365', 'football', 'hello-world', '1-2', 0, 2.5)     # doesn't send anything due to bookie weights
set_outcome('betano', 'football', 'hello-world', '1-2', 1, 1.2)     # sends first offer
set_outcome('betano', 'football', 'hello-world', '1-2', 1, 1.2)     # doesn't send anything due to not enough changes
set_outcome('betano', 'football', 'hello-world', '1-2', 1, 1.3)     # sends offer
set_outcome('betano', 'football', 'hello-world', '1-2', 1, 1.45)    # sends offer
set_outcome('betano', 'football', 'hello-world', '1-2', 0, 2.2)     # sends offer
set_outcome('betano', 'football', 'hello-world', '1-2', 0, 2.25)    # doesn't send due to not enough changes
set_outcome('betano', 'football', 'hello-world', '1-2', 0, 2.25)    # doesn't send due to not enough changes
set_outcome('betano', 'football', 'hello-world', '1-2', 0, 2.5)     # sends offer

send_offer('1x2', 'real-vs-liverpool', 'football')

expire() # should not generate updates

import time
time.sleep(6)

expire() # should generate updates
