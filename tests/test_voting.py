import random

from workforce.protocol.voting import DEFAULT_CONFIDENCE, anonymous_vote


def rng():
    return random.Random(0)


def test_plurality_wins_without_tie_break():
    outcome = anonymous_vote(["a", "b", "a"], [10, 99, 10], rng())
    assert (outcome.winner, outcome.tie_break) == ("a", "none")
    assert outcome.counts == {"a": 2, "b": 1}


def test_tie_broken_by_summed_confidence():
    outcome = anonymous_vote(["a", "b", "b", "a", "c"], [60, 30, 40, 5, 99], rng())
    # a and b tie on two votes; a carries 65, b carries 70; c's single vote is not in the tie.
    assert (outcome.winner, outcome.tie_break) == ("b", "confidence")
    assert outcome.weights == {"a": 65, "b": 70}


def test_remaining_tie_is_a_seeded_random_draw():
    winners = {anonymous_vote(["x", "y"], [50, 50], random.Random(s)).winner for s in range(20)}
    assert winners == {"x", "y"}
    first = anonymous_vote(["x", "y"], [50, 50], random.Random(3))
    again = anonymous_vote(["x", "y"], [50, 50], random.Random(3))
    assert first.tie_break == "random" and first.winner == again.winner


def test_abstentions_do_not_vote_and_missing_confidence_is_neutral():
    outcome = anonymous_vote(["a", None, "b"], [None, 100, 49], rng())
    assert outcome.counts == {"a": 1, "b": 1}
    assert outcome.weights == {"a": DEFAULT_CONFIDENCE, "b": 49}
    assert outcome.winner == "a"


def test_no_votes():
    outcome = anonymous_vote([None, None], [80, 90], rng())
    assert outcome.winner is None and outcome.tie_break == "no_votes"


def test_vote_is_anonymous_order_does_not_matter():
    keys, confs = ["a", "b", "c", "b", "a"], [10, 20, 30, 40, 50]
    base = anonymous_vote(keys, confs, rng())
    for seed in range(5):
        order = list(range(5))
        random.Random(seed).shuffle(order)
        shuffled = anonymous_vote([keys[i] for i in order], [confs[i] for i in order], rng())
        assert shuffled.winner == base.winner
