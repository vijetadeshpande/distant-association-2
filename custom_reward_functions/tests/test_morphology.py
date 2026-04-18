from custom_reward_functions.morphology import (
    clue_violates_morphology,
    is_morph_variant,
)


def test_exact_match():
    assert is_morph_variant("swim", "SWIM") is True


def test_progressive():
    assert is_morph_variant("SWIMMING", "swim") is True


def test_agentive():
    assert is_morph_variant("SWIMMER", "swim") is True


def test_past_tense_irregular_is_best_effort():
    # "SWAM" is irregular; our simple stemmer may miss it — current
    # behaviour is documented here.  Either outcome is acceptable; we
    # assert the function does not error.
    is_morph_variant("SWAM", "swim")


def test_containment():
    assert is_morph_variant("tired", "tire") is True


def test_unrelated():
    assert is_morph_variant("continent", "africa") is False


def test_short_substrings_rejected():
    # "or" inside "africa" shouldn't trip containment (shorter must be >=3).
    assert is_morph_variant("or", "africa") is False


def test_board_sweep():
    assert clue_violates_morphology("AFRICAN", ["africa"], ["pyrophobia"]) is True
    assert clue_violates_morphology("continent", ["africa"], ["pyrophobia"]) is False
