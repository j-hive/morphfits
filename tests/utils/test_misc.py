import pytest
from morphfits.utils import misc


def test_get_unique_batch_limits():
    """Testing the Unique Batch Limit Function"""

    # Testing Base Case

    min_i, max_i = misc.get_unique_batch_limits(0, 1, 5)

    assert min_i == 0
    assert max_i == 5

    # Testing Out of Range

    with pytest.raises(ValueError):
        min_i, max_i = misc.get_unique_batch_limits(5, 1, 5)

    # Testing No Remainder Case:

    min_i_0, max_i_0 = misc.get_unique_batch_limits(0, 2, 4)
    min_i_1, max_i_1 = misc.get_unique_batch_limits(1, 2, 4)

    assert (max_i_1 - min_i_1) == (max_i_0 - min_i_0)

    # Testing the Maximum Remainder Case
    min_i_0, max_i_0 = misc.get_unique_batch_limits(0, 4, 7)
    min_i_1, max_i_1 = misc.get_unique_batch_limits(1, 4, 7)
    min_i_2, max_i_2 = misc.get_unique_batch_limits(2, 4, 7)
    min_i_3, max_i_3 = misc.get_unique_batch_limits(3, 4, 7)

    assert (max_i_0 - min_i_0) == (max_i_3 - min_i_3) + 1
    assert max_i_0 == min_i_1
    assert max_i_1 == min_i_2
    assert max_i_2 == min_i_3
