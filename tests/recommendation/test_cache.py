import pytest
from minisgl.recommendation.cache import PrefixPool


def test_shared_prefix_stays_alive_until_last_reader_releases():
    pool = PrefixPool(5)
    slots = pool.allocate(3)
    pins, adopted = pool.insert([1, 2, 3], slots)
    reader = pool.match([1, 2], pin=True)
    pool.release(pins)
    pool.free_owned(slots, adopted)
    assert pool.allocate(3) == [3, 4, 2]  # only the unpinned leaf can be evicted
    assert [n.slot for n in reader] == [0, 1]
    with pytest.raises(MemoryError):
        pool.allocate(1)
    pool.release(reader)
    assert len(pool.allocate(2)) == 2


def test_duplicate_prefill_frees_its_own_copies_once():
    pool = PrefixPool(8)
    first, second = pool.allocate(3), pool.allocate(3)
    pins1, adopted1 = pool.insert([1, 2, 3], first)
    pins2, adopted2 = pool.insert([1, 2, 3], second)
    assert adopted2 == set()
    pool.release(pins1)
    pool.free_owned(first, adopted1)
    pool.release(pins2)
    pool.free_owned(second, adopted2)
    assert len(pool.free) == 5
    assert len(set(pool.free)) == 5
    assert pool.available == 8
