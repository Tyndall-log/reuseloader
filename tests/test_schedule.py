"""재사용 규칙의 성질 검사."""

from collections import Counter

import pytest

from reuseloader import ReusePolicy, permute


@pytest.mark.parametrize("size", [1, 2, 3, 12, 100, 1200])
def test_permute_is_a_permutation(size: int) -> None:
    for key in (0, 12345):
        assert sorted(permute(index, size, key) for index in range(size)) == list(range(size))


def test_permute_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError):
        permute(5, 5, 0)


def test_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ReusePolicy(lifetime=0)
    with pytest.raises(ValueError):
        ReusePolicy(lifetime=2).key(-1, 0)
    with pytest.raises(ValueError):
        ReusePolicy(lifetime=2).key(0, -1)


def test_lifetime_two_alternates_by_phase() -> None:
    policy = ReusePolicy(lifetime=2, seed=1)
    assert [policy.key(0, epoch).generation for epoch in range(5)] == [0, 0, 1, 1, 2]
    assert [policy.key(1, epoch).generation for epoch in range(5)] == [0, 1, 1, 2, 2]


@pytest.mark.parametrize("lifetime", [1, 2, 3, 5])
def test_lifetimes_are_staggered_and_bounded(lifetime: int) -> None:
    policy = ReusePolicy(lifetime=lifetime, seed=3)
    size, epochs = 4 * lifetime, 4 * lifetime
    uses: Counter[tuple[int, int]] = Counter()
    seeds: dict[tuple[int, int], int] = {}
    for epoch in range(epochs):
        new = 0
        for index in range(size):
            key = policy.key(index, epoch)
            identity = (index, key.generation)
            uses[identity] += 1
            assert seeds.setdefault(identity, key.seed) == key.seed
            assert key.first_epoch <= epoch <= key.last_epoch
            assert key.use_index == epoch - (key.generation * lifetime - index % lifetime)
            assert key.is_last_use == (epoch == key.last_epoch)
            assert 0 <= key.seed < 2**63
            new += key.first_epoch == epoch
        # 0 epoch은 전부 만들고, 이후에는 epoch마다 1 / lifetime만 새로 만든다.
        assert new == (size if epoch == 0 else size // lifetime)
    assert len(set(seeds.values())) == len(seeds)  # 표본마다 seed가 다르다
    # 0세대의 수명은 lifetime - 위상으로 1~lifetime에 흩어진다.
    first = Counter(uses[(index, 0)] for index in range(size))
    assert first == Counter({value: 4 for value in range(1, lifetime + 1)})
    # 끝까지 쓰인 이후 세대는 정확히 lifetime번 쓰인다.
    for (index, generation), count in uses.items():
        if generation > 0 and policy.key(index, epochs - 1).generation > generation:
            assert count == lifetime


def test_order_is_a_permutation_that_changes_with_epoch_and_seed() -> None:
    policy = ReusePolicy(lifetime=2, seed=7)
    orders = [[policy.order(position, epoch, 24) for position in range(24)] for epoch in range(4)]
    for order in orders:
        assert sorted(order) == list(range(24))
    assert len({tuple(order) for order in orders}) == 4
    assert ReusePolicy(2, 7).order(5, 3, 24) == policy.order(5, 3, 24)
    other = ReusePolicy(lifetime=2, seed=8)
    assert [other.order(position, 0, 24) for position in range(24)] != orders[0]


def test_values_are_stable_across_releases() -> None:
    # 규칙은 재현성의 근거다. 식이 바뀌면 이 값이 달라져 알아챌 수 있다.
    policy = ReusePolicy(lifetime=2, seed=42)
    key = policy.key(356, 3)
    assert (key.generation, key.use_index, key.first_epoch, key.last_epoch) == (1, 1, 2, 3)
    assert key.seed == 2589139544276501644
    assert [policy.order(position, 3, 1200) for position in range(4)] == [694, 465, 728, 340]
