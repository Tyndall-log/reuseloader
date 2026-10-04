"""(index, epoch)에서 재사용 여부를 닫힌 식으로 정하는 규칙. torch에 의존하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def _mix64(value: int) -> int:
    """splitmix64의 마무리 함수. 64비트 정수를 고르게 섞는다."""
    value = (value + _GOLDEN) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def mix(*values: int) -> int:
    """여러 정수를 순서대로 섞어 64비트 값 하나로 만든다. Python 해시와 달리 실행마다 같다."""
    state = 0
    for value in values:
        state = _mix64(state ^ (value & _MASK64))
    return state


def permute(index: int, size: int, key: int) -> int:
    """``[0, size)``의 순열에서 ``index``번째 값을 표 없이 구한다(원소당 O(1)).

    Feistel 구조는 2의 거듭제곱 범위의 전단사이고, ``size`` 밖으로 나온 값은 다시 넣는다.
    """
    if not 0 <= index < size:
        raise ValueError("index는 [0, size) 안에 있어야 합니다")
    if size == 1:
        return 0
    half = ((size - 1).bit_length() + 1) // 2
    mask = (1 << half) - 1
    value = index
    while True:
        left, right = value >> half, value & mask
        for round_index in range(4):
            left, right = right, left ^ (mix(key, round_index, right) & mask)
        value = (left << half) | right
        if value < size:
            return value


@dataclass(frozen=True, slots=True)
class ReuseKey:
    """한 index가 한 epoch에 쓰는 표본. ``build``에 넘긴다."""

    index: int
    generation: int  # 이 index에 차례로 들어가는 표본의 번호
    seed: int  # 이 표본을 만드는 seed. (규칙 seed, index, 세대)에서만 정해진다. 63비트
    epoch: int  # 이 사용이 속한 epoch
    first_epoch: int  # 이 표본이 처음 쓰이는 epoch
    last_epoch: int  # 마지막으로 쓰이는 epoch
    use_index: int  # 이 표본의 몇 번째 사용인지(0부터). 0세대는 0에서 시작하지 않을 수 있다

    @property
    def is_last_use(self) -> bool:
        return self.epoch == self.last_epoch


@dataclass(frozen=True, slots=True)
class ReusePolicy:
    """표본을 몇 epoch 동안 재사용할지 정하는 규칙. 값이 같으면 어디서 계산해도 결과가 같다.

    - 수명(``lifetime``): 표본 하나가 쓰이는 epoch 수.
    - 위상: index별 교체 시점. ``index % lifetime``. 0세대의 수명은 ``lifetime - 위상``이라
      index마다 1~lifetime으로 흩어지고, epoch마다 약 ``1 / lifetime``의 index만 교체된다.
    """

    lifetime: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lifetime <= 0:
            raise ValueError("lifetime은 양수여야 합니다")

    def key(self, index: int, epoch: int) -> ReuseKey:
        """주어진 epoch에 index가 쓰는 표본."""
        if index < 0 or epoch < 0:
            raise ValueError("index와 epoch은 0 이상이어야 합니다")
        phase = index % self.lifetime
        generation, use_index = divmod(epoch + phase, self.lifetime)
        return ReuseKey(
            index=index,
            generation=generation,
            seed=mix(self.seed, 1, index, generation) >> 1,
            epoch=epoch,
            first_epoch=max(0, generation * self.lifetime - phase),
            last_epoch=(generation + 1) * self.lifetime - phase - 1,
            use_index=use_index,
        )

    def order(self, position: int, epoch: int, size: int) -> int:
        """epoch마다 seed로 섞은 순서에서 ``position``번째로 쓰는 index."""
        return permute(position, size, mix(self.seed, 2, epoch))
