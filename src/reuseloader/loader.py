"""저장된 표본을 다시 쓸지 새로 만들지를 정하는 Dataset과, 그것을 감싼 DataLoader."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .schedule import ReuseKey, ReusePolicy


def _load(path: Path) -> Any:
    # 이 라이브러리가 직접 쓴 파일만 읽는다. 임의 객체를 담을 수 있어야 하므로 pickle을 허용한다.
    return torch.load(path, weights_only=False)


class ReuseDataset(Dataset):
    """index마다 저장된 표본을 읽을지, 새로 만들어 덮어쓸지 정하는 Dataset.

    지금 epoch에 필요한 세대를 ``policy.key(index, epoch)``로 구한다. 저장된 파일의 세대가 같으면
    읽고, 다르거나 없으면 ``build(key)``로 만들어 같은 파일에 덮어쓴다. 파일은 index당 하나다.

    ``build(key)``의 결과는 key의 값(``seed``, ``first_epoch`` 등)만으로 정해져야 한다.
    그래야 어느 worker가 언제 만들어도, 재개 후 다시 만들어도 같은 표본이 된다.

    저장된 파일은 세대뿐 아니라 표본 seed, ``lifetime``, ``fingerprint``까지 같아야 다시 쓴다.
    seed나 수명을 바꾸면 예전 파일은 무시하고 새로 만든다. ``build``의 동작을 바꿨을 때는
    ``fingerprint``를 바꿔 예전 파일을 무효로 만든다. 한 디렉터리는 한 loader만 쓴다.

    epoch은 공유 메모리에 두어 worker 프로세스도 같은 값을 본다. ``ReuseDataLoader``를 쓰면
    한 바퀴 돌 때마다 자동으로 올라가고, 일반 ``DataLoader``와 쓸 때는 ``set_epoch``을 부른다.
    """

    def __init__(
        self,
        build: Callable[[ReuseKey], Any],
        size: int,
        directory: Path,
        policy: ReusePolicy,
        *,
        fingerprint: str = "",
        save: Callable[[Any, Path], None] = torch.save,
        load: Callable[[Path], Any] = _load,
    ) -> None:
        if size <= 0:
            raise ValueError("size는 양수여야 합니다")
        self.build = build
        self.size = int(size)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # 이전 실행이 쓰다가 중단돼 남긴 임시 파일을 지운다.
        for leftover in self.directory.glob("*.tmp"):
            leftover.unlink(missing_ok=True)
        self.policy = policy
        self.fingerprint = str(fingerprint)
        self.save = save
        self.load = load
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return self.size

    @property
    def epoch(self) -> int:
        return int(self._epoch)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch은 0 이상이어야 합니다")
        self._epoch.fill_(int(epoch))

    def key(self, index: int) -> ReuseKey:
        """지금 epoch에 index가 쓰는 표본."""
        return self.policy.key(index, self.epoch)

    def path(self, index: int) -> Path:
        return self.directory / f"{index:08d}.pt"

    def _identity(self, key: ReuseKey) -> dict[str, Any]:
        """저장된 파일이 지금 필요한 표본인지 가리는 값."""
        return {
            "generation": key.generation,
            "seed": key.seed,
            "lifetime": self.policy.lifetime,
            "fingerprint": self.fingerprint,
        }

    def _read(self, path: Path, key: ReuseKey) -> tuple[bool, Any]:
        """저장된 표본이 지금 필요한 것이면 ``(True, 표본)``, 아니면 ``(False, None)``."""
        if not path.exists():
            return False, None
        try:
            stored = self.load(path)
            if stored["identity"] == self._identity(key):
                return True, stored["item"]
        except Exception:  # noqa: BLE001
            # 손상돼 읽을 수 없는 경우다. 같은 key로 다시 만들면 결과가 같다.
            # 저장 형식을 바꿀 수 있어 읽기 오류의 종류를 미리 좁힐 수 없다.
            return False, None
        return False, None

    def __getitem__(self, index: int) -> Any:
        if not 0 <= index < self.size:
            raise IndexError(index)
        key = self.key(index)
        path = self.path(index)
        found, item = self._read(path, key)
        if found:
            return item
        item = self.build(key)
        if not key.is_last_use:  # 다시 쓰일 표본만 저장한다
            # 임시 파일에 쓰고 이름을 바꾼다. 읽는 쪽은 쓰다 만 파일을 보지 않는다.
            temporary = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
            self.save({"identity": self._identity(key), "item": item}, temporary)
            try:
                os.replace(temporary, path)
            except OSError:
                # 다른 프로세스가 대상 파일을 열고 있으면 바꿀 수 없는 OS가 있다(Windows).
                # 표본은 이미 만들었으니 저장만 건너뛴다. 다음에 필요할 때 다시 만들어도 결과가 같다.
                temporary.unlink(missing_ok=True)
        return item


class ReuseSampler(Sampler[int]):
    """epoch마다 seed로 섞은 순서를 닫힌 식으로 내놓는 sampler.

    ``policy.order(position, epoch, size)``로 어떤 epoch의 몇 번째가 어떤 index인지 바로 구할 수 있다.
    """

    def __init__(self, dataset: ReuseDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        epoch, size, policy = self.dataset.epoch, len(self.dataset), self.dataset.policy
        for position in range(size):
            yield policy.order(position, epoch, size)


class ReuseDataLoader(DataLoader):
    """표본을 ``lifetime`` epoch 동안 재사용하는 DataLoader.

    재사용 판단만 맡고 나머지는 표준 ``DataLoader`` 그대로다. ``batch_size``, ``sampler``,
    ``num_workers``, ``prefetch_factor``, ``collate_fn`` 등 모든 옵션을 그대로 쓸 수 있다.
    한 바퀴 돌 때마다 epoch이 1 올라간다. 재개할 때는 ``start_epoch``을 준다.

    ``shuffle=True``이고 sampler를 주지 않으면 ``ReuseSampler``를 쓴다. 순서가 epoch과 seed에서
    닫힌 식으로 정해져 추적할 수 있다. 다른 순서를 원하면 ``sampler``를 직접 준다.
    """

    def __init__(
        self,
        build: Callable[[ReuseKey], Any],
        size: int,
        directory: Path,
        *,
        lifetime: int,
        seed: int = 0,
        start_epoch: int = 0,
        shuffle: bool = False,
        fingerprint: str = "",
        save: Callable[[Any, Path], None] = torch.save,
        load: Callable[[Path], Any] = _load,
        **dataloader_options: Any,
    ) -> None:
        if start_epoch < 0:
            raise ValueError("start_epoch은 0 이상이어야 합니다")
        dataset = ReuseDataset(
            build,
            size,
            directory,
            ReusePolicy(lifetime, seed),
            fingerprint=fingerprint,
            save=save,
            load=load,
        )
        dataset.set_epoch(start_epoch)
        own_order = (
            shuffle
            and dataloader_options.get("sampler") is None
            and dataloader_options.get("batch_sampler") is None
        )
        if own_order:
            dataloader_options["sampler"] = ReuseSampler(dataset)
            shuffle = False
        super().__init__(dataset, shuffle=shuffle, **dataloader_options)
        self.epoch = int(start_epoch)  # 다음에 돌 epoch

    @property
    def policy(self) -> ReusePolicy:
        return self.dataset.policy

    def __iter__(self):
        # worker가 보는 epoch을 먼저 맞춘 뒤 표준 반복을 시작한다.
        self.dataset.set_epoch(self.epoch)
        iterator = super().__iter__()
        self.epoch += 1
        return iterator
