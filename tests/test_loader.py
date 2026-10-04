"""재사용 Dataset과 DataLoader의 성질 검사."""

import uuid
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from reuseloader import ReuseDataLoader, ReuseDataset, ReuseKey, ReusePolicy


def _build(key: ReuseKey) -> dict:
    """seed에서만 정해지는 표본. 만든 횟수를 세려고 흔적 파일을 남길 수 있다."""
    generator = torch.Generator().manual_seed(key.seed)
    trace = _build.trace_directory
    if trace is not None:
        (trace / f"built-{key.index}-{key.generation}-{uuid.uuid4().hex}").touch()
    return {
        "index": key.index,
        "generation": key.generation,
        "value": torch.rand(3, generator=generator),
    }


_build.trace_directory = None


def _collate(items: list[dict]) -> dict:
    return {
        "ids": [(item["index"], item["generation"]) for item in items],
        "value": torch.stack([item["value"] for item in items]),
    }


def _run(loader: DataLoader, epochs: int) -> list[list[dict]]:
    return [list(loader) for _ in range(epochs)]


def _assert_same(expected: list[list[dict]], actual: list[list[dict]]) -> None:
    assert len(expected) == len(actual)
    for expected_epoch, actual_epoch in zip(expected, actual, strict=True):
        assert len(expected_epoch) == len(actual_epoch)
        for want, got in zip(expected_epoch, actual_epoch, strict=True):
            assert want["ids"] == got["ids"]
            assert torch.equal(want["value"], got["value"])


def test_builds_once_per_generation_and_reuses_from_disk(tmp_path: Path) -> None:
    store, trace = tmp_path / "store", tmp_path / "trace"
    trace.mkdir()
    _build.trace_directory = trace
    try:
        loader = ReuseDataLoader(
            _build, 8, store, lifetime=2, seed=11, batch_size=4, collate_fn=_collate
        )
        epochs = _run(loader, 4)
    finally:
        _build.trace_directory = None
    built = Counter(tuple(path.name.split("-")[1:3]) for path in trace.iterdir())
    assert set(built.values()) == {1}  # 표본마다 한 번만 만든다
    assert len(built) == 8 + 3 * 4  # 0 epoch 8개 + 이후 3개 epoch에 4개씩
    values: dict[tuple[int, int], torch.Tensor] = {}
    for epoch, batches in enumerate(epochs):
        for batch in batches:
            for (index, generation), value in zip(batch["ids"], batch["value"], strict=True):
                assert generation == loader.policy.key(index, epoch).generation
                if (index, generation) in values:
                    assert torch.equal(values[(index, generation)], value)  # 재사용 때 같은 내용
                values[(index, generation)] = value
    # 파일은 index당 하나를 덮어쓴다.
    assert len(list(store.iterdir())) <= 8
    assert loader.epoch == 4


def test_lifetime_one_never_writes(tmp_path: Path) -> None:
    loader = ReuseDataLoader(_build, 6, tmp_path, lifetime=1, batch_size=3, collate_fn=_collate)
    epochs = _run(loader, 3)
    assert list(tmp_path.iterdir()) == []
    assert [batch["ids"] for batch in epochs[2]] == [
        [(0, 2), (1, 2), (2, 2)],
        [(3, 2), (4, 2), (5, 2)],
    ]


def test_standard_dataloader_options_work(tmp_path: Path) -> None:
    loader = ReuseDataLoader(
        _build,
        10,
        tmp_path,
        lifetime=2,
        batch_size=4,
        drop_last=True,
        sampler=SequentialSampler(range(10)),
        collate_fn=_collate,
    )
    batches = list(loader)
    assert len(loader) == 2
    assert [[index for index, _ in batch["ids"]] for batch in batches] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_shuffle_follows_the_closed_form_order(tmp_path: Path) -> None:
    loader = ReuseDataLoader(
        _build, 12, tmp_path, lifetime=3, seed=9, shuffle=True, batch_size=4, collate_fn=_collate
    )
    for epoch, batches in enumerate(_run(loader, 3)):
        order = [index for batch in batches for index, _ in batch["ids"]]
        assert order == [loader.policy.order(position, epoch, 12) for position in range(12)]
        assert sorted(order) == list(range(12))


def test_resume_gives_the_same_batches(tmp_path: Path) -> None:
    options = {"lifetime": 2, "seed": 5, "shuffle": True, "batch_size": 4, "collate_fn": _collate}
    reference = _run(ReuseDataLoader(_build, 16, tmp_path / "a", **options), 5)
    # 디스크가 비어 있어도, 이전 실행의 파일이 남아 있어도 같은 결과다.
    fresh = _run(ReuseDataLoader(_build, 16, tmp_path / "b", start_epoch=2, **options), 3)
    reused = _run(ReuseDataLoader(_build, 16, tmp_path / "a", start_epoch=2, **options), 3)
    _assert_same(reference[2:], fresh)
    _assert_same(reference[2:], reused)


@pytest.mark.parametrize("persistent", [False, True])
def test_workers_give_the_same_batches(tmp_path: Path, persistent: bool) -> None:
    options = {"lifetime": 2, "seed": 5, "shuffle": True, "batch_size": 4, "collate_fn": _collate}
    reference = _run(ReuseDataLoader(_build, 16, tmp_path / "a", **options), 3)
    parallel = _run(
        ReuseDataLoader(
            _build,
            16,
            tmp_path / "b",
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=persistent,
            **options,
        ),
        3,
    )
    _assert_same(reference, parallel)


def test_unreadable_file_is_rebuilt(tmp_path: Path) -> None:
    options = {"lifetime": 2, "batch_size": 2, "collate_fn": _collate}
    first = _run(ReuseDataLoader(_build, 4, tmp_path, **options), 2)
    for path in tmp_path.glob("*.pt"):
        path.write_bytes(b"broken")
    again = _run(ReuseDataLoader(_build, 4, tmp_path, **options), 2)
    _assert_same(first, again)


def test_dataset_works_with_a_plain_dataloader(tmp_path: Path) -> None:
    dataset = ReuseDataset(_build, 4, tmp_path, ReusePolicy(lifetime=2, seed=1))
    loader = DataLoader(dataset, batch_size=2, collate_fn=_collate)
    generations = []
    for epoch in range(3):
        dataset.set_epoch(epoch)
        generations.append([generation for batch in loader for _, generation in batch["ids"]])
    assert generations == [[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]]
    with pytest.raises(IndexError):
        dataset[4]


def test_invalid_arguments_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ReuseDataLoader(_build, 0, tmp_path, lifetime=2)
    with pytest.raises(ValueError):
        ReuseDataLoader(_build, 4, tmp_path, lifetime=2, start_epoch=-1)


def _generations_built(trace: Path) -> Counter:
    return Counter(tuple(path.name.split("-")[1:3]) for path in trace.iterdir())


@pytest.mark.parametrize(
    "changed",
    [{"seed": 2}, {"lifetime": 3}, {"fingerprint": "v2"}],
    ids=["seed", "lifetime", "fingerprint"],
)
def test_changed_settings_do_not_reuse_old_files(tmp_path: Path, changed: dict) -> None:
    """seed, 수명, fingerprint가 바뀌면 같은 디렉터리의 예전 파일을 다시 쓰지 않는다."""
    store = tmp_path / "store"
    base = {"lifetime": 2, "seed": 1, "fingerprint": "v1", "batch_size": 3, "collate_fn": _collate}
    _run(ReuseDataLoader(_build, 6, store, **base), 1)
    assert len(list(store.glob("*.pt"))) == 3  # 위상 0인 index만 저장된다
    expected = _run(ReuseDataLoader(_build, 6, tmp_path / "clean", **{**base, **changed}), 2)
    trace = tmp_path / "trace"
    trace.mkdir()
    _build.trace_directory = trace
    try:
        actual = _run(ReuseDataLoader(_build, 6, store, **{**base, **changed}), 2)
    finally:
        _build.trace_directory = None
    _assert_same(expected, actual)
    # 0 epoch의 표본 6개를 모두 새로 만들었다(예전 파일을 읽지 않았다).
    assert sum(1 for index, generation in _generations_built(trace) if generation == "0") == 6


def test_same_settings_reuse_old_files(tmp_path: Path) -> None:
    options = {"lifetime": 2, "seed": 1, "batch_size": 3, "collate_fn": _collate}
    _run(ReuseDataLoader(_build, 6, tmp_path / "store", **options), 1)
    trace = tmp_path / "trace"
    trace.mkdir()
    _build.trace_directory = trace
    try:
        _run(ReuseDataLoader(_build, 6, tmp_path / "store", **options), 1)
    finally:
        _build.trace_directory = None
    # 저장돼 있던 위상 0의 3개는 읽고, 저장하지 않았던 위상 1의 3개만 다시 만든다.
    assert sorted(_generations_built(trace)) == [("1", "0"), ("3", "0"), ("5", "0")]


def test_failed_replace_still_returns_the_item(tmp_path: Path, monkeypatch) -> None:
    def refuse(source, destination):
        raise PermissionError("대상 파일이 열려 있음")

    monkeypatch.setattr("reuseloader.loader.os.replace", refuse)
    dataset = ReuseDataset(_build, 4, tmp_path, ReusePolicy(lifetime=2, seed=3))
    item = dataset[0]
    assert item["index"] == 0
    assert list(tmp_path.iterdir()) == []  # 임시 파일을 남기지 않는다
    monkeypatch.undo()
    assert torch.equal(dataset[0]["value"], item["value"])
    assert [path.name for path in tmp_path.iterdir()] == ["00000000.pt"]


def test_leftover_temporary_files_are_removed(tmp_path: Path) -> None:
    (tmp_path / "00000000.pt.123-abc.tmp").write_bytes(b"partial")
    (tmp_path / "notes.txt").write_text("keep")
    ReuseDataset(_build, 4, tmp_path, ReusePolicy(lifetime=2))
    assert [path.name for path in tmp_path.iterdir()] == ["notes.txt"]


def test_epoch_counts_each_started_pass(tmp_path: Path) -> None:
    loader = ReuseDataLoader(_build, 4, tmp_path, lifetime=2, batch_size=2, collate_fn=_collate)
    first = next(iter(loader))  # 도중에 멈춘 한 바퀴도 epoch으로 센다
    assert [generation for _, generation in first["ids"]] == [0, 0]
    second = list(loader)
    assert loader.epoch == 2
    assert [generation for batch in second for _, generation in batch["ids"]] == [0, 1, 0, 1]
