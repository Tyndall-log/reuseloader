# ReuseLoader

ReuseLoader는 입력 준비가 학습의 병목일 때, 준비한 표본을 **정해진 규칙에 따라 여러 epoch
동안 재사용**하게 해 주는 PyTorch DataLoader입니다.

하는 일은 하나입니다. index마다 **저장된 표본을 읽을지, 새로 만들어 덮어쓸지**를 정합니다.
그 밖의 것(batch 구성, 순서, worker, prefetch, collate)은 표준 `DataLoader` 그대로입니다.
판단은 `(index, epoch)`에서 닫힌 식으로 나오므로, worker 수나 준비 속도와 무관하게 입력을
그대로 재현할 수 있습니다.

## 주요 기능

- `(index, epoch)` → (세대, seed, 첫 사용 epoch, 사용 차수)를 O(1)로 계산하는 `ReusePolicy`
- 표본마다 정해진 수명(최대 사용 횟수)과, 한꺼번에 만료되지 않도록 엇갈린 교체
- 저장된 세대가 맞으면 읽고, 아니면 만들어 덮어쓰는 `ReuseDataset`
- 표준 `torch.utils.data.DataLoader`의 모든 옵션을 그대로 쓰는 `ReuseDataLoader`
- 순서까지 닫힌 식으로 추적할 수 있는 `ReuseSampler`
- 상태가 epoch 번호 하나뿐인 재개

## 설치

```bash
pip install reuseloader
```

Python 3.11 이상과 PyTorch 2.1 이상을 지원합니다.

## 사용법

```python
from pathlib import Path

import torch

from reuseloader import ReuseDataLoader, ReuseKey


def build(key: ReuseKey) -> dict:
    # 결과는 key의 값만으로 정해져야 한다.
    generator = torch.Generator().manual_seed(key.seed)
    return {"x": torch.rand(3, generator=generator)}


loader = ReuseDataLoader(
    build,
    1200,                 # 한 epoch의 표본 수
    Path("cache/train"),  # 재사용할 표본을 저장할 곳
    lifetime=2,           # 표본 하나를 2 epoch 동안 쓴다
    seed=42,
    shuffle=True,
    batch_size=4,
    num_workers=8,
    prefetch_factor=2,
)
for epoch in range(100):
    for batch in loader:
        ...
```

`build`, 표본 수, 저장 위치, `lifetime`, `seed`, `start_epoch`, `fingerprint`, `save`, `load`
외의 인자는 `DataLoader`에 그대로
넘어갑니다. 새 표본은 worker가 prefetch 범위 안에서 미리 만들고, 재사용할 표본은 읽기만 합니다.

중단한 학습을 이어서 하려면 `start_epoch`만 주면 됩니다.

```python
loader = ReuseDataLoader(build, 1200, Path("cache/train"), lifetime=2, seed=42, start_epoch=37)
```

디스크에 남아 있는 표본이 필요한 세대이면 그대로 읽고, 아니면 같은 seed로 다시 만듭니다.

## 읽을지 새로 만들지

index `i`가 epoch `e`에 쓰는 표본은 다음 식으로 정해집니다.

```text
위상         = i % lifetime
세대 g       = (e + 위상) // lifetime
사용 차수    = (e + 위상) %  lifetime
표본 seed    = mix(seed, i, g)
첫 사용 epoch = max(0, g * lifetime - 위상)
```

저장된 파일의 세대가 `g`이면 읽고, 아니면 `build`로 만들어 같은 파일에 덮어씁니다.

`lifetime = 2`이면 index의 절반은 위상 0, 절반은 위상 1입니다.

| epoch | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 위상 0인 index의 세대 | 0 | 0 | 1 | 1 | 2 |
| 위상 1인 index의 세대 | 0 | 1 | 1 | 2 | 2 |

- 모든 표본은 연속한 `lifetime`개 epoch에 쓰입니다.
- 0세대만 수명이 `lifetime - 위상`이라 index마다 1~`lifetime`으로 흩어집니다. 그래서 1 epoch부터는
  epoch마다 약 `1 / lifetime`의 표본만 새로 만듭니다. 한꺼번에 만료되지 않습니다.
- `loader.policy.key(i, e)`가 이 값을 담은 `ReuseKey`를 돌려줍니다.

## 순서

- `shuffle=True`이고 sampler를 주지 않으면 `ReuseSampler`를 씁니다. epoch `e`의 `p`번째 index는
  `loader.policy.order(p, e, size)`로 바로 구할 수 있습니다. 앞선 epoch을 다시 돌려 볼 필요가
  없습니다.
- `sampler`나 `batch_sampler`를 직접 주면 그 순서를 따릅니다. 재사용 판단은 순서와 무관합니다.

## 재현성을 위한 규칙

- **`build(key)`의 결과는 key의 값만으로 정해져야 합니다.** 무작위 요소는 `key.seed`에서만
  얻습니다. 시간에 따라 바뀌는 설정(예: curriculum)을 반영하려면 `key.first_epoch`으로 그 epoch의
  값을 찾는 표를 두고, 그 표를 학습 기록에 남기세요.
- epoch은 loader를 한 바퀴 돌기 시작할 때마다 1씩 올라갑니다. 도중에 멈추고 다시 돌아도 다음
  epoch으로 셉니다. 특정 epoch부터 다시 하려면 `start_epoch`으로 새 loader를 만드세요.

## 디스크 사용

- 파일은 index당 하나입니다(`00000123.pt`). 세대가 바뀌면 덮어쓰므로 파일 수가 표본 수를 넘지
  않고, 따로 정리할 것이 없습니다.
- 다시 쓰일 표본만 저장합니다. `lifetime = 1`이면 아무것도 쓰지 않습니다.
- 임시 파일에 쓰고 이름을 바꾸므로 읽는 쪽이 쓰다 만 파일을 보지 않습니다.
- 손상돼 읽을 수 없는 파일은 다시 만듭니다. 같은 key라 결과가 같습니다.
- 저장된 파일은 세대, 표본 seed, `lifetime`, `fingerprint`가 모두 같을 때만 다시 씁니다.
  seed나 `lifetime`을 바꾸고 같은 디렉터리를 써도 예전 표본이 섞이지 않습니다.
- **`build`의 동작을 바꿨으면 `fingerprint`를 바꾸세요**(예: `fingerprint="v2"`). 라이브러리는
  `build` 함수가 바뀌었는지 알 수 없습니다. 바꾸지 않으면 예전 방식으로 만든 표본을 다시 씁니다.
- 한 디렉터리는 한 loader만 쓰세요. 시작할 때 이전 실행이 남긴 임시 파일(`*.tmp`)을 지웁니다.
- 기본 저장 형식은 `torch.save`입니다. `save`와 `load` 인자로 바꿀 수 있습니다. 기본 `load`는
  임의 객체를 읽으므로 이 라이브러리가 직접 쓴 디렉터리에만 사용하세요.

## 일반 DataLoader와 함께 쓰기

`ReuseDataset`만 써도 됩니다. 이때는 epoch을 직접 알려 줍니다.

```python
from torch.utils.data import DataLoader

from reuseloader import ReuseDataset, ReusePolicy

dataset = ReuseDataset(build, 1200, Path("cache/train"), ReusePolicy(lifetime=2, seed=42))
loader = DataLoader(dataset, batch_size=4, num_workers=8)
for epoch in range(100):
    dataset.set_epoch(epoch)
    for batch in loader:
        ...
```

epoch 값은 공유 메모리에 있어서 `persistent_workers=True`인 worker도 바뀐 값을 봅니다.

## 언제 쓰는가

GPU 계산보다 CPU 입력 준비가 느려 GPU가 쉬는 경우입니다. `lifetime = 2`면 worker가 만들어야 할
표본이 절반이 됩니다. 같은 표본을 여러 번 보는 것이므로 step 수 기준 수렴은 느려질 수 있습니다.
실제 이득은 wall time 기준으로 확인해야 합니다.

## 라이선스

MIT
