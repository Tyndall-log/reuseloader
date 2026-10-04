# PyPI 배포

LazeTable과 같은 GitHub Actions + PyPI Trusted Publisher 방식으로 배포한다. 장기 API
token은 GitHub secret이나 파일에 저장하지 않는다.

## 최초 PyPI 연결

PyPI 계정의 pending publisher에 다음 값을 등록한다.

```text
PyPI project name: reuseloader
GitHub owner: Tyndall-log
GitHub repository: reuseloader
Workflow name: build.yml
Environment name: pypi
```

GitHub repository에도 `pypi` environment를 만든다. 필요하면 이 environment에 수동
승인자를 설정한다.

## 로컬 검증

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv build
uvx twine check dist/*
```

PyPI에 이미 공개한 version은 교체할 수 없다. 새 배포 전에 `pyproject.toml`의 version을
올리고 변경사항을 commit한다.

## 자동 배포

package version과 같은 `v<version>` tag를 push한다.

```powershell
git tag v0.1.0
git push origin v0.1.0
```

workflow는 다음 순서로 동작한다.

1. Windows, macOS, Linux와 Python 3.11~3.13 테스트
2. wheel과 source distribution 빌드
3. distribution metadata 검사
4. OIDC Trusted Publishing으로 PyPI 업로드
5. 같은 파일을 첨부한 GitHub Release 생성

publish job만 `id-token: write` 권한을 가진다. build job과 publish job을 합치지 않는다.

## 재사용 규칙의 호환성

`ReusePolicy`의 식과 `mix`·`permute`는 입력 재현의 근거다. 이 식을 바꾸면 같은 seed, index,
epoch에서 다른 표본과 다른 순서가 나온다. 바꿀 때는 호환되지 않는 변경으로 다루고 version과
변경 기록에 명시한다. `tests/test_schedule.py`의 고정값 테스트가 이를 지킨다.
