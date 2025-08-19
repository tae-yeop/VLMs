from pathlib import Path, PurePosixPath, PurePath
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict
import hashlib
import requests
import shutil
import subprocess
import shutil, sys

# ---------------------------------------------------------------------------
# Constants – REST endpoints.
# ---------------------------------------------------------------------------
UNSPLASH_URL = "https://api.unsplash.com/search/photos"  # Unsplash v1 search
FLICKR_URL = "https://api.flickr.com/services/rest"      # Flickr REST API

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def sha1(path: Path) -> str:
    """Return the SHA‑1 hex digest of a file.

    파일의 SHA-1 해시 값을 계산하여 반환합니다.
    다른 제공자나 쿼리에서 동일한 이미지를 찾아 중복을 제거하는 데 사용됩니다.
    """
    h = hashlib.sha1()
    with path.open("rb") as f:
        # 파일을 8192바이트(8KB)씩 읽어 해시를 업데이트합니다.
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download *url* into *dest*.

    Returns *True* if successful, *False* otherwise.  A small wrapper around
    `requests` with basic exception handling so the main loop can continue on
    failures (404, time‑outs, etc.).
    주어진 URL에서 파일을 다운로드하여 지정된 경로에 저장합니다.
    성공하면 True, 실패하면 False를 반환합니다.
    기본적인 예외 처리를 포함하여 다운로드 실패 시에도 메인 루프가 계속 진행되도록 합니다.
    """
    try:
        r = requests.get(url, timeout=timeout) # 지정된 URL로 GET 요청을 보냅니다.
        r.raise_for_status() # HTTP 오류 응답(4xx 또는 5xx)이 발생하면 예외를 발생시킵니다.
        with dest.open("wb") as f: # 바이너리 쓰기 모드로 대상 파일을 엽니다.
            f.write(r.content) # 응답 내용을 파일에 씁니다.
        return True
    except Exception as exc:  # noqa: BLE001, S110
        # 다운로드 실패 시 오류 메시지를 표준 오류 출력으로 보냅니다.
        print(f"[!] Download failed {url[:80]}… {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Provider‑specific download helpers
# ---------------------------------------------------------------------------
def fetch_unsplash(query: str, n: int, out_dir: Path, size: str = "regular") -> List[Dict]:
    """Download *n* images for *query* from Unsplash.

    *size* can be one of Unsplash sizes (`raw`, `full`, `regular`, `small`,
    `thumb`).  Unsplash requires an *access key* stored in the
    `UNSPLASH_ACCESS_KEY` environment variable.
    Unsplash에서 주어진 쿼리에 해당하는 이미지를 *n*개 다운로드합니다.
    'size'는 Unsplash에서 제공하는 이미지 크기 중 하나입니다.
    `UNSPLASH_ACCESS_KEY` 환경 변수에 저장된 액세스 키가 필요합니다.
    """
    key = os.getenv("UNSPLASH_ACCESS_KEY") # 환경 변수에서 Unsplash API 키를 가져옵니다.
    if not key:
        print("[Unsplash] UNSPLASH_ACCESS_KEY not set – skipping", file=sys.stderr)
        return []

    per_page = 30  # Unsplash API의 페이지당 최대 결과 수
    page = 1
    items: List[Dict] = [] # 다운로드된 이미지 메타데이터를 저장할 리스트

    while len(items) < n: # 목표 개수만큼 이미지를 수집할 때까지 반복
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "client_id": key,
            "content_filter": "high", # 높은 수준의 콘텐츠 필터 적용
        }
        resp = requests.get(UNSPLASH_URL, params=params, timeout=30)
        resp.raise_for_status() # HTTP 오류가 있으면 예외 발생
        data = resp.json() # JSON 응답을 파싱

        for shot in data["results"]:
            url = shot["urls"][size] # 지정된 크기의 이미지 URL을 가져옵니다.
            fname = out_dir / f"unsplash_{shot['id']}.jpg" # 저장할 파일 경로 생성
            if download(url, fname): # 이미지 다운로드 시도
                items.append( # 성공하면 매니페스트에 추가할 정보 저장
                    {
                        "id": shot["id"],
                        "provider": "unsplash",
                        "query": query,
                        "file": str(fname.relative_to(out_dir.parent)), # 상대 경로 저장
                        "license": "Unsplash‑license",
                    }
                )
            if len(items) >= n: # 목표 개수에 도달하면 루프 종료
                break

        if not data["results"]:  # 더 이상 결과가 없으면 루프 종료
            break
        page += 1 # 다음 페이지로 이동

    return items

def fetch_flickr(
    query: str,
    n: int,
    out_dir: Path,
    license_allow: set[str] | None = None,
) -> List[Dict]:
    """Download *n* images for *query* from Flickr.

    By default only grabs licenses 4/5/6/9/10 which are CC‑BY or more open.
    Flickr에서 주어진 쿼리에 해당하는 이미지를 *n*개 다운로드합니다.
    기본적으로 CC-BY 또는 더 개방적인 라이선스(4, 5, 6, 9, 10)의 이미지만 가져옵니다.
    """
    key = os.getenv("FLICKR_API_KEY") # 환경 변수에서 Flickr API 키를 가져옵니다.
    if not key:
        print("[Flickr] FLICKR_API_KEY not set – skipping", file=sys.stderr)
        return []

    if license_allow is None:
        # 4 = CC‑BY, 5 = CC‑BY‑SA, 6 = CC‑BY‑ND, 9/10 = CC0 / PublicDomain
        # 기본 허용 라이선스 목록 설정
        license_allow = {"4", "5", "6", "9", "10"}

    per_page = 100 # Flickr API의 페이지당 최대 결과 수
    page = 1
    items: List[Dict] = [] # 다운로드된 이미지 메타데이터를 저장할 리스트

    while len(items) < n: # 목표 개수만큼 이미지를 수집할 때까지 반복
        params = {
            "method": "flickr.photos.search", # Flickr 검색 메서드
            "api_key": key,
            "text": query,
            "sort": "relevance", # 관련성 기준으로 정렬
            "content_type": 1,  # 사진만 검색
            "license": ",".join(license_allow), # 허용된 라이선스 필터링
            "extras": "url_o,url_c,url_l,license", # 추가 정보 요청 (원본, 중간, 작은 URL, 라이선스)
            "per_page": per_page,
            "page": page,
            "format": "json", # JSON 형식 응답 요청
            "nojsoncallback": 1, # JSONP 콜백 방지
        }
        resp = requests.get(FLICKR_URL, params=params, timeout=30)
        resp.raise_for_status() # HTTP 오류가 있으면 예외 발생
        data = resp.json() # JSON 응답을 파싱

        for p in data["photos"]["photo"]:
            # 사용 가능한 가장 큰 이미지 URL을 우선적으로 선택
            url = p.get("url_o") or p.get("url_l") or p.get("url_c")
            if not url:
                continue  # 다운로드 가능한 URL이 없으면 건너뜁니다.
            fname = out_dir / f"flickr_{p['id']}.jpg" # 저장할 파일 경로 생성
            if download(url, fname): # 이미지 다운로드 시도
                items.append( # 성공하면 매니페스트에 추가할 정보 저장
                    {
                        "id": p["id"],
                        "provider": "flickr",
                        "query": query,
                        "file": str(fname.relative_to(out_dir.parent)), # 상대 경로 저장
                        "license": p.get("license"), # 라이선스 정보 (있으면) 저장
                    }
                )
            if len(items) >= n: # 목표 개수에 도달하면 루프 종료
                break

        if page >= data["photos"]["pages"]: # 모든 페이지를 탐색했으면 루프 종료
            break
        page += 1 # 다음 페이지로 이동

    return items


def fetch_openimages(query: str, n: int, out_dir: Path) -> List[Dict]:
    """Download *n* images for *query* using the `openimages` CLI.

    The Open‑Images dataset (V7) is stored on GCS buckets; the third‑party
    `openimages` Python package wraps all the hard work.  We shell out to it to
    avoid re‑implementing the logic.
    `openimages` CLI 도구를 사용하여 Open-Images 데이터셋에서 이미지를 다운로드합니다.
    Open-Images 데이터셋은 GCS 버킷에 저장되어 있으며, `openimages` Python 패키지가 다운로드 로직을 캡슐화합니다.
    """
    # label = query.replace(" ", "_") # 쿼리 문자열을 파일 시스템 친화적인 레이블로 변환 (공백을 밑줄로)
    # cmd = [
    #     "openimages",
    #     "download",
    #     "--label",
    #     query, # 검색할 레이블 (쿼리)
    #     "--limit",
    #     str(n), # 다운로드할 이미지 개수 제한
    #     "--dataset_dir",
    #     str(out_dir), # 이미지를 저장할 디렉토리
    # ]

    fmt = "pascal"                        # or "darknet"
    if shutil.which("openimages"):        # 콘솔 스크립트 有
        cmd = ["openimages",
               "--base_dir", str(out_dir),
               "--labels",   query,
               "--limit",    str(n),
               "--format",   fmt]         # ★ 필수 옵션
    else:                                 # 콘솔 스크립트 無 → 모듈 호출
        cmd = [sys.executable, "-m", "openimages.download",
               "--base_dir", str(out_dir),
               "--labels",   query,
               "--limit",    str(n),
               "--format",   fmt]         # ★ 필수 옵션

    try:
        subprocess.run(cmd, check=True, timeout=300)
    except Exception as exc:
        print(f"[OpenImages] {exc}", file=sys.stderr)
        return []

    # Collect freshly downloaded JPGs (PNG fallback)
    items: List[Dict] = []
    # 다운로드된 JPG 및 PNG 파일을 수집합니다.
    # glob()은 해당 디렉토리 내의 파일만 찾고, rglob()은 재귀적으로 하위 디렉토리까지 찾습니다.
    # OpenImages CLI의 실제 저장 방식에 따라 rglob을 고려할 수 있습니다.
    for img in out_dir.glob("*.jpg") | out_dir.glob("*.png"):
        items.append(
            {
                "id": img.stem, # 파일 이름에서 확장자를 제외한 부분 (ID로 사용)
                "provider": "openimages",
                "query": query,
                "file": str(img.relative_to(out_dir.parent)), # 상대 경로 저장
                "license": "OpenImages", # OpenImages 라이선스
            }
        )
        if len(items) >= n: # 목표 개수에 도달하면 루프 종료
            break
    return items

# ---------------------------------------------------------------------------
# Utility helpers – de‑duplication & main orchestrator
# ---------------------------------------------------------------------------

def dedupe_images(root: Path) -> None:
    """주어진 루트 디렉토리 아래의 중복 이미지 파일(동일한 SHA-1 해시 값을 가진)을 삭제합니다."""
    seen: Dict[str, Path] = {} # 이미 본 이미지들의 SHA-1 해시와 파일 경로를 저장할 딕셔너리
    # 루트 디렉토리 아래의 모든 JPG 파일을 재귀적으로 탐색합니다.
    for img in root.rglob("*.jpg"):
        h = sha1(img) # 현재 이미지 파일의 SHA-1 해시를 계산합니다.
        if h in seen: # 동일한 해시를 가진 이미지가 이미 있다면
            print(f"[dup] removing {img} (duplicate of {seen[h]})") # 중복 메시지 출력
            img.unlink(missing_ok=True) # 현재 이미지를 삭제합니다.
        else: # 새로운 해시라면
            seen[h] = img # 해시와 파일 경로를 'seen' 딕셔너리에 추가합니다.

def main() -> None:
    """Command‑line entry point."""
    parser = argparse.ArgumentParser(prog="collect_airport_bird_images")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write images & manifest (will be created if absent).",
    )
    parser.add_argument(
        "--keywords", # 리스트로 전달받기
        nargs="+",
        required=True,
        help="One or more search queries (wrap in quotes if containing spaces).",
    )
    # 허용되는 제공자 목록
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["unsplash", "flickr", "openimages"],
        choices=["unsplash", "flickr", "openimages"],
        help="Which image sources to use.",
    )
    # 키워드당, 제공자당 이미지 수
    parser.add_argument(
        "--per-provider",
        type=int,
        default=300,
        help="Number of images per provider *per keyword*.",
    )
    # 전체 이미지 수 제한 (per-provider보다 우선)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Global limit on total images (overrides --per-provider).",
    )

    # 인수를 파싱합니다.
    args = parser.parse_args()

    # 출력 디렉토리 경로 객체 생성
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # 다운로드된 모든 이미지의 메타데이터를 저장할 리스트
    manifest: List[Dict] = []

    # Iterate over all keyword / provider pairs
    for kw in args.keywords:
        for provider in args.providers:
            # 현재 키워드/제공자 조합에 대해 다운로드할 이미지 개수를 결정합니다.
            n_target = args.per_provider
            if args.limit is not None:
                # 간단하게 (키워드 수 × 제공자 수)로 총 제한을 나눕니다.
                n_each = args.limit // (len(args.keywords) * len(args.providers))
                # per-provider와 계산된 개수 중 더 작은 값을 사용
                n_target = min(n_target, n_each)

            # 제공자별 하위 디렉토리 경로 생성
            out_sub = root / provider
            # 하위 디렉토리 생성
            out_sub.mkdir(parents=True, exist_ok=True)

            # 각 제공자에 따라 적절한 fetch 함수를 호출하여 이미지 다운로드
            if provider == "unsplash":
                manifest += fetch_unsplash(kw, n_target, out_sub)
            elif provider == "flickr":
                manifest += fetch_flickr(kw, n_target, out_sub)
            elif provider == "openimages":
                manifest += fetch_openimages(kw, n_target, out_sub)

    print(f"Downloaded {len(manifest)} images in total.")

    # 제공자 간의 중복 이미지를 제거합니다.
    dedupe_images(root)

    # Save manifest
    manifest_file = root / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)
    # print(f"Manifest saved to {manifest_file.relative_to(Path.cwd())}")
    print(f"Manifest saved to {manifest_file}")

if __name__ == "__main__":
    main()
