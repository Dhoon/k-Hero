"""원본 LP 데이터(xlsx / HTML-disguised xls)를 pandas DataFrame으로 읽어오는 모듈.

파일 구조 (실측 기준):
  1행: 제목 (읽지 않음)
  2행: 메타정보 — 계기번호/계기접속시간/단위 (읽지 않음)
  3행: 병합헤더 — "수전" / "송전" 그룹 (읽지 않음)
  4행: 실제 컬럼명 (읽지 않음)
  5행~: 데이터 (역순: 순번1이 최신 시각)

계측기 소프트웨어는 종종 HTML 표를 .xls 확장자로 저장하므로
파일 앞부분 바이트로 실제 포맷을 먼저 감지한 뒤 파서를 분기한다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

# 파일명에서 계량기 번호 추출 — 두 가지 패턴 지원:
#   "02530046335_2026.07.06 ..."  → 02530046335
#   "LP02530185548_2026.07.06 ..." → 02530185548  (LP 접두어 무시)
_METER_ID_RE = re.compile(r'^(?:LP)?(\d+)_')

# skiprows=4 후 실제 나타나는 컬럼 순서 (수전/송전 병합헤더 수동 해소)
FINAL_COLUMNS = [
    "순번",
    "일자시간",
    "LP발생횟수",
    "수전_유효전력량",
    "수전_지상무효전력량",
    "수전_진상무효전력량",
    "수전_피상전력량",
    "상태정보",
    "송전_유효전력량",
    "송전_지상무효전력량",
    "송전_진상무효전력량",
    "송전_피상전력량",
]

POWER_COLS = [
    "수전_유효전력량", "수전_지상무효전력량", "수전_진상무효전력량", "수전_피상전력량",
    "송전_유효전력량", "송전_지상무효전력량", "송전_진상무효전력량", "송전_피상전력량",
]

SEND_COLS = [
    "송전_유효전력량", "송전_지상무효전력량", "송전_진상무효전력량", "송전_피상전력량",
]

# 파일 포맷 판별에 사용하는 앞부분 바이트 크기
_PROBE_BYTES = 1024

# OLE2 Compound Document (Excel 97-2003 .xls) 매직 바이트
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

FileFormat = Literal["xlsx", "xls_ole2", "html", "unknown"]


# -------------------------------------------------------------------------
# 포맷 감지
# -------------------------------------------------------------------------

def _detect_format(path: Path) -> FileFormat:
    """파일 앞부분 바이트로 실제 포맷을 감지한다.

    - ZIP magic (PK..)         → 'xlsx'      진짜 xlsx
    - OLE2 magic (D0 CF 11 E0) → 'xls_ole2'  Excel 97-2003 바이너리 .xls
    - HTML 태그 포함            → 'html'      계측기가 HTML을 .xls로 저장한 케이스
    - 그 외                     → 'unknown'
    """
    with open(path, "rb") as f:
        raw = f.read(_PROBE_BYTES)

    if raw[:2] == b"PK":
        return "xlsx"

    if raw[:8] == _OLE2_MAGIC:
        return "xls_ole2"

    # HTML 감지: errors='ignore'로 디코드해서 태그 여부만 확인
    for enc in ("utf-8-sig", "utf-8", "euc-kr", "cp949"):
        snippet = raw.decode(enc, errors="ignore").lower().strip()
        if any(tag in snippet for tag in ("<html", "<table", "<!doctype")):
            return "html"
        break  # 첫 번째 인코딩으로 충분 (errors='ignore'라 실패 없음)

    return "unknown"


# -------------------------------------------------------------------------
# OLE2 바이너리 .xls 파서 (xlrd)
# -------------------------------------------------------------------------

def _read_xls_ole2_lp(path: Path) -> pd.DataFrame:
    """OLE2 binary .xls (Excel 97-2003) 파일을 xlrd 엔진으로 파싱."""
    df = pd.read_excel(path, skiprows=4, header=None, engine="xlrd")

    if df.shape[1] < len(FINAL_COLUMNS):
        raise ValueError(
            f"{path.name}: 컬럼 수 불일치 (예상 {len(FINAL_COLUMNS)}, 실제 {df.shape[1]})"
        )

    df = df.iloc[:, : len(FINAL_COLUMNS)].copy()
    df.columns = FINAL_COLUMNS
    df = df.dropna(how="all").reset_index(drop=True)

    for col in POWER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -------------------------------------------------------------------------
# HTML 파서
# -------------------------------------------------------------------------

def _read_html_lp(path: Path) -> pd.DataFrame:
    """HTML-disguised .xls 파일을 LP DataFrame으로 파싱.

    pd.read_html()은 <table> 단위로 파싱하므로:
      - 여러 테이블(범례, 데이터 등) 중 LP 데이터 테이블을 선택한다.
      - 선택 기준: FINAL_COLUMNS 개수(12) 이상 컬럼을 갖는 테이블 중 행 수 최대.
      - 앞 4행(제목/메타/병합헤더/컬럼명)은 슬라이싱으로 제거한다.
    """
    raw_bytes = path.read_bytes()

    # 인코딩 자동 감지 — euc-kr / utf-8 순서로 시도
    tables: list[pd.DataFrame] | None = None
    tried_encs: list[str] = []
    for enc in ("utf-8-sig", "utf-8", "euc-kr", "cp949"):
        try:
            html_str = raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
        tried_encs.append(enc)
        for flavor in ("lxml", "html.parser"):
            try:
                tables = pd.read_html(html_str, header=None, flavor=flavor)
                used_enc = enc
                used_flavor = flavor
                break
            except Exception:
                continue
        if tables is not None:
            break

    if tables is None:
        raise ValueError(
            f"HTML 파싱 실패 (시도한 인코딩: {tried_encs}): {path.name}"
        )

    # LP 데이터 테이블 선택: 컬럼 수 ≥ 12, 행 수 최대
    candidates = [t for t in tables if t.shape[1] >= len(FINAL_COLUMNS)]
    if not candidates:
        # 조건 완화: 컬럼 수 최대인 테이블
        candidates = [max(tables, key=lambda t: t.shape[1])]

    df = max(candidates, key=len)

    # 앞 4행 제거 (제목/메타/병합헤더/컬럼명)
    df = df.iloc[4:].reset_index(drop=True)

    # 컬럼 수 검증
    if df.shape[1] < len(FINAL_COLUMNS):
        raise ValueError(
            f"{path.name}: 컬럼 수 부족 (예상 {len(FINAL_COLUMNS)}, 실제 {df.shape[1]}). "
            f"테이블 후보 수: {len(tables)}, 선택된 테이블 shape: {df.shape}"
        )

    df = df.iloc[:, : len(FINAL_COLUMNS)].copy()
    df.columns = FINAL_COLUMNS
    df = df.dropna(how="all").reset_index(drop=True)

    for col in POWER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -------------------------------------------------------------------------
# xlsx 파서 (기존 openpyxl 경로)
# -------------------------------------------------------------------------

def _read_xlsx_lp(path: Path) -> pd.DataFrame:
    """진짜 xlsx 파일을 openpyxl로 파싱."""
    df = pd.read_excel(path, skiprows=4, header=None, engine="openpyxl")

    if df.shape[1] < len(FINAL_COLUMNS):
        raise ValueError(
            f"{path.name}: 컬럼 수 불일치 (예상 {len(FINAL_COLUMNS)}, 실제 {df.shape[1]})"
        )

    df = df.iloc[:, : len(FINAL_COLUMNS)].copy()
    df.columns = FINAL_COLUMNS
    df = df.dropna(how="all").reset_index(drop=True)

    for col in POWER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -------------------------------------------------------------------------
# 공개 인터페이스
# -------------------------------------------------------------------------

def read_lp_excel(path: str | Path) -> pd.DataFrame:
    """단일 LP 파일을 정규화된 DataFrame으로 반환.

    포맷(xlsx / HTML-disguised xls)을 자동 감지해 적절한 파서를 호출한다.
    어떤 포맷으로 읽혔는지 로그를 출력한다.

    Returns:
        DataFrame with FINAL_COLUMNS. 전력 컬럼은 float64.
    """
    path = Path(path)
    fmt = _detect_format(path)

    if fmt == "xlsx":
        print(f"    [format=xlsx]      {path.name}")
        return _read_xlsx_lp(path)

    if fmt == "xls_ole2":
        print(f"    [format=xls_ole2]  {path.name}  ← Excel 97-2003 바이너리")
        return _read_xls_ole2_lp(path)

    if fmt == "html":
        print(f"    [format=html]      {path.name}  ← HTML-disguised xls")
        return _read_html_lp(path)

    # unknown: xlsx → xls_ole2 → html 순서로 시도
    print(f"    [format=unknown]   {path.name}  — 포맷 자동 탐색 중...")
    for label, fn in [("xlsx", _read_xlsx_lp), ("xls_ole2", _read_xls_ole2_lp), ("html", _read_html_lp)]:
        try:
            df = fn(path)
            print(f"    [format=unknown→{label} 성공] {path.name}")
            return df
        except Exception as e:
            print(f"    [format=unknown→{label} 실패] {e}")

    raise ValueError(f"지원하지 않는 파일 포맷 (xlsx/xls_ole2/html 모두 실패): {path.name}")


def extract_meter_id(filename: str) -> str:
    """파일명 앞 숫자를 계량기 번호로 추출.

    '02530046335_202407.xlsx' → '02530046335'
    숫자가 없으면 확장자 제외 전체 이름 반환.
    """
    stem = Path(filename).stem
    m = _METER_ID_RE.match(stem)
    return m.group(1) if m else stem


def load_all_meters(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """raw_dir 내 모든 xls/xlsx 파일을 계량기별로 합쳐서 반환.

    Returns:
        {meter_id: DataFrame}  — 중복 timestamp 제거 전. 전처리는 preprocessing.py에서 수행.
    """
    raw_dir = Path(raw_dir)
    xlsx_files = sorted(raw_dir.glob("*.xls*"))

    if not xlsx_files:
        raise FileNotFoundError(f"xls/xlsx 파일 없음: {raw_dir}")

    buckets: dict[str, list[pd.DataFrame]] = {}
    for f in xlsx_files:
        meter_id = extract_meter_id(f.name)
        df = read_lp_excel(f)
        buckets.setdefault(meter_id, []).append(df)
        print(f"  [load] {f.name}  rows={len(df)}  meter={meter_id}")

    result: dict[str, pd.DataFrame] = {}
    for meter_id, dfs in buckets.items():
        combined = pd.concat(dfs, ignore_index=True)
        result[meter_id] = combined
        print(f"  [merge] meter={meter_id}  합산 rows={len(combined)}")

    # meter_id 검증 출력 — 순수 숫자인지 / 이전 버전과 병합 달라진 케이스 확인
    print("\n  [meter_id 목록 검증]")
    all_pure_numeric = True
    for f in sorted(xlsx_files):
        mid = extract_meter_id(f.name)
        is_numeric = mid.isdigit()
        flag = "" if is_numeric else "  ← [경고] 숫자 아님"
        if not is_numeric:
            all_pure_numeric = False
        print(f"    {f.name:55s} → {mid}{flag}")
    if all_pure_numeric:
        print("  → 모든 meter_id 순수 숫자. 이상 없음.")

    return result


def check_송전_nonzero(meter_dfs: dict[str, pd.DataFrame]) -> None:
    """송전 컬럼에 0이 아닌 값이 존재하는지 계량기별로 출력.

    결과가 있으면 해당 채널을 pretrain 입력에 포함할지 사용자에게 알림.
    """
    print("\n[송전 컬럼 비-zero 검사]")
    found_any = False
    for meter_id, df in meter_dfs.items():
        for col in SEND_COLS:
            if col not in df.columns:
                continue
            nonzero = df[col].dropna()
            nonzero = nonzero[nonzero != 0]
            if not nonzero.empty:
                found_any = True
                print(
                    f"  meter={meter_id}  {col}: 비-zero {len(nonzero)}건"
                    f"  (max={nonzero.max():.4f}, mean={nonzero.mean():.4f})"
                )
    if not found_any:
        print("  → 모든 계량기에서 송전 컬럼은 전부 0 또는 NaN. 수전 4채널로 충분.")
    else:
        print("  ★ 비-zero 송전 값 존재 — 해당 채널을 feature_cols에 추가할지 검토 필요.")
