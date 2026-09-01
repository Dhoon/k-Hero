"""datetime 파싱, 정렬, 중복 제거, gap 기반 segment 분리, 리샘플링, 결측치 보간.

처리 순서 (preprocess_meter):
  1. parse_datetime / sort_and_dedup — 역순 정렬 해제, 중복 제거
  2. report_status_info              — 상태정보 분류 및 의심 이벤트 CSV 저장
  3. split_by_gap                    — 대형 gap 기준 segment 분리, 짧은 segment 버림
  4. resample_and_fill (per segment) — segment 내 짧은 결측만 linear 보간
  반환: list[DataFrame]  (segment별 전처리 완료본)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_DOW_PATTERN = re.compile(r'[\(（][월화수목금토일][\)）]')
_BRACKET_PREFIX = re.compile(r'\[[^\]]*\]')

# 상태정보 코드 분류
_ROUTINE_CODES: frozenset[str] = frozenset({"LN", "LF", "SR", "SSR", "TC", "T", "P", "LP", "ISR"})
_SUSPICIOUS_CODES: frozenset[str] = frozenset({"CO", "OV", "LV", "SE", "WE", "O", "LE", "MF", "B"})


# -------------------------------------------------------------------------
# 기본 정제
# -------------------------------------------------------------------------

def parse_datetime_col(series: pd.Series) -> pd.Series:
    """'2026-07-06 14:15:00(월)' → pd.Timestamp. 파싱 실패 시 NaT."""
    cleaned = (
        series.astype(str)
        .str.replace(_DOW_PATTERN, "", regex=True)
        .str.strip()
    )
    return pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def sort_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """datetime 오름차순 정렬 + 중복 timestamp 제거."""
    df = df.copy()
    df["일자시간"] = parse_datetime_col(df["일자시간"])
    df = df.dropna(subset=["일자시간"])
    df = df.sort_values("일자시간").reset_index(drop=True)
    df = df.drop_duplicates(subset="일자시간", keep="first").reset_index(drop=True)
    return df


# -------------------------------------------------------------------------
# 상태정보 파싱 헬퍼
# -------------------------------------------------------------------------

def _extract_suspicious_codes(value) -> list[str]:
    """상태정보 단일 값에서 의심 코드 목록 추출.

    "[A]LN, OV" → ["OV"]
    "CO, WE"    → ["CO", "WE"]
    "LN"        → []
    """
    if pd.isna(value):
        return []
    cleaned = _BRACKET_PREFIX.sub("", str(value))
    parts = [p.strip() for p in cleaned.split(",")]
    return [p for p in parts if p in _SUSPICIOUS_CODES]


# -------------------------------------------------------------------------
# 상태정보 리포트
# -------------------------------------------------------------------------

def report_status_info(
    df: pd.DataFrame,
    meter_id: str = "",
    status_events_dir: Path | None = None,
) -> None:
    """상태정보 분류 출력 + 의심 이벤트 CSV 저장.

    출력:
      1. 전체 value_counts
      2. 의심 이벤트 rows (value_counts + 전체 대비 비율)
      3. 개별 의심 코드별 발생 횟수
    저장:
      status_events_dir/<meter_id>.csv  (의심 이벤트 rows)
    """
    if "상태정보" not in df.columns:
        return

    label = f"(meter={meter_id})" if meter_id else ""
    total = len(df)

    # 1. 전체 value_counts
    vc = df["상태정보"].value_counts(dropna=False)
    print(f"\n  [상태정보 value_counts] {label}")
    print(vc.to_string())

    # 의심 코드 추출
    matched_series = df["상태정보"].map(_extract_suspicious_codes)
    suspicious_mask = matched_series.map(bool)
    n_suspicious = suspicious_mask.sum()

    if n_suspicious == 0:
        print("  → 의심 이벤트 없음.")
        return

    pct = n_suspicious / total * 100

    # 2. 의심 이벤트 value_counts + 비율
    sus_vc = df.loc[suspicious_mask, "상태정보"].value_counts(dropna=False)
    print(f"\n  [의심 이벤트] {label}  {n_suspicious}/{total} rows ({pct:.1f}%)")
    print(sus_vc.to_string())

    # 3. 개별 의심 코드별 빈도
    all_codes: list[str] = []
    for codes in matched_series[suspicious_mask]:
        all_codes.extend(codes)
    code_vc = pd.Series(all_codes).value_counts()

    print(f"\n  [의심 코드별 빈도] {label}")
    for code, cnt in code_vc.items():
        code_pct = cnt / total * 100
        print(f"    {code:5s}: {cnt:6d}건  ({code_pct:.1f}%)")

    # 4. CSV 저장
    if status_events_dir is not None and meter_id:
        status_events_dir = Path(status_events_dir)
        status_events_dir.mkdir(parents=True, exist_ok=True)

        events_df = df.loc[suspicious_mask, ["일자시간", "상태정보"]].copy()
        events_df["matched_codes"] = matched_series[suspicious_mask].map(
            lambda codes: ", ".join(codes)
        )
        out_path = status_events_dir / f"{meter_id}.csv"
        events_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  → 저장: {out_path}  ({len(events_df)}건)")


# -------------------------------------------------------------------------
# gap 기반 segment 분리
# -------------------------------------------------------------------------

def split_by_gap(
    df: pd.DataFrame,
    gap_threshold_hours: float,
    window_size: int,
    meter_id: str = "",
) -> list[pd.DataFrame]:
    """실측 타임스탬프 간 gap을 기준으로 연속 segment 분리."""
    threshold = pd.Timedelta(hours=gap_threshold_hours)
    gaps = df["일자시간"].diff()
    mask = (gaps > threshold).fillna(False)
    segment_id = mask.cumsum()

    kept: list[pd.DataFrame] = []
    dropped_info: list[tuple[int, pd.Timedelta]] = []

    all_segs = list(df.groupby(segment_id, sort=True))
    label = f"meter={meter_id}" if meter_id else ""

    print(f"\n  [segment 분리] {label}  threshold={gap_threshold_hours}h  "
          f"총 {len(all_segs)}개 segment")

    for sid, seg_df in all_segs:
        seg_df = seg_df.reset_index(drop=True)
        n = len(seg_df)
        t_start = seg_df["일자시간"].iloc[0]
        t_end = seg_df["일자시간"].iloc[-1]
        duration = t_end - t_start

        if n >= window_size:
            kept.append(seg_df)
            status = "[KEEP]"
        else:
            dropped_info.append((n, duration))
            status = f"[DROP]  rows={n} < window_size={window_size}"

        print(f"    seg{sid+1:02d}: {t_start}  ~  {t_end}  "
              f"rows={n:5d}  duration={duration}  {status}")

    if dropped_info:
        total_dropped = sum(d[0] for d in dropped_info)
        print(f"  → 버린 segment: {len(dropped_info)}개  합계 rows={total_dropped}")
    print(f"  → 사용 segment: {len(kept)}개")

    return kept


# -------------------------------------------------------------------------
# 리샘플링 + 보간 (단일 segment)
# -------------------------------------------------------------------------

def resample_and_fill(
    df: pd.DataFrame,
    freq: str = "15min",
    method: str = "linear",
) -> pd.DataFrame:
    """freq 간격 리샘플링 후 결측치 보간 (segment 단위로 호출)."""
    df = df.set_index("일자시간").sort_index()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    df_num = df[numeric_cols].resample(freq).mean()
    if method == "linear":
        df_num = df_num.interpolate(method="linear", limit_direction="both")
    else:
        df_num = df_num.ffill().bfill()

    if cat_cols:
        df_cat = df[cat_cols].resample(freq).ffill()
        df_out = pd.concat([df_num, df_cat], axis=1)
    else:
        df_out = df_num

    return df_out.reset_index().rename(columns={"index": "일자시간"})


# -------------------------------------------------------------------------
# 계량기 전처리 파이프라인
# -------------------------------------------------------------------------

def preprocess_meter(
    df: pd.DataFrame,
    freq: str = "15min",
    method: str = "linear",
    gap_threshold_hours: float = 6.0,
    window_size: int = 96,
    meter_id: str = "",
    status_events_dir: Path | None = None,
) -> list[pd.DataFrame]:
    """단일 계량기 DataFrame → 전처리 완료된 segment DataFrame 리스트."""
    df = sort_and_dedup(df)
    report_status_info(df, meter_id=meter_id, status_events_dir=status_events_dir)

    kept_segs = split_by_gap(
        df,
        gap_threshold_hours=gap_threshold_hours,
        window_size=window_size,
        meter_id=meter_id,
    )

    label = f"meter={meter_id}" if meter_id else ""
    print(f"\n  [보간 비율] {label}")

    resampled: list[pd.DataFrame] = []
    for i, seg in enumerate(kept_segs):
        raw_n = len(seg)
        t_start = seg["일자시간"].iloc[0]
        t_end = seg["일자시간"].iloc[-1]

        seg_r = resample_and_fill(seg, freq=freq, method=method)
        rs_n = len(seg_r)

        interp_pct = (rs_n - raw_n) / rs_n * 100 if rs_n > 0 else 0.0
        flag = "  *** 높음" if interp_pct > 20 else ""
        print(
            f"    seg{i+1}: {t_start} ~ {t_end}"
            f"  raw={raw_n}  resampled={rs_n}"
            f"  보간비율={interp_pct:.1f}%{flag}"
        )
        resampled.append(seg_r)

    return resampled
