"""CLI: 합성 이상치를 주입해 레이블 있는 검증셋을 생성.

Usage:
    python scripts/inject_synthetic_anomalies.py --config configs/eval/synthetic.yaml

TODO: argparse --config -> adt.data.anomaly_injection.inject_synthetic_anomalies(...)
      결과는 configs/eval/synthetic.yaml 의 output_dir (data/processed/synthetic_eval/)에 저장
      -> scripts/evaluate.py 에서 이 폴더를 로드해 auroc/auprc/f1/precision_at_k 계산
"""
