"""torch Dataset / DataLoader 정의.

TODO:
- LPWindowDataset(Dataset): processed 텐서를 로드해 (x, y) 또는 (x,) 반환
  - pretrain 단계: 레이블 없이 x만 반환
  - finetune 단계: 정상/이상 레이블(y) 또는 재구성 대상 반환
- build_dataloader(cfg, split) -> DataLoader
"""
