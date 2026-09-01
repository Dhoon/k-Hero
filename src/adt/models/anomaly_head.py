"""이상탐지(다운스트림) head.

TODO:
- ReconstructionAnomalyHead: encoder(pretrained) 위에서 재구성 오차를 이상 점수로 사용
- ClassificationHead: 레이블이 있는 경우 정상/이상 이진분류 head
- encoder는 models/encoder.py 의 백본을 그대로 재사용 (freeze_encoder 옵션은 configs/finetune 참고)
"""
