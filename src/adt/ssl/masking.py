"""SSL masking 전략.

TODO:
- random_mask(x, mask_ratio)   : 타임스텝을 무작위로 마스킹
- segment_mask(x, mask_ratio)   : 연속 구간(segment) 단위로 마스킹 (시계열에서 더 어려운/유의미한 과제)
- channel_mask(x, mask_ratio)    : 특정 채널 전체를 마스킹
전략 선택은 configs/pretrain/default.yaml 의 ssl.mask_mode 로 제어
"""
