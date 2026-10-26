#ifndef LSTM_JOINT_H
#define LSTM_JOINT_H

#include "model_weights.h"

typedef struct {
    float p_attack;            /* sigmoid(det_logit) */
    int   is_attack;           /* p_attack > LJ_THRESHOLD */
    int   cls;                 /* argmax(cls_logits) — is_attack일 때만 의미 있음 */
    float cls_logits[LJ_K];
} lj_result;

/* x: 정규화 전 구간 사용량 window, [LJ_T][LJ_C] row-major (시간 순서, 오래된 것 먼저).
 * 내부에서 scaler 정규화 후 추론한다. */
void lj_infer(const float x[LJ_T][LJ_C], lj_result *out);

#endif /* LSTM_JOINT_H */
