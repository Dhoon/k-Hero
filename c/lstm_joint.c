/* LSTM Joint 추론 — PyTorch nn.LSTM(bidirectional, batch_first) + MLP head 와 동일 연산.
 * 게이트 순서 i, f, g, o (PyTorch 규약). */
#include <math.h>
#include <string.h>

#include "lstm_joint.h"

#define G4 (4 * LJ_H)
#define D2 (2 * LJ_H)

static float sigmoidf_(float v) { return 1.0f / (1.0f + expf(-v)); }

/* 한 방향 LSTM. in: [T][din], out의 열 [off, off+H) 에 hidden 기록.
 * reverse=1 이면 t=T-1 → 0. 반환값 h_last: 마지막으로 처리한 step의 hidden. */
static void lstm_dir(const float *in, int din,
                     const float *wih, const float *whh, const float *b,
                     int reverse, float *out, int off, float *h_last)
{
    float h[LJ_H], c[LJ_H], g[G4];
    memset(h, 0, sizeof h);
    memset(c, 0, sizeof c);

    for (int s = 0; s < LJ_T; s++) {
        int t = reverse ? LJ_T - 1 - s : s;
        const float *xt = in + t * din;

        for (int r = 0; r < G4; r++) {
            float acc = b[r];
            const float *wi = wih + r * din;
            for (int k = 0; k < din; k++) acc += wi[k] * xt[k];
            const float *wh = whh + r * LJ_H;
            for (int k = 0; k < LJ_H; k++) acc += wh[k] * h[k];
            g[r] = acc;
        }
        for (int j = 0; j < LJ_H; j++) {
            float ig = sigmoidf_(g[j]);
            float fg = sigmoidf_(g[LJ_H + j]);
            float gg = tanhf(g[2 * LJ_H + j]);
            float og = sigmoidf_(g[3 * LJ_H + j]);
            c[j] = fg * c[j] + ig * gg;
            h[j] = og * tanhf(c[j]);
        }
        memcpy(out + t * D2 + off, h, sizeof h);
    }
    memcpy(h_last, h, sizeof h);
}

/* y = W2 · relu(W1 · z + b1) + b2 */
static void mlp(const float *z, const float *w1, const float *b1, int hid,
                const float *w2, const float *b2, int nout, float *y)
{
    float a[64];
    for (int i = 0; i < hid; i++) {
        float acc = b1[i];
        for (int k = 0; k < D2; k++) acc += w1[i * D2 + k] * z[k];
        a[i] = acc > 0.0f ? acc : 0.0f;
    }
    for (int o = 0; o < nout; o++) {
        float acc = b2[o];
        for (int i = 0; i < hid; i++) acc += w2[o * hid + i] * a[i];
        y[o] = acc;
    }
}

void lj_infer(const float x[LJ_T][LJ_C], lj_result *out)
{
    static float xn[LJ_T * LJ_C];
    static float buf[2][LJ_T * D2];   /* layer 출력 ping-pong */
    float z[D2];

    for (int t = 0; t < LJ_T; t++)
        for (int k = 0; k < LJ_C; k++)
            xn[t * LJ_C + k] = (x[t][k] - LJ_MEAN[k]) / LJ_STD[k];

    const float *in = xn;
    int din = LJ_C;
    for (int l = 0; l < LJ_L; l++) {
        float *o = buf[l & 1];
        lstm_dir(in, din, LJ_WIH[l][0], LJ_WHH[l][0], LJ_B[l][0], 0, o, 0,    z);
        lstm_dir(in, din, LJ_WIH[l][1], LJ_WHH[l][1], LJ_B[l][1], 1, o, LJ_H, z + LJ_H);
        in  = o;
        din = D2;
    }
    /* z = [마지막 레이어 fwd 최종 hidden (t=T-1), bwd 최종 hidden (t=0)] */

    float det;
    mlp(z, LJ_DET_W1, LJ_DET_B1, LJ_DH, LJ_DET_W2, LJ_DET_B2, 1, &det);
    mlp(z, LJ_CLS_W1, LJ_CLS_B1, LJ_CH, LJ_CLS_W2, LJ_CLS_B2, LJ_K, out->cls_logits);

    out->p_attack  = sigmoidf_(det);
    out->is_attack = out->p_attack > LJ_THRESHOLD;
    out->cls = 0;
    for (int k = 1; k < LJ_K; k++)
        if (out->cls_logits[k] > out->cls_logits[out->cls]) out->cls = k;
}
