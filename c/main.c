/* lstm_joint — CSV 입력 → 이상 탐지 + 공격 유형 분류.
 *
 * 사용법:
 *   ./lstm_joint [옵션] [input.csv]      (파일 생략 시 stdin)
 *
 * 입력: 한 줄에 한 시점(15분 간격, 오래된 것 먼저). 쉼표/탭/세미콜론 구분.
 *       각 줄의 "마지막 4개 필드"를 수전 4채널(유효, 지상무효, 진상무효, 피상)로 사용.
 *       숫자로 읽히지 않는 줄(헤더 등)은 건너뛴다.
 * 옵션:
 *   --cumulative  입력이 적산값이면 1-step diff(음수→0)로 구간 사용량 변환
 *                 (첫 줄은 diff 기준으로만 쓰이므로 97줄 이상 필요)
 *   --all         모든 sliding window(stride 1) 결과 출력. 기본은 마지막 window만
 *   --bench N     마지막 window 추론을 N회 반복해 평균 시간(ms) 출력
 *   --no-header   헤더 줄 생략
 *
 * 출력 (헤더 1줄 + window 당 한 줄, 공백 정렬 — 공백 기준으로 split 하면 됨):
 *   end_row  p_attack  is_attack  type  score_<유형> x5
 *   end_row 는 window 마지막 시점의 입력 행 번호(0-based, 데이터 행 기준).
 *   score_* 는 유형별 classification logit (is_attack=1 일 때만 의미 있음).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lstm_joint.h"

static int parse_line(char *line, float v[LJ_C])
{
    char *fields[64];
    int n = 0;
    for (char *tok = strtok(line, ",\t;\r\n"); tok && n < 64; tok = strtok(NULL, ",\t;\r\n"))
        fields[n++] = tok;
    if (n < LJ_C) return 0;
    for (int k = 0; k < LJ_C; k++) {
        char *s = fields[n - LJ_C + k], *end;
        while (*s == ' ') s++;
        v[k] = strtof(s, &end);
        while (*end == ' ') end++;
        if (end == s || *end != '\0') return 0;
    }
    return 1;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* 컬럼 폭: 헤더 이름과 값 중 긴 쪽. type 은 가장 긴 유형 이름 기준. */
static int type_w(void)
{
    int w = (int)strlen("normal");
    for (int k = 0; k < LJ_K; k++)
        if ((int)strlen(LJ_CLASS_NAMES[k]) > w) w = (int)strlen(LJ_CLASS_NAMES[k]);
    return w;
}

static int score_w(int k)
{
    int w = (int)strlen("score_") + (int)strlen(LJ_CLASS_NAMES[k]);
    return w > 10 ? w : 10;
}

static void print_header(void)
{
    printf("%7s  %8s  %9s  %-*s", "end_row", "p_attack", "is_attack", type_w(), "type");
    for (int k = 0; k < LJ_K; k++) {
        char name[64];
        snprintf(name, sizeof name, "score_%s", LJ_CLASS_NAMES[k]);
        printf("  %*s", score_w(k), name);
    }
    printf("\n");
}

static void print_result(long end_row, const lj_result *r)
{
    printf("%7ld  %8.6f  %9d  %-*s", end_row, r->p_attack, r->is_attack, type_w(),
           r->is_attack ? LJ_CLASS_NAMES[r->cls] : "normal");
    for (int k = 0; k < LJ_K; k++) printf("  %*.6f", score_w(k), r->cls_logits[k]);
    printf("\n");
}

int main(int argc, char **argv)
{
    int cumulative = 0, all = 0, bench = 0, header = 1;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cumulative")) cumulative = 1;
        else if (!strcmp(argv[i], "--all")) all = 1;
        else if (!strcmp(argv[i], "--no-header")) header = 0;
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
        else if (argv[i][0] == '-') { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
        else path = argv[i];
    }

    FILE *fp = path ? fopen(path, "r") : stdin;
    if (!fp) { perror(path); return 1; }

    size_t cap = 1024, n = 0;
    float (*rows)[LJ_C] = malloc(cap * sizeof *rows);
    char line[4096];
    while (fgets(line, sizeof line, fp)) {
        float v[LJ_C];
        if (!parse_line(line, v)) continue;
        if (n == cap) { cap *= 2; rows = realloc(rows, cap * sizeof *rows); }
        memcpy(rows[n++], v, sizeof v);
    }
    if (fp != stdin) fclose(fp);

    long row0 = 0;   /* rows[0] 에 해당하는 원본 데이터 행 번호 */
    if (cumulative) {
        if (n < 2) { fprintf(stderr, "need >= 2 rows for --cumulative\n"); return 1; }
        for (size_t t = n - 1; t > 0; t--)
            for (int k = 0; k < LJ_C; k++) {
                float d = rows[t][k] - rows[t - 1][k];
                rows[t][k] = d > 0.0f ? d : 0.0f;
            }
        rows++; n--; row0 = 1;
    }
    if (n < LJ_T) {
        fprintf(stderr, "need >= %d rows (got %lu)%s\n", LJ_T, (unsigned long)n,
                cumulative ? " after diff" : "");
        return 1;
    }

    lj_result r;
    if (header) print_header();
    size_t first = all ? 0 : n - LJ_T;
    for (size_t s = first; s + LJ_T <= n; s++) {
        lj_infer((const float (*)[LJ_C])rows[s], &r);
        print_result(row0 + (long)(s + LJ_T - 1), &r);
    }

    if (bench > 0) {
        double t0 = now_ms();
        for (int i = 0; i < bench; i++) lj_infer((const float (*)[LJ_C])rows[n - LJ_T], &r);
        fprintf(stderr, "bench: %.3f ms / window (%d runs)\n", (now_ms() - t0) / bench, bench);
    }
    return 0;
}
