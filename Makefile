# make 기본 CC(cc)가 없는 환경(Windows MinGW 등)을 위해 gcc 로 고정
ifeq ($(origin CC),default)
CC = gcc
endif
CFLAGS ?= -O2
CFLAGS += -std=gnu99 -Wall -Wextra
LDLIBS += -lm

# 보드(AM43xx, Cortex-A9)에서 빌드할 때 NEON 사용:  make ARM=1
ifeq ($(ARM),1)
CFLAGS += -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -ffast-math
endif

# 실행 파일은 이 폴더(프로젝트 최상위)에 생성:  ./lstm_joint examples/sample_normal.csv
SRC = c/main.c c/lstm_joint.c
DEP = $(SRC) c/lstm_joint.h c/model_weights.h

lstm_joint: $(DEP)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

clean:
	rm -f lstm_joint lstm_joint.exe

.PHONY: clean
