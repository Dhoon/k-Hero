CC     ?= gcc
CFLAGS ?= -O2
CFLAGS += -std=gnu99 -Wall -Wextra
LDLIBS += -lm -lrt

# 보드(AM43xx, Cortex-A9)에서 빌드할 때 NEON 사용:  make ARM=1
ifeq ($(ARM),1)
CFLAGS += -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -ffast-math
endif

lstm_joint: main.c lstm_joint.c lstm_joint.h model_weights.h
	$(CC) $(CFLAGS) -o $@ main.c lstm_joint.c $(LDLIBS)

clean:
	rm -f lstm_joint lstm_joint.exe

.PHONY: clean
