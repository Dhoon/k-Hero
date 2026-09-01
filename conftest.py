import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (pytest가 어느 위치에서 실행되든 동작)
sys.path.insert(0, str(Path(__file__).resolve().parent))
