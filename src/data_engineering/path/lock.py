import os
import sys
import fcntl      # POSIX 계열에서 파일 락 사용
import threading
import weakref
from pathlib import Path
import builtins   # open을 monkey-patch하기 위해 import

class LockedPath:
    """
    - 동일한 절대경로를 싱글턴으로 유지
    - 다른 모듈(pandas 등)에서 그냥 str 처럼 쓸 수 있도록 __fspath__ 구현
    - OS 차원의 파일잠금은 monkey_patch_open()을 통해 open(...) 시점에 적용
    """
    _instances = {}
    _instances_lock = threading.Lock()

    def __new__(cls, raw_path: str):
        resolved = str(Path(raw_path).resolve())
        with cls._instances_lock:
            if resolved not in cls._instances:
                obj = super().__new__(cls)
                obj._init_once(resolved)
                cls._instances[resolved] = obj
            return cls._instances[resolved]

    def _init_once(self, resolved_str: str):
        self._path_str = resolved_str  # 절대경로 문자열
        self._deleted = False

    def __fspath__(self):
        """os.fspath(path) 혹은 str()로 사용될 때 호출"""
        return self._path_str

    def __str__(self):
        return self._path_str

    def __repr__(self):
        return f"LockedPath({self._path_str})"

    def mark_deleted(self):
        """cleanup 시에 싱글턴 레지스트리에서도 지우고 싶다면 호출"""
        with self._instances_lock:
            self._deleted = True
            # 아래처럼 레지스트리에서 삭제할 수도 있음
            if self._path_str in self._instances:
                del self._instances[self._path_str]

    def is_deleted(self) -> bool:
        return self._deleted


# -------------------------------------------------------------------------
# OS 차원의 파일 잠금을 “open” 때마다 자동 적용하기 위한 monkey patch 예시
# -------------------------------------------------------------------------
_original_open = builtins.open  # 원본 open 함수 보관

def locked_open(file, mode='r', buffering=-1, encoding=None,
                errors=None, newline=None, closefd=True, opener=None):
    """
    file이 LockedPath이면:
      - 읽기 모드('r', 'rb' 등) => 공유락(LOCK_SH)
      - 쓰기/추가/변경 모드('w', 'a', 'x', 'r+') => 배타락(LOCK_EX)
    file이 일반 str이면 그냥 원본 open 호출
    """
    # file이 LockedPath인지 체크
    if isinstance(file, LockedPath):
        path_str = os.fspath(file)  # 실제 문자열 경로
        # 원본 open으로 FD 생성
        f = _original_open(path_str, mode, buffering, encoding, errors, newline, closefd, opener)

        # 어떤 락 모드를 쓸지 결정
        # 단순화: 'r'이면 공유락, 그 외는 배타락
        # 세부적으로 'r+' 같은 모드는 배타락이어야 할 수도 있고, 상황에 맞춰 조정
        if 'r' in mode and '+' not in mode and 'w' not in mode and 'a' not in mode:
            lock_mode = fcntl.LOCK_SH
        else:
            lock_mode = fcntl.LOCK_EX

        # 블로킹 락
        fcntl.flock(f.fileno(), lock_mode)
        return f
    else:
        # 그냥 일반 str 경로라면 원본 open
        return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

def monkey_patch_open():
    """내장 open을 locked_open으로 갈아끼우기"""
    builtins.open = locked_open

def restore_open():
    """open을 원복"""
    builtins.open = _original_open


# -------------------------------------------------------------------------
# DataFrame이 살아있는 동안 락 유지 시연 (약식)
# -------------------------------------------------------------------------
import weakref

class LockedFileHolder:
    """
    - 파일 객체(열린 FD)와 '공유락 or 배타락'을 보유
    - 이 객체가 살아있는 동안엔 FD가 닫히지 않으므로, OS 락 유지
    - weakref.finalize 등을 통해 DataFrame과 생명주기 동기화 가능
    """
    def __init__(self, locked_path: LockedPath, mode: str = 'r'):
        self._locked_path = locked_path
        self._mode = mode
        self._file = None
        self._lock_mode = None

    def open_and_lock(self):
        path_str = os.fspath(self._locked_path)
        # FD 열기
        self._file = _original_open(path_str, self._mode)
        # 읽기 모드 => 공유락, 아니면 배타락
        if self._mode.startswith('r') and '+' not in self._mode:
            self._lock_mode = fcntl.LOCK_SH
        else:
            self._lock_mode = fcntl.LOCK_EX
        fcntl.flock(self._file.fileno(), self._lock_mode)

    def close_and_unlock(self):
        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        """GC될 때 자동 close"""
        self.close_and_unlock()

    def fileobj(self):
        return self._file


# -------------------------------------------------------------------------
# 간단한 사용 예시
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) open()을 monkey-patch
    monkey_patch_open()

    try:
        # LockedPath 생성 (싱글턴)
        p = LockedPath("test_data.csv")

        # 예시1) pandas 등에서 그냥 read (읽기 락 -> 파일 닫힘 -> 락 해제)
        import pandas as pd
        df = pd.read_csv(p)  # 내부적으로 open(p, 'r') -> locked_open -> flock(LOCK_SH)
        print("DataFrame loaded:", df.head())

        # 이 시점에 파일은 이미 닫혔고(= 락도 해제), df는 메모리에 존재
        # => df가 살아있어도 파일에 대한 OS 락은 해제된 상태
        # => 만약 df가 살아있는 동안 락을 유지하고 싶으면, 아래처럼 FD를 계속 쥐고 있어야 함

        holder = LockedFileHolder(p, mode='r')
        holder.open_and_lock()
        # 이제 holder 객체가 살아있는 동안에는 공유락이 유지됨 (파일이 닫히지 않음)
        # df가 살아있는 동안 holder를 같이 참조하면, df가 소멸되면서 holder도 GC → 파일 닫힘(락 해제)

        # 예시2) 다른 프로세스/스레드에서 p에 대해 write 모드로 열면,
        # 공유락이 걸려 있으므로 flock(LOCK_EX) 시도 시 대기하게 됨

        # holder를 스스로 닫으려면:
        # holder.close_and_unlock()

        # 예시3) 삭제 시도 (non-blocking) 예: 간단히 구현
        # 배타락 try
        with _original_open(os.fspath(p), 'a') as f:  # 'a' 모드는 배타락
            # non-blocking 락
            # 실패시 예외 발생 -> 곧바로 except로 넘어가서 "삭제 스킵"
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 여기까지 왔으면 락 성공 -> 파일 삭제 가능
                # 삭제 후 close하면 락 해제
                print("File locked exclusively. Now we can safely remove it if we want.")
                # os.remove(os.fspath(p))
                # p.mark_deleted()
            except BlockingIOError:
                print("Someone else is reading or writing. Skip delete.")
    finally:
        # 2) monkey-patch 원복
        restore_open()
