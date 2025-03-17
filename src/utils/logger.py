import os
from pathlib import Path
import psutil
from datetime import datetime

from loguru import logger



# 로그 파일 설정 함수
def setup_logger(LOG_DIR, now):
    """
    INFO, DEBUG, ERROR 로그를 각각 다른 파일에 저장하도록 설정.
    """
    INFO_LOG_FILE = os.path.join(LOG_DIR, f"{now}_info.log")
    DEBUG_LOG_FILE = os.path.join(LOG_DIR, f"{now}_debug.log")
    ERROR_LOG_FILE = os.path.join(LOG_DIR, f"{now}_error.log")

    # INFO 로그 설정
    logger.add(
        INFO_LOG_FILE,
        level="INFO",  # INFO 레벨 이상의 로그만 기록
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="1 day",  # 하루 단위로 로그 파일 회전
        retention="7 days",  # 로그 7일 동안 유지
        compression="zip",  # 오래된 로그 압축
    )
    
    # DEBUG 로그 설정
    logger.add(
        DEBUG_LOG_FILE,
        level="DEBUG",  # DEBUG 레벨 이상의 로그만 기록
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="1 day",
        retention="7 days",
        compression="zip",
    )
    
    # ERROR 로그 설정
    logger.add(
        ERROR_LOG_FILE,
        level="ERROR",  # ERROR 레벨 이상의 로그만 기록
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="1 day",
        retention="30 days",  # 오류 로그는 더 오래 보관
        compression="zip",
    )

# 리소스 상태 수집
def inspect_server_resources(unit = "MB", round = 2):
    """
    리소스 사용량(현재 사용량 / 총량)을 반환합니다.
    """
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()

    unit_converter = {
        "B" : 1024**0,
        "KB" : 1024**1,
        "MB" : 1024**2,
        "GB" : 1024**3,
    }

    converter = lambda x: x*(10**(round+1))//unit_converter[unit]/(10**(round+1))

    metrics = {
        # "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 현재 시간
        "cpu_usage": psutil.cpu_percent(interval=0),  # CPU 사용률 (%)
        "memory_used": converter(memory.used),  # 메모리 사용량 (Bytes)
        "memory_total": converter(memory.total),  # 메모리 총량 (Bytes)
        "disk_used": converter(disk.used),  # 디스크 사용량 (Bytes)
        "disk_total": converter(disk.total),  # 디스크 총량 (Bytes)
        "network_sent": converter(network.bytes_sent),  # 누적 송신량 (Bytes)
        "network_received": converter(network.bytes_recv),  # 누적 수신량 (Bytes)
    }
    return metrics

# 로그 기록
def logging_resource():
    """
    현재 리소스 상태를 로그에 기록합니다.
    """
    metrics = inspect_server_resources()
    message = " | ".join(list(map(str, [metrics['cpu_usage'], metrics['memory_used'], metrics['memory_total'], metrics['disk_used'], metrics['disk_total'], metrics['network_sent'], metrics['network_received']])))
    logger.info(message)


# 테스트용 코드
if __name__ == "__main__":
    # python -m src.utils.loguri
    from dotenv import load_dotenv
    load_dotenv(verbose=False)

    from .now import get_now



    # 로그 디렉토리 및 파일 설정
    LOG_DIR = os.getenv("PATH_LOG_VIRTUAL")
    now = get_now()
    setup_logger(LOG_DIR, now)
    
    # 로그 테스트
    logger.info("This is an INFO log.")
    logger.debug("This is a DEBUG log.")
    logger.error("This is an ERROR log.")
    print(inspect_server_resources())

    logging_resource()