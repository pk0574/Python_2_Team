import string
import threading
import zipfile
from threading import Thread, Lock
import logging
import time
import datetime

charset = string.digits + string.ascii_lowercase  # '0123456789abcdefghijklmnopqrstuvwxyz'
max_val = 36**6
#Current_Value = 0

logging.basicConfig(filename="./P2-1/log_file.txt", level=logging.DEBUG, 
                    format="[ %(asctime)s | %(levelname)s ] %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S",
                    encoding="utf-8")

logger = logging.getLogger()

def int_to_base36(n: int) -> str:
    s = []
    for _ in range(6):
        n, r = divmod(n, 36)
        s.append(charset[r])
    return ''.join(reversed(s))


def try_password(password, id):
    zip_path = '2-1-emergency_storage_key.zip'
    output_dir = 'P2-1/extracted_files/'
    res = False

    # 1) 비밀번호 입력 받기 (문자열 → 바이트로 변환)
    password = password.encode('utf-8')

    try:
        # 2) ZIP 파일 열기
        with zipfile.ZipFile(zip_path) as zf:
            # 3) 전체 파일 압축 해제
            #    pwd 인자는 bytes 타입이어야 합니다.
            zf.extractall(path=output_dir, pwd=password)
        
        logger.info("[{0}]압축 해제 완료: {1} -{2}-".format(id, output_dir, password.decode()))
        res = True
    except RuntimeError as e:
        # 비밀번호가 틀린 경우 zipfile.BadZipFile 또는 RuntimeError 발생
        logger.info("[{0}]압축 해제 실패 1: -{1}-".format(id, password.decode()))
        # print("[{0}]압축 해제 실패 1: -{1}-".format(id, password))
    except zipfile.BadZipFile:
        #print("[{0}]압축 해제 실패. ZIP 파일이 손상되었거나, 올바른 ZIP 파일이 아닙니다.".format(id))
        logger.info("[{0}]압축 해제 실패 2: -{1}-".format(id, password.decode()))
        # print("[{0}]압축 해제 실패 2: -{1}-".format(id, password))
    finally:
        # print(password)
        
        return res

        
def worker(mutex, worker_num, thread_num, thread_safe):
    start_time = time.time()
    last_print = 5
    thread_num += 1

    gap = round(max_val/worker_num)
    start = gap*thread_num
    current = start
    print("스레드 {0}: {1}\n".format(thread_num, start))

    end = min(start + gap, max_val)
    while current < end:
        current += 1
            
        try:
            pwd = int_to_base36(current)
        except Exception as e:
            print("Error: {e}")
            logger.info("Error: {e}")


        if thread_safe:
            mutex.acquire()

        if try_password(pwd, thread_num):  # 사용자의 암호 검증 함수
            print("[{0}]비밀번호 발견: {1}".format(thread_num, pwd))
            logger.info("[{0}]비밀번호 발견: {1}".format(thread_num, pwd))
            logger.info("- 소요시간: {0}".format(time.time()-start_time))
            break

        if thread_safe:
            mutex.release()

        
        now = time.time()
        if now - last_print >= 5:
            elapsed = str(datetime.timedelta(seconds=round(now-start_time)))
            print("[{0}] 경과 시간 : {1} - 진행률 {2:5.1f}%".format(thread_num, elapsed, current/(start+gap)))
            last_print = now




#메인 쓰레드 
if __name__ == "__main__":
    threads = []
    thread_safe = False
    mutex =Lock()

    n = input("스레드 개수 설정: ")
    # print(try_password(n,0))
    # n = input("스레드 개수 설정: ")
    # n = int(n)
    # print(int_to_base36(n))

    try:
       n = int(n)
    except ValueError:
       n = 2
    except TypeError:
       n = 2

    if n < 2:
       n = 2
    
    print("스레드 개수 : {0}".format(n))

    for i in range(n):
       t = Thread(target=worker, args=(mutex, n, i, thread_safe))
       t.start()
       threads.append(t)

    for t in threads:
       t.join()

    