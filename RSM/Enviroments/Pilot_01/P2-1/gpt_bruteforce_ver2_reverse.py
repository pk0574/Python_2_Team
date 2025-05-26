import os
import string
import zipfile
import threading
import zlib
import time
import logging
import sys
import multiprocessing as mp

# ======================================
# Configuration
# ======================================
ZIP_PATH     = '2-1-emergency_storage_key.zip'
OUTPUT_DIR   = 'P2-1/extracted_files'
CHARSET      = string.digits + string.ascii_lowercase
MAX_VAL      = 36 ** 6
THREAD_COUNT = min(os.cpu_count() * 2, 30)
LOG_FILE     = './P2-1/bruteforce.log'

# ======================================
# Logging setup
# ======================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='[%(asctime)s | %(threadName)s | %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8'
)
logger = logging.getLogger()

# ======================================
# Convert integer to 6-digit base36 string
# ======================================
def int_to_base36(n: int) -> str:
    s = []
    for _ in range(6):
        n, r = divmod(n, 36)
        s.append(CHARSET[r])
    return ''.join(reversed(s))

# ======================================
# Shared data for progress
# ======================================
progress = {i+1: 0.0 for i in range(THREAD_COUNT)}
elapsed_time = {i+1: 0.0 for i in range(THREAD_COUNT)}
found_event = threading.Event()
found_password = None
lock = threading.Lock()

# ======================================
# Worker thread (reverse order)
# ======================================
def worker(start: int, end: int, thread_idx: int):
    global found_password
    thread_name = f"T{thread_idx}"
    threading.current_thread().name = thread_name
    start_time = time.time()

    try:
        zf = zipfile.ZipFile(ZIP_PATH)
    except Exception as e:
        logger.error(f"ZIP 열기 실패: {e}")
        return

    logger.info(f"{thread_name} 범위 {start}~{end} 시작 (역순)")
    # iterate from end-1 down to start
    for i in range(end-1, start-1, -1):
        if found_event.is_set():
            break

        pwd_str = int_to_base36(i)
        pwd_bytes = pwd_str.encode('utf-8')
        zf.setpassword(pwd_bytes)

        try:
            bad_file = zf.testzip()
        except (RuntimeError, zlib.error):
            continue

        if bad_file is None:
            with lock:
                if not found_event.is_set():
                    found_password = pwd_str
                    found_event.set()
                    logger.info(f"{thread_name} 비밀번호 발견: {pwd_str}")
                    print(f"[*] {thread_name} found password: {pwd_str}")
            break

        # Update progress in reverse
        now = time.time()
        progress[thread_idx] = ((end-1 - i) / (end - start)) * 100
        elapsed_time[thread_idx] = now - start_time

    zf.close()
    logger.info(f"{thread_name} 종료")

# ======================================
# Monitor thread: display per-thread and overall progress
# ======================================
def monitor():
    monitor_start = time.time()
    rows = (THREAD_COUNT + 3) // 4
    # Initial grid
    for start in range(1, THREAD_COUNT+1, 4):
        line = ''
        for tid in range(start, min(start+4, THREAD_COUNT+1)):
            line += f"T{tid}:   0.00% el: 00:00:00    "
        sys.stdout.write(line.rstrip() + "\n")
    sys.stdout.write("Overall progress: 0.00% elapsed: 00:00:00\n")
    sys.stdout.flush()

    while not found_event.is_set():
        sys.stdout.write(f"\033[{rows+1}F")
        now = time.time()
        for start in range(1, THREAD_COUNT+1, 4):
            line = ''
            for tid in range(start, min(start+4, THREAD_COUNT+1)):
                pr = progress[tid]
                et = elapsed_time[tid]
                h, rem = divmod(int(et), 3600)
                m, s = divmod(rem, 60)
                elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
                line += f"T{tid}: {pr:6.2f}% el: {elapsed_str}    "
            sys.stdout.write(line.rstrip() + "\n")
        overall = sum(progress.values()) / THREAD_COUNT
        total_elapsed = now - monitor_start
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        overall_el = f"{h:02d}:{m:02d}:{s:02d}"
        sys.stdout.write(f"Overall progress: {overall:6.2f}% elapsed: {overall_el}\n")
        sys.stdout.flush()
        time.sleep(5)

# ======================================
# Main execution (threaded)
# ======================================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"브루트포스 시작 with {THREAD_COUNT} threads (역순)")
    print(f"Starting brute-force with {THREAD_COUNT} threads (reverse)...")

    chunk = MAX_VAL // THREAD_COUNT
    threads = []

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    for idx in range(THREAD_COUNT):
        start = idx * chunk
        end = (idx + 1) * chunk if idx < THREAD_COUNT - 1 else MAX_VAL
        t = threading.Thread(target=worker, args=(start, end, idx+1))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    found_event.set()
    mon.join(0)

    if found_password:
        print(f"[+] Password found: {found_password}")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(path=OUTPUT_DIR, pwd=found_password.encode('utf-8'))
        logger.info(f"Files extracted to '{OUTPUT_DIR}'")
        print(f"[+] Files extracted to '{OUTPUT_DIR}'")
    else:
        logger.info("Password not found.")
        print("[-] Password not found.")
