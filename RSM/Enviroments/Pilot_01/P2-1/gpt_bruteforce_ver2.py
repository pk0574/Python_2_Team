import os
import string
import zipfile
import threading
import zlib
import time
import logging
import sys

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
# Worker thread
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

    logger.info(f"{thread_name} 범위 {start}~{end} 시작")
    for i in range(start, end):
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

        # Update progress
        now = time.time()
        progress[thread_idx] = (i - start) / (end - start) * 100
        elapsed_time[thread_idx] = now - start_time

    zf.close()
    logger.info(f"{thread_name} 종료")

# ======================================
# Monitor thread: display per-thread progress and overall progress
# ======================================
def monitor():
    monitor_start = time.time()
    # Initial display of thread grid and overall
    # Clear screen lines equal to grid rows + 1 overall line
    rows = (THREAD_COUNT + 3) // 4
    # Print empty grid
    for start in range(1, THREAD_COUNT+1, 4):
        line = ''
        for tid in range(start, min(start+4, THREAD_COUNT+1)):
            line += f"T{tid}:   0.00% el: 00:00:00    "
        sys.stdout.write(line.rstrip() + "\n")
    # Print initial overall line
    sys.stdout.write("Overall progress: 0.00% elapsed: 00:00:00\n")
    sys.stdout.flush()

    while not found_event.is_set():
        # Move cursor up to redraw
        sys.stdout.write(f"[{rows+1}F")
        now = time.time()
        # Per-thread rows
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
        # Overall
        overall = sum(progress.values()) / THREAD_COUNT
        total_elapsed = now - monitor_start
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        overall_el = f"{h:02d}:{m:02d}:{s:02d}"
        sys.stdout.write(f"Overall progress: {overall:6.2f}% elapsed: {overall_el}\n")
        sys.stdout.flush()
        time.sleep(5)

# ======================================
# Main execution
# ======================================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"브루트포스 시작 with {THREAD_COUNT} threads")
    print(f"Starting brute-force with {THREAD_COUNT} threads...")

    chunk = MAX_VAL // THREAD_COUNT
    threads = []

    # Start monitor thread
    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    # Start worker threads
    for idx in range(THREAD_COUNT):
        start = idx * chunk
        end = (idx + 1) * chunk if idx < THREAD_COUNT - 1 else MAX_VAL
        t = threading.Thread(target=worker, args=(start, end, idx+1))
        t.start()
        threads.append(t)

    # Wait for workers to finish
    for t in threads:
        t.join()

    # Signal monitor to finish
    found_event.set()
    mon.join(0)

    # Final output
    if found_password:
        print(f"[+] Password found: {found_password}")
        try:
            with zipfile.ZipFile(ZIP_PATH) as zf:
                zf.extractall(path=OUTPUT_DIR, pwd=found_password.encode('utf-8'))
            logger.info(f"Files extracted to '{OUTPUT_DIR}'")
            print(f"[+] Files extracted to '{OUTPUT_DIR}'")
        except Exception as e:
            logger.error(f"최종 추출 실패: {e}")
            print(f"Extraction failed: {e}")
    else:
        logger.info("Password not found.")
        print("[-] Password not found.")

# ======================================
# Multiprocessing version
# ======================================
import multiprocessing as mp

def mp_worker(start, end, flag, result_ns):
    try:
        zf = zipfile.ZipFile(ZIP_PATH)
    except Exception as e:
        return
    for i in range(start, end):
        if flag.is_set():
            break
        pwd_str = int_to_base36(i)
        pwd_bytes = pwd_str.encode('utf-8')
        zf.setpassword(pwd_bytes)
        try:
            bad = zf.testzip()
        except (RuntimeError, zlib.error):
            continue
        if bad is None:
            result_ns.password = pwd_str
            flag.set()
            break
    zf.close()

if __name__ == '__main__':
    # Setup multiprocessing
    cpu_cnt = mp.cpu_count()
    manager = mp.Manager()
    found_flag = manager.Event()
    result_ns = manager.Namespace()
    procs = []
    chunk = MAX_VAL // cpu_cnt
    for idx in range(cpu_cnt):
        s = idx * chunk
        e = (idx+1)*chunk if idx < cpu_cnt-1 else MAX_VAL
        p = mp.Process(target=mp_worker, args=(s, e, found_flag, result_ns))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    if hasattr(result_ns, 'password'):
        print(f"[MP] Password found: {result_ns.password}")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(path=OUTPUT_DIR, pwd=result_ns.password.encode('utf-8'))
        print(f"[MP] Files extracted to '{OUTPUT_DIR}'")
    else:
        print("[MP] Password not found.")
