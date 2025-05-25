import zipfile, threading, os

# 전역 플래그
found_event    = threading.Event()
found_password = None

def int_to_base36(n, chars):
    s = []
    for _ in range(6):
        n, r = divmod(n, 36)
        s.insert(0, chars[r])
    return ''.join(s)

def worker(zip_path, output_dir, start, end, chars):
    global found_password

    # ZIP 열기
    zf = zipfile.ZipFile(zip_path)
    zf.setpassword(None)  # 초기화

    for i in range(start, end):
        if found_event.is_set():
            return

        pwd = int_to_base36(i, chars).encode('utf-8')
        zf.setpassword(pwd)

        # **비밀번호 검증만 수행** (디스크에 쓰지 않음)
        bad_file = 0
        try:
            bad_file = zf.testzip()
        except Exception as e:
            None
        
        if bad_file is None:
            # 모든 파일 CRC 체크 통과 → 진짜 압축 해제
            found_password = pwd.decode('utf-8')
            print(f"[*] Thread {threading.current_thread().name} found password: {found_password}")
            found_event.set()
            return
        # else: bad_file != None → 틀린 비밀번호

def brute_force_multi(zip_path, output_dir, num_threads=4):
    global found_password

    # 1) 출력 디렉토리 미리 생성 (exist_ok=True 로 동시 접근 안전)
    os.makedirs(output_dir, exist_ok=True)

    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    total = 36 ** 6
    chunk = total // num_threads
    threads = []

    # 2) 쓰레드 생성 및 시작
    for idx in range(num_threads):
        start = idx * chunk
        end   = (idx + 1) * chunk if idx < num_threads - 1 else total
        t = threading.Thread(
            target=worker,
            name=f"T{idx+1}",
            args=(zip_path, output_dir, start, end, chars)
        )
        threads.append(t)
        t.start()

    # 3) 결과 대기
    for t in threads:
        t.join()

    if found_password:
        # 메인 쓰레드에서 한 번만 압축 해제
        zf = zipfile.ZipFile(zip_path)
        zf.extractall(path=output_dir, pwd=found_password.encode('utf-8'))
        print("[*] 최종 비밀번호:", found_password)
        return found_password
    else:
        print("[!] Password not found.")
        return None

if __name__ == '__main__':
    ZIP_FILE    = '2-1-emergency_storage_key.zip'
    OUTPUT_DIR  = 'extracted'
    THREAD_COUNT = 256
    
    brute_force_multi(ZIP_FILE, OUTPUT_DIR, THREAD_COUNT)

