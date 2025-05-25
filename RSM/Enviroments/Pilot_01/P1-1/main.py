# 문제 1
print("Hello Mars")

try:
    with open("1-1-mission_computer_main.log", 'r', encoding="utf-8") as file_log:
        tmp_log = []
        res_log = ""
        for line in file_log:
            tmp_log.append(line)
        #역순 출력
        for line_str in tmp_log[::-1]:
            #문제 상황 저장
            if ("explosion" in line_str) | ("unstable" in line_str ):
                try:
                    with open("Problems.md", mode="x", encoding="utf-8") as problems:
                        problems.write(line_str)
                except FileExistsError:
                    with open("Problems.md", "a", encoding="utf-8") as problems:
                        problems.write(line_str)
            res_log += line_str
        print(res_log) #출력
#예외처리
except FileNotFoundError:
    print("파일이 경로에 없음")
except IsADirectoryError:
    print ("파일이 아닌 폴더에 접근")
except Exception as e:
    print(e)