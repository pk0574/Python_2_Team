# 문제 2
import json

try:
    with open("1-1-mission_computer_main.log", 'r', encoding="utf-8") as file_log:
        str_log =""
        datetime_log = []
        content_log = []
        dict = {}

        tmp =[]
        #res_log = ""
        for line in file_log:
            str_log += line
            try:
                tmp = line.split(',',2)
                datetime_log.append(tmp[0]) #date time
                content_log.append(tmp[2]) # content
            except IndexError:
                None
            except Exception as e:
                print("문자 분할 에러 : {e}")

        print(datetime_log) #출력    
        print(content_log) #출력    

        datetime_log.sort(reverse=True)

        if len(datetime_log) != len(content_log):
            raise ValueError("키·값 리스트의 길이가 서로 다릅니다.")
        
        dict =  {k: v for k, v in zip(datetime_log, content_log)}

        
        with open("mission_computer_main.json", "w", encoding="utf-8") as j:
            json.dump(dict, j, ensure_ascii=False, indent=2)


#예외처리
except FileNotFoundError:
    print("파일이 경로에 없음")
except IsADirectoryError:
    print ("파일이 아닌 폴더에 접근")
except Exception as e:
    print(e)