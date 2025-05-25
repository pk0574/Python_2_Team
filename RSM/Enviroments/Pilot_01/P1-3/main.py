import csv

file_path = "P1-3/1-3-Mars_Base_Inventory_List.csv"
file_save_path = "P1-3/Mars_Base_Inventory_danger.csv"

rows: list[list[str]] = []   

try:
    with open(file_path, mode="r", encoding="utf-8", newline="") as file: #newline ="" 윈도우에서 빈 줄이 끼는 것을 방지함
        reader = csv.reader(file)           # 각 행을 리스트로 변환, 기본 구분자 = 콤마(,)  
        _rows = list(reader)    
        #print(_rows[0][0])

        try:
        # 2) 헤더 · 데이터 분리
            header, data = _rows[0], _rows[1:]
        except ValueError:
            # 숫자가 아닌 경우 그대로 놔둠
            print("정렬 에러")

        data.sort(key= lambda inner : inner[4]) # 내부 리스트[1] 값을 기준으로 정렬

        data = list(zip(*data)) # 예: 0번 열 → ['apple', 'banana', 'cherry'] 열과 행을 전치


    with open(file_save_path, mode="w",encoding="utf-8", newline="") as save_file:
        writer = csv.writer(save_file)
        writer.writerow(header)     # 헤더 쓰기
        writer.writerows(data)      # 수정된 데이터 쓰기

#예외처리
except FileNotFoundError:
    print("파일이 경로에 없음")
except IsADirectoryError:
    print ("파일이 아닌 폴더에 접근")
except Exception as e:
    print(e)
