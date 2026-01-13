import os
import re
from deep_translator import GoogleTranslator

# 설정: 번역할 폴더 경로 (현재 폴더면 '.' 입력)
# 현재 위치(.) 대신 특정 폴더 이름 입력
TARGET_FOLDER = '3.local_extinction_project'

def contains_korean(text):
    # 한글이 포함되어 있는지 정규표현식으로 확인
    return bool(re.search('[가-힣]', text))

def translate_comment(text):
    try:
        # Google 번역기 사용 (auto -> english)
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"번역 실패: {e}")
        return text

def process_file(file_path):
    print(f"Processing: {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    modified = False
    
    for line in lines:
        # 주석(#)이 있는 라인인지 확인
        if '#' in line:
            # 주석 부분만 분리 (코드는 건드리지 않음)
            code_part, separator, comment_part = line.partition('#')
            
            # 주석에 한글이 포함된 경우에만 번역
            if contains_korean(comment_part):
                translated_comment = translate_comment(comment_part.strip())
                # 원본 주석 형식을 유지하며 영문으로 교체
                new_line = f"{code_part}# {translated_comment}\n"
                new_lines.append(new_line)
                modified = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    # 변경된 내용이 있으면 파일 덮어쓰기
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Done: {file_path}")

def main():
    # 하위 폴더까지 모든 .py 파일 탐색
    for root, dirs, files in os.walk(TARGET_FOLDER):
        for file in files:
            if file.endswith('.py') and file != 'translate_comments.py': # 자기 자신은 제외
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == '__main__':
    main()