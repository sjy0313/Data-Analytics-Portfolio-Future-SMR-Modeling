# 원하는 항목을 다중 테이블에서 검색방법 : 

select s.stu_no, stu_name, sub_code, p.prof_code, prof_name
	from student s, attend a, professor p 
	where s.stu_no = a.stu_no and 
	a.prof_code = p.prof_code; 
# Inner join : 조인을 하는 테이블에 동일한 이름의 컬럼이 있을 떄 그 컬럼을 select에서 사용하면 오류발생
# student.stu_no : 로 attend테이블의 stu_no와의 구분이 필요 안해줄 시 ambiguous error 발생
# 1 # 각 학생의 학번과 이름, 수강년도, 학기, 수강과목코드, 교수코드를 나타내라.
select student.stu_no, stu_name, att_year, att_term, sub_code, prof_code
	from student, attend
    where student.stu_no = attend.stu_no;
# 2 FROM 절에 여러 개의 테이블이 사용되는 경우에 가명(alias) 사용하면 편리
select s.stu_no, stu_name,
    att_year, att_term, sub_code, prof_code
	from student s, attend a
    where s.stu_no = a.stu_no;
# 테이블의 가명(별칭)을 반드시 사용해야 하는 경우 
#[예제 8-14] 장수인(1999년 03월 14일) 학생보다 먼저 태어난 학생의 이름과 생년월일을 나타내어라.
select stu_name, birthday from student;
describe student; # birthday 의 type은 varchar(8)
select s.stu_name, s.birthday 
	from student s, student st # st 보다 나이가어린
	where st.stu_name = '전형배' #  1996/01/16 보다 이전에 태어난 사람은 연세It미래교육원연세It미래교육원
    and s.birthday < st.birthday; # 19930303, 19930314 < 19960116	
# 출력결과 : 연세It미래교육원
SELECT s.stu_name, s.birthday
	FROM student s, student st
	WHERE s.birthday < (SELECT st.birthday FROM student st WHERE st.stu_name = '전형배')
	#and s.birthday > '연세IT미래교육원'; < s.birthday는 student table에서 stu_name에 해당하므로 sql이 인식
    # 할 수 있도록 아래와 같이 출처를 명확히 밝혀야함. > 
    AND s.birthday > (SELECT birthday FROM student WHERE stu_name = '연세IT미래교육원');
# 출력결과 : 홍길동

# 3 수강신청 구분
select s.stu_no, stu_name, att_div
	from student s, attend a 
	where s.stu_no = a.stu_no;
    
#[예제 8-7] 테이블의 학번과 이름, 수강테이블의 수강년도, 학기, 수강교과목코드, 교과목테이블의 교과목명을 나타내어라.
select * from subject;

select s.stu_no, stu_name, att_year, att_term, a.sub_code, sub_name
	from student s, attend a, subject su
    # 조건문 추가 : 학번 = 학생의 해당 수강과목 / 수강교과목코드 = 학생의 교과목 별 수강과목코드
	where s.stu_no = a.stu_no and 
	a.sub_code = su.sub_code;
describe attend; 

#[예제 8-7A] 테이블의 학번과 이름, 수강테이블의 수강년도, 학기, 수강교과목코드, 교과목테이블의 교과목명 교수테이블의 교수명을 나타내어라 
-- professor 에 attend.prof_code 에 해당하는 교수코드가 검색되지 않음.
select * from professor;
select s.stu_no, stu_name, att_year, att_term, a.sub_code, sub_name, prof_name
	from student s, attend a, subject su, professor p
	where s.stu_no = a.stu_no 
    and a.sub_code = su.sub_code
    and a.prof_code = p.prof_code;

-- left outer join
# 수강과 교수 테이블을 조인
# 수강(attend) 테이블에 있는 professor table 에 존재하지 않아도 출력
select a.sub_code, a.prof_code, p.prof_name
	from attend a
    left outer join professor p
    on a.prof_code = p.prof_code;
    
#The LEFT OUTER JOIN ensures that all rows from the attend table are included in the result, even if there are no matching rows in the professor table.
# 4001 과목에 대해 교수 존재x 
# 존재하지 않는 교수이름 null 출력

# 학생을 기준으로 교과목과 교수가 없는 경우 모두 출력 : 
select s.stu_no, s.stu_name, a.att_year, a.att_term, a.sub_code, su.sub_name, p.prof_name
	from  student s left outer join attend a on s.stu_no = a.stu_no
    left outer join subject su on a.sub_code = su.sub_code
    left outer join professor p on a.prof_code = p.prof_code;

# 학생을 교과목을 기준으로 교수가 없는 경우 출력 : 
# join, outer join 결합
select s.stu_no, s.stu_name, a.att_year, a.att_term, a.sub_code, su.sub_name, p.prof_name
	from  student s left outer join attend a on s.stu_no = a.stu_no
    left join subject su on a.sub_code = su.sub_code
    left outer join professor p on a.prof_code = p.prof_code;
    
# 수강신청과 교과목 조인
# 방법 1 :
select a.att_year, a.att_term, a.sub_code, su.sub_name
	from attend a, subject su
    where a.sub_code = su.sub_code;
# 방법 2 :  
select a.att_year, a.att_term, a.sub_code, su.sub_name
	from attend a join subject su on a.sub_code = su.sub_code;
# [예제 8-8] 학적테이블의 학번과 이름, 보관성적테이블의 성적 취득년도, 학기, 신청학점, 취득학점, 평점평균을 나타내어라.
select s.stu_no, stu_name, sco_year, sco_term,
	req_point, take_point, exam_avg
	from student s, score sc
	where s.stu_no = sc.stu_no;









    
