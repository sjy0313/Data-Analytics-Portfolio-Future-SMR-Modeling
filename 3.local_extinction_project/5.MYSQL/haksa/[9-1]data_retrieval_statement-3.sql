# 원하는 항목 다중조건연산자 통한 검색방법 : 
select stu_no, stu_name, gender, birthday
	from student
	where gender = 2 and substring(birthday , 1, 4) < 2000 ;

# SQL 스칼라 함수 SUBSTRING(string, start_position, length)
# birthday 8자리 중 첫4자리 

# 부속 질의어에서 IN 연산자 => 중복처리
select * from student; 
select distinct grade from student;
select juya, class from student;
select juya, class from student order by 1,2;  # 중복포함
select distinct juya, class from student order by 1,2; # 중복제거
# 1은 select 목록의 juya 2는 class
# 중복되지 않는 값을 출력
select distinct juya, class, grade from student order by 1,2,3; 

select stu_no, stu_name from student where stu_no in ('20141001', '20241001')

# 휴대폰이 없는 사람 출력 + 휴대폰 앞자리가 010을 제외한 나머지 학생 출력

select stu_no, stu_name, mobile from student
	where substring(mobile, 1, 3) <> '010'
	or mobile is null;

# 02 는 서울 / 031 경기도 번호조합 :
select stu_no, stu_name concat(tel1, '-', tel2, '-', tel3) from student where tel1 = '02' or tel1 = '031';
    
# [예제 9-21] 부속질의어를 이용하여 등록을한 각 학생의 학번, 이름을 출력
# 등록을 한 번이라도 지불한 학생의 정보 
select stu_no from fee order by 1;
select distinct stu_no from fee order by 1;
# in 은 반드시 하나의 query에 대해 사용가능
 select stu_no, stu_name from student
	where stu_no in (select distinct stu_no from fee); 


# 당해 연도의 등록금을 낸 학생 정보
select * from fee;
select year(now());
select * from fee where fee_year = year(now());
# in의 서브쿼리의 결과는 반드시 유일한 하나의 컬럼이 지정되어야함. 
select stu_no, stu_name from student
	where stu_no in (select distinct stu_no from fee where fee_year = year(now())); 

# 당해 연도의 등록금을 낸 학생을 제외한 나머지 학생 page14 [9번 pdf 파일] not in
# 등록금을 지불한 학생 포함.  
 select stu_no, stu_name from student
	where stu_no not in (select distinct stu_no from fee where fee_year = year(now())); 
# sub-query 조건에 따라 결과 1개 출력
select stu_no, stu_name from student
	where stu_no = (select distinct stu_no from fee where fee_year = '2014');  
    
# [예제 9-23] “20141001”인 학생이 가입한 동아리를 제외한 다른 동아리에 적어도 한 번 가입을 한 학생의 학번과 이름을 출력하라.
select * from circle;
select stu_no, stu_name 
	from student
	where stu_no in (
    select stu_no 
    from circle 
		where cir_name not in (
			select cir_name 
            from circle 
            where stu_no = '20141001')
);

# 가장 나이가 많은 사람 제외한 나머지 출력
# all 과 any문은 비교대상이 여러가지인 경우 :
use haksa;
select stu_no, stu_name, birthday from student order by birthday desc;
# 방법 1 : 
select stu_no, stu_name, birthday
	from student
	where birthday > any (select birthday from student);
# 방법 2 : 
 select stu_no, stu_name, birthday
	from student
	where birthday > (select min(birthday) from student);
# 가장 나이가 어린 학생 출력 : 
 select stu_no, stu_name, birthday
	from student
	where birthday <= all (select birthday from student);
    
# exists  : 등록을 한 학생 출력 : 
# 방법 1 : 
select stu_no, stu_name
	from student
	where stu_no in (select stu_no from fee);
# 방법 2 : 
select stu_no, stu_name
	from student
	where exists (select * from fee where stu_no = student.stu_no);

-- 조인을 하면 결합되는 데이터의 갯수만큼 결과가 출력 
select * from fee;
select s.stu_no, s.stu_name, f.fee_year, f.fee_term
	from student s, fee f
	where s.stu_no = f.stu_no; # 등록금을 낸 인원 동일한 이름 여러 번 출력 가능.
    
# 등록되지 않은 학생 : 
# 방법 1: 
 select stu_no, stu_name 
 from student where not exists (select * from fee where stu_no = student.stu_no);
# 방법 2: 
 select stu_no, stu_name 
 from student where stu_no not in (select distict stu_no from fee);

use haksa;
desc student;
select * from student;
select * from student where stu_name like '전%';


