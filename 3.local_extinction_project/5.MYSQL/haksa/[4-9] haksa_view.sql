# views 테이블은 필요에 따라 사용자가 재 정의하여 생성해주는 테이블
-- 어떤 공간도 차지하지 않으며 derived/virtual table이라고도 함.
-- 실제 테이터 행을 가지고 있는 것처럼 동작하지만 데이터 행은 없음.
# view 의 사용경우
-- 반복문(routine)을 간단히 사용할 때
-- 테이블의 출력 방법을 재구성할 때
-- 여러 단계에서 select명령문이 사용될 때
-- 데이터를 보호할 떄

select now() "오늘날짜", year(now()) "당해년도";
# 생일만 출력
select birthday from student; 
# substring 역할 "20000501"에서 문자열 4개뽑아냄.
select substring("20000501", 1, 4); -- 2000 
select year(now()) - substring("20000501", 1, 4); -- 24 

select stu_no "학번", stu_name "이름", birthday "생년월일",
	year(now()) - substring(birthday, 1, 4) "나이"
	from student;
    
# 가상테이블 ages 생성하기
create view ages(학번, 이름, 생년월일, 나이) as 
	select stu_no "학번", stu_name "이름", birthday "생년월일",
	year(now()) - substring(birthday, 1, 4) "나이" 
	from student;


desc ages;

select * from ages;
select * from ages where 나이 > 30;

 drop view ages;
 -- 뷰 생성 : ages30
 -- 검색조건을 지정하여 뷰를 생성[ 30세 미만제외 조건 추가]
 create or replace view ages30(학번, 이름, 생년월일, 나이) as 
	select stu_no "학번", stu_name "이름", birthday "생년월일",
	year(now()) - substring(birthday, 1, 4) "나이" 
	from student
    where year(now()) - substring(birthday, 1, 4) > 30;

select * from ages30;

commit;



