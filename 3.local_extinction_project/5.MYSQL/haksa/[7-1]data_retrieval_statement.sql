use haksa; 
#등록 테이블(" FEE")에서 장학금 백만원 이상을 지급 받은 학생 중에서 2회 이상 지급받은 
select * from fee;
select * from student;
select stu_no, count(*) from fee -- "장학금 수령횟수" 
	#where jang_total >= 1000000 -- 백만원 이상
    where jang_total > 1000000 -- 백만원초과
	group by stu_no -- 학번 기준 그룹핑
    having count(*) >= 1 -- 2회 이상
    order by stu_no desc;
    
select stu_no, count(*) from fee -- "장학금 수령횟수" 
    where jang_total > 1000000 -- 백만원초과
	group by stu_no -- 학번 기준 그룹핑
    having count(*) >= 1 -- 1회 이상
    order by stu_no desc;
    
-- 그룹 : '학번'기준으로 카운트
select stu_no, count(*) from fee group by stu_no;

-- 장학금 1백만 이상 개수
select jang_total from fee;
select jang_total from fee where jang_total > 1000000;
    # having jang_total >= 2000000 and jang_total <= 4500000;

