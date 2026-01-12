savepoint spl;
select * from insa;
SET SQL_SAFE_UPDATES = 0; #disable save updatemode

update insa set e_name = 'CYOG' where name = '최영';
SET SQL_SAFE_UPDATES = 1;
select * from insa;

-- 전체 원상복구
-- rollback;

-- 세이브 포인트까지 원복
-- 최영의 이름변경만 취소, 입력된 전우치는 존재 
rollback to sp1;
select * from insa;

commit;
-- 전체 롤백
rollback;

select * from insa;


