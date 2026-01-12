-- 데이터베이스 선택
use hellodb;
-- 검색
SELECT * FROM hello;

create table hello2(
hid integer,
name varchar(40),
age integer);
-- hello의 테이블에서 hello2로 모든 데이터를 카피
select * from hello2;
insert into hello2 (select * from hello);

-- hello의 테이블에서 hello2로 모든 데이터를 카피
insert into hello2(select * from hello);

