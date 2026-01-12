create database hellobd default character set utf8 collate utf8_general)_ci;
show databases;

-- 데이터베이스 선택
use hellobd;

-- 테이블 생성
create table hello(
hid integer,
name varchar(40),
age integer);

-- 데이터 입력
insert into hello values (3000, '이혁진', 27);
insert into hello values (4000, '김삿갓', 25);
commit;