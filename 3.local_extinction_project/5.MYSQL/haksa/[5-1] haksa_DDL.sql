-- DDL : Data Definition Language 명령문 종류
-- db가 포함할 수 있는 table 수/행의 수 : 무한
-- db가 포함할 수 있는 열의 수(number of fields) : 254개까지 
use haksa;
show tables;

rename table ages30 to age_over30;
# alter table allows to add/delete/modify columns
# as well as apply various constraint to the table
#ALTER TABLE age_over30 ADD/DROP/MODIFY column_name datatype;
 
# sub_code char(5) Not null,
-- Not null  무결성 규칙이 지정되면 반듯이 데이터 값 입력 
-- 유일성 규칙(uniqueness rule) : 테이블에서 서로 다른 행은 동일한 값을 기본키로 가질 수 없다
-- 최소화 규칙(minimal rule) : 기본 키는 불필요하게 많은 열로 구성하지 않아야 한다 

diplomas CREATE TABLE DIPLOMAS 
(COURSE VARCHAR(20) NOT NULL, 
STUDENT VARCHAR(10) NOT NULL, 
COU_NUM INT(2), 
END_DATE DATE NOT NULL, 
PRIMARY KEY (COURSE, STUDENT, END_DATE));
 insert into diplomas values('웹프로그래밍','공자',2,'2007/07/25');
 insert into diplomas values('웹프로그래밍','맹자',3,'2007/07/25');
 SELECT * FROM diplomas;
 
 SELECT * FROM haksa.STUDENT;
 # db 소유자 = db(haksa)
 SELECT user FROM mysql.db WHERE db = 'haksa';

# one schemas owning two tables with same name is permitted. 
# name of table/column have limit of 64letters 
 
# 열 추가 및 변경
ALTER TABLE DIPLOMAS ADD SEX CHAR(2);
SELECT * FROM diplomas;
ALTER TABLE DIPLOMAS ADD GENDER CHAR(2);
# 열의 길이를 2에서 4로 증가시켜라.
ALTER TABLE DIPLOMAS MODIFY GENDER CHAR(4);
ALTER TABLE DIPLOMAS MODIFY GENDER INT;

# 테이블 복사
CREATE TABLE learner AS SELECT * FROM STUDENT;
# 원하는 열만 선택하여 테이블 생성
create table subject_copy as select sub_code, sub_name from subject;
desc subject_copy;
# 테이블 이름 변경
alter table subject_copy rename test_subject;
# 테이블과 데이터 사전정보 보기(views+tables 대한 정보를 지정한 중앙 저장소)
# schema는 데이터의 구조적 특성을 의미하며, instance에 의해 규정

show databases;
use mysql;
use information_schema; # 데이터베이스를 모니터링하고 관리(e.g. checking size/column in the tables)
desc tables; -- table view
desc columns; -- column view
use mysql;
desc user; -- user view

# user shin의 인증문자열과 파일권한 출력:
use mysql;
select user authentication_string, file_priv from user where user = 'shin'; -- N(권한없음)
GRANT FILE ON *.* TO 'shin'@'localhost' IDENTIFIED BY '1234';
grant all privileges on haksa.* to shin;

CREATE USER 'shin'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON haksa.* TO 'shin'@'localhost';




 
 
 