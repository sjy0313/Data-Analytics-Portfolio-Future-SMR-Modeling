-- 학사 관리
CREATE DATABASE haksa DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
 #  This determines how text data is stored within your database.\
 #CHARACTER SET : This defines the set of characters that can be stored in the database.
 #utf8 is a popular character set that includes most characters from many languages around the world.
 
#  데이터 정렬 방식으로 COLLATE utf8_general_ci 로 설정하면, 문자열을 비교할 떄 대소문자를 구분하지 않고 사용할 수 있음을 의미.  
show databases;
-- 사용자 관리
use mysql;

-- 사용자 조회
select user, host from user;
select user();

-- 사용자 추가 
-- id : shin@localhost
-- pw : 1234
create user shin@localhost identified by 'shin';

-- 사용자 권한 부여 
grant all privileges on haksa.* to shin@localhost;

