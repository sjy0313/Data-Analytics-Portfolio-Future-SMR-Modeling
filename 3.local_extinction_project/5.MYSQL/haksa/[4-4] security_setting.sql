-- 보안설정
-- root 사용자의 데이터 보안

use mysql; 
-- select password('1234');

create user shin identified by 'shin123';
-- 권한부여
grant all privileges on haksa.* to shin;
-- 권한확인
show grants for shin;

desc user;

select host, user from user;
# 보안 업데이트로 조회불가 password from user;

-- 권한회수
revoke all privileges on haksa.* from shin;
flush privileges;
-- 권한회수 확인
select host, db, user, select_priv, update_priv from db; 
-- 사용자 삭제
drop user shin;
