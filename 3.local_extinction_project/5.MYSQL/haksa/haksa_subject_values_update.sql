update subject set CREATE_YEAR = '2006' where SUB_NAME = '운영체제';
# drop database haksa; -> db날리기 

select * from subject;

select sub_code, sub_name, sub_ename, create_year from subject;

delete from subject where sub_name = 'UML';
commit;
#You are using safe update mode and you tried to update a table without a WHERE that uses a KEY column. 
# safe mode 해지
set sql_safe_updates = 0; 
update subject set create_year = '2002' where sub_name = '운영체제';

#select * from subject order by create_year; ->오름차순 정렬 
# select * from subject order by create_year desc; ->연도 별 내림차순 정렬

set sql_safe_updates = 1; 
# 제약조건의 종류파악하기 table에서 열이 Key column(primary/unique)
SELECT CONSTRAINT_TYPE
FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
WHERE TABLE_SCHEMA = 'haksa'
AND TABLE_NAME = 'subject'
AND CONSTRAINT_NAME = 'PRIMARY'; -- primary key로 설정되어 있으면 constraint_type 밑에 뜸 
# unique key인지 확인할 떄는 AND CONSTRAINT_NAME = 'unique' 아무것도 뜨지 않음 왜냐 아니기 때문.

# 결론은 sub_name이 primary key로 지정되어있기 때문에 수정할 떄 safe_update mode가 default로 설정되있음.

