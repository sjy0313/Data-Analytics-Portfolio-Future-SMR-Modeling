show databases;
use hellodb;

craete TABLE insa(
	bunho int(1) auto_increment, 
	name char(8) not null, 
	e_name char(4), 
	town char(6) not null, 
	primary key(bunho) 
); 

show tables;
DESC insa;

INSERT INTO insa VALUES('1', '홍길동', 'HGD', '서울');
INSERT INTO insa VALUES(null, '이순신', 'LSS', '대전'); -- 자동번호부여
Insert into insa Values(null, '강감찬','KGC','신의주');
insert into insa values(null,'최영', '','광주');

savepoint spl;

insert into insa values(null,'정윤', 'JY','수원');


