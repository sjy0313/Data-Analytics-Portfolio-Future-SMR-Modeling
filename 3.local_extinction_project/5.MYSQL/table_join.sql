show tables;
desc student;

# Department 입력
insert into department values(10, '간호학과','Dept. of Nursing', '1991-02-01');
insert into department values(40, '컴퓨터정보학과','Dept. of Computer Information', '1997-02-01');

# student 학생 테이블 입력
insert into student values('20141001','박도상','Park Do-Sang',40,4,1,'주','19960116','1', '01066','101동 203호','02','744','6126','010-0611-9884')
insert into student values('20141001','홍길동','Hong Gil-Dong',40,4,1,'주','19960116','1', '01066','101동 203호','02','744','6126','010-0611-9884')


select * from department;
select * from student;

select s.stu_no, s.stu_name, s.dept_code, d.dept_name, d.dept_ename
	from student s, department d -- 축소 2개의 코드에서 일치하는 것을 찾아서 join
	where s.dept_code = d.dept_code; -- pandas의 join과 유사