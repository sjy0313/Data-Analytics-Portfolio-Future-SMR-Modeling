# 전체 실행 ctrl + shift + enter 
# join 학생정보와 학과정보 
select s.stu_no, s.stu_name, s.dept_code, d.dept_name, d.dept_ename
	from student s, department d
    where s.dept_code = d.dept_code; 
    
# 뷰 : stud_dept_vw
create or replace view stud_dept_vw as 
	select s.stu_no, s.stu_name, s.dept_code, d.dept_name, d.dept_ename
		from student s, department d
		where s.dept_code = d.dept_code;
   
