use scott;
# 급여
select ename, sal from emp;

# 연봉
select ename, sal, sal * 12 "연봉" from emp;

# 수당
select ename, sal, comm from emp;

# 총 연봉 : 급여 * 12 + 수당
# comm(수당)이 없는 경우 총연봉이 계산되지 않음. 
select ename, sal, sal * 12, sal * 12 + comm "총연봉" from emp;
# ename(사원이름) / sal(월급) / sal*12 / sal*12+comm(보너스)

# comm 의 결측값 처리 + 총연봉 column 추가
# ifnull(expr1, expr2)
select ename, sal, comm, sal * 12, sal * 12 + ifnull(comm, 0) "총연봉" from emp;

# 건수
select count(*) from emp; -- 14 
select count(comm) from emp; -- 수당이 있는 사원 수 4
# 수당이 있는 사원 정보 
# 수당이 0인 경우도 선택
select * from emp where comm is not null; 

# 수당이 등록되어 있으면서 0보다 큰 사원
select * from emp
	where comm is not null
    and comm > 0;

# join 
# 사원의 부서정보 출력
# 사원 정보이름 부서정보
use scott;
# 정통방식
select e.deptno, d.dname, e.empno, e.ename, d.loc from dept d, emp e 
	where e.deptno = d.deptno;
# 표준화(ANSI형태)
select e.deptno, d.dname, e.empno, e.ename, d.loc from emp e join dept d 
	on(e.deptno = d.deptno);

# 부서정보
select * from dept;
# 사원정보
select * from emp;

# 사원정보 중에서 사원이름 A로 시작하는 사원과 해당부서 정보
select e.deptno, d.dname, e.empno, e.ename, d.loc from emp e join dept d 
	on(
		e.deptno = d.deptno
        and e.ename like 'A%');
# 사원정보 중에서 사원이름에 'S'가 포함된 사원 
select * from emp where ename like '%S%';

select cast(empno as char), emp.* from emp where ename like '%S%';

select cast(e.empno as char), e.* from emp e where ename like '%S%';
select convert(e.empno, char), e.* from emp e where ename like '%S%';

# 사원정보 중에서 사원이름이 'S'로 끝나는 사원
select * from emp where ename like '%S';

# 사원정보 중에서 사원이름의 두 번째 글자가 'A'인 사원
select * from emp where ename like '_A%';

# 사원정보 중에서 사원이름의 세 번째 글자가 'R'인 사원
select * from emp where ename like '__R%';

# 사원의 급여총액/평균금액/최대급여/최소급여 : 
use scott;
select sum(sal), avg(sal), max(sal), min(sal)
	 from emp;

# 최근에 입사한 사원과 가장 오래된 사원의 입사일
select max(hiredate), min(hiredate) from emp;

# 사원의 직책(job)의 종류의 갯수:
select count(distinct job) from emp; -- 5건

# 사원의 직책(job)의 종류의 갯수: 
select distinct job from emp;

# 사원의 직책별 건수
select job, count(*)
	from emp
    group by job;

# 사원의 부서별 평균 급여
select avg(sal) as "평균급여"
	from emp
    group by deptno;
# 개별적인 empno와 그룹핑된 deptno 는 결합이 불가
select deptno, empno, avg(sal) as "평균급여"
	from emp
    group by deptno;    
# 부서별/ 직책 별 모두 그룹핑 후 결합가능
select deptno, empno, avg(sal) as "평균급여"
	from emp
    group by deptno, empno;    

# 부서별, 직책별 사원수
select deptno, job, count(*) as "사원수"
	from emp
    group by deptno, job
    order by 1, 2;

# 부서별, 직책별 사원수, 평균급여
select deptno, job, count(*) as "사원수", avg(sal) as "평균급여"
	from emp
    group by deptno, job
    order by deptno, job;

# 그룹함수 
# 그룹함수 결과 제한
# 부서별 급여 총액이 3000이상인 부서의 번호와 부서별 급여 총액 : 
select deptno, max(sal)
	from emp
    group by deptno
    having max(sal) >= 3000;

# 부서별로 내림차순 정렬 
select deptno, max(sal)
	from emp
    group by deptno
    having max(sal) >= 3000
    order by deptno desc;

# 사원정보에서 직책이 'manager'를 제외하고 급여 총액이 5000이상인 직책별 급여 총액 
select job, count(*), sum(sal)
	from emp
    where job not like '%MANAGER%'
    group by job
    having sum(sal) >= 5000
    order by sum(sal);

# 부서별 평균 급여
select deptno, avg(sal) from emp group by deptno;

# 부서별 평균 급여 중 가장 급여가 많은 부서의 급여액은? 
#select max(avg(sal)) from emp group by deptno; 안됨.
#Invalid use of group function
# 2916.666667 10
select max(x.avg_sal)
	from (select avg(sal) as avg_sal
        from emp 
        group by deptno) x;

# 부서별 평균급여와 부서정보(부서이름)를 출력
select deptno, round(avg(sal)) from emp group by deptno;

select d.deptno, d.dname, x.avgsal
	from dept d,
    (select deptno, round(avg(sal)) as avgsal 
		from emp group by deptno) x
	where d.DEPTNO = x.deptno;
    
select d.deptno, d.dname, x.avgsal
	from dept d
		join (select deptno, round(avg(sal)) as avgsal 
			from emp group by deptno) x
		on(d.deptno = x.deptno);

# view 
create or replace view view_dept_avgsal as 
select d.deptno, d.dname, x.avgsal
	from dept d
		join (select deptno, round(avg(sal)) as avgsal
			from emp group by deptno) x 
		on(d.deptno = x.deptno);

desc view dept_avgsal;
select * from view_dept_avgsal;
select max(avgsal) from view_dept_avgsal;

select avgsal from view_dept_avgsal where deptno = 10 or deptno = 20;
select avgsal from view_dept_avgsal where deptno in(10, 20);
select avgsal from view_dept_avgsal where deptno between 10 and 20;










    
    
    




	










