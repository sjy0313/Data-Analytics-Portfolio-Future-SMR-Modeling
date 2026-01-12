# 서브쿼리(sub-query)

select * from emp;
select * from emp where empno = 7654; -- 'MARTIN'

# 'MARTIN'의 급여보다 많이 급여를 많이 받는 사원은? 
# 'MARTIN'의 급여 : 1250
select * from emp where sal > 1250;
# 'MARTIN'의 사번을 이용해서 'MARTIN'의 급여보다 급여를 많이 받는 사원을 구하라
select * from emp
	where sal > (select sal from emp where empno = 7654)
	order by sal;
    
# 부서코드가 20,30에 속한 사원중에서 TURNER(7844)보다 높은 급여를 받는 사원정보와
# 소속부서 정보를 출력 : 
select * from emp e, dept d
	where e.deptno in (20,30)
    and e.deptno = d.deptno
    and e.sal > (select sal from emp where empno = 7844) -- Turner 1500급여
    order by e.sal;
# ----------------------------------------------------------------------------------
# 급여 등급 
select * from salgrade;

# 사원의 급여등급
select e.EMPNO, e.ENAME, e.SAL, s.GRADE
	from emp e, salgrade s
	where e.sal between s.LOSAL and s.HISAL
    order by e.sal;

# Turner의 급여 등급
select e.EMPNO, e.ENAME, e.SAL, s.GRADE from emp e, salgrade s
	where e.EMPNO = 7844 and e.sal between s.LOSAL and s.HISAL;
 
# Turner(7844)의 급여 등급에 속한 급여를 받는 사원 정보와 소속부서 정보
# grade 3에 속한 다른 사원정보와 소속 부서정보

select e.EMPNO, e.ENAME, e.sal, e.DEPTNO, d.DNAME, s.GRADE 
	from emp e, dept d, salgrade s
	where e.DEPTNO = d.DEPTNO
    and (e.sal between s.LOSAL and s.HISAL)
    and s.grade = (select s.GRADE  # turner 의 급여등급인 3을 구하기 위한 sub-query
		from emp e, salgrade s
        where e.empno = 7844 and e.sal between s.losal and s.hisal)
	order by e.sal;
    
# 전체 사원의 평균 급여보다 같거나 많은 급여를 받는 사원정보와 부서정보 
select avg(sal) as "평균급여"from emp; -- 2073.214286
select e.EMPNO, e.ENAME, e.SAL, e.DEPTNO, d.DNAME
	FROM emp e 
    join dept d on e.deptno = d.deptno
	where e.sal >= (select avg(sal) from emp where d.deptno = e.deptno)
    order by e.sal;





    
