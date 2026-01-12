use infrastructure;
select * from cheap;

# views 테이블 School_level_c /  School_level_f 생성 : 
create or replace view School_level_c as 
	select c.district, c.Primary_School, c.Middle_School, c.High_School
		from cheap c, fancy f
		where c.district = f.district;
        
select c.district, c.Primary_School from cheap c order by Primary_School asc;

SELECT AVG(Primary_School) AS average_value FROM cheap;


create or replace view School_level_f as 
	select f.district, f.Primary_School, f.Middle_School, f.High_School
		from cheap c, fancy f
		where c.district = f.district;
        


-- 평균 구하기
SELECT AVG(Primary_School) AS average_value
FROM fancy;

SELECT 
    AVG(Primary_School) AS avg_Primary,
    AVG(Middle_School) AS avg_Middle,
    AVG(High_School) AS avg_High
FROM fancy;




select c.district, c.Primary_School, f.Primary_School from School_level_c c, School_level_f f
	where c.district = f.district and 


select * from School_level;


SELECT Primary_School from School_level ORDER BY Primary_School ASC;


select sl.Primary_School, c.district 
	from School_level sl , cheap c 
    where sl.Primary_School >= any(select sl.Primary_School from School_level);
    

SELECT * FROM cheap ORDER BY Subway asc;
SELECT * FROM cheap ORDER BY Primary_School  asc;
SELECT * FROM cheap ORDER BY Middle_School asc;
SELECT * FROM cheap ORDER BY High_School asc;
SELECT * FROM cheap ORDER BY General_Hospital asc;
SELECT * FROM cheap ORDER BY Supermarket asc;
SELECT * FROM cheap ORDER BY Park asc;



