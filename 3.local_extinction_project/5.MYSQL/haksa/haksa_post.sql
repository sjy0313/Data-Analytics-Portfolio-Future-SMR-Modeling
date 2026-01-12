use haksa;

show tables;

show variables like 'secure_file_priv';

load data infile 'C:/MySQL/8.4/Data/Uploads/zipcode_DB/Gyeonggi-do.txt' into table post fields terminated by '|';

select count(*) from post; -- 1026153

select * from post;
SELECT * FROM post WHERE road_name LIKE '정조로%';
# 학원 위치
select * from post where post_no = '16269' and building_bon = 940 and building_boo = 1;


select post_no, management_no from post;
