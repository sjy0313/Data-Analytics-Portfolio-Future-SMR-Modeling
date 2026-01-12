
use extinction;
CREATE TABLE extinct (
    sigun_gu VARCHAR(20),           -- 지역 이름
    extinction_index FLOAT(25) NOT NULL,  -- 소멸 지수 
    extinction_grade VARCHAR(3) NOT NULL  -- 소멸 등급 
);
show variables like 'secure_file_priv';
load data infile 'C:/MySQL/8.4/Data/Uploads/extinct/grade.csv' into table extinct fields terminated by ',';


-- 교육테이블 생성
CREATE TABLE cofog_education (
    sigun_gu VARCHAR(20),           -- 지역 이름
    student_per_teacher_kin INT(10) NOT NULL,  -- 교사 당 유치원 학생 수
    teacher_per_student_pri INT(10) NOT NULL,  -- 교사 당 초등학생 수
    teacher_per_student_mid INT(10) NOT NULL,  -- 교사 당 중학생 수
    teacher_per_student_high INT(10) NOT NULL, -- 교사 당 고등학생 수
    student_per_class_kin INT(10) NOT NULL,    -- 학급 당 유치원 학생 수
    student_per_class_pri INT(10) NOT NULL,    -- 학급 당 초등학생 수
    student_per_class_mid INT(10) NOT NULL,    -- 학급 당 중학생 수
    student_per_class_high INT(10) NOT NULL,   -- 학급 당 고등학생 수
    extra_curr_school INT(10) NOT NULL,        -- 학교 부속 특별 활동 프로그램 수
    extra_curr_lifelong INT(10),               -- 평생 교육 관련 프로그램 수
    student_per_extra_curr INT(10) NOT NULL,   -- 사설학원 당 학생 수
    kin_stu INT(10) NOT NULL,                  -- 유치원 학생 수
    pri_stu INT(10) NOT NULL                   -- 초등학생 수
);

-- CSV 데이터 로드
LOAD DATA INFILE 'C:/MySQL/8.4/Data/Uploads/extinct/education.csv' 
INTO TABLE cofog_education
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';

-- IGNORE 1 ROWS;  -- 첫 번째 행이 헤더라면 무시

-- 보건 테이블 생성
-- 테이블 생성
CREATE TABLE cofog_health (
    sigun_gu VARCHAR(20),         
    general_hospital INT(10) NOT NULL,  
    hospital INT(10) NOT NULL,
    clinic INT(10) NOT NULL,
    dental_clinic INT(10) NOT NULL,
    medicine_hospital INT(10) NOT NULL,
    koreanmedical_clinic int(10) not null,
    beds_per_thousandpeople FLOAT(5,1) NOT NULL,  -- 소수점 자리 포함
    total_beds FLOAT(10,1) NOT NULL  -- 총 침대 수, FLOAT으로 수정
);

-- CSV 데이터 로드
LOAD DATA INFILE 'C:/MySQL/8.4/Data/Uploads/extinct/health.csv' 
INTO TABLE cofog_health
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'; 
-- IGNORE 1 ROWS;  -- 첫 번째 행이 헤더라면 무시

-- 주거행정 테이블 생성
-- 테이블 생성
CREATE TABLE cofog_dwellingadministration (

    sigun_gu VARCHAR(20),
    Number_of_Births
    Total Fertility Rate
    Sex Ratio

-- CSV 데이터 로드
LOAD DATA INFILE 'C:/MySQL/8.4/Data/Uploads/extinct/health.csv' 
INTO TABLE health
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n' 
IGNORE 1 ROWS;  -- 첫 번째 행이 헤더라면 무시





