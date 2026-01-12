CREATE TABLE cofog_dwelling (
    sigun_gu VARCHAR(50),
    Sewerage
    Water_Supply
    Number_of_Houses
    Resident_Registered
    Number
    Number_of_births INT(10) NOT NULL,
    Total_Fertility_Rate FLOAT(10,2) NOT NULL,
    Sex_ratio FLOAT(5,1) NOT NULL,
    Population_Growth_Rate FLOAT(10,2) NOT NULL,
    Resident_Registered_Population INT(10) NOT NULL,
    Public_Administration_Budget FLOAT(10,2) NOT NULL
);
-- CSV 데이터 로드
load data infile 'C:/MySQL/8.4/Data/Uploads/extinct/administration.csv' into table cofog_administration fields terminated by ',';




Number_of_Houses