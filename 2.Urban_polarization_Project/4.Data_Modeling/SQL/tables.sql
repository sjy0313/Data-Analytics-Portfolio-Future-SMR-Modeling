use infrastructure;
create table ulsan  (
	Io_Date varchar(10),         
	inflow_total int(7) Not null,       
	outflow_total int(7) Not null,             
	net_movement int(7) Not null,   
	primary key (Io_Date)
);


show variables like 'secure_file_priv';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Busan_mv.csv' into table Busan fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Seoul_mv.csv' into table seoul fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table daegu fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table chungbuk fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table chungnam fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table daejeon fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table gangwondo fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table gwangju fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table gyeongbuk fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table incheon fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table jeju fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table jeonbuk fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table jeonnam fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table sejoong fields terminated by ',';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Daegu_mv.csv' into table ulsan fields terminated by ',';


create table parameter(
	Pm_Date varchar(10),         
	Transaction_Price_index char(10) Not null,       
	President_support_rate char(10)Not null,             
	PIR_national_index char(10) Not null, 
	Consumer_price_index char(10) Not null, 
    realestate_tax_for_individual char(10) Not null, 
    total_numberof_transaction char(10) Not null, 
   mortgage_loan char(10) Not null, 
    Supply_demand_trend char(10) Not null,
	primary key (Pm_Date)
);
show variables like 'secure_file_priv';
load data infile 'C:/MySQL/8.4/Data/Uploads/csvfd/Parameter.csv' into table parameter fields terminated by ',';

