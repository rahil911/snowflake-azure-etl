/*****************************************
Course: IMT 577
Assignment: Module 6
Notes: Example of Dim_Date table creation
*****************************************/

--===================================================
-------------DIM_DATE
--==================================================
-- Create table script for Dimension DIM_DATE
create or replace table DIM_DATE (
	DATE_PKEY				number(9) PRIMARY KEY,
	DATE					date not null,
	FULL_DATE_DESC			varchar(64) not null,
	DAY_NUM_IN_WEEK			number(1) not null,
	DAY_NUM_IN_MONTH		number(2) not null,
	DAY_NUM_IN_YEAR			number(3) not null,
	DAY_NAME				varchar(10) not null,
	DAY_ABBREV				varchar(3) not null,
	WEEKDAY_IND				varchar(64) not null,
	US_HOLIDAY_IND			varchar(64) not null,
	COMPANY_HOLIDAY_IND     varchar(64) not null,
	MONTH_END_IND			varchar(64) not null,
	-- Additional fields omitted for brevity
	SQL_TIMESTAMP			timestamp_ntz,
	CURRENT_ROW_IND			char(1) default 'Y',
	EFFECTIVE_DATE			date default to_date(current_timestamp),
	EXPIRATION_DATE			date default To_date('9999-12-31') 
)
comment = 'Type 0 Dimension Table Housing Calendar and Fiscal Year Date Attributes'; 

-- Note: Actual insertion script contains complex date generation logic
-- Example: INSERT INTO DIM_DATE SELECT ... FROM (SELECT ... date generation logic); 