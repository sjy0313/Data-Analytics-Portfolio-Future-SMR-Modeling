
CREATE VIEW test_data AS
SELECT
    ext.sigun_gu,
    ext.extinction_point
FROM
    extinct ext
WHERE 
	ext.sigun_gu LIKE '%2015%' 
    OR ext.sigun_gu LIKE '%2016%' 
    OR ext.sigun_gu LIKE '%2017%' 
    OR ext.sigun_gu LIKE '%2018%' 
    OR ext.sigun_gu LIKE '%2019%' 
    OR ext.sigun_gu LIKE '%2020%'
	OR ext.sigun_gu LIKE '%2021%';

