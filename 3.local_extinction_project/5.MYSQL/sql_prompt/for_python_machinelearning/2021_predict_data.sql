CREATE VIEW test_data AS
SELECT
    ext.sigun_gu,
    ext.extinction_point
FROM
    extinct ext
WHERE 
	ext.sigun_gu LIKE '%2021%';

