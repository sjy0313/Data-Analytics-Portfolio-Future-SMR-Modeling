-- coef>0.05 & P-value < 0.05 인 변수 11개 리스트 DBMS에서 추출:
CREATE VIEW integrated_data AS
SELECT
    e.sigun_gu,
    e.teacher_per_student_mid,
    e.teacher_per_student_high,
    e.student_per_class_kin,
    e.extra_curr_lifelong,
    h.medicine_hospital,
    h.total_beds,
    s.Perceived_Health_Status_Rate,
    a.Sex_Ratio,
    a.Population_Growth_Rate,
    a.Resident_Registered_Population,
    d.Sewerage,
    ext.extinction_point
FROM 
    cofog_education e
LEFT JOIN 
    extinct ext ON e.sigun_gu = ext.sigun_gu
LEFT JOIN 
    cofog_health h ON e.sigun_gu = h.sigun_gu
LEFT JOIN
    cofog_socialprotection s ON e.sigun_gu = s.sigun_gu
LEFT JOIN
    cofog_administration a ON e.sigun_gu = a.sigun_gu   
LEFT JOIN
    cofog_dwelling d ON e.sigun_gu = d.sigun_gu   
WHERE 
    e.sigun_gu LIKE '%2015%' 
    OR e.sigun_gu LIKE '%2016%' 
    OR e.sigun_gu LIKE '%2017%' 
    OR e.sigun_gu LIKE '%2018%' 
    OR e.sigun_gu LIKE '%2019%' 
    OR e.sigun_gu LIKE '%2020%'
    OR e.sigun_gu LIKE '%2021%';

