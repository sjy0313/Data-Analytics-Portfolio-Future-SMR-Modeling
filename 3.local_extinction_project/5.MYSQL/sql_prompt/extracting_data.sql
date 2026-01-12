SELECT 
    e.*,  -- 교육테이블의 모든 데이터
    h.*,  -- 보건테이블의 모든 데이터
    dw.*, -- 주거행정테이블의 모든 데이터
    ext.extinction_grade
   
FROM 
    cofog_education e
LEFT JOIN 
    extinct ext ON e.sigun_gu = ext.sigun_gu
LEFT JOIN 
    cofog_dwellingadministration dw ON e.sigun_gu = dw.sigun_gu
LEFT JOIN 
    cofog_health h ON e.sigun_gu = h.sigun_gu;
    
