# 2015~2021 학습 데이터 / 예측 데이터 결합
CREATE VIEW integrate_data AS
SELECT
	tr.sigun_gu,
    tr.teacher_per_student_mid,
    tr.teacher_per_student_high,
    tr.student_per_class_kin,
    tr.extra_curr_lifelong,
    tr.medicine_hospital,
    tr.total_beds,
    tr.Perceived_Health_Status_Rate,
    tr.Sex_Ratio,
    tr.Population_Growth_Rate,
    tr.Resident_Registered_Population,
    tr.Sewerage,
    te.extinction_point

FROM 
    train_data tr
LEFT JOIN 
    test_data te ON tr.sigun_gu = te.sigun_gu

