use haksa;
desc post;
select count(*) from post where post_no = '16269'; -- 360
# select * from post where post_no = '16269'; 이렇게 써도 관계없음.

# create index statement is a powerful tool for optimizing the performance of db
# 즉 post_no 컬럼을 사용하는 검색쿼리의 기능을 향상
# 파일의 크기가 방대할 때 유용함
# 둘다 사용가능.
create index idx_post_no on post(post_no);

CREATE INDEX CREATEIX ON post (post_no);

# 그러나 행을 삽입하거나 업데이트할 때 db는 index도 업데이트를 해주어야함 index유지관리 위함.
# index를 신중하게(judiciously)사용하여 빠른 쿼리의 이점과 업데이트/삽입 성능 및 디스크 공간
# (인덱스는 실제 테이블 데이터와 별도로 저장) 사용에 대한 잠재적 비용 사이의 균형을 맞추는 것이 중요합니다
-- drop index post_sk_post_no on post;

