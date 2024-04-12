#### dacon 학습 플랫폼 구독자 예측 AI 해커톤

- 링크 : https://dacon.io/competitions/open/236179/overview/description
- 사용한 모델 : DecisionTreeClassifier모델, Label인코딩, log1p스케일링
- 분류모델 세가지를 사용, 하이퍼파라미터 설정을 하자 점수가 떨어져서 생략 
- XGBClassifier --> 0.511469062751114
- RandomForestClassifier --> 0.46178450513381275
- DecisionTreeClassifier  --> 0.5270466552850682 
- 가장 점수가 높은 의사결정나무모델을 선택
