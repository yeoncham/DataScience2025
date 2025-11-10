# 갑상선 암 진단 이진 분류 모델  
**팀명:** Mango

**팀구성:** 김종민, 김석민,김수민, 윤세혁, 홍종효   

**과목:** 데이터 사이언스

---

## 프로젝트 개요  
- 이 프로젝트는 **의료 데이터와 머신러닝 기법을 활용하여 갑상선 암(Thyroid Cancer)의 악성·양성 여부를 예측**하기 위한 연구입니다.  
- 기존 초음파 및 조직검사 기반 진단은 **침습적**이며 **시간이 많이 소요되는 문제**가 존재합니다.  
- 이에 따라, **간단한 임상 정보만으로 암 진단 여부를 예측할 수 있는 머신러닝 모델**을 구현하는 것을 목표로 합니다.  

---

## 연구 목표 및 가설  

**연구 목표:**  
갑상선 암 여부(Cancer)를 예측할 수 있는 이진 분류 모델을 개발하고  
통계적 가설 검증을 통해 암 발병에 유의미한 영향을 미치는 파생변수를 찾아낸다.  

**가설:**  
1. **T3·T4 호르몬 수치**는 갑상선 암 발병과 관련이 있다.  
2. **인종(Race)** 과 **T3 호르몬 수치**는 암 발병과 관련이 있다.  
3. **가족력(Family_Background)** 과 **요오드 결핍(Iodine_Deficiency)** 은 암 발병과 관련이 있다.  

---

## 데이터 개요  

- **출처:** 의료 데이터셋 (DACON)  
- **데이터 구성:** 15개 피처 + 1개 타깃 (Cancer)  
- **학습 형태:** 분류(Classification)  

| Feature | 설명 |
|:--|:--|
| ID | 샘플별 고유 ID |
| Age | 환자의 나이 |
| Gender | 성별 |
| Country | 국적 |
| Race | 인종 |
| Family_Background | 가족력 여부 |
| Radiation_History | 방사선 노출 이력 |
| Iodine_Deficiency | 요오드 결핍 여부 |
| Smoke | 흡연 여부 |
| Weight_Risk | 체중 관련 위험도 |
| Diabetes | 당뇨병 여부 |
| Nodule_Size | 갑상선 결절 크기 |
| TSH_Result | TSH 호르몬 검사 결과 |
| T4_Result | T4 호르몬 검사 결과 |
| T3_Result | T3 호르몬 검사 결과 |
| Cancer | 라벨 (0=양성, 1=악성) |

---

## 데이터 전처리  

1. **범주형 변수 변환**  
   - 수치형 데이터를 범주형 데이터로 변환  
   ```python
   def age_category(self, x): # 예시
    if x < 30: return "Young"
    elif x < 50: return "Middle"
    elif x < 65: return "Senior"
    else: return "Elderly"
   ```
2. **결측치 및 이상치 확인**  
   - boxplot & describe()로 이상치 탐색  
   - 연속형 변수의 분포를 시각화하여 임상 기준 임계값(Thresholds) 설정  

3. **파생 변수 생성 (Feature Engineering)**  
   - 가설들을 토대로 파생 변수 생성

---

## 특성 중요도 분석  

- 트리 기반 모델을 사용하여 **각 변수의 중요도(Feature Importance)** 평가  
- 중요도가 낮은 특성(T3/T4/TSH 등)은 제거 → 과적합 방지  

| Feature | Importance |
|:--|--:|
| Country | 0.315 |
| Race | 0.146 |
| Family_Background | 0.112 |
| Weight_Risk | 0.095 |
| Smoke | 0.092 |
| Diabetes | 0.089 |
| Iodine_Deficiency | 0.072 |
| **Age·TSH/T3/T4** | 0.002~0.004 (낮음) |

---

## 모델링 및 실험  

1. **베이스라인 모델 설정**  
   - Random Forest Classifier 기반 이진 분류  
   - **StratifiedKFold** 로 10-Fold 교차검증 진행  

2. **Feature Engineering 적용 전·후 성능 비교**  

   | 구분 | 정확도(Accuracy) |
   |:--|:--:|
   | 파생 변수 추가 전 | 0.838 |
   | **파생 변수 추가 후** | **0.857 (+1.9%p)** |

3. **GridSearchCV 하이퍼파라미터 튜닝**  
   - 탐색 파라미터
   ```python
     param_grid = {
    'n_estimators': [100, 200], 
    'max_depth': [10, None], 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4], 
    'max_features': ['sqrt', 'log2'], 
    'class_weight': ['balanced', None] 
    }
   ``` 

   - 최적 조합 적용 후 정확도 **0.863 (+0.6%p)** 달성  

5. **최종 모델 평가 (전체 데이터셋 기준)**  

   | 모델 | 정확도 | 개선 폭 |
   |:--|:--:|:--:|
   | Base | 0.836 | - |
   | Feature Eng. + Tuning | **0.860** | +2.3%p |

---

## 통계적 검정 결과 (Chi-Square Test)

| 가설 | χ² 통계량 | p-val | 결론 |
|:--|:--|:--|:--|
| T3 ↔ Cancer | 1.397 | 0.497 | 관련 없음 |
| T4 ↔ Cancer | 1.639 | 0.441 | 관련 없음 |
| T3+T4 ↔ Cancer | 19.103 | 0.014 | 유의미 |
| Race ↔ Cancer | 1384.24 | 0.000 | 유의미 |
| Iodine deficiency ↔ Cancer | 556.36 | 0.000 | 유의미 |
| Family history ↔ Cancer | 1000.58 | 0.000 | 유의미 |
| Iodine deficiency+Family history ↔ Cancer | 1554.4941 | 0.000 | 유의미 |

---

## 결론 및 시사점  

- **복합적 특성 조합이 모델 성능 향상에 효과적일 수 있음**  
- 초음파나 조직검사 없이, **임상 정보만으로도 기본적인 암 예측 가능성 확인**  
- 통계 검정 및 특성 중요도 분석을 통해 **임상적으로 유의미한 가능성을 갖고 있는 변수들**을 도출함    

---

## 역할 분담  

| 이름 | 역할 |
|:--:|:--|
| **김종민** | 조장, 가설 검증, 데이터 분석, 회의록 작성, 최종 PPT 제작 및 발표 |
| **김석민** | 가설 검증, 데이터 분석, 회의록 및 보고서 검토, 최종 PPT 제작 및 발표 |
| **김수민** | 가설 검증, 데이터 분석, PPT 검토, 최종 PPT 제작 및 발표 |
| **윤세혁** | 가설 검증, 데이터 분석 및 시각화, 전처리, 자료조사, 제안 PPT 제작 및 발표 |
| **홍종효** | 가설 검증, 데이터 시각화, 보고서 작성, 제안 PPT 제작 및 발표 |

---

## 프로젝트 회고  

- 의료 도메인 지식 부족으로 **결과 해석에 한계**가 있었으나  
  **통계 검정을 통해 유의미한 변수 도출**과 성능 향상 경험을 얻음.  
- **SMOTE 기법**으로 클래스 불균형 문제 해결,  
  **Random Forest 앙상블** 기반 학습으로 일반화 성능 향상.  
- 단순 모델링보다 **문제 정의 및 방향 설정의 중요성**을 체득함.  

---

## 주요 지표 요약  

| 항목 | 정확도 | 개선 폭 | 비고 |
|:--|:--:|:--:|:--|
| Baseline | 0.838 | - | 파생 전 |
| Feature Added | 0.857 | +1.9%p | Feature Eng. 적용 |
| Tuned Model | 0.863 | +0.6%p | 하이퍼파라미터 최적화 |
| Final Test Accuracy | 0.860 | +2.3%p | 전체 데이터 기준 |

---

## 결론 요약  

> “의료 데이터에서도 **Feature Engineering**과  
> **적절한 모델 선택**은 예측 정확도 향상에 핵심적인 역할을 한다.”
