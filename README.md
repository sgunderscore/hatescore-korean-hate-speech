<img src = "https://github.com/sgunderscore/hatescore-korean-hate-speech/blob/main/rsc/zoomed_HateScore_transparent.png" width="75%">  

**HateScore : Human-in-the-Loop Korean Multi-label Hate Speech Dataset (feat. Smilegate Unsmile Dataset)**  

- 본 데이터는 SmilegateAI에서 공개한 1.8만 건의 [Korean Unsmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)의 데이터 및 분류 모델을 기반으로 합니다.
- 본 데이터의 크기는 약 1만 건으로, Unsmile base model을 활용해 HITL(Human-in-the-Loop) 방식으로 태깅된 8천 건과 위키피디아에서 수집한 혐오 이슈 관련 중립 문장 2.2천 건으로 구성됩니다.
- 언더스코어는 Smilegate Korean Unsmile Dataset의 개발과 레이블링 작업을 진행했으며, 본 HateScore 데이터셋 역시 당시의 참여 인원 및 레이블링 기준을 동일하게 유지했습니다.
- 데이터 수집 및 레이블링 방식, 혐오발언 유형 선정 기준 등 보다 상세한 정보는 [이 논문](https://ojs.aaai.org/index.php/ICWSM/article/view/18059)에서 확인하실 수 있습니다.

## 1. 예제 (KcBERT-base)
문장 | 여성 | 성소수자 | 남성 | 인종/국적 | 지역 | 종교 | 연령 |
--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
여자는 집에서 애나 봐라 | **0.86** | 0.01 | 0.03 | 0.03 | 0.01 | 0.01 | 0.01 |
좆족은 21세기의 홍어다 | 0.03 | 0.02 | 0.03 | **0.68** | **0.89** | 0.04 | 0.03 | 
너는 전라도 사람이니? | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
상폐 한남들 다 재기하라고 | 0.09 | 0.02 | **0.88** | 0.05 | 0.05 | 0.04 | **0.55** | 
도심에서 변태성욕 축제라니 말세 | 0.06 | **0.79** | 0.02 | 0.01 | 0.13 | 0.01 | 0.01 |
쉰내나는 태극기들 틀니 압수 | 0.07 | 0.03 | 0.06 | 0.09 | 0.03 | 0.05 | **0.94** |
개독이나 짱깨나 거기서 거기 | 0.05 | 0.03 | 0.02 | **0.84** | 0.09 | **0.92** | 0.05 |
저 친구는 필리핀 출신이다 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
쿵쾅이들도 필리핀 그지는 싫지? | **0.74** | 0.01 | 0.04 | **0.71** | 0.02 | 0.01 | 0.01 |

## 2. 데이터셋 및 모델 성능 비교 (LRAP, Label Ranking Average Precision)
모델명 | Unsmile | Unsmile+HateScore |
--- | :---: | :---: |
KcBERT-base | .886 | .914 |
**KcBERT-large** | **.892** | **.919** |
KcELECTRA-large | .884 | .912 |

## 3. 인용 방식
```
Kang, TaeYoung, et al. "Korean Online Hate Speech Dataset for Multilabel Classification : How Can Social Science Aid Developing Better Hate Speech Dataset?" arXiv preprint arXiv:0000.00000 (2022).
```
```
@misc{SmilegateAI2022KoreanUnSmileDataset,
  title         = {HateScore: Human-in-the-Loop Korean Multi-label Hate Speech Dataset},
  author        = {Underscore},
  year          = {2022},
  howpublished  = {\url{https://github.com/sgunderscore/hatescore-korean-hate-speech}},
}
```

## 4.권장사항
- 

## 5.FAQ
- 

## 6. 프로젝트 참여 연구원
- 강태영 (KAIST 경영공학 석사)
- 권은낭 (연세대학교 사회학 박사과정)
- 김학준 (서울대학교 사회학 석사)
- 남영은 (Ph.D. candidate in Sociology at Purdue University)
- 서정규 (Ph.D. candidate in Political Science at University of Houston)
- 송준모 (연세대학교 사회학 박사과정)
- 이준범 (서울대학교 데이터사이언스학 석사)

**레이블링 보조참여자**
- 강태영 (KAIST 경영공학 석사)
- 권은낭 (연세대학교 사회학 박사과정)
- 김학준 (서울대학교 사회학 석사)
- 남영은 (Ph.D. candidate in Sociology at Purdue University)
- 서정규 (Ph.D. candidate in Political Science at University of Houston)
- 송준모 (연세대학교 사회학 박사과정)
- 이준범 (서울대학교 데이터사이언스학 석사)
