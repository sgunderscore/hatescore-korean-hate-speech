<img src = "https://github.com/sgunderscore/hatescore-korean-hate-speech/blob/main/rsc/zoomed_HateScore_transparent.png" width="75%">  

**HateScore : Human-in-the-Loop Korean Multi-label Hate Speech Dataset (feat. [Smilegate Unsmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset))**  

- 본 데이터는 SmilegateAI에서 공개한 1.8만 건의 [Korean Unsmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)의 분류 모델을 기반으로 합니다.
- 본 데이터의 크기는 약 1만 건으로, Unsmile base model을 활용해 HITL(Human-in-the-Loop) 방식으로 태깅된 8천 건과 위키피디아에서 수집한 혐오 이슈 관련 중립 문장 2.2천 건으로 구성됩니다.
- 언더스코어는 Smilegate Korean Unsmile Dataset의 개발과 레이블링 작업을 진행했으며, 본 HateScore 데이터셋 역시 당시의 참여 인원 및 레이블링 기준을 동일하게 유지했습니다. 다만 이하의 *4.권장사항* 및 *5.FAQ* 항목은 Smilegate와는 무관한 언더스코어의 독립적인 의견입니다.
- 데이터 수집 및 레이블링 방식, 혐오발언 유형 선정 기준 등 보다 상세한 정보는 [이 논문](https://ojs.aaai.org/index.php/ICWSM/article/view/18059)에서 확인하실 수 있습니다.

## 1. 예제 (KcBERT-base)
문장 | 여성 | 성소수자 | 남성 | 인종 | 지역 | 종교 | 연령 |
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

## 2. 데이터셋 비교
**LRAP(Label Ranking Average Precision)**
모델명 | Unsmile | Unsmile+HateScore |
--- | :---: | :---: |
KcBERT-base | .886 | .914 |
**KcBERT-large** | **.892** | **.919** |
KcELECTRA-large | .884 | .912 |
  
**Base Model 기준 비교 예제**
혐오발언 분류 확률 | Unsmile | Unsmile+HateScore |
--- | :---: | :---: |
저 사람 중국인이네 | **0.87** | 0.20 |
너 페미니스트니? | 0.03 | 0.01 |
동성혼은 논쟁적이지 | 0.35 | 0.01 |
무슬림을 다 죽인다고?\* | **0.84** | **0.76** |

\*두 모델 모두 오분류한 사례

## 3. 인용 방식
**논문**
```
Kang, TaeYoung, et al. "Korean Online Hate Speech Dataset for Multilabel Classification : How Can Social Science Aid Developing Better Hate Speech Dataset?" arXiv preprint arXiv:0000.00000 (2022).
```
**깃헙(Github)**
```
@misc{Underscore2022KoreanHateScoreDataset,
  title         = {HateScore: Human-in-the-Loop Korean Multi-label Hate Speech Dataset},
  author        = {Underscore},
  year          = {2022},
  howpublished  = {\url{https://github.com/sgunderscore/hatescore-korean-hate-speech}},
}
```

## 4. 권장사항
- HateScore는 중립 문장을 포함하고 2021년도 하반기 이후의 댓글 데이터 역시 포함한다는 강점이 있으나, 3인의 다수결 투표로 최종 레이블을 결정한 Unsmile과 달리 Human-in-the-loop 방식으로 '모델의 분류 확률'과 '연구원 한 명의 의견'의 두 가지 값만을 활용했습니다. 이에 응용 시에는 HateScore와 Unsmile 데이터를 함께 학습하는 것이 좋습니다.
- HateScore는 온라인 '댓글' 데이터만을 다룹니다. 그렇기에 "아까 학력 인증한 연베대게이다. 학점 ㅁㅌㅊ?"나 "페미니스트들의 실체.png"와 같은 웹 커뮤니티 제목 텍스트에 모델을 적용할 경우, 혐오발언 여부를 오분류할 가능성이 높습니다. 이에 댓글 텍스트에만 적용하는 것을 권장합니다.
- 각 혐오발언 카테고리를 독립적으로 간주하지 않고, 멀티레이블(multi-label) 방식의 분류기 개발을 권장합니다.
- 입력한 텍스트가 혐오발언 카테고리에 해당되지 않더라도 '단순 악플'에는 해당될 수 있으니, 멀티레이블 분류기에서 주어진 댓글의 공격성을 단순히 "1-(Clean 분류 확률)"만으로 계산하는 것은 부적합합니다.

## 5. FAQ
- 혐오발언 유형은 어떻게 되나요?  
→ 여성, 성소수자, 지역, 인종/국적, 종교, 연령, 남성의 7가지 이며 기타 혐오발언, 단순 악플, 일반 댓글(clean)의 3가지 유형이 추가로 제공됩니다.
- 혐오발언 카테고리 별 데이터 수는 중요도와 비례하나요?  
- → 아니요. 그렇지 않습니다.
- 기타 혐오발언의 경우 어떤 내용을 포함하나요?  
- → 외모에 대한 조롱, 특정 직업군에 대한 비하, 장애 희화화 등 위 7가지 대분류에 포함되지 않는 혐오발언들이 이에 해당됩니다.
- 기타 혐오발언은 7가지 유형보다 중요하지 않은가요?  
- → 아니요. 그렇지 않습니다. 예산 및 시간의 제약으로 인해 우리 사회에 존재하는 모든 유형의 혐오발언에 대해 충분한 수의 데이터셋을 개발할 수는 없었습니다.
- 왜 '남성'이 혐오발언의 유형 중 한 가지에 포함되어 있나요?  
→ 좁은 의미에서의 혐오발언은 사회적 소수자(social minority)에 대한 적대적 발언을 지칭합니다.

## 6. 프로젝트 참여 연구원
**혐오발언 유형 설정, 레이블링 매뉴얼 수립, 모델 개발**
- 강태영 (KAIST 경영공학 석사)
- 권은낭 (연세대학교 사회학 박사과정)
- 김학준 (서울대학교 사회학 석사)
- 남영은 (Ph.D. candidate in Sociology at Purdue University)
- 서정규 (Ph.D. candidate in Political Science at University of Houston)
- 송준모 (연세대학교 사회학 박사과정)
- 이준범 (서울대학교 데이터사이언스학 석사)

**레이블링 보조참여자**
- 권혜윤 (서울대학교 인류학 석사)
- 박형준 (싱가포르국립대학교 심리학 석사)
- 이보미 (서강대학교 정치외교학 석사)
- 이성일 (서강대학교 사회학 석사)
- 지소연 (서강대학교 사회학 석사)
- 홍수민 (연세대학교 정치외교학 석사과정)
- 황지영 (서강대학교 사회학 석사과정)

## 7. 문의
master@underscore.kr
