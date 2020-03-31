---
layout: post
title: "롤은 못하지만 딥러닝은 잘하지 - 1"
date: 2019-11-21
excerpt: "딥러닝으로 게임 리그 오브 레전드의 결과를 예측하려는 시도. 그리고 champ2vec"
tag:
- Development
- Deep Learning
- Machine Learning
- League of Legend
- Champ2Vec
- Pytorch

comments: true
---

제목은 어그로다. 난 딥러닝을 잘 하는건 아니지만 [난 드럼은 못치지만 편집은 잘하지 영상](https://youtu.be/tFKMUDp86mk)에서 제목을 차용했을 뿐이다. 다시 한 번 말하지만 난 딥잘알이 아니다. 물론 롤은 더 못한다. 솔랭 기준으로 고작 실2따리다. 그래도 롤 중계를 보는 것은 즐겨해서 전부터 롤을 보면서 느꼈던 부분이고 특히 2019년 롤드컵을 보면서 증폭되었던 의문이 있다. 바로 '밴픽만으로 유불리를 얼마나 따질 수 있느냐' 인데 사실 해설자들이 '난이도가 어려운 조합'이라고 하면 밴픽이 망했다고 하더라. 그러나 얼마나 수치적으로 불리한지는 모르고 또한 밴픽보다는 그런 밴픽의 의미를 살리는 인게임 플레이가 훨씬 더 중요한 것도 맞다. 그럼에도 불구하고 2019년 롤드컵 FPX와 GAM의 조별리그 2차전을 보면 밴픽이 경기에서 얼마나 치명적으로 작용할 때가 있다는 것도 깨달을 수 있다. 도인비 선수가 레넥톤, 녹턴, 클레드라는 돌진 3AD를 카운터치기 위해 마지막 픽으로 미드 말파이트라는 일반적인 예상을 뛰어넘는 픽으로 FPX는 해당 경기를 쉽게 가져간다. 이런 식으로 어떤 조합에 따라 승률을 계산할 수 있다면 '9개의 픽을 보고 앞서 나온 9개 픽과 밴된 10개 픽을 제외한 모든 챔피언을 딥러닝 네트워크에 넣고 승률을 계산했을 때 승률이 가장 높은 챔피언이 좋지 않을까?'라는 결론에 도달하게 되었다. 또한 더 나아가서 관전자 입장에서 본 인게임 정보(글로벌 골드, 레벨, 챔피언들이 보유하고 있는 아이템 등)를 취합하여 딥러닝을 학습하고 test할 수 있는 환경을 만든다면 바둑처럼 실시간 승률 예측 또한 할 수 있을 것이다.


### 비슷한 생각을 했던 사람들을 찾아보자
이런 생각을 한 사람이 분명 내가 처음이 아니었을 것이다. 검색을 해보자 아니나 다를까 비슷한 생각을 한 분들의 블로그가 나왔다. [League of Legends: Predicting Wins In Champion Select With Machine Learning](https://hackernoon.com/league-of-legends-predicting-wins-in-champion-select-with-machine-learning-6496523a7ea7)라는 제목의 블로그에서 Neural Network를 포함하여 SVM, random forest 등의 고전 머신러닝 기법까지 활용하였다. 이 블로그는 데이터의 수가 1400개 정도로 적고 각각의 챔피언을 one-hot으로 했는지 embedding을 했는지 정확히 표기되어 있지 않았다. 챔피언의 조합만을 input으로 했을 때 솔로 랭크 게임의 승률을 최대 60%가량의 확률로 예측했다.

다른 [Deep LoL](https://ckcks12.com/dev-reviews/deep-lol-review/)이라는 프로젝트도 찾을 수 있었다. 이 포스트에서는 솔로 랭크 데이터를 취합하는 방법부터 시작하여 모델의 input을 결정하고 CNN을 사용하여 챔피언을 추천하는 단계까지 나아갔다. 마지막 부분에서는 야스오와의 조합으로 말파이트, 자크. 알리스타를 추천하는 모습 또한 보여준다. 해당 프로젝트에서 한 가지 특이한 점은 챔피언을 임베딩하여 모델 혹은 알고리즘에 적용하는 것 보다 one-hot으로 된 챔피언의 데이터를 적용하는 것이 더 나은 결과를 나타내는 것이었다. 포스트에서 어떤 방식으로 임베딩을 진행하였는지는 나오지 않았지만 조금 의아한 느낌이 들었다.

[챔피언을 embedding 한 프로젝트](https://medium.com/@yuan_tian/predicting-league-of-legends-match-outcome-with-embeddings-and-deep-learning-b7d1446c710f)또한 찾을 수 있었다. 해당 글은 champion을 임베딩 한 후 그래도 그 값을 mlp에 적용하여 마지막 regression으로 승률을 예측한 듯 한데 마지막 결론에서 기존 54%였던 정확도가 66%정도로 상승했다고 한다. 그러나 이 포스트에서는 데이터 셋이 솔로 랭크 데이터가 아니라 2015년~2018년 전 셰계 롤 리그의 밴픽을 담고 있어 리메이크, 패치 등으로 인한 챔피언의 변화를 포착할 수 없어 데이터의 통일성이 비교적 떨어진다고 볼 수 있다.

### 첫 번째 단계 champ2vec
딥러닝을 어떤 모델에 넣든 넣을 방법이 필요하다. 예를 들어 위의 두 번째 블로그에 언급된 one-hot 벡터와 같은 방법이다. 그러나 나는 각각의 챔피언의 특성을 녹아낸 벡터를 모델에 삽입하는 것이 더 나은 결과를 만들어내지 않을까 생각했다. 따라서 내놓은 결론이 NLP의 word2vec 기법을 적용하여 champ2vec을 사용하기로 했다.

<img src="http://mblogthumb3.phinf.naver.net/MjAxODEyMTlfMTcz/MDAxNTQ1MjA0MTk4NDQy.-lCTSpFhyK1yb6_e8FaFoZwZmMb_-rRZ04AnFmNijB4g.ID8x5cmkX8obTOxG8yoq39JRURXvKBPjbxY_z5M90bkg.JPEG.cine_play/707211_1532672215.jpg?type=w800" style="max-width: 50%; height: auto;">  

어떻게 word2vec을 LoL에 적용하는지 의문이 들 것이다. 내 생각은 이렇다. word2vec은 주변 단어와의 관계를 활용하여 자신의 벡터를 학습한다. 롤의 포지션을 봤을 때 탑, 정글, 미드, 바텀, 서포터가 있고 탑 챔프 하나의 벡터를 학습하기 위해서 나머지 포지션의 챔프와의 관계를 고려하여 학습하도록 한다면 각각의 챔피언의 특징이 벡터 안으로 들어갈 수 있을 것이다. 예를 들어 바텀 듀오를 봤을 때 카이사는 룰루, 소라카 등 유틸챔프 보다는 노틸러스, 레오나 등의 cc기 있는 챔피언과의 궁합이 좋다고 한다. 따라서 해당 방법으로 챔프벡터를 학습시켰을 때(특정 조합에서 같이 더 자주 쓰이므로)  룰루, 소라카의 챔피언 벡터의 유사도는 높게 나타날 것이고 룰루와 노틸러스, 레오나 등의 유사도는 낮게 나타날 것이다. 

word2vec은 NLP에서 단어의 전후관계를 파악하여 각각의 단어를 벡터로 바꾸어준다. [word2vec 알고리즘에 대한 자세한 설명](https://dreamgonfly.github.io/machine/learning,/natural/language/processing/2017/08/16/word2vec_explained.html)은 다음 블로그를 참조했다. 수학적 설명 이전까지는 쉽게 이해할 수 있는 글이라 직관적으로 이해할 수 있다. CBOW와 Skip-Gram 중 일반적으로 성능이 더 좋다고 알려져 있는 Skip-Gram을 사용하여 champ2vec을 구현할 것이다.

### gensim을 사용하여 구현하기

파이썬 3.6과 빠르고 안정적인 word2vec 라이브러리 gensim을 이용하여 구현을 진행하였고 데이터는 [다음 Kaggle 링크](https://www.kaggle.com/chuckephron/leagueoflegends)에서 2015~2018년의 전세계 롤 대회(LCK, LPL, LCS 등)의 픽과 경기기록을 모아 놓은 데이터를 사용했다. 사실 이 데이터는 정확하지 않다. 챔피언들의 패치, 리메이크 등을 거치며 특성이 바뀌는 경우가 많기 때문이지만 기본적인 챔피언의 특징이 변화되는 경우는 비교적 적기 때문에 학습은 될 거라는 희망을 품고 champ2vec을 실행했다.

먼저 구미호를 모티브로만든 챔피언 아리의 챔피언 벡터의 일부를 보자.

![아리 벡터](https://raw.githubusercontent.com/queez0405/queez0405.github.io/master/_posts/lol_project/Ahri_vector.png){:width="50%" height="50%"}  
챔피언의 임베딩 벡터 크기는 256으로 진행하였다. 이 숫자들만 봐서는 제대로 학습이 되었는지 아리송하다. 그러면 서포터와 그랩의 상징하는 챔프인 블리츠크랭크와 유사한 챔피언 벡터 10개를 골라보았다.
![블리츠 유사도](https://raw.githubusercontent.com/queez0405/queez0405.github.io/master/_posts/lol_project/Blitzcrank_similar.png)  
꽤나 고무적인 결과가 나왔다. 그랩형 서포터의 쌍두마차 쓰레쉬가 가장 벡터가 유사한 챔피언으로 나왔고 이외에도 라칸, 자이라, 브라움 등 서포터 포지션으로 주로 기용되는 챔피언들이 그 유사도가 높다. 그러나 이 모델이 완벽하다고 할 수 있을까? 우선 다른 그랩형 서포터인 노틸러스를 찾기가 힘들다(2018년 까지의 데이터이므로 파이크는 출시되기 전이다). 그러나 이전에 노틸러스는 서포터 보다는 탑에서 많이 사용되었으므로 그럴 수 있다.
![쉔 유사도](https://raw.githubusercontent.com/queez0405/queez0405.github.io/master/_posts/lol_project/Shen_similar.png)
완벽하지 않다는 것을 다음 챔피언인 쉔의 유사도 벡터를 통해 확실하게 확인할 수 있다. 쉔은 탱커 챔피언으로 설계되었으며 최근(2019년 기준)에는 서포터 포지션에서도 많이 기용되지만 우리의 데이터 셋에서는 대부분 탑솔러 포지션으로 기용되었다. 이 쉔가 가장 유사한 챔피언은 나르였다. 나르는 탱커가 아닌 딜탱인 브루저 포지션으로 약간의 의구심을 갖게 하는데 두 번째로 유사한 챔피언인 럼블이 더 이상하다. 럼블은 탱커가 아니라 딜러 챔피언이면서도 AD가 아닌 AP 위주의 딜로 구성된다. 이는 쉔이 대회에 나올 만한 기간에 럼블 또한 다른 챔피언들보다 성능이 좋아 동시에 나왔다는 것을 의미한다고 추정했다. 그래도 다른 유사 챔피언인 마오카이, 뽀삐 등은 탑에서 기용되는 탱커 챔피언으로 아주 엉터리까지는 아니다.

코드는 어느 정도 정리가 된 후 github에 업로드 할 예정이다.

### 다음 단계
좋지 않은 데이터로 champ2vec이 꽤나 괜찮은 성과를 냈다. 이제 본격적인 딥러닝 모델에 들어갈 준비가 되었다고 생각한다. 내가 하고 싶은 task는 10개의 챔피언 데이터가 차례대로 들어갔을 때 그 승률을 계산하는 것이 text classification task의 목표와 매우 유사하다. 이렇게 텍스트를 분류하는 방법 중 CNN text classification과 RNN text classification 방법을 적용하여 마지막에 softmax layer 대신 logistic regression을 삽입하여 승률을 계산하고 어느 모델이 성능이 좋은지 알아볼 것이다.

또한 라이엇 개발자 api를 통해 한 패치버전의 매치데이터만으로 학습시켰을 때 벡터가 어떻게 나타나는지 알아보고자 한다.