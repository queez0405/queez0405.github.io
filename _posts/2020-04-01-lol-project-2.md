---
layout: post
title: "롤은 못하지만 딥러닝은 잘하지 - 2"
date: 2020-04-01
excerpt: "딥러닝으로 게임 리그 오브 레전드의 결과를 예측하려는 시도. 그리고 champ2vec -2"
tag:
- Development
- Deep Learning
- Machine Learning
- League of Legend
- Champ2Vec
- Pytorch

comments: true
---

이 포스팅을 잊은 건 아니다. 다만 이 주제로는 딥러닝 코딩 공부로 시작했지만 이런 저런 조언을 듣고 논문 주제가 되었기 때문에 논문을 쓰느라 바쁜 시간을 보냈다.

논문에 대해서 얘기는 차치하고 내가 진행하고 삽질한 내용들과 champ2vec의 결과 그리고 그를 이용한 분류기의 설계와 그 결과에 대해 기술해보고자 한다.

### 고난과 역경의 데이터 모으기
첫 번째 데이터를 모으는 데만 족히 한 달은 걸렸다. 이때만 하더라도 사이드 프로젝트라고 생각해 퇴근 후와 주말을 이용한 시간을 사용해서 그런가 싶다. 어떻게 데이터를 모으냐가 가장 우선한 과제였는데 [라이엇 개발자 페이지](https://developer.riotgames.com/)에서 API key를 발급받아야  데이터 모으는 것을 시작할 수 있었다. 분명 상기한 페이지에는 키 발급에 10 영업일 정도 걸린다고 했으나 나의 경우에는 3주가 되어도 발급되지 않아서 개발자 디스코드로 들어가 디스코드의 봇에다가 키를 발급해달라고 징징거리니 하루이틀 뒤 프로젝트가 승인되고 키가 발급되었다.

또한 API 키를 사용해 서버에 데이터를 요청하는 것은 개인 용도라면 2분에 100개의 요청에 대한 제한이 있기에 이를 핸들링할 수 있는 라이브러리 중 [카시오페아](https://github.com/meraki-analytics/cassiopeia)를 선택했다. 예제도 충분하고 문서화도 잘 되어 있어 코딩이 비교적 편하다고 볼 수도 있다(물론 난 내 목적에 맞게 코딩할때도 엄청나게 삽질했다.). 그러나 메모리 누수문제와 데이터를 불러올 때 간헐적으로 내가 원인을 찾을 수 없는 에러가 발생하여 그 부분을 try except 문으로 처리해야 하는 점은 확실히 문제점이기에 처음부터 프로젝트를 진행한다고 하면 다른 라이브러리를 쓰는 것도 고민하지 않았을까 싶다.

그래서 한 달여가 지난 시점에 첫 번째 한국, 북미, 유럽 서부, 유럽 북동부 9.19 패치 데이터의 마스터 이상 플레이어의 경기에서 사용된 챔피언들의 조합을 기록한 csv 파일을 얻을 수 있었다. 9.19 패치를 선택한 이유는 별거없다. 2019년 롤드컵 패치 버전이라서 그렇다. 이 데이터로 champ2vec을 시행한 결과는 다음에 소개하겠다.

### champ2vec의 결과

![champ2vec 결과](https://raw.githubusercontent.com/queez0405/queez0405.github.io/master/_posts/lol_project/tsne.PNG){: width="50%}

위 그림은 champ2vec으로 만들어진 128차원의 챔피언 벡터 145개를 t-sne로 차원을 줄인 결과다. 탑, 정글, 미드, 바텀, 서폿 모두 아주 이쁘게 모여 있다. 9.19 버전 중 멀티포지션이 가능한 사일러스나 하이머딩거 등은 그 사이 어딘가에 위치해 있고 미드 챔피언을 보면 AD 암살자라고 할 수 있는 탈론, 제드, 키아나 그리고 조금 애매하지만 야스오까지 아주 가까운 위치에 몰있다는 굉장히 재미있는 결과가 나왔다. 이걸 봤을 때 임베딩 자체는 꽤 잘 된 것 아닌가?

### 임베딩 해서 뭘 할 거야?
임베딩을 한 기법이 뭔가? 원래 이름이 word2vec이니까 word를 임베딩했다. 그러면 word를 임베딩한 text classification task를 그대로 champ2vec에 적용하여 레드 팀과 블루 팀의 승패를 예측할 수 있을 것이라는 결론에 이르렀다. 그래서 사용하기로 한 방법이 [cnn task classification](https://arxiv.org/pdf/1408.5882.pdf)과 rnn task classification이다. rnn 방법 중에서는 [reference github](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch)에서 일반적으로 가장 성능이 좋은 bi-lstm을 사용했다.

### 결과는 시궁창

![실험 결과](https://raw.githubusercontent.com/queez0405/queez0405.github.io/master/_posts/lol_project/pre_accuracy.PNG){: width="100%}

하. 한숨이 나오는 결과다. multi hot, cnn, rnn 모두 51%의 accuracy를 넘지 못했다. lol이라는 게임은 단순 챔피언의 조합만 가지고는 승패를 예측할 수 없다는 결론에 이르렀다. 또 한편으로는 꽤나 잘 만든 게임이라는 것도.

### 하고싶은 것들
사실 위의 실험 결과 그림만 봤을 때 latex 표를 떼어다 왔다는 것을 눈치챈 사람이 있을 것이다. 내가 논문으로 준비하고 있었던 내용이 맞으나 위의 실험 결과를 보고 나가리 된 것이다. 물론 지금은 조금 다른 주제로 피벗하여 계속 논문을 쓰고 있다.

또 하나 하고 싶은 연구는 한 팀을 조합하고 있는 5개의 챔피언의 라인 (탑, 정글, 미드, 바텀, 서폿)을 맞히는 네트워크를 설계하는 것이다. 이번 시도의 하나 큰 약점은 팀의 조합만 안다는 것이지 각 챔피언의 포지션을 정확히 알 수는 없다는 것이다. 라이엇 API에서 제공할 때 포지션이라는 변수도 있으나 탑이 2명이거나 미드가 2명이거나 하여튼 각 포지션이 1명씩 있지 않아 그것을 활용하는 것을 포기했기 때문이다.

코드도 정리해서 깃에 올려야 하나.... 흠... 올렸다면 이 문장이 없겠지....?
