---
layout: post
title:  "머신러닝에 방금 입문한 사람이 쓰는 <br />머신러닝에 입문하는 방법"
date:   2019-04-25
excerpt: "'개구리 올챙이 시절 기억 못한다'라는 말이 있다."
tag:
- Development
- Deep Learning
- Machine Learning
- CNN
- Reinforcement Learning
- GAN

comments: true
---

### 본격적인 이야기에 앞서

'개구리 올챙이 시절 기억 못한다'라는 말이 있다. 이 말은 머신러닝을 배울 때도 적용되는데 구글에서 머신러닝 입문이라는 키워드로 검색한다면 머신러닝에 대해 한가락 하시는 분들이 쓴 책과 블로그 그리고 유튜브가 주르륵 나온다. 그러나 어떤 글이나 자료는 시작할 때 'cross entropy'나 'image convolution' 등 처음부터 내가 알아먹지 못하는 소리만 한다. 이를 알아보기 위해 구글에 다시 'cross entropy'를 검색하면 또 'probabilistic classification'이라는 단어가 등장한다. 그리고 이러한 무한루프가 등장하여 아무것도 이해하지 못하는 상황까지 오기 마련이다(정확히 필자가 겪은 상황이다!). 이런 내가 겪었던 상황을 바탕으로 입문자가 이해하기 쉬운 자료를 정리해 보고 머신러닝에 현재까지의 갈래가 어떤 식으로 있는지 간단히 정리해 보려고 한다.

그러나 입문이라고 해도 어느 정도의 진입 장벽이 있는 것은 사실인데, 유튜브에서 영어 자막이 있는 영어 강좌 정도는 이해하는 수준의 영어 실력이 권장되고 공과대학교 1~2학년 수준의 미적분학(편미분과 chain rule의 이해), 그리고 비슷한 수준의 선형대수와 확률변수론 지식이 필요하다.

사족으로 '머신러닝을 공부하려면 위의 수학적 지식이 꼭 필요한가?'라는 질문에 대해서는 이용자가 아닌 설계자가 되기 위해서는 반드시 필요하다고 생각한다. 그저 텐서플로우 등의 라이브러리 사용자라도 수학적 원리를 이해한다면 유리한 점이 많다는 것은 자명하다.

## 그래서 어떻게 입문해야 하는데?
### 큰 그림 보기

이 글을 읽는 당신이 머신러닝 입문자라면 CNN, RNN, GAN, Reinforcement Learning(이하 RL) 등등의 기법 전혀 모를 것이다. 그러나 모든 머신러닝과 딥러닝에서 공통적으로 사용되는 방법이 있는데 바로 Loss함수를 계산하여 backpropagation을 실행한다는 것이다. 이 마지막 문장에서 음? 하는 의문점이 생긴다면 다음 3blue1brown의 머신러닝 관련 유튜브를 보는 것보다 직관적으로 이해하는 방법을 찾기 힘들다고 생각한다.
<iframe width="560" height="315" src="https://www.youtube.com/embed/aircAruvnKk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

이 시리즈의 가장 큰 단점은 neural network관련 네 개의 영상 중 첫 번째 동영상을 제외하고는 한글 자막이 없다는 점이다. 그러나 이만큼 시각적 직관화를 시킨 다른 영상은 찾기 힘들다. 필자는 해당 유튜브 채널과 같이 시각적 직관화를 시키는 것을 매우 좋아하는데 해당 채널의 다른 영상들도 공학분야 수학을 이해하는데 있어 큰 도움이 된다.

### 입문 바이블. 스탠포드 cs231n, 모두의 딥러닝
스탠포드 대학교의 [Convolution Neural Network for Visual Recognition](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) 강의의 재생목록을 하루에 한 강의씩 볼 수 있다면 이보다 좋은 머신러닝 입문법을 찾기는 힘들 것이다. 그러나 우리는 수많은 인터넷 강의를 완강하지 못하고 그 이유 중 하나는 분명 강의가 모두 영어라는 데 있다. 필자도 5~6강 정도 보다 포기했다. 한국어로 된 강의만 아니 강의자료만 있었어도 끝까지 볼 수 있었을 텐데.

그래서 찾은 것이 불완전하지만 조금이나마 [변역된 자료](https://aikorea.org/cs231n/)이다. 한국어로 된 자료는 우리의 학습 속도를 비약적으로 높여 주는 것은 당연한 사실이다.

알다시피 이런 류의 공부는 영어로 하는 것이 가장 좋지만 귀에 들리는 짧은 시간의 수많은 영어는 우리를 강의에 집중하기 힘들게 하고 효율이 나기 힘들다. 그래서 찾은 것이 홍콩과기대 김성훈 교수님의 [모두의 딥러닝](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)이다. 이 강의를 듣고 잘 이해가 가지 않는다면 이 강의 자체를 정리해 놓은 블로그도 구글에 검색하면 많으니 참고하기도 쉽다.(최근에 알게 되어 모두의 [딥러닝 시즌 2](https://www.youtube.com/watch?v=7eldOrjQVi0&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C)가 있다는 것을 첨언한다.)

하루에 하나씩 강의를 보고 있는데 어제의 강의 내용 일부분이 기억이 나지 않는다면 어제 강의를 다시 보는 것도 매우 좋다. 그러나 우리는 불행히도 좀 더 빨리 다음 내용을 보고 싶은 마음이 크다. 그렇다면 다음 [데이터 분석하는 문과생](https://sacko.tistory.com/category/Data%20Science) 블로그는 지나간 기억을 상기시키는데 매우 좋다.

여기까지 왔다면 축하한다. 당신은 이제 머신러닝을 입문했다고 말할 수 있다.

## 입문자 그 다음...
### 여러 CNN 모델들
[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [VGGNet](https://arxiv.org/abs/1409.1556), [ResNet](http://www.arxiv.org/abs/1512.03385), [SqueezeNet](https://arxiv.org/abs/1602.07360) 등은 모두 ILSVRC라는 이미지 분류 대회에서 우수한 성적을 낸 모델이며 그 동작 원리를 알 만한 가치가 있다. 링크가 되어었는 각각의 논문을 영어로 읽는다면 더할 나위 없이 좋겠지만 우리는 빨리 최근 연구의 경향을 알아보고 싶다. 따라서 'VGGNet'설명, 'ResNet'설명 등의 키워드로 구글에 검색하여 마음에 드는 블로그가 있다면 따라 읽으면 된다. 이쯤되면 해당 블로그의 설명을 어느정도 알아 들을 수 있을 것이다. 또 알아 듣지 못한다 하더라도 어떤가 우리에겐 구글이 곁에 있는데!

단 여기서 개인적인 시행착오를 겪은 조언이 있는데 앞서 이야기한 키워드로 구글에 검색한다면 '라온피플'이라는 회사에서 [운영중인 블로그](https://blog.naver.com/PostList.nhn?blogId=laonple)가 거의 가장 먼저 검색된다. 그러나 개인적으로는 이 블로그에서 크게 도움받을 수 있는 점이 적었다. 말 그대로 일을 위해서 블로그를 운영하는 느낌이라 선뜻 이해하기 쉽지 않았다. 목차를 쭉 둘러보고 어떤 순서로 모델이 발전되었는가 정도의 도움은 충분히 받을 만 하다.

### 강화학습(Reinforcement Learning)
2016년 이후 우리나라에 머신러닝 열풍을 불러일으키게 한 한 가지 단어라면 단연 '알파고'일 것이다. 이 알파고는 강화학습을 기반으로 한 구글에 인수된 DeepMind 사에서 개발한 바둑 프로그램으로 이 강화학습한 제대로 안다면 당신도 알파고를 만들 수 있다!

여기서 우리는 상술한 김성훈 교수님께 다시 한 번 감사를 표해야 한다. 모두의 딥러닝 뿐 아니라 [모두의 RL](https://www.youtube.com/watch?v=dZ4vw6v3LcA&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG) 또한 있으니까.
또한 알파고를 만든 딥마인드에 재직중인 David Silver 박사가 [직접 강의한 영상](https://www.youtube.com/watch?v=2pWv7GOvuf0)도 있으니 영어 실력에 자신이 있다면 이쪽도 굉장히 좋다.

그러나 성격 급한 필자는 또다시 강의를 보지 못하고 [다음 깃북](https://dnddnjs.gitbooks.io/rl/content/index.html)을 통해 강화학습을 공부하기 이른다. 해당 링크는 위의 David Silver 박사의 강의를 한글로 옮긴 강의로 중반부까지는 조금 읽을 만 하다가도 후반부는... 직접 읽어 보시길 바란다.

### Generative Model(GAN, VAE, autoencoder)
이 항목에 있는 딥러닝 모델들은 classification 이나 detect의 범주에서 벗어나 새로운 이미지를 생성해낸다. 기술적 뉴스에 조금이라도 관심이 있다면 [인공지능이 고흐 스타일로 그림을 재해석](http://www.asiae.co.kr/news/view.htm?idxno=2017041911030791522)한다거나 [실제로는 존재하지 않는 사람의 얼굴을 생성](https://www.sciencetimes.co.kr/?news=ai%EB%A1%9C-%EC%A7%84%EC%A7%9C-%EA%B0%99%EC%9D%80-%EA%B0%80%EC%A7%9C-%EB%A7%8C%EB%93%A4%EA%B8%B0-%EA%B2%BD%EC%9F%81)한다던가 하는 뉴스를 본 적이 있을텐데 이러한 작업을 하는 모델을 Generative Model이라고 한다. 그리고 이런 GAN 구조에 대해 한국어로 가장 잘 설명된 블로그는 의심할 여지 없이 [유재준님의 블로그](http://jaejunyoo.blogspot.com/search/label/GAN)이다.

그렇다고 블로그에 있는 여러 GAN 구조를 모두 이해할 필요는 없을 듯 하다.(다시 얘기하지만 필자 또한 방금 입문했을 뿐이다.)그 중 wGAN 이나 f-GAN 같은 구조는 상당한 수학적 이론이 뒷받침되어야 이해할 수 있을 것 같다. 절대 내가 이해하지 못해서 그런건 아니다. 이런 친절한 설명에도 불구하고 유재준님의 블로그에 설명된 논문들은 주로 GAN 구조의 mode collapse 문제를 해결하는 구조 자체의 안정성에 관한 논문을 주로 리뷰하는 편이다. 상기되어있는 GAN의 응용과 그 예시를 보고 이해하고 싶다면 Pix2Pix, DiscoGAN, cycleGAN 등의 논문이 리뷰되어 있는 연세대학교에서 박사과정을 밟고 있는 [김태오님의 블로그](https://taeoh-kim.github.io/#blog)를 참조하는 것이 많은 도움이 될 것이다. 또한 Variational AutoEncoder인 VAE는 [이기창님의 블로그](https://ratsgo.github.io/generative%20model/2018/01/27/VAE/)에 좋은 설명이 있다.

상기된 블로그들 모두 링크를 제외하고 다른 포스팅들도 딥러닝 관련 글이 많으니 참조하면 좋을 것이다.

## 다음으로 공부할 것
여기까지 잘 따라왔다면 축하한다. 당신은 대충 2017년까지의 딥러닝 연구 동향을 안 것이다. 그러나 현재는 2019년이며 매년 새로운 연구결과가 나오는 소위 '핫'한 분야의 특성상 현재까지 배운 것들은 꽤나 오래전 지식이라고 할 수 있다.

최근 CNN은 autoML구조를 통해 CNN의 하이퍼파라미터를 머신러닝을 통해 결정하며 GAN에 그 구조를 적용한 듯 한 autoGAN이라는 논문도 존재한다.(물론 아직 읽지 않았다) 여기서부터는 한글로 정리된 자료는 없다고 보는 것이 옳으며 ICPR이나 JRML과 같은 저명한 저널에 실린 논문을 읽는 것이 연구 동향을 따라갈 수 있는 길이다.

부디 머신러닝 입문자에게 조금이나마 도움이 되었으면 한다. 아래 내가 공부하면서 많은 도움을 얻었던 몇 가지 링크를 첨부한다.

[Cross-Entropy에 관한 간단한 설명](http://blog.naver.com/PostView.nhn?blogId=gyrbsdl18&logNo=221013188633&redirect=Dlog&widgetTypeCall=true)  
[조대협의 블로그](https://bcho.tistory.com/)  
[동신한의 조재성](https://nittaku.tistory.com/category/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20%26%20%EB%94%A5%EB%9F%AC%EB%8B%9D/%EB%94%A5%EB%9F%AC%EB%8B%9D%20-%20Image%20classification)  
[수학적 물리학적 그리고 기계학습에서 텐서란 무엇인가](https://blog.naver.com/rlaghlfh/220914107525)  
[KL-Divergence 설명](https://brunch.co.kr/@chris-song/69)  
[텐서플로우로 50줄만에 GAN 구현하기](https://taeoh-kim.github.io/blog/tensorflow%EB%A1%9C-50%EC%A4%84%EC%A7%9C%EB%A6%AC-original-gan-code-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0/)  
[EM 알고리즘 for VAE](http://sanghyukchun.github.io/70/)  
[구글에서 제작한 머신러닝 용어집](https://developers.google.com/machine-learning/glossary/?hl=ko)  
