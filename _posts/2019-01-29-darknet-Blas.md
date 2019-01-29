---
layout: post
title:  "OpenBlas를 이용한 Darknet yolo 가속방법"
date:   2019-01-29
excerpt: "이미지 처리 딥러닝 오픈소스 Darknet을 OpenBlas를 이용하여 가속"
tag:
- Development
- Deep Learning
- Machine Learning
- Darknet
- OpenBlas

comments: true
---

## 실험 세팅

실험은 intel i7-6700에 ubuntu 18.04LTS, openCV 3.4.2, Darknet yolo v3, OpenBlas 0.2.20 을 사용했다. openCV설치는 [이 블로그](https://jsh93.tistory.com/53)를 통해 진행했으며 darknet은 [공식 홈페이지](https://www.darnet.net)의 installation. 부분을 참고했다.

참고사항으로 yolo V3를 사용하려면 openCV 3.4.2버전 이상을 필요로 한다. (필자는 3.x.x 번대 중 가장 최신 버전인 3.4.3 설치를 시도해보았으나 실패하고 3.4.2를 설치하였다.)

## OpenBLAS 설치

[OpenBlas 공식 홈페이지](https://www.openblas.net/)에서 원하는 경로에 다음 명령어를 통해 tar.gz 파일을 다운받고 압축을 해체한다.
~~~
$ wget http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
$ tar -zxvf v0.2.20.tar.gz
$ cd OpenBlas-0.2.20
$ cmake .
$ make
~~~
다음 명령어들을 통해 OpenBLAS 설치를 완료할 수 있다.

OpenBLAS 공식 깃헙의 [Installation Guide](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide)를 참고할 수도 있다. 허나 필자는 해당 링크에서 Precompile installtion packages의 Linux 부분을 참고 한 후 참고해서 진행하였으나 Linux의 마지막 줄
~~~
$ sudo update-alternatives --config libblas.so.3
~~~
를 실행하는 과정에서 아무리 다른 방법을 찾아봐도 libblas.so.3파일을 찾울 수 없다는 오류가 떳다. 심지어 구글링을 통해 libblas.so.3파일의 경로를 찾아 추가해줬는데도 말이다!
결과적으로 OpenBlas를 사용하여 작동했지만 찝찝함을 금할 수 없었다.

## Darknet에 OpenBLAS 적용

[다음 링크](https://github.com/pjreddie/darknet/issues/332)의 질문과 답변을 참고했다. darknet의 convolution 구조 중 gemm(general matrix multiply)에서 가속하는 방법인데 답변을 그대로 옮겨 쓰자면 다음과 같다.
>1).build openblas from source.  
>2).refine "Makefile"  
adding...  
OPENBLAS=1  

>ifeq ($(OPENBLAS), 1)  
COMMON+= -I/opt/OpenBLAS/include/  
CFLAGS+= -DOPENBLAS  
LDFLAGS+= -L/opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran  
endif  

>3).refine gems.c  
#include "cblas.h"  

>void gemm(int TA, int TB, int M, int N, int K, float ALPHA,  
float *A, int lda,  
float *B, int ldb,  
float BETA,  
float *C, int ldc)  
{  
#ifdef OPENBLAS  
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);  
#else  
gemm_cpu( TA, TB, M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);  
#endif  
}  

답변에 따르면 첫 번째로 실험 세팅 부분에서 행한 OpenBLAS를 빌드해야하고 그 다음으로 darknet의 Makefile에서 다음 코드를 추가해야 한다.
~~~
OPENBLAS=1


ifeq ($(OPENBLAS), 1)
COMMON+= -I/opt/OpenBLAS/include/
CFLAGS+= -DOPENBLAS
LDFLAGS+= -L/opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran
endif
~~~
다음으로 darknet의 src 디렉토리로 이동 후 gemm.c 파일에 코드를 다음과 같이 추가하고 수정한다. (위의 gems.c는 답변자의 오타로 예상된다.)

~~~c
#include "cblas.h"  

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,  
float *A, int lda,  
float *B, int ldb,  
float BETA,  
float *C, int ldc)  
{  
#ifdef OPENBLAS  
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);  
#else  
gemm_cpu( TA, TB, M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);  
#endif  
} 
~~~

이후 darknet폴더에서 make clean과 make를 진행하면 필자의 경우 다음과 같이 5배 이상 빨라진 것을 확인할 수 있었다.

![Blas미적용](https://github.com/queez0405/queez0405.github.io/blob/master/assets/img/darknetBlas/darknet.JPG?raw=true)
##### OpenBLAS 미적용 작동시간
![Blas적용](https://github.com/queez0405/queez0405.github.io/blob/master/assets/img/darknetBlas/darknetBlas.JPG?raw=true)
##### OpenBLAS 적용시 작동시간
