**Network Dissection:\
Quantifying Interpretability of Deep Visual Representations**

1.  Abstract

2.  Introduction

    A.  Related Work

3.  Network Dissection

    A.  Broden: Broadly and Densely Labeled Dataset

    B.  Scoring Unit Interpretability

4.  Experiments

    A.  Human Evaluation of Interpretations

    B.  Measurement of Axis-Aligned Interpretability

    C.  Disentangled Concepts of Layer

    D.  Network Architectures and Supervisions

    E.  Training Conditions vs. Interpretability

    F.  Discrimination vs. Interpretability

    G.  Layer Width vs. Interpretability

5.  Conclusion

```{=html}
<!-- -->
```
1.  Abstract

-   Neural Dissection

    -   개별 은닉 유닛과 semantic 개념의 집합 사이의 alignment를
        평가하여 CNN의 latent representation 해석성을 정량화 하기 위한
        틀

    -   CNN 모델을 고려하여 각 중간 Convolution Layer에서 은닉 유닛의
        semantic을 기록하기 위해, 광범위한 시각 개념의 데이터셋 사용

    -   Semantic이 있는 유닛들은 물체, 부분, 장면, 텍스쳐, 재료, 색
        등으로 label이 주어짐

-   가설

    -   유닛의 해석성이 유닛의 무작위 선형 조합과 동일

        -   다양한 supervised learning과 self-supervised learning에 적용

        -   반복 트레이닝의 효과를 추가로 분석

        -   다양한 초기 설정의 네트워크 비교

        -   깊이와 너비에 따른 네트워크의 영향 조사

        -   Deep visual representation의 dropout, batch normalization의
            효과 측정

2.  Introduction

-   은닉 유닛에 대한 관찰

    -   인간이 해석 가능한 개념이 네트워크 내에서 잠재적인 변수로 나타남

        -   장소를 인식하도록 훈련된 네트워크 내에서 물체 감지기가
            나타남

![](media/image1.png){width="4.109600831146107in"
height="1.4485717410323709in"}

-   네트워크가 문제 해결 시 방법이 제한되지 않을 때 나타남

-   심층 네트워크가 자연스럽게 분리된 표현을 배우고 있을 수 있음

```{=html}
<!-- -->
```
-   문제점

    -   Disentangled Representation은 무엇이며, 어떻게 정량화하고 찾을
        수 있는가?

    -   해석 가능한 은닉 유닛은 Feature Space의 특정 alignment를
        반영하는 것인가?

    -   어떤 훈련 조건이 더 크거나 덜 얽힌 표현으로 읽을 수 있는가?

-   해결 방안

    -   시각적 표현을 해석하고 정량화하기 위해 분석 프레임워크Network
        Dissection 제안

        -   다양한 라벨이 붙여진 데이터셋인Broden을 사용, 특정 CNN에
            대한 숨겨진 유닛의 의미 파악, 인간이 해석 가능한 개념들과
            일치

        -   물체와 장면 인식에 대해 훈련된 다양한 CNN(AlexNet, VGG,
            GugLeNet, ResNet)에서 평가

        -   훈련 데이터 세트, dropout 및 batch normalize와 같은 훈련
            기법 등에 의해 해석이 어떻게 영향을 받는지 검토

    A.  Related Work

    -   시각화를 통한 convolutional neural network의 이해

        -   은닉 유닛의 활성화를 극대화하는 이미지 패치 샘플링

        -   역전파를 사용하여 중요한 이미지 기능 식별, 생성하여 시각화

        -   네트워크의 일부를 격리하고, 전송 또는 제한하며, 문제에 대해
            능력을 시험

    -   이 논문과 관련 있는 내용

        -   장면 분류를 위해 훈련된 네트워크에서 개별 장치가 물체
            감지기로 동작하는 것을 결정하기 위해 사용

        -   피쳐 반전 매핑을 학습하여 개별 장치에 대한 프로토타입
            이미지를 자동으로 생성

        -   레이어 간의 정보 역학관계와 최종 예측에 미치는 영향을
            분석하는 간단한 선형 탐침을 교육하여 중간 레이어를 시험하는
            접근법

3.  Network Dissection

-   Deep visual representation을 측정하는 세 단계

    -   Human Lable의 광범위한 시각적 개념 식별

    -   알려진 개념에 대한 은닉 변수의 반응 수집

    -   은닉 변수-개념 쌍의 alignment 정량화

    A.  Broden: Broadly and Densely Labeled Dataset

    -   데이터 세트 조립

        -   낮은 수준의 개념(색상)부터, 높은 수준의 개념(개체)의
            alignment를 확인할 수 있도록 하는 것이 목적

    -   목적

        -   광범위한 시각적 개념에 대한 예시들의 ground truth 세트를
            제공

    -   개념 라벨

        -   모든 클래스가 영어 단어와 일치하도록 표준화되고 원본 데이터
            세트에서 병합

        -   라벨은 공유된 동의어에 기초하여 병합되며, '왼쪽'과 '위'와
            같은 위치적 구별을 무시

        -   29개의 지나치게 일반적인 동의어('자동차'의 경우 '기계'와
            같은)의 블랙리스트를 회피

        -   표 1은 라벨 클래스당 영상 샘플의 평균 수

![스크린샷이(가) 표시된 사진 자동 생성된
설명](media/image2.png){width="5.708333333333333in"
height="1.8888888888888888in"}

![](media/image3.png){width="4.031340769903762in"
height="2.307871828521435in"}

B.  Scoring Unit Interpretability

![텍스트, 지도이(가) 표시된 사진 자동 생성된
설명](media/image4.png){width="6.268055555555556in"
height="2.0548611111111112in"}

-   Borden은 Back Propagation 없이도 어떤 네트워크 층에서든 적용 가능함

-   IoU: 객관적 신뢰도 점수

    -   서로 다른 표현의 해석성을 비교할 수 있게 하며 아래 실험의 근거를
        제시

    -   네트워크 해부는 기본 데이터 세트와 마찬가지로 작동한다는 점에
        유의

    -   유닛은 Borden에 없는 인간 이해 가능한 개념과 일치하면 해석성에
        대해 점수가 잘 나오지 않음

    -   Borden의 미래 버전은 더 많은 종류의 시각적 개념을 포함하도록
        확장될 것

4.  Experiments

    A.  Human Evaluation of Interpretations

![](media/image5.png){width="3.950495406824147in"
height="1.576353893263342in"}

B.  Measurement of Axis-Aligned Interpretability

-   가설

    -   해석 가능한 개념을 독립된 유닛에 할당하는 것이 의미 있는지
        여부를 결정하기 위한 실험

    -   가설 1. 해석 가능한 단위가 나타나는 이유는 사전준비 가능한
        개념이 대부분의 배치 공간에 나타나기 때문

        -   표현법이 관련된 개념을 축에 독립적인 방식으로 로컬화
            하면,어떤 방향으로든 투영하는 것은 해석 가능한 개념을 드러낼
            수 있으며, 자연적 근거에서 단일 단위를 해석하는 것은 표현을
            이해하는 의미 있는 방법이 아닐 수 있음

    -   가설 2 해석 가능한 정렬은 흔치 않으며, 해석 가능한 유닛이 나타남

        -   학습은 설명적 요소를 개별 단위와 정렬하는 특별한 기초에
            통합되기 때문이다. 이 모델에서 자연적 근거는 네트워크에 의해
            학습된 의미 있는 분해를 나타냄

![지도, 텍스트이(가) 표시된 사진 자동 생성된
설명](media/image6.png){width="4.029702537182852in"
height="3.147285651793526in"}

-   결론

    -   해석성이discriminative power의 필연적인 결과도 아니며,
        discriminative power의 전제조건도 아님

    -   해석성은 별도로 측정해야 이해할 수 있는 다른 품질이라는 것을
        알게 됨

C.  Disentangled Concepts of Layer

-   네트워크 dissection으로 Places-AlexNet 및 ImageNet-AlexNet의 모든
    convolutional layer 내의 유닛의 해석성을 분석하고 비교

    -   Places-AlexNet은 Places205의 장면 분류를 위해 트렝이닝

    -   ImageNet-AlexNet은 ImageNet\[15\]의 객체 분류를 위해 트레이닝

![텍스트, 스크린샷이(가) 표시된 사진 자동 생성된
설명](media/image7.png){width="6.268055555555556in"
height="3.529861111111111in"}

-   그 결과는 그림 5에 요약

-   단위의 표본은 자동으로 유추된 사전 및 수동 할당된 해석과 함께 표시

-   예측된 라벨이 인간의 주석과 잘 일치한다는 것을 알 수 있다. 비록
    그것들이 때때로 알고리즘에 의해 예측된 \'횡단선\'과 같은 시각적
    개념에 대한 다른 설명을 포착

-   직관, 색, 질감은 하위 계층 conv1과 conv2에서 주로 검출되고, 반면
    conv5에서는 더 많은 물체 및 유닛 검출

D.  Network Architectures and Supervisions

-   서로 다른 네트워크 아키텍처와 지도 학습이 학습된 표현의 해석성에
    어떤 영향을 미치는가?

    -   다양한 네트워크 아키텍처와 감독을 평가하기 위해 네트워크
        dissection을 적용

    -   단순성을 위해 다음 실험은 semantic detector가 가장 많이 등장하는
        각 CNN의 마지막 convolution layer에 초점

![스크린샷이(가) 표시된 사진 자동 생성된
설명](media/image8.png){width="3.712871828521435in"
height="2.3319761592300963in"}

![자동차, 담장이(가) 표시된 사진 자동 생성된
설명](media/image9.png){width="6.268055555555556in"
height="3.145138888888889in"}

-   네트워크 아키텍처의 관점에서, ResNet\> VGG \>GugLeNet\>AlexNet의
    해석 가능성을 발견

    -   더 깊은 구조들은 더 큰 해석성을 허용하는 것으로 보임

    -   교육 데이터 세트를 비교하면 장소 \> ImageNet을 찾을 수 있음

    -   한 장면은 여러 개체로 구성되므로 장면을 인식하도록 훈련된
        CNN에서 더 많은 물체 감지기가 나타나는 것이 유익

![스크린샷이(가) 표시된 사진 자동 생성된
설명](media/image10.png){width="3.8910892388451446in"
height="2.8368536745406825in"}

-   다양한 감독 업무와 자체 감독 업무에 대해 훈련된 네트워크의 결과

    -   네트워크 아키텍처는 각 모델에 대한 AlexNet이며, 우리는
        Places365에 대한 교육이 가장 많은 수의 고유 검출기를 생성한다는
        것을 관찰

    -   자체 감독 모델은 많은 텍스쳐 감지기를 생성하지만 상대적으로 적은
        감지기를 생성

    -   자체 학습된 일차 과제의 감시는 큰 주석 데이터 세트에 대한 감독
        교육보다 해석 가능한 개념을 추론하는 데 훨씬 약함

    -   자기 감시의 형태는 차이를 만든다: 예를 들어, 컬러 이미지 모델은
        그레이 스케일 이미지 모델에 대해 훈련

5.  Conclusion

-   CNN의 해석성을 계량화하기 위한 일반적인 프레임워크인 네트워크 섹션을
    제안

-   해석성이 축에 독립적인 현상인지 측정하기 위해 네트워크 dissection를
    적용했는데, 그렇지 않음

    -   이는 해석 가능한 단위가 부분적으로 분리된 표현을 나타낸다는
        가설과 일치

-   네트워크 dissection를 적용하여 최첨단 CNN 훈련 기법의 해석에 미치는
    영향을 조사

-   서로 다른 계층의 표현들이 다른 범주의 의미를 분리하고, 다른 훈련
    기법이 숨겨진 유닛에 의해 학습된 표현의 해석성에 중요한 영향을 미칠
    수 있다는 것을 확인
