Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization

Grad-CAM: 딥 네트워크 이용, 지역을 그라디언트로 표현하는 시각적 설명
방법

작성자: 김양선

작성일: 2019. 12. 4.

**목차**

1.  Abstract

2.  Introduction

    A.  What makes a good visual explanation?

3.  Related Work

    A.  Visualizing CNNs.

    B.  Assessing Model Trust

    C.  Weakly supervised localization

4.  Approach

    A.  Grad-CAM as a generalization to CAM

    B.  Guided Grad-CAM

5.  Evaluation Localization

    A.  Weakly-supervised Localization

        i.  Weakly-supervised Segmentation

    B.  Pointing Game

6.  Evaluation Visualizations

    A.  Evaluation Class Discrimination

    B.  Evaluation Trust

    C.  Faithfulness vs. Interpretability

7.  Diagnosing image classification CNNS

    A.  Analyzing Failure Modes for VGG-16

    B.  Effect of adversarial noise on VGG-16

    C.  Identifying bias in dataset

8.  Counterfactual Explanations

9.  Image Captioning and VQA

    A.  Image Captioning

        i.  Comparison to dense captioning

    B.  Visual Question Answering

        i.  Comparison to Human Attention

        ii. Visualizing ResNet-Based VQA model with attention

10. Conclusion

```{=html}
<!-- -->
```
1.  Abstract(요약)

-   CNN 기반 의사 결정에 대한 '시각적 설명' 제공

    -   AI의 투명성 제공

-   Grad-CAM -- Gradient weighted Class Activation Mapping

    -   최종 Convolution 레이어에 유입되는 모든 타겟 컨셉을 사용

    -   개념 예측을 위해 이미지의 중요 부분을 표현하는 coarse
        localization map 작성

    -   이전 접근 방식과 다른 점

        -   Fully-Connected 레이어s(ex: VGG)

        -   구조화된 출력에 사용하는 CNN(ex: 캡션)

        -   다중 모달 입력(ex: VQA) 또는 강화 학습에도 사용 가능

    -   이미지 분류 모델로서 장점

        -   모델들의 failure mode에 대한 통찰력

            -   비 합리적인 예측이 가장 합리적인 설명일 가능성

        -   대립하는 이미지에 대한 강건함(robust)

        -   ILSVRC-15에서 취약했던 지역화에 비해 우수한 약-지도 지역화
            작업 제공

        -   기반 모델에 더 충실

        -   데이터셋의 편향을 식별, 모델 일반화 달성

-   이미지 캡션 붙이기 및 VQA

    -   비-주의 기반 모델도 입력을 지역화 가능

-   예측에 대한 신뢰도

    -   Grad-CAM의 캡션으로 인간이 딥 네트워크에 대한 신뢰를 확립하는데
        도움

    -   훈련 받지 않은 사용자로 하여금 신뢰할 수 있는 딥 네트워크를
        식별할 수 있는지 확인

2.  Introduction(소개)

-   CNN의 한계

    -   딥 뉴럴 네트워크는 강력하지만, 그에 비해 사용자가 이해하기에는
        어려움

    -   시스템 고장의 경우에도 설명이나 해석이 어려움

    A.  Interpretability matters(대화 능력의 문제)

    -   시스템에 대한 신뢰를 쌓기 위해, 왜 그런 선택을 했는지 설명하는
        '투명(Transparent)' 모델을 필요로 함

    -   인공지능에서의 유용성

        1.  아직 신뢰성을 확인할 수 없는 약-인공지능에서는 투명성과
            설명을 통해 실패를 판정할 수 있음

        2.  인공지능이 인간과 동등한 경우에는 이용자에 대한 적절한 신뢰
            구축 가능

        3.  인공지능이 인간보다 강할 때, 더 나은 방법을 인간에게
            가르치기 가능

    -   Accuracy & simplicity or interpretability tradeoff(정확도와
        단순성, 대화 능력의 트레이드 오프)

        -   규칙 기반 고전 시스템

            -   충분히 해석 가능하지만, 정확도 면에서 떨어짐

            -   각 단계를 손으로 설정하고, 외부의 변수 등을 미리
                예측하기 때문에 추측이 쉬운 것으로 추측

        -   ResNet 같은 심층 모델

            -   200개 이상의 레이어 깊이로 성능은 최첨단이지만 해석은
                어려움

        -   이 연구에서 한 일

            -   최첨단 심층 모델의 아키텍처를 변경하지 않고 설명을 제공

            -   CAM의 일반화로 접근

                i.  완전히 연결된 층(Fully-connected 레이어, ex: VGG)

                ii. 구조 출력에 사용되는 CNN(ex: 캡션)

                iii. 다중 모달 입력(ex:VQA) 또는 강화 학습에 사용하는
                     CNN 모델에도 사용 가능

    B.  What makes a good visual explanation?(좋은 시각적 설명이란?)

    -   세분화된 구분(이미지의 대상 범주를 지역에 표현)

    -   고 해상도(세부 정보를 세밀하게 포착)

    C.  요약

        i.  아키텍처 변경 또는 재 트레이닝 없이 대다수의 CNN-based
            네트워크에 적용할 수 있는 시각화 설명을 위해 클래스 식별
            가능한 지역화 기법인 Grad-CAM을 제안

        ii. Grad-CAM을 최고의 분류, 캡션, VQA 모델로 적용

        iii. 이미지 분류 및 VQA에 적용된 ResNets을 시각화

        iv. 인간 연구를 통해 훈련 받지 않은 인간에게도 Grad-CAM이
            효과적으로 설명 가능

3.  Related Work(관련 연구)

    A.  Visualizing CNNs.(CNN의 시각화)

    -   CNN 예측기가 중요한 픽셀을 하이라이팅하게 함

    -   네트워크 장치를 최대로 활성화 시키거나 잠재한 재설정을
        반전시키기 위해 이미지를 합성

    B.  Assessing Model Trust(평가 모델의 신뢰)

    -   대화 가능성의 개념

    -   모델에 대한 신뢰 보장

    C.  Weakly supervised localization(약-지도 지역화)

    -   CNN을 이용한 약-지도 지역화는 모든 이미지 클래스 라벨에서 이미지
        안의 오브젝트의 위치를 잡아내는 것

4.  Approach(접근)

-   Convolutional feature에서 기대할 수 있는 것

    -   더 깊은 표현으로 더 높은 수준의 시각적 구조 포착

    -   Fully-connected 레이어에서 손실된 공간 정보를 포함하므로 마지막
        Convolutional 레이어에서 높은 수준의 정보와 의미를 담고 있음을
        기대

    -   Grad-CAM에서는 마지막 Convolutional 레이어에서 각각의 뉴런의
        선택이나 흥미를 이해할 수 있음

![지도, 텍스트이(가) 표시된 사진 자동 생성된
설명](media/image1.png){width="6.270833333333333in"
height="2.3055555555555554in"}

-   그림에서

    -   클래스 식별 지역화 맵인 Grad-CAM은
        $L_{Grad - CAM}^{c}\  \in \ \mathbb{R}^{u \times u}$, 너비 $u$와
        높이 $v$, 그리고 어떤 클래스 $c$로 표현

    -   소프트 맥스 하기 전인 클래스에 대한 스코어 $y^{c}$와 feature
        map에 관련된 Convolutional 레이어 $A^{k}$(예를들어
        $\frac{\partial y^{c}}{\partial A_{\text{ij}}^{k}}$ )를 먼저
        계산

    -   그라디언트는 뉴런의 중요도에 대한 가중치를 얻기 위해
        global-average-pooled 되어 있으며 수식으로 표현하면:

$$a_{k}^{c} = \ \frac{1}{Z}\sum_{i}^{}\ \sum_{j}^{}\ \frac{\partial y^{c}}{\partial A_{\text{ij}}^{k}}$$

앞부분은 global average pooling이며, 뒷부분은 gradients(backpropagation)

-   가중치 $a_{k}^{c}$는 A에서부터 내려오는 심층 네트워크의 부분을
    선형화한 것이며, $k$는 피쳐 맵에서의 중요한 $c$라는 클래스 부분을
    포착한 것

-   Forward activation map의 가중치 조합을 활성화 후, ReLU 함수를
    적용하면:

$$L_{Grad - CAM}^{C} = ReLU(\sum_{k}^{}{\alpha_{k}^{c}A^{k}})$$

-   Convolution feature map의 사이즈와 동일한 heat map 생성(VGG와
    AlexNet의 경우에는 14 \* 14 사이즈의 레이어 생성)

-   ReLU를 적용한 이유는 $y^{c}$를 증가시키는 positive influence가 있는
    클래스만 보기 위함

-   일반적으로 $y^{c}$는 CNN을 이용한 이미지 분류에 의해 생성된 클래스
    별 스코어일 필요는 없음

A.  Grad-CAM as a generalization to CAM(Grad-CAM을 CAM으로 일반화)

-   CAM의 역할

    -   Global-average-pooled 된 CNN 피쳐 맵을 SoftMax 함수에 직접
        공급하여, 특정한 아키텍처가 이미지 분류용 맵을 제작

-   Panultimate 레이어가 생성한 $K$라는 피쳐 맵이 있을 때:

$$A^{k} \in \ \mathbb{R}^{u \times v}$$

-   그런 다음 GAP(Global Average Pooling)를 사용하여 풀링 후 선형
    변환하여 각 클래스 $c$에 대한 점수 $S^{c}$를 생성하면:

$$S^{c} = \ \sum_{k}^{}{\ \varpi_{k}^{c}}\frac{1}{Z}\sum_{i}^{}{\ \sum_{j}^{}{\ A_{\text{ij}}^{k}}}$$

-   위와 같이 수정된 이미지 분류용 아키텍처에 대한 지역화 맵을 제작하기
    위해 순서를 바꾸면:

$$S^{c} = \ \frac{1}{Z}\sum_{k}^{}{\sum_{i}^{}{\ \sum_{j}^{}{\ \varpi_{k}^{c}\ A_{\text{ij}}^{k}}}}$$

-   수정한다고 해서 모든 맵에 가중치인 $\ \varpi_{k}^{c}$가 적용되는
    것은 아님

-   $\alpha_{k}^{c} = \ \varpi_{k}^{c}$ 같은 아키텍처에 Grad-CAM을
    적용할 경우 CAM은 Strict Generalization이 됨

-   일반화 하는 경우, 복잡한 상호 작용과 함께 CNN 기반의 모델들로부터
    Convolutional 레이어를 얻어내어 시각적 설명을 볼 수 있음

-   이미지 분류를 넘어서 이미지 캡션 및 VQA(시각적 질문 답변)에도
    Grad-CAM을 사용

B.  Guided Grad-CAM(Grad-CAM 가이드)

-   Grad-Cam 시각화는 클래스 분별 능력이 뛰어나지만, 픽셀 공간을 시각화
    하는 것에서는 부족(guided back propagation & deconvolution 같은)

-   포인트 별 곱하기를 통해 Guided Backpropagation과 Grad-CAM의 시각화를
    결합

    -   $L_{Grad - CAM}^{c}$는 bi-linear interpolation을 통해 입력
        이미지의 resolution으로 up-sampling

    -   Deconvolution보다 guided backpropagation이 조금 더 노이즈가 없는
        결과를 만들 수 있음

5.  Evaluation Localization(지역화에 대한 평가)

    A.  Weakly-supervised Localization(약-지도 지역화)

    -   Grad-CAM의 지역화 능력 평가

        -   ImageNet의 지역화 능력 평가

            -   분류 라벨 외의 Bounding Box를 제공하는 경쟁적 접근방식
                요구

            -   Top-1과 Top-5 예측 범주에 모두 평가

        -   지역화 방식

            -   네트워크를 통해 클래스를 예측

            -   각 예측 클래스에 대한 Grad-CAM 맵을 생성

            -   최대 밀도의 15% 임계값으로 Binarize

            -   픽셀의 세그먼트가 이어짐

            -   단일한 가장 큰 세그먼트에 Bounding Box 그리기

        -   Backprop, c-MWP, Grad-CAM과 비교했을 때 Grad-CAM이 제일
            성능이 좋음

        i.  Weakly-supervised Segmentation(약-지도의 분할)

        -   Grad-CAM의 지역화를 약한 지도로 사용하여, SEC의 Segmentation
            architecture를 훈련

    B.  Pointing Game(포인팅 게임)

    -   지역화 정확도에 대한 점수는
        $Acc = \ \frac{\# Hits}{\# Hits + \# Misses}$로 평가

    -   C-MWP와 비교했을 때 Grad-CAM은 엄청난 차이로 점수가 높음

6.  Evaluation Visualizations(시각화에 대한 평가)

    A.  Evaluation Class Discrimination(클래스 식별에 대한 평가)

    -   평가 방법

        -   두개의 레이블이 있는 이미지를 VOC 2007 트레이닝 세트에서
            선택

        -   각각에 대해서 시각화 작성

        -   VGG-16과 AlexNet의 CNN에 대해서

            -   Deconvolution

            -   Guided Back Propagation

        -   Grad-CAM 또한 두가지 방법 사용

        -   작업자 43명에게 "이미지에 묘사된 물체가 있는지" 테스트

    -   평가 결과

        -   Guided Grad-CAM: 61.23%의 경우 구별할 수 있음

    B.  Evaluation Trust(신뢰에 대한 평가)

    -   평가 항목

        -   두가지 예측에 대한 설명을 볼 때, 어느 것이 더 신뢰 할만
            한지에 대한 평가

    -   평가 방법

        -   동일한 예측을 한 시각화 자료에 대해, 확실히 더 신뢰할 수
            있는 (+/-2), 약간 더 신뢰할 수 있는 (+/-1) 및 동일하게
            신뢰할 수 있는 (0)의 척도로 서로 상대적인 모델의 신뢰성을
            평가하도록 지시

    -   평가 결과

        -   Guidded Grad-CAM: 1.27의 달성율

    C.  Faithfulness vs. Interpretability(결과에 대한 신뢰도 vs. 설명
        능력)

    -   평가 기준

        -   지역적으로 선택된 곳의 설명이 정확해야 함

    -   평가 방법

        -   Image occlusion

        -   이미지가 Masked 되었을 때, CNN의 평가 점수를 확인

    -   평가 결과

        -   CNN 점수를 변화시키는 마스크 패치들이 있음에도 Grad-CAM과
            Guided Grad-CAM이 높은 강도 할당

        -   이는 더 신뢰할 수 있다는 의미

7.  Diagnosing image classification CNNS(이미지 분류용 CNN 진단)

    A.  Analyzing Failure Modes for VGG-16(VGG-16 실패 모드에 대한 분석)

    -   분석 방법

        -   VGG-16이 잘 못 분류하고 있는 이미지를 확인

        -   이미지에 대해 guided Grad-CAM을 사용하여 올바른 클래스와
            클래스를 시각화

    -   장점

        -   고해상도

        -   고도의 분류 능력

    -   특징

        -   겉보기에는 비합리적인 예측이 종종 합리적인 설명 같이 보일 수
            있음

    B.  Effect of adversarial noise on VGG-16(VGG-16에 있는 대립
        노이즈의 영향)

    -   심층 네트워크의 취약성

        -   없는 레이블에 대해 높은 신뢰도를 갖게 함으로써 이미지를 잘
            못 분류하게 만듬

    -   Grad-CAM의 특징

        -   이미지에 높은 확률로 없는 레이블에 높은 신뢰도를 할당 후,
            있는 레이블에 대해 낮은 신뢰도를 할당

        -   그럼에도 불구하고 Grad-CAM은 존재하지 않는 레이블임을 알고
            있음에도 시각화의 범위를 정확히 지역화 할 수 있었음

    C.  Identifying bias in dataset(데이터셋의 편향 찾기)

    -   데이터셋의 편향

        -   실제 존재하는 시나리오에 대해 일반화 되지는 않을 수 있음

        -   다만 현실에서 존재하는 편견과 고정관념에 대해 학습할 수 있음

    -   분석 방법

        -   ImageNet에서 훈련된 VGG-16 모델

            -   의사와 간호사를 구별하게 함

            -   인기 있는 이미지 상위 250개를 트레이닝

            -   트레이닝 모델은 상위 250개의 이미지에 대해서 높은 정확도
                달성

        -   Grad-CAM의 간호사/의사 구별 방법

            -   헤어 스타일 및 얼굴을 학습

            -   성별-편향적임

        -   시각화를 통해 얻은 데이터를 통해 트레이닝 이미지 변경

            -   균형 잡힌(90%) 상태로 잘 일반화

    -   특징

        -   Grad-CAM이 있으면 데이터의 편향을 확인할 수 있고, 트레이닝
            세트를 수정할 수 있음

8.  Counterfactual Explanations

-   Counterfactual Explanation 제안

    -   Grad-CAM을 약간 수정하면 Counterfactual 설명을 얻게 됨

    -   네트워크가 결정을 바꾸도록 하는 지역을 하이라이팅

    -   이 지역에서 발생하는 개념을 제거하면 더욱 명확한 결과를 얻을 수
        있음

-   구체적인 표현

    -   Convolutional 레이어 $A$에 대해 $y^{c}$(c로 표현하는 class에
        대한 스코어)를 부정하면 중요도 가중치인 $\alpha_{k}^{c}$는:

$$\alpha_{k}^{c} = \ \frac{1}{Z}\sum_{i}^{}{\ \sum_{j}^{}{- \ \frac{\partial y^{c}}{\partial A_{\text{ij}}^{k}}}}$$

![개, 실내, 앉아있는, 보는이(가) 표시된 사진 자동 생성된
설명](media/image2.png){width="5.902777777777778in"
height="2.361111111111111in"}

9.  Image Captioning and VQA(이미지 캡션 달기와 VQA)

-   기존에 존재하는 이미지 분류 기법이나, 아키텍처에는 사용이 어려울 수
    있음

    A.  Image Captioning(이미지 캡션 달기)

    -   사용 모델

        -   LSTM 기반 언어 모델을 위해 최종 처리되어 VGG-16 CNN 사용,
            [NeuralTalk2](https://github.com/karpathy/neuraltalk2)를
            사용

        -   명확한 attention에 대한 메커니즘이 없음 주의

    -   구현 방법

        -   주어진 캡션에 따라 CNN의 마지막 레이어(VGG-16의 경우
            conv5\_3)의 로그 확률 구하기

        -   Grad-CAM 시각화 이미지 생성

![사람, 실내, 전자기기, 표시중이(가) 표시된 사진 자동 생성된
설명](media/image3.png){width="6.263888888888889in"
height="2.0694444444444446in"}

i.  Comparison to dense captioning(dense captioning과의 비교)

![다채로운, 잔디, 사진, 나무이(가) 표시된 사진 자동 생성된
설명](media/image4.png){width="6.263888888888889in"
height="1.9861111111111112in"}

-   특정 이미지에서 중요 부위를 포착, 캡션을 다는 DensCap

    -   FCLN(Fully Convolutional Localization Network)과 LSTM 기반 언어
        모델로 구성

    -   관심 영역과 관련 캡션을 위한 bounding box를 단일 포워드 패스로
        표시

    -   흥미롭게도, NeuralTalk2와 병행으로 작동하는 Grad-CAM에서도
        비슷하게 포착하는 양상이 발견됨

B.  Visual Question Answering(시각 질문에 대한 답변)

-   전형적인 VQA 파이프라인

    -   이미지를 모델링하는 CNN과 질문에 대한 RNN 언어 모델

    -   이미지와 질문은 1000-way classification으로 답을 하기 위해
        융합됨

    -   분류에 대한 문제이므로, 정답을 뒷받침하는 이미지와 함께 답을
        선택할 수 있게 $y^{c}$를 사용

    i.  Comparison to Human Attention(인간의 관심도와 비교)

    -   Grad-CAM과 Human Attention 맵의 상관관계는 0.136으로 높은 편

    ii. Visualizing ResNet-Based VQA model with attention(ResNet을
        기반으로 한 관심도에 따른 VQA 모델의 시각화)

    -   ResNet의 더 깊은 레이어를 시각화할 때, 인접 레이어에 대한
        Grad-CAM의 적은 변화와 차원의 감소를 보이는 계층 간 큰 변화 확인
        가능

10. Conclusion(결론)

-   CNN 기반 모델을 시각적 설명을 통해 보다 투명하게 만들 수 있는
    Grad-CAM 제안

-   기존의 고해상도 시각화와 Grad-CAM 지역화 결합, 고해상도의 클래스
    식별을 가능하게 하는 Guided Grad-CAM 시각화 획득

-   Weakly-Supervised 지역화와 포인팅, 원래 모델의 충실도 면에서 기존
    방식보다 뛰어남

-   Human Study에서도 효과가 좋았으며, 분류자의 신뢰도를 믿을만 하다고
    판단하게 만듬

-   이미지 분류, 이미지 캡션 및 VQA를 포함한 작업을 위한 모델들에 즉시
    적용 가능
