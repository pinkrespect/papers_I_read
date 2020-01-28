Learning What and Where to draw

1.  Abstract

2.  Introduction

3.  Related Work

4.  Preliminaries

    A.  *Generative Adversarial Network*

    B.  *Structured joint embedding of visual description and images*

5.  Generative Adversarial What-Where Networks(GAWWN)

    A.  *Bounding-box-conditional text-to-image model*

    B.  *Keypoint-conditional text-to-image model*

    C.  *Conditional keypoint generation model*

6.  Experiments

    A.  *Controlling bird location via bounding boxes*

    B.  *Controlling individual part locations via keypoints*

    C.  *Generating both bird keypoints and images from text alone*

    D.  *Comparison to previous work*

    E.  *Beyond birds: generating images of humans*

7.  Discussion

```{=html}
<!-- -->
```
1.  Abstract

-   기존 모델의 한계

    -   Generative Adversarial Networks(GANs)

        -   실내장식, 앨범 커버, 만화, 얼굴, 새, 꽃과 같은 현실 물건을
            합성 가능

        -   클래스 라벨이나 캡션과 같은 전역적 제약조건에 근거하여
            이미지를 합성할 수 있음

            -   반면 포즈나 객체 위치에 대한 제어를 제공하지 않음

-   Generative Adversarial What-Where Network(GAWWN) 제안

    -   어떤 콘텐츠를 어떤 위치에 그릴지 설명하는 지시를 받은 이미지를
        합성

        -   Caltech-UCSD 조류 데이터 세트에 대한 고품질 128 \* 128
            이미지 합성을 비공식적인 텍스트 설명과 객체 위치를 조건화

        -   새 주위에 있는 Bounding Box와 새의 부분에 대한 통제권을 넘김

        -   새의 부분에 대한 조건부 분포를 모델링하여, MPIII 휴먼 포즈
            데이터셋에 있는 인간 포즈 이미지의 텍스트와 위치에 대해 제어
            가능한 합성과 영역의 조건부 분포를 가능하게 함

2.  Introduction

-   비정형적 설명에서 현실적인 이미지를 생성하면 광범위한 적용이 가능

-   현대 컴퓨터 그래픽은 이미 현저히 사실적인 장면을 생성할 수 있지만,
    여전히 높은 수준의 concepts와 최종 제품 사이의 간격을 좁히기 위해
    상당한 노력이 필요

-   심층 네트워크에는 빠르게 제어 가능한 영상 합성 능력이 있음

-   영상 생성 시스템이 유용하려면 생성할 장면의 내용에 대한 높은 수준의
    제어를 지원해야 함

-   convolutional GANS(Generative Adversarial Network) 사용, 제어력을
    갖춘 영상 합성 기능가능

-   Variational Autoencoders도 조건부 영상 합성, 특히 DRAW와 같은
    recurrent version에 대한 어떤 가능성을 보여줌

-   현재 접근방식은 지금까지 클래스 라벨이나 비 로컬화 캡션과 같은
    단순한 조건화 변수만 사용했으며, 장면에 물체가 나타나는 곳을
    제어하는 것을 허용하지 않음

-   현실적이고 복잡한 장면을 생성하는 방법

    -   영상 합성 모델은 부분적인 개체의 개념을 통합함으로써 이익을 얻을
        수 있음

    -   동일한 유형의 물체는 다양한 크기, 포즈 및 구성으로 많은 위치에서
        나타날 수 있음

    -   계산의 각 단계에서 이미지를 수정하기 위해 \"what\"와 \"where\"의
        질문을 분리

    -   매개변수 효율성에 더해, 네트워크가 각 위치에서 묘사하는 것을
        추적할 수 있다는 점이 장점

-   이미지 데이터 세트

    -   Caltech-USCD Birds(CUB)

        -   클래스 라벨과 같은 글로벌 라벨

        -   관련 텍스트 캡션 존재

        -   조류 부분 키 포인트

    -   MPIII Human Pose 데이터 세트(MHP)

        -   인간 포즈 같은 Localized 라벨

        -   이미지당 3개의 캡션이 포함된 새로운 데이터 세트

-   이 논문의 모델은 위의 데이터셋에서 위치 및 콘텐츠 제어 가능한 이미지
    합성을 수행하는 것을 학습

    -   Spatial transformer을 사용하여 텍스트 조건의 GAN에 모듈을
        결합하여, 조류의 위치를 조절

    -   우리는 새와 인간의 일부 위치를 Normalized된 좌표(x, y)의 형태로
        조절 가능

        -   Generator와 discriminator는 관련 파츠의 위치를 준수하기 위해
            multiplicative gating 메커니즘을 사용

-   주요 주제

    -   보다 사실적이고 고해상도 CUB 샘플을 산출하는 텍스트 및 위치 제어
        가능한 영상 합성

    -   파츠 위치를 지정하기 위한 사용자 인터페이스를 가능하게 하는
        텍스트 조건의 객체 파츠 완성 모델

    -   인간 이미지 합성에서 포즈 조건 텍스트에 대한 새로운 데이터 세트
        및 탐색 결과

3.  Related Work

-   이미지를 생성

    -   모두 결정론적(즉, 기존의 피드-포워드 및 recurrent
        신경망)이었으며, 잠재된 공간에서 픽셀 공간으로 일대일 매핑을
        배우도록 훈련

    -   Dosovitskiy et al: 형태, 위치 및 조명을 나타내는 그래픽 코드
        세트에 조건화된 3D 의자 렌더링을 생성하기 위해 Deconvolution
        network Training

    -   Yang et al: 회전하는 의자와 얼굴 이미지의 시퀀스를 생성하기 위해
        incremental 3D 회전을 적용하는 법을 배운 반복적인 convolutional
        인코더-디코더를 사용

    -   Oh et al: 아타리 게임의 액션 조건부 미래 프레임을 예측하기 위해
        유사한 접근법을 사용

    -   Reed et al: 시각적 유추 문제를 해결하는 이미지를 생성하기 위해
        네트워크를 훈련

-   가변 오토엔코더로 확률론적 모델을 학습(Kigma 및 Wellding, 2014년,
    Rezende 외, 2014년)

    -   Kulkarni et al: 그래픽 코드에 해당하는 창 모양의 단위 블록으로
        잠복된 공간을 \"해제\"하는 원통형 변형 자동인코더를 개발

    -   Gregor et al: 각 시간 단계에서 이미지 캔버스의 부분을 읽고 쓰는
        데 주의를 기울이는 메커니즘을 가진 recurrent variational
        auto-encoder을 만들었다. (DRAW)

-   VAE 기반 이미지 생성 모델 외의 단순하고 효과적인 Generative
    Adversarial Networks \[Goodfellow et al, 2014\]

    -   Denton et al., 2015: Class-Conditional GAN은 Residual Image의
        Laplacian pyramid를 generator network에 통합

    -   Radfor et al. 2016: deep convolutional GAN Training을 안정화하고
        얼굴과 인테리어 이미지를 합성할 수 있는 방법을 제안

-   Spatial Transformer Network (STN) \[Jaderberg et al.\]: visual
    attention 메커니즘으로 입증, 최신 심층 생성 모델에 통합

    -   Eslami et al: STN을 영상 의존적인 추론 단계 수를 사용하는
        Air(Attend, Infer, Repeat)라고 하는 반복적인 VAE의 형태로
        통합하여 단순한 다중 객체 2D 및 3D 장면을 생성하는 방법을 학습

    -   Rezende et al: 인상적인 샘플 복잡성 시각화 특성을 가진 DRAW와
        같은 반복적인 네트워크에 STN을 구축

-   Larochelle & Murray \[2011\]: 조건의 산물로 이미지 픽셀에 대한
    분포를 정밀하게 모델링하기 위해 NADE(Neural Autorientive Density
    Estimator) 제안

    -   Theis and Gethge, 2015, Van Den Oord et al, 2016: 최근에 소품
        처리된 공간 그리드 구조 반복 네트워크로 영상 학습 결과 향상

4.  Preliminaries

    A.  *Generative Adversarial Network(GANs)*

    -   Generator G, Discriminator D: 둘이 경쟁하는 Minimax Game

        -   Discriminator: 입력을 실제 또는 합성 입력으로 올바르게 분류

        -   Generator: Discriminator가 실제로 분류할 이미지를 합성

    -   Value Function V(D, G):

$$\begin{matrix}
\min \\
G \\
\end{matrix}\ \begin{matrix}
\max \\
D \\
\end{matrix}\text{\ V}\left( D,\ G \right) = \ \mathbb{E}_{\mathcal{X\sim}P\ data(\mathcal{X)}}\lbrack log{D\left( x \right)}{+ \ \mathbb{E}_{\mathcal{X\sim}P_{z}\left( z \right)}}\lbrack\log\left( 1 - D\left( G\left( z \right) \right) \right)\rbrack$$

-   Z: Nosie vector(Gaussian or uniform distribution)

-   G, D가 충분한 Capacity를 갖고, $p_{\text{data}}$에 수렴할 때,
    $p_{g} = p_{\text{data}}$인 경우 global 최적화 됨

```{=html}
<!-- -->
```
-   조건부 GAN 훈련

    -   G(z, c)와 D(x, c)를 산출하는 추가 입력 C를 Generator,
        Discriminator에게 제공

    -   입력 튜플(x, c)을 \"실제\"로 해석하려면 이미지 x가 사실적으로
        보일 뿐 아니라 컨텍스트 c와 모두 일치해야 함

    -   G는 logD(g(z, c))를 최대화하도록 훈련

B.  *Structured joint embedding of visual description and images*

-   텍스트 설명에서 시각적 내용을 인코딩

    -   Corresponding function 학습하기

        -   Sentence Embedding: Structure loss을 최적화

$$\frac{1}{N}\sum_{n = 1}^{N}{\mathrm{\Delta}\left( y_{n},\ f_{v}\left( v_{n} \right) \right) + \ \mathrm{\Delta}\left( y_{n},\ f_{t}\left( t_{n} \right) \right)}$$

-   Training data set:
    $\{\left( v_{n},\ t_{n},\ y_{n} \right),\ n = 1,\ \ldots,\ N\}$

-   ![](media/image1.emf): 0-1 loss

-   $v_{n}$: 이미지

-   $t_{n}$: Corresponding 텍스트 설명

-   $y_{n}$: 클래스 레이블

```{=html}
<!-- -->
```
-   $f_{v},\ f_{t}$는 아래 수식에 의해 유도됨

$$f_{v}\left( v \right) = \begin{matrix}
\arg\max \\
y \in \Upsilon \\
\end{matrix}\mathbb{E}_{t\sim\tau(y)}\lbrack{\phi\left( v \right)}^{T}\varphi\left( t \right))\rbrack,\ f_{t}(t) = \begin{matrix}
\text{argmax} \\
y \in \Upsilon \\
\end{matrix}\mathbb{E}_{\upsilon\sim\Upsilon\left( y \right)}\lbrack{\phi\left( v \right)}^{T}\varphi(t))\rbrack$$

-   $\phi$: 이미지 인코더

-   $\varphi$: 텍스트 인코더

-   $\tau(y)$: 클래스 y와 V(y)의 텍스트 설명의 집합

```{=html}
<!-- -->
```
-   텍스트 인코더의 학습

    -   다른 클래스에 비해 해당 클래스의 이미지와 더 높은 정확도 점수를
        산출하는 방법 학습

    -   방정식과 관련된 손실을 최소화

        -   char-CNN-RNN 대신 char-CNN-GRU를 사용

        -   방정식 1 대신 이미지당 평균 4개의 샘플링 캡션을 사용하여
            방정식 2의 기대치를 추정함

5.  Generative Adversarial What-Where Networks(GAWWN)

    A.  *Bounding-box-conditional text-to-image model*

![지도, 텍스트이(가) 표시된 사진 자동 생성된
설명](media/image2.tiff){width="6.270833333333333in"
height="2.076388888888889in"}

-   Input noise $z \in \mathbb{R}^{z}$와 텍스트 embedding
    $t \in \mathbb{R}^{T}$(사전 훈련된 인코더에 의해 캡션에서 추출된
    $\varphi(t)$)에서 시작

    -   텍스트 임베딩(녹색)을 공간적으로 복제, MXMXT feature map 구성

    -   공간적으로 Wraping, 정규화된 bounding box 좌표에 맞도록 함

    -   상자 밖의 feature map 항목은 모두 0

    -   위 그림에서는 단일 객체를 나타내지만, 복수의 localized caption인
        경우 feature map의 average를 구하

    -   공간 치수를 1X1로 다시 줄이기 위해 convolution과 풀링 작업 적용

    -   직관적으로, 이 feature 벡터는 이미지의 coarse spatial
        structure를 암호화

    -   이것을 noise vecotr z와 결합

-   generator가 local 및 global 처리 단계로 분기

    -   global 경로는 공간을 1X1에서 MxM으로 늘리기 위한 일련의
        Striide-2 deconvolution

    -   local 경로에서 공간 차원 MxM에 도달, 객체 bounding box 외부의
        영역이 0으로 설정되도록 마스킹 작업 적용

    -   지역 경로와 전역 경로는 depth concatenation에 의해 merge

        -   Deconvolution의 Final spatial series에 도달하기 위해
            Deconvolution 레이어의 final series layer를 사용

        -   final 레이어에서는 출력을 \[-1, 1\]로 제한하기 위해 tanh의
            비선형성을 적용

-   Discriminator에서 텍스트는 MxMxT 텐서를 형성하기 위해 공간적으로
    유사하게 복제됨

-   한편 이미지는 로컬 및 글로벌 pathway에서 처리

-   최소 경로에서 이미지는 Striide-2 confolution을 통해 MxM 공간
    차원으로 전달

    -   이 때 텍스트에 텐서를 포함하는 깊이와 결합

    -   결과 텐서는 경계 상자 좌표 이내로 잘림

    -   공간 치수가 1x1이 될 때까지 잘린 후, 원문 텍스트 임베딩 t의
        addictive contribution와 함께 벡터 변환

-   local 및 global pathaway output 벡터는 부가적으로 결합, 최종
    layer으로 공급

    -   스칼라 discriminator score를 산출

B.  *Keypoint-conditional text-to-image model*

![텍스트, 지도이(가) 표시된 사진 자동 생성된
설명](media/image3.tiff){width="6.263888888888889in"
height="2.2222222222222223in"}

-   위치 키포인트

    -   채널이 파츠에 해당하는 M X M X K 공간 형상 맵(즉, 채널 1의 머리,
        채널 2의 좌측 다리 등)으로 인코딩

    -   키포인트 Tensor는 네트워크의 여러 Stage로 공급

-   local 및 global pathaway에서 noise-텍스트-키포인트 벡터는
    deconvolution을 통해 공급

    -   또 다른 MXMXH 텐서를 생성

    -   Local pathaway activations는 같은 크기의 keypoint Tensor와
        pointwise 곱셈과 결합

    -   원래의 MxMxK 키포인트 Tensor는 local 및 global Tensor와 깊이
        결합

    -   추가 deconvolutions으로 처리하여 최종 이미지를 생성

    -   tanh nonlinearity 적용

-   Discriminator에서의 변화

    -   텍스트 embedding t는 두 단계로 공급

        -   이미지를 정교하게 처리하여 벡터 출력을 생성하는 글로벌
            pathway와 추가적으로 결합

        -   공간적으로 M x M에 복제, 다음 로컬 경로에 있는 다른 M x M
            feature map으로 심층 분석

        -   이 local 텐서는 generator와 정확히 같이 binary 키포인트
            마스크와 곱

        -   결과 Tensor는 M x M x T와 깊이 결합

        -   글로벌 pathway 출력 벡터와 추가 결합

        -   스칼라 식별자 점수를 생성하는 최종 레이어로 결합

C.  *Conditional keypoint generation model*

-   사용자-경험의 관점에서, 사용자가 그리기를 원하는 물체 부분의 모든
    핵심 지점을 입력하도록 요구하는 것은 최적이 아님

    -   관찰된 키포인트의 하위 집합과 텍스트 설명을 제공하는 관찰되지
        않은 키포인트의 모든 조건부 분포에 접근할 수 있다면 매우 유용할
        것

-   데이터 공급에서도 유사한 문제발생

    -   예를 들어 누락된 레코드 작성 또는 이미지 삽입

    -   가장 가능성이 높은 값을 채우기 보다는 설득력 있는 샘플을 뽑기를
        원함

-   예를 들어 새의 부리의 위치에만 맞춰진 채, 그 제약을 만족시키는 아주
    다른 몇 가지 그럴듯한 포즈가 있을 수 있음

    -   DBM 또는 가변 자동인코더는 이론에서는 사용할 수 있을 것 같지만
        복잡함

    -   단순성을 위해 이 문제에 동일한 일반 GAN 프레임워크를 적용하여
        달성한 결과를 입증

-   각 물체의 파츠의 할당을 관측(예: 조건화 변수)하거나, 또는 관측되지
    않은 것을 게이트 메커니즘으로 사용하는 것

    -   단일 영상의 키포인트를
        $k_{i} ≔ \left\{ x_{i},\ y_{i},\ v_{i} \right\},\ i = 1,\ \ldots,\ K$로
        표현

        -   x와 y는 각각 행과 열 위치를 나타내고, v는 파츠가 보이는 경우
            1, 그렇지 않으면 0 설정

        -   파츠가 보이지 않으면 x와 y도 0으로 설정

        -   k가 키포인트를 행렬로 인코딩하도록 하고, i번째 부분이 조건화
            변수인 경우 i번째 항목이 1로 설정, 그렇지 않으면 0으로
            설정된 상태에서 조건화 변수(예: 사용자가 지정한 부리 위치)를
            스위치 장치 s의 벡터로 인코딩하도록 함

        -   텍스트 T와 키포인트 k, s의 하위 집합에 맞춰 키포인트
            $G_{k}$를 통해 제너레이터 네트워크를 공식화 가능

$$G_{k}\left( z,\ \mathbf{t,\ k,\ }s \right) ≔ s \odot \mathbf{k} + \left( 1 - s \right) \odot f(z,\ \mathbf{t},\mathbf{\text{\ k}})$$

-   여기서 ![](media/image4.emf)는 pointwise multiplication을 의미

-   $f:\mathbb{R}^{Z + T + 3K} \rightarrow \mathbb{R}^{3K}$는 MLP

-   실제로 우리는 z, t, flattened k를 concatenate

-   f를 3층 Fully-connected로 선택

```{=html}
<!-- -->
```
-   Discriminator $D_{k}$는 진짜 요점과 텍스트($k_{\text{real}}$,
    $t_{\text{real}}$)를 Synthetic과 구별하는 법을 학습

-   $G_{k}$가 모든 조건부 분포를 키포인트와 비교하여 포착하기 위해 훈련
    중에 각 미니 배치의 스위치 유닛을 무작위로 샘플링

    -   보통 1 또는 2개의 요점을 명시하고 싶기 때문에, 우리의 실험에서
        \"on\" 확률을 0.1로 설정

        -   즉, 15개의 새 부품이 다른 부위로 움직일 확률이 10%

6.  Experiments

-   Caltech-UCSD Birds(CUB) 및 MPII Human Pose(MHP) 데이터 세트에 대해
    텍스트 설명에서 이미지를 생성하기 위한 실험 진행

    -   CUB

        -   200종 중 하나에 속하는 새의 11,788 이미지

        -   조류 이미지당 10개의 한 단어 설명을 포함한 텍스트 데이터
            존재

        -   각 이미지에는 경계 상자를 통한 조류 위치 및 15개의 조류 부품
            각각에 대한 키포인트 (x, y) 좌표가 포함

        -   각 이미지에는 모든 파츠가 보이지는 않으며, 보이는 핵심
            데이터 또한 파츠 당 보이는지 여부를 나타내는 값을 제공한다.

    -   MHP

        -   410개의 서로 다른 공통 활동을 가진 25K 이미지

        -   각 이미지에 대해 Mechanical Turk를 사용하여 3개의 한 단어
            설명을 수집

        -   노동자들에게 그 사람의 가장 독특한 면과 그들이 하고 있는
            활동에 대해 설명해줄 것을 요청

            -   예를 들어, \"골프를 휘두를 준비를 하고 있는 노란 옷을
                입은 남자\"

        -   각 이미지에는 16개의 조인트가 있으며, 각각에 대해 여러 개의
            (x, y) 키포인트 세트가 잠재적으로 존재

        -   훈련하는 동안 우리는 여러 사람과 함께 이미지를 걸러냄

        -   나머지 19K 이미지들은 그 사람의 Bounding Box에 이미지를 자름

    -   미리 훈련된 Char-CNN-GRU를 사용하여 캡션을 암호화

        -   훈련하는 동안, 주어진 이미지에 대해 1024 차원 텍스트
            임베딩은 해당 이미지에 해당하는 무작위로 샘플링된 4개의 캡션
            인코딩의 평균이 됨

        -   이미지당 여러 캡션을 샘플링하여, 객체를 그리는 데 필요한
            추가 정보 획득

        -   테스트 시간에는 하나의 자막을 포함하여 모든 수의 설명에
            평균을 낼 수 있음

    -   배치 크기 16과 학습 속도 0.0002의 ADAM solver를 사용, GAWWN을
        교육

        -   모델들은 모든 카테고리에서 훈련

        -   일련의 캡션에 대한 샘플을 제공

        -   Spatial Transformer 모듈의 경우, Oquab에서 제공하는 torch
            구현을 사용

        -   dcgan.torch를 느슨하게 기반으로 함

    -   실험에서 GAWWN 샘플이 텍스트와 위치 제약조건을 얼마나 정확하게
        반영하는지 분석

        -   우선 경계상자와 keypoint을 통해 보간법으로 새의 위치를 통제

        -   \(1\) 데이터 세트의 지상-진실 키포인트와 (2) 본문에 조건화된
            우리 모델에 의해 생성된 합성 키포인트의 경우를 모두 고려.

            -   (2)는 가상 사용자(즉, 15개의 키포인트 위치 입력)의
                노력이 덜 필요하기 때문에 유리

        -   그런 다음 이전 작업의 대표적인 샘플에 CUB 결과를 반영

        -   마지막으로, 인간의 행동에 대한 텍스트와 포즈 조건 생성에
            대한 샘플을 제공

    A.  *Controlling bird location via bounding boxes*

![사진이(가) 표시된 사진 자동 생성된
설명](media/image5.tiff){width="6.263888888888889in"
height="2.076388888888889in"}

B.  *Controlling individual part locations via keypoints*

![스크린샷이(가) 표시된 사진 자동 생성된
설명](media/image6.tiff){width="6.263888888888889in"
height="3.8541666666666665in"}

C.  *Generating both bird keypoints and images from text alone*

![나무, 사진, 표시중, 실외이(가) 표시된 사진 자동 생성된
설명](media/image7.tiff){width="6.263888888888889in"
height="1.9097222222222223in"}

D.  *Comparison to previous work*

![사진, 표시중이(가) 표시된 사진 자동 생성된
설명](media/image8.tiff){width="6.263888888888889in"
height="3.1458333333333335in"}

E.  *Beyond birds: generating images of humans*

![잔디, 사진이(가) 표시된 사진 자동 생성된
설명](media/image9.tiff){width="6.263888888888889in"
height="2.0347222222222223in"}

7.  Discussion

-   비공식적인 텍스트 설명과 객체 위치에 조건화된 이미지를 생성

    -   위치는 경계 상자 또는 파츠 키포인트 세트로 정확하게 제어 가능

    -   CUB에서는 위치 제약 조건을 추가하여 128 \* 128 이미지를 정확하게
        생성할 수 있었지만 이전 모델은 64 \* 64 영상만 생성

    -   위치 조절은 부품 위치의 텍스트 조건 생성 모델을 학습하고 테스트
        시간에 생성할 수 있어 시간적 이점 있음

-   문제를 더 쉬운 하위 문제로 분해하는 것이 현실적인 고해상도 이미지를
    생성하는 데 도움이 될 수 있음을 알 수 있음

    -   전체 텍스트에서 이미지 파이프라인을 GAN으로 쉽게 훈련시킬 수
        있을 뿐만 아니라, 이미지 합성을 제어하는 추가 방법 도출

-   Future Work

    -   Unsupervised 또는 Weak Supervised 방법으로 물체나 부품 위치를
        학습
