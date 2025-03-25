# 2025-02-26

## 오늘 한 일 ✅

AWS EC2를 이용했지만, 선생님의 소개로 Featurize.cn 이라는 곳을 알아냄.
Featurize의 장점:

- 초기설정 간편
- 중국 플랫폼
- 저렴한 가격
- GUI 지원
- SSH접속, port개방, Vscode연동 매우 편리
- Conda와 같은 기본 라이브러리들 자체 탑제

저번에 만들었던 Flask서버를 clone하고 서버 테스트까지 성공, featurize port export 5000 를 실행하면 외부에서 접속할 수 있는 주소와 포트를 알려줌.

deepseek-vl2-tiny 모델을 테스트 해봤지만, GPU 요구 성능에 도달하지 못해서 실패, 포기.

bczhou/tiny-llava-v1-hf 를 테스트 하는 과정에서 알 수 없는 오류가 계속 뜸

docs의 requirements and installation에서 설명한대로 새로운 conda 환경 구축 후 필요 패키지 설치하니 잘 작동함.

## 🎯 다음 목표

- api request 수신 시 bczhou 모델로 process하여 response 보내기
- 약간의 prompt 공부

# 2025-03-02

## 오늘 한 일 ✅

- 수신한 이미지를 bczhou/llava 모델로 process하여 전송해주는 api 서버 구축

## 🎯 다음 목표

- 서버 body의 params들 확립 (그러기 위해선 'index' 자료 구조를 정의하고, 'index' 값들을 정할 것)
- 약간의 prompt 공부

# 2025-03-11

## 오늘 한 일 ✅

- Prompt 공부. 원하는 출력값을 얻기 위해서 여러 시도를 해봤지만 기대한 출력이 나오지 않음. 
선생님이 알려준대로 temperature, top_p와 같은 parameter들도 건드려 봤지만 별 차이가 없음.
혹시 몰라서 다른 모델을 사용해봤는데 훨씬 더 작동을 잘함.
(bczhou/tiny-llava-v1-hf -> deepseek-ai/deepseek-vl-7b-chat)

## 🎯 다음 목표

- 같은 환경에서 두 모델들의 성능 비교 및 기록.

# 2025-03-18

## 오늘 한 일 ✅

- "/process" 엔드포인트에 body를 ("image", "text")를 필수로 받았었는데, "text"는 선택으로 받게 바꿨다("text"가 없을 시 None값으로 설정)
- Featurize 특성상 `/home/featurize/work`이하의 디렉토리를 제외한 모든 파일들은 재부팅 시 사라졌는데, transformer cache파일 경로를 `/home/featurize/work/openvino-flask-server/model_cache`으로 설정하니 재부팅마다 모델을 재다운받는 불편한 일이 없어졌다.

# 2025-03-19

## 오늘 한 일 ✅

- 다음과 같은 코드 작성:
    1. 클라이언트에서 **(이미지/객체)** 또는 **이미지**를 받아온다.
    2. **이미지/객체**를 받았다면:  
    - 객체가 `dangerous_objects` 리스트에 있는지 확인한다.  
    - 있으면 `content_danger` 프롬프트 적용, 없으면 `content_safe` 프롬프트 적용.  
    3. **이미지**를 받았다면:  
    - CLIP 모델로 객체를 인식하고, 정확도가 가장 높은 객체를 추출한다.  
    - 이후 2번 과정과 동일하게 진행.

## 🎯 다음 목표

- 같은 환경에서 deepseek와 bczhou 두 모델들의 성능 비교 및 기록하기.
- featurize에서 transformers의 모델 캐싱에 실패했는데, 캐싱을 통해 모델 불러오는 시간을 단축하는 방법 연구하기.
- Openvino로 가속화 하는 방법 연구하기.

# 2025-03-24

## 오늘 한 일 ✅

## 1. 패키지 버전 관리

- 버전 충돌 해결: `torch==2.2.2`, `transformers==4.38.2` 호환성 맞춤
- 주요 패키지 버전:
    - Python 3.11.8
    - Flask 3.1.0
    - Pillow 10.3.0
    - Featurize 플랫폼 대응: 모든 설치에 `--user` 옵션 적용 (`pip install --user <패키지>`)

## 2. 프롬프트 생성 시스템 구현

- **객체 감지 파이프라인**:
- `detected_objects`가 비었을 경우 CLIP 기반 `detect_objects()`로 자동 채움
- **동적 프롬프트 생성**:
- `generate_prompt()` 함수가 `dangerous_objects` 포함 여부에 따라:
  - 위험물 존재 → `dangerous_prompt` 반환 (즉시 경고)
  - 안전 → `safe_prompt` 반환 (TTS 최적화된 출력)

## 3. 이미지 처리 유틸리티

- `load_image()` 함수 추가:
- 지원 형식: 파일 경로, 파일 객체, BytesIO, 이미지 URL
- 멀티포맷 입력 대응 가능

## 🎯 다음 목표

- 같은 환경에서 deepseek와 bczhou 두 모델들의 성능 비교 및 기록하기.
- featurize에서 transformers의 모델 캐싱에 실패했는데, 캐싱을 통해 모델 불러오는 시간을 단축하는 방법 연구하기.
- Openvino로 가속화 하는 방법 연구하기.

# 2025-03-25

## 오늘 한 일 ✅

- 서버 시작 시 전역 변수로 DeepSeek 모델 1회 로드해 추론 시간 20초 → 2초로 90% 단축

## 🎯 다음 목표

- Openvino로 가속화 하는 방법 연구하기.
- 같은 환경에서 deepseek와 bczhou 두 모델들의 성능 비교 및 기록하기.

# 2025-03-26

## 오늘 한 일 ✅

- 여러 모델들의 기본 추론 시간/Openvino 가속화 추론 시간을 비교 (DeepSeek-VL-7B-Chat, Qwen2.5-VL-7B-Instruct, Llama-3.2-11B-Vision-Instruct)

## 🎯 다음 목표

- Intel CPU/GPU 환경에서 추가 테스트