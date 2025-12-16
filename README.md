# 🧸 Object Persona AI

Object Persona AI는 사용자가 업로드한 **일상 사물 이미지**를  
딥러닝 기반 객체 분석과 이미지 합성을 통해  
**의인화된 캐릭터 이미지**로 변환해주는 웹 서비스입니다.

---

## 📌 프로젝트 개요
- 과목명: 딥러닝
- 프로젝트 주제: AI 코딩을 이용한 프로그램 설계
- 프로젝트명: Object Persona AI
- 학번 / 이름: 2021145035 배재환

---

## 🎯 프로젝트 목표
- 사물 이미지에서 객체 영역을 자동으로 분리
- 객체 중심을 기준으로 얼굴(눈/입) 자동 배치
- 스타일(귀여움 / 잔잔함 / 액션)에 따라 다른 캐릭터 생성
- 웹 기반 인터페이스로 결과 확인 및 다운로드 제공

---

## 🧠 사용 기술
- Python
- PyTorch (U-Net 기반 구조)
- OpenCV
- PIL (Pillow)
- Streamlit

---

## ⚙️ 시스템 구성
1. 사용자가 사물 이미지를 업로드
2. 딥러닝 모델을 통해 객체 마스크 생성
3. OpenCV 기반 후처리로 마스크 품질 향상
4. 객체 중심(Centroid)을 계산하여 얼굴 위치 자동 정렬
5. 스타일에 맞는 눈/입 템플릿을 합성
6. 최종 캐릭터 이미지 및 페르소나 설명 출력

---

## 🖼️ 주요 기능
- 객체 기반 의인화 캐릭터 생성
- 얼굴 위치 자동 조정 (정밀 센터링)
- 스타일별 랜덤 눈/입 템플릿 적용
- 결과 이미지 PNG 다운로드 기능
- 웹 서비스 형태로 즉시 실행 가능

---
## 🎬 데모 동영상

아래 링크에서 **Object Persona AI의 실제 실행 과정**을 확인할 수 있습니다.

▶️ https://youtu.be/U8FDCI-Jr5A

---
📁 프로젝트 구조
Object-Persona-AI/
├── assets/
│   ├── eyes/
│   └── mouths/
├── models/
│   └── unet_model.py
├── utils/
│   ├── mask_inference.py
│   ├── compose_character.py
│   └── persona_text.py
├── web_app.py
└── .gitignore

---
🚀 실행 방법
pip install -r requirements.txt
streamlit run web_app.py
