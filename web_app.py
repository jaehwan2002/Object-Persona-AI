import streamlit as st
from PIL import Image
import io
import random
import time

from utils.mask_inference import get_object_mask
from utils.compose_character import compose_character, extract_dominant_color
from utils.persona_text import generate_persona

st.set_page_config(page_title="Object Persona AI", layout="centered")

st.title("🧸 Object Persona AI")
st.write("일상 사물을 의인화된 캐릭터로 바꿔주는 딥러닝 웹 서비스")

uploaded_file = st.file_uploader("사물 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

style = st.selectbox("원하는 캐릭터 스타일을 선택하세요", ["귀여움", "잔잔함", "액션"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    st.subheader("⬇ 원본 이미지")
    st.image(original_img, use_container_width=True)

    if st.button("캐릭터 생성하기"):
        with st.spinner("AI가 캐릭터를 생성하는 중입니다..."):
            random.seed(time.time())

            mask_img = get_object_mask(original_img)
            final_img = compose_character(original_img, mask_img, style)

            dom_color = extract_dominant_color(original_img)
            persona_text = generate_persona(style, dom_color)

        st.subheader("✨ 의인화된 캐릭터 이미지")
        st.image(final_img, use_container_width=True)

        st.subheader("🧠 AI 페르소나 분석")
        st.write(persona_text)

        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        st.download_button(
            label="📥 결과 이미지 다운로드 (PNG)",
            data=buf.getvalue(),
            file_name="object_persona_ai_result.png",
            mime="image/png"
        )
else:
    st.info("👆 사물 이미지를 업로드하면 캐릭터를 생성할 수 있습니다.")
