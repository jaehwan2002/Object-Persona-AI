# utils/persona_text.py
def generate_persona(style: str, color: tuple[int, int, int]) -> str:
    r, g, b = color

    if r >= g and r >= b:
        tone = "따뜻하고 적극적인"
    elif g >= r and g >= b:
        tone = "안정적이고 편안한"
    else:
        tone = "차분하고 깊이 있는"

    if style == "귀여움":
        mood = "사람들을 볼 때마다 작은 웃음을 건네는 귀여운 캐릭터입니다."
    elif style == "잔잔함":
        mood = "조용히 자기 자리를 지키며 주변을 지켜보는 든든한 캐릭터입니다."
    else:  # 액션
        mood = "언제든지 튀어나와 모험을 떠날 준비가 되어 있는 에너지 넘치는 캐릭터입니다."

    return f"이 사물은 {tone} 분위기를 가진 존재로, {mood}"
