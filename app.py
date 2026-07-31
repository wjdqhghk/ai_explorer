import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import random
import os
import io
import math
import base64
import textwrap
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageDraw, ImageFont

# ── media 폴더 위치 찾기 (페이지 아이콘에 쓰기 위해 가장 먼저 계산한다) ──
APP_DIR_ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_media_root():
    """media 폴더 위치를 찾는다.
    - 배포 폴더 구조: (루트)/program/app.py + (루트)/media/...  → 상위 폴더의 media
    - 혹시 app.py와 같은 폴더에 media를 두고 실행해도 동작하도록 함께 지원한다."""
    candidates = [
        os.path.join(os.path.dirname(APP_DIR_ROOT), "media"),  # ../media (기본 배포 구조)
        os.path.join(APP_DIR_ROOT, "media"),                    # ./media
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


MEDIA_DIR = _find_media_root()
BADGE_ASSET_DIR = os.path.join(MEDIA_DIR, "images", "badges")
AUDIO_ASSET_DIR = os.path.join(MEDIA_DIR, "sound")
IMAGE_ASSET_DIR = os.path.join(MEDIA_DIR, "images")
MOVIE_DIR = os.path.join(MEDIA_DIR, "movie")


def _load_page_icon():
    """브라우저 탭 아이콘으로 쓸 로봇 마스코트 이미지를 불러온다.
    파일이 없으면 기본 이모지(🧸)로 안전하게 대체한다."""
    icon_path = os.path.join(IMAGE_ASSET_DIR, "robot_mascot.png")
    try:
        if os.path.exists(icon_path):
            return Image.open(icon_path)
    except Exception:
        pass
    return "🧸"


# 1. 페이지 설정
st.set_page_config(
    page_title="손끝에서 배우는 인공지능 원리 AI 탐험대 (초등 6학년)",
    layout="wide",
    page_icon=_load_page_icon(),
    initial_sidebar_state="expanded",
)


def md_html(s):
    """들여쓰기된 여러 줄 HTML을 st.markdown으로 안전하게 렌더링.
    (들여쓰기가 있으면 마크다운이 코드블럭으로 착각해 HTML 태그가 그대로 노출되는 문제를 방지)"""
    st.markdown(textwrap.dedent(s), unsafe_allow_html=True)


# =========================================================
# Streamlit 버전 호환 레이어
# =========================================================
# 최신 Streamlit(대략 1.42+)은 **_STRETCH 를 쓰지만, 그보다 낮은 버전에서는
# 이 인자가 없어 TypeError가 발생한다("어? 이거 이미 알아" 탭에서 나던 에러의 원인).
# 설치된 버전을 확인해서 알맞은 인자를 자동으로 골라 쓰도록 호환 레이어를 둔다.
def _get_stretch_kwargs():
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return {"use_container_width": True}
    if (major, minor) >= (1, 42):
        return {"width": "stretch"}
    return {"use_container_width": True}


_STRETCH = _get_stretch_kwargs()


# =========================================================
# 디자인 토큰 & 전역 스타일
# =========================================================
# 팔레트: 하늘(배경) / 인디고(사이드바·강조) / 6개 섹션 고유 색
PALETTE = {
    "sky": "#EAF6FF",
    "card": "#FFFFFF",
    "navy": "#1B2A4A",
    "coral": "#FF6F59",
    "sun": "#FFC93C",
    "mint": "#2EC4B6",
}
SECTION_COLORS = {
    "home": "#2D9CDB",
    "reg": "#2D9CDB",      # 1. 선형회귀 - 하늘색
    "gd": "#4FC3F7",       # 2. 경사하강법 - 눈 덮인 얼음색
    "dt": "#43A047",       # 3. 결정트리 - 숲 초록
    "knn": "#8E24AA",      # 4. KNN - 외계 보라
    "km": "#2EC4B6",       # 5. K-Means - 무인도 민트
    "nn": "#FF6F59",       # 6. 신경망 - 로켓 코랄
}

md_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Jua&family=Gowun+Dodum&display=swap');

html, body, [class*="css"], .stMarkdown, p, span, label, li { font-family: 'Gowun Dodum', sans-serif; }
h1, h2, h3, h4 { font-family: 'Jua', sans-serif !important; letter-spacing: -0.5px; }

.stApp { background: #EAF6FF; }

/* 사이드바 */
section[data-testid="stSidebar"] { background: #1B2A4A; }
section[data-testid="stSidebar"] * { color: #FFFFFF !important; font-family: 'Gowun Dodum', sans-serif; }
section[data-testid="stSidebar"] h1 { font-family: 'Jua', sans-serif !important; }
section[data-testid="stSidebar"] label { color: #EAF6FF !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
section[data-testid="stSidebar"] .stRadio label { background: #24365C; border-radius: 12px; padding: 8px 10px; margin-bottom: 2px; }

/* ── 사이드바 글자만 살짝 줄여 공간을 아낀다 (폭은 Streamlit 기본값 유지) ── */
section[data-testid="stSidebar"] h1 { font-size: 1.35rem !important; }
section[data-testid="stSidebar"] .stRadio label { padding: 6px 9px; font-size: 0.92rem; }

/* ── 화면을 컴팩트하게: 여백을 줄여 한 화면에 더 많이 담기게 한다 ── */
.block-container { padding-top: 1.2rem !important; padding-bottom: 1.5rem !important; }
.block-container h1 { margin-top: 0.2rem !important; margin-bottom: 0.4rem !important; }
.block-container h2 { margin-top: 0.5rem !important; margin-bottom: 0.35rem !important; }
.block-container h3 { margin-top: 0.5rem !important; margin-bottom: 0.3rem !important; }
.block-container h4 { margin-top: 0.4rem !important; margin-bottom: 0.25rem !important; }
.block-container hr { margin-top: 0.7rem !important; margin-bottom: 0.7rem !important; }
[data-testid="stVerticalBlock"] { gap: 0.55rem !important; }
.stAlert { padding-top: 0.6rem !important; padding-bottom: 0.6rem !important; }
[data-testid="stExpander"] { margin-bottom: 0.4rem !important; }

/* 버튼 */
.stButton>button {
    border-radius: 999px; border: none; font-family: 'Jua', sans-serif;
    background: #FF6F59; color: white; padding: 0.55em 1.3em;
    box-shadow: 0 4px 0 #C6503F; transition: all 0.08s ease;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 0 #C6503F; }
.stButton>button:active { transform: translateY(2px); box-shadow: 0 1px 0 #C6503F; }

/* 카드형 요소: metric, expander, alert, dataframe */
[data-testid="stMetric"] {
    background: white; border-radius: 18px; padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(27,42,74,0.08);
}
.stAlert { border-radius: 16px; border: none; }
.streamlit-expanderHeader, [data-testid="stExpander"] {
    border-radius: 16px !important; overflow: hidden;
}
[data-testid="stExpander"] { box-shadow: 0 2px 10px rgba(27,42,74,0.06); }

/* 탭 */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: white; border-radius: 14px 14px 0 0; font-family: 'Jua', sans-serif;
    padding: 10px 16px; box-shadow: 0 -2px 8px rgba(27,42,74,0.05);
}
.stTabs [aria-selected="true"] { background: #2D9CDB !important; color: white !important; }
.stTabs [aria-selected="true"] p { color: white !important; }

/* 진행바 */
.stProgress > div > div > div { background: linear-gradient(90deg, #2EC4B6, #2D9CDB); }

/* 슬라이더 손잡이 */
.stSlider [data-baseweb="slider"] > div > div { background: #2D9CDB; }

/* 태블릿 터치 조작을 위해 슬라이더 손잡이를 더 크게 (손가락으로 잡기 쉽게) */
.stSlider [role="slider"] {
    width: 26px !important; height: 26px !important;
    box-shadow: 0 2px 6px rgba(27,42,74,0.35) !important;
}

/* =========================================================
   태블릿·좁은 화면 대응 (반응형)
   - 컴퓨터실 데스크탑은 넓은 화면(와이드 레이아웃)을 그대로 쓰고,
     태블릿처럼 화면 폭이 좁을 때만 여러 칸(columns)을 세로로 쌓아
     글자·버튼이 너무 작아지지 않게 한다.
   ========================================================= */
@media (max-width: 820px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* 버튼과 라디오 터치 영역을 조금 더 넓게 */
    .stButton>button { padding: 0.7em 1.3em; font-size: 16px; }
    section[data-testid="stSidebar"] .stRadio label { padding: 12px 12px; }
}
</style>
""")


def hero_card(emoji, title, subtitle, color="#2D9CDB", term=None):
    """섹션 상단에 표시되는 카드형 히어로 헤더"""
    term_html = ""
    if term:
        term_html = (f'<div style="display:inline-block; margin-top:14px; background:rgba(255,255,255,0.28); '
                     f'border:2px solid rgba(255,255,255,0.75); border-radius:999px; padding:8px 22px; '
                     f'font-family:\'Jua\', sans-serif; font-size:20px; color:white;">'
                     f'🏷️ 오늘의 AI 원리: <b>{term}</b></div>')
    card_html = (f'<div style="background: linear-gradient(135deg, {color}, {color}CC); '
                 f'border-radius: 26px; padding: 30px 34px; margin-bottom: 20px; '
                 f'box-shadow: 0 10px 28px rgba(27,42,74,0.18);">'
                 f'<div style="font-size: 46px; line-height: 1;">{emoji}</div>'
                 f'<div style="font-family: \'Jua\', sans-serif; font-size: 30px; color: white; margin-top: 8px;">{title}</div>'
                 f'<div style="font-family: \'Gowun Dodum\', sans-serif; font-size: 16px; color: white; opacity: 0.92; margin-top: 8px; line-height: 1.6;">{subtitle}</div>'
                 f'{term_html}'
                 f'</div>')
    st.markdown(card_html, unsafe_allow_html=True)


def section_card(content_html, color="#2D9CDB"):
    """옅은 색 배경의 보조 카드 (성취기준 등)"""
    md_html(f"""
    <div style="background: {color}14; border-left: 5px solid {color};
                border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;">
        {content_html}
    </div>
    """)


def add_sweetness_axis_hints(fig):
    """KNN 그래프에 축 의미를 직관적으로 알려주는 안내 문구 추가"""
    fig.add_annotation(x=0, y=-0.14, xref="paper", yref="paper", text="🍋 전혀 안 달다", showarrow=False, font=dict(size=11, color="#3A4A6B"))
    fig.add_annotation(x=1, y=-0.14, xref="paper", yref="paper", text="엄청 달다 🍯", showarrow=False, font=dict(size=11, color="#3A4A6B"))
    fig.add_annotation(x=-0.12, y=0, xref="paper", yref="paper", text="작다", showarrow=False, textangle=-90, font=dict(size=11, color="#3A4A6B"))
    fig.add_annotation(x=-0.12, y=1, xref="paper", yref="paper", text="크다", showarrow=False, textangle=-90, font=dict(size=11, color="#3A4A6B"))
    return fig


# (media 폴더 경로 및 APP_DIR_ROOT는 페이지 아이콘 설정을 위해 파일 상단에서 이미 정의했다)

# =========================================================
# 오디오 설정 — 파일명을 여기서 한 번에 바꿀 수 있어요.
# =========================================================
# 배경음악(무한반복)으로 쓸 곡과 부품 획득 효과음.
BGM_HOME_FILE = "music1.mp3"     # 도입(홈) 화면 배경음악
BGM_CERT_FILE = "music2.mp3"     # 수료증 화면 배경음악
SFX_BADGE_FILE = "key.mp3"       # 부품(배지) 획득 효과음


@st.cache_data(show_spinner=False)
def _file_b64_cached(path, _mtime):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _img_b64_cached(path, _mtime):
    return _file_b64_cached(path, _mtime)


def _img_b64(filename):
    """media/images/badges 안의 이미지를 base64로 인코딩해서 반환 (없으면 None).
    로컬 파일을 base64로 직접 HTML에 심으면 배포 환경/오프라인 시연에서도 항상 안전하게 보인다.
    파일 수정시각을 캐시 키에 넣어, 파일을 바꾸면 새 이미지가 즉시 반영된다."""
    path = os.path.join(BADGE_ASSET_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    return _img_b64_cached(path, mtime)


def _audio_b64(filename):
    """media/sound 안의 오디오를 base64로 반환 (없으면 None)."""
    path = os.path.join(AUDIO_ASSET_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    return _file_b64_cached(path, mtime)


def _asset_img_b64(filename):
    """media/images 안의 이미지를 base64로 반환 (없으면 None). 마스코트 등에 사용."""
    path = os.path.join(IMAGE_ASSET_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    return _file_b64_cached(path, mtime)


def render_bgm_player(filename, section_key, label="🎵 배경음악", compact=False):
    """무한반복 배경음악 + 켜기/끄기 버튼.
    - 브라우저 자동재생 정책상 처음엔 꺼진 상태로 두고, 버튼을 눌러야 재생된다.
    - 켜고/끄기 상태는 session_state에 저장돼, 같은 화면에 머무는 동안 유지된다.
    - 컴퓨터실에서 여러 대가 겹치지 않도록 기본은 '꺼짐'이 안전하다.
    - compact=True 면 작은 버튼 하나만(오른쪽 상단 배치용) 보여준다."""
    b64 = _audio_b64(filename)
    if not b64:
        st.caption(f"🎵 (배경음악 파일 {filename} 을 찾지 못했어요. media/sound 폴더를 확인하세요.)")
        return

    state_key = f"_bgm_on_{section_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    playing = st.session_state[state_key]

    if compact:
        # 오른쪽 상단에 작은 버튼만. (col 비율로 오른쪽 끝에 몰아 배치)
        spacer, col_btn = st.columns([4, 1])
        with col_btn:
            btn_label = "🔇 음악끄기" if playing else "🎵 음악켜기"
            if st.button(btn_label, key=f"bgm_btn_{section_key}"):
                st.session_state[state_key] = not playing
                st.rerun()
    else:
        col_btn, col_txt = st.columns([1, 2])
        with col_btn:
            if playing:
                if st.button("🔇 음악 끄기", key=f"bgm_btn_{section_key}", **_STRETCH):
                    st.session_state[state_key] = False
                    st.rerun()
            else:
                if st.button("🎵 음악 켜기", key=f"bgm_btn_{section_key}", **_STRETCH):
                    st.session_state[state_key] = True
                    st.rerun()
        with col_txt:
            st.caption("켜면 배경음악이 잔잔하게 반복돼요. 컴퓨터실에서는 헤드폰을 쓰거나 소리를 줄여주세요.")

    # 켜진 상태일 때만 오디오 태그를 심어서 자동재생(무한반복). 끄면 태그 자체가 사라져 소리도 멈춘다.
    if playing:
        components.html(f"""
        <audio autoplay loop id="bgm_{section_key}">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
        <script>
            const a = document.getElementById("bgm_{section_key}");
            if (a) {{ a.volume = 0.5; a.play().catch(()=>{{}}); }}
        </script>
        """, height=0)


def play_sfx(filename):
    """짧은 효과음을 한 번 재생 (부품 획득 순간 등).
    이미 사용자가 페이지에서 버튼을 눌러 상호작용한 뒤라 소리 재생이 허용된다."""
    b64 = _audio_b64(filename)
    if not b64:
        return
    # 매번 다른 id를 줘서, 연속 획득 시에도 확실히 재생되도록 한다.
    uid = random.randint(0, 1_000_000)
    components.html(f"""
    <audio autoplay id="sfx_{uid}">
        <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>
        const s = document.getElementById("sfx_{uid}");
        if (s) {{ s.volume = 0.7; s.play().catch(()=>{{}}); }}
    </script>
    """, height=0)




# 6개 섬을 클리어하면 '로봇 네오'의 부품을 하나씩 모으는 컨셉으로 배지를 구성한다.
# (emoji, 배지 이름[기존 session_state 값과 동일하게 유지], 이미지 파일명, 부품 이름, 부품 한 줄 설명)
ALL_BADGES = [
    ("🎯", "🎯 붕어빵 가격 박사", "r_sensor1.png", "센서 부품",
     "데이터를 재고 예측하는 눈"),
    ("🏂", "🏂 골짜기 탈출 영웅", "r_leg1.png", "다리 부품",
     "한 걸음씩 착실하게 내려가는 다리"),
    ("🕵️‍♂️", "🕵️‍♂️ 초고속 명탐정", "r_an1.png", "안테나 부품",
     "좋은 질문의 신호를 캐치하는 안테나"),
    ("🍇", "🍇 외계 알 구출대원", "r_heart1.png", "하트 부품",
     "가장 가까운 친구를 알아보는 마음"),
    ("🧹", "🧹 분리수거 마스터", "r_arm1.png", "팔 부품",
     "비슷한 것끼리 척척 정리하는 팔"),
    ("🤖", "🤖 로봇 네오 발사 성공", "r_brain1.png", "뇌 부품",
     "여러 판단을 모아 결론을 내리는 뇌"),
]


# =========================================================
# 복습 게임용 데이터 — "로봇 네오 종합 점검 미션"
# =========================================================
# 6개 인공지능 원리. (원리 이름, 짧은 별명) — 문제의 보기(선택지)로도 쓰인다.
AI_PRINCIPLES = [
    "선형회귀 (선 긋기)",
    "경사하강법 (한 걸음씩)",
    "결정트리 (스무고개)",
    "KNN (가까운 이웃)",
    "K-평균 군집화 (비슷한 끼리)",
    "인공신경망 (생각 주머니)",
]

# 각 부품 점검 문제.
# part_idx: ALL_BADGES/부품 순서와 동일 (0=센서, 1=다리, 2=안테나, 3=하트, 4=팔, 5=뇌)
# answer:  AI_PRINCIPLES 안의 정답 인덱스
# 선생님이 문구/문제를 쉽게 고치거나 추가할 수 있도록 리스트로 정리해두었습니다.
# (부품별로 문제가 여러 개면, 게임 시작 때마다 그중 하나가 랜덤으로 뽑힙니다.)
REVIEW_QUESTIONS = [
    # 0) 센서 부품 = 선형회귀
    {"part_idx": 0, "answer": 0,
     "scenario": "🍞 붕어빵이 클수록 값이 비싸져요. '크기'라는 숫자로 '가격'을 미리 예측하려고 해요.",
     "hint": "점들 사이에 곧은 '선'을 그어서 미래 값을 맞히던 원리예요."},
    {"part_idx": 0, "answer": 0,
     "scenario": "📏 공부한 시간이 늘어날수록 시험 점수가 올라가는 관계를 하나의 '선'으로 나타내 예측해요.",
     "hint": "원인이 커지면 결과도 비례해서 변하는 규칙을 '선'으로 찾아요."},

    # 1) 다리 부품 = 경사하강법
    {"part_idx": 1, "answer": 1,
     "scenario": "⛰️ 틀린 정도가 가장 작은 지점(골짜기 밑)을 찾으려고, 한 번에 못 가고 조금씩 걸음을 옮겨 내려가요.",
     "hint": "보폭이 너무 크면 튕겨나가던, 그 '한 걸음씩' 내려가던 원리예요."},
    {"part_idx": 1, "answer": 1,
     "scenario": "🎿 AI가 정답을 단번에 맞히지 않고, 오차를 조금씩 줄여가며 여러 번 고쳐 나가요.",
     "hint": "산을 더듬더듬 내려가며 가장 낮은 곳을 찾던 방법이에요."},

    # 2) 안테나 부품 = 결정트리
    {"part_idx": 2, "answer": 2,
     "scenario": "🦁 '알을 낳나요?' '물속에서만 사나요?' 예/아니오 질문으로 후보를 좁혀 동물을 맞혀요.",
     "hint": "좋은 질문으로 갈래를 나누며 정답을 찾던 스무고개 원리예요."},
    {"part_idx": 2, "answer": 2,
     "scenario": "🌿 나뭇잎을 '바늘 모양인가요?' 같은 질문으로 갈래갈래 나눠서 종류를 알아내요.",
     "hint": "질문마다 길이 두 갈래로 갈라지던 그 원리예요."},

    # 3) 하트 부품 = KNN
    {"part_idx": 3, "answer": 3,
     "scenario": "🍎 정체를 모르는 새 과일이 있어요. 주변에 가장 가까이 있는 이웃 과일들을 보고 다수결로 정해요.",
     "hint": "가까운 친구들에게 물어보고 따라가던 원리예요."},
    {"part_idx": 3, "answer": 3,
     "scenario": "👟 전학 온 친구가 늘 태권도부 친구들 옆에 있으면 '이 친구도 태권도 하나 봐!' 하고 짐작해요.",
     "hint": "가장 가까운 이웃이 누구인지 보고 판단하던 방법이에요."},

    # 4) 팔 부품 = K-평균 군집화
    {"part_idx": 4, "answer": 4,
     "scenario": "🧹 이름표가 하나도 없는 나뭇잎들을, 생김새가 비슷한 것끼리 알아서 몇 개의 모둠으로 나눠요.",
     "hint": "정답(이름표) 없이 비슷한 것끼리 스스로 묶던 원리예요."},
    {"part_idx": 4, "answer": 4,
     "scenario": "🎨 정답을 알려주지 않았는데도 AI가 색깔·모양이 닮은 것끼리 스스로 무리를 지어요.",
     "hint": "'비슷한 친구끼리' 모으던 그 원리예요."},

    # 5) 뇌 부품 = 인공신경망
    {"part_idx": 5, "answer": 5,
     "scenario": "🧠 여러 '판단 담당(뉴런)'이 각자 의견을 내고, 그 의견들을 다시 한 번 모아서 최종 결론을 내려요.",
     "hint": "요정 여러 명의 판단을 합쳐 로봇 파워를 정하던 원리예요."},
    {"part_idx": 5, "answer": 5,
     "scenario": "🤖 입력을 여러 층의 뉴런이 조금씩 다르게 계산하고 종합해서 똑똑한 결론을 만들어요.",
     "hint": "뉴런을 여러 층으로 쌓아 만든 '생각 그물망'이에요."},
]


# =========================================================
# 사전/사후 진단 평가 문항 — 학습 효과 측정용
# =========================================================
# 학습 전(사전)과 후(사후)에 똑같은 문항을 풀게 해서 점수 변화를 비교한다.
# answer는 AI_PRINCIPLES의 인덱스. (0선형회귀 1경사하강 2결정트리 3KNN 4K평균 5신경망)
DIAGNOSTIC_QUESTIONS = [
    {"q": "🍞 붕어빵 크기로 가격을 예측하는 것처럼, 숫자 데이터로 '선'을 그어 미래 값을 맞히는 방법은?",
     "answer": 0},
    {"q": "⛰️ 정답을 한 번에 못 맞히고, 오차를 조금씩 줄이며 한 걸음씩 가장 낮은 곳으로 내려가는 방법은?",
     "answer": 1},
    {"q": "🦁 예/아니오 질문으로 후보를 확 좁혀가며 답을 찾는 방법은?",
     "answer": 2},
    {"q": "🍎 새 데이터의 정체를, 가장 가까운 이웃들에게 물어보고 다수결로 정하는 방법은?",
     "answer": 3},
    {"q": "🧹 이름표(정답)가 없어도 비슷한 것끼리 스스로 무리를 짓는 방법은?",
     "answer": 4},
    {"q": "🧠 여러 뉴런이 각자 계산한 값을 다시 합쳐서 최종 결론을 내리는 방법은?",
     "answer": 5},
]


# =========================================================
# 사전/사후 설문 문항 (5점 척도) — 보고서 정량 분석용
# =========================================================
# 같은 영역을 학습 '전'과 '후'에 서로 다른 문장으로 묻는다.
# key: CSV 저장 및 대시보드 표시에 쓰이는 영역 이름
SURVEY_ITEMS = [
    {"key": "흥미도",
     "pre": "인공지능(AI)이 작동하는 원리를 배우는 것에 흥미가 얼마나 있나요?",
     "post": "로봇 네오의 부품을 모으는 미션을 수행한 후, AI 원리 학습에 대한 흥미가 얼마나 증가했나요?"},
    {"key": "AI이해력",
     "pre": "선형회귀, 경사하강법 등 인공지능이 데이터를 학습하는 원리를 얼마나 잘 이해하고 있나요?",
     "post": "인공지능이 정답을 찾고 데이터를 분류하는 원리를 이해하는 데 얼마나 도움이 되었나요?"},
    {"key": "문제해결력",
     "pre": "변수를 직접 조작하며 결과를 스스로 예측하고 발견하는 학습 활동에 얼마나 자신이 있나요?",
     "post": "슬라이더를 직접 움직여보며 오차를 줄여가는 활동이 스스로 탐구하는 데 얼마나 도움이 되었나요?"},
    {"key": "앱만족도",
     "pre": "현재 학교에서 사용하는 AI 교육용 소프트웨어 도구에 얼마나 만족하고 있나요?",
     "post": "앱의 전반적인 만족도(실시간 그래프 반영, 수료증 발급 등)는 얼마나 되었나요?"},
]

# 5점 척도 보기 (초등학생이 고르기 쉽도록 이모지+말로 표현)
SURVEY_SCALE = [
    "1점 😞 전혀 그렇지 않다",
    "2점 🙁 그렇지 않다",
    "3점 😐 보통이다",
    "4점 🙂 그렇다",
    "5점 😄 매우 그렇다",
]


# =========================================================
# 학습 데이터 저장 (교사용 분석) — 로컬 CSV
# =========================================================
def _find_data_dir():
    """학습 결과 CSV를 저장할 data 폴더를 찾는다.
    - 배포 폴더 구조: (루트)/data 폴더가 있으면 그곳을 사용
    - 없으면 app.py 옆의 program/data 를 사용 (자동 생성)"""
    root_data = os.path.join(os.path.dirname(APP_DIR_ROOT), "data")
    if os.path.isdir(root_data):
        return root_data
    return os.path.join(APP_DIR_ROOT, "data")


DATA_DIR = _find_data_dir()
RESULTS_CSV = os.path.join(DATA_DIR, "learning_results.csv")
# 교사 대시보드 접근 비밀번호 (선생님이 필요하면 이 값만 바꾸면 됨)
TEACHER_PASSWORD = "teacher"


def _ensure_data_dir():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except OSError:
        pass


def save_student_record(name):
    """현재 학생의 학습 결과 한 줄을 CSV에 저장/갱신한다.
    '학급코드 + 이름'을 기준으로 같은 학생이면 최신 값으로 덮어쓴다(중복 방지).
    학급코드로 구분해 저장하므로, 여러 학급·학교가 같은 앱을 써도 서로 섞이지 않는다."""
    _ensure_data_dir()
    now_kr = datetime.now(timezone(timedelta(hours=9)))
    record = {
        "학급코드": st.session_state.get("class_code", "").strip() or "미입력",
        "이름": name.strip(),
        "사전점수": st.session_state.get("pre_score", ""),
        "사후점수": st.session_state.get("post_score", ""),
        "향상점수": (st.session_state.get("post_score", 0) - st.session_state.get("pre_score", 0))
                     if (st.session_state.get("pre_score") is not None
                         and st.session_state.get("post_score") is not None) else "",
        "획득배지수": len(st.session_state.get("badges", set())),
        "보너스배지수": len(st.session_state.get("bonus_badges", set())),
        "복습미션_틀린횟수": st.session_state.get("review_wrong", "") if st.session_state.get("review_done") else "",
        "복습미션_완료": "완료" if st.session_state.get("review_done") else "",
    }
    # 4개 영역 설문(5점 척도)의 사전·사후 응답과 변화량을 함께 저장
    pre_sv = st.session_state.get("pre_survey", {}) or {}
    post_sv = st.session_state.get("post_survey", {}) or {}
    for item in SURVEY_ITEMS:
        k = item["key"]
        pv, qv = pre_sv.get(k), post_sv.get(k)
        record[f"{k}_사전"] = pv if pv is not None else ""
        record[f"{k}_사후"] = qv if qv is not None else ""
        record[f"{k}_변화"] = (qv - pv) if (pv is not None and qv is not None) else ""
    record["기록시각"] = now_kr.strftime("%Y-%m-%d %H:%M")
    try:
        if os.path.exists(RESULTS_CSV):
            df = pd.read_csv(RESULTS_CSV, dtype=str)
        else:
            df = pd.DataFrame(columns=list(record.keys()))
        # 같은 학급 + 같은 이름의 기존 행만 제거 후 추가 (최신값 유지)
        if "학급코드" not in df.columns:
            df["학급코드"] = "미입력"
        df = df[~((df["이름"] == record["이름"]) & (df["학급코드"] == record["학급코드"]))]
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
        return True
    except Exception:
        return False


def load_all_records():
    if os.path.exists(RESULTS_CSV):
        try:
            return pd.read_csv(RESULTS_CSV, dtype=str)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def badge_img_tag(idx, size=64, grayscale=False, variant="sm"):
    """ALL_BADGES[idx]의 이미지를 <img> 태그 문자열로 반환. 파일이 없으면 이모지로 대체.
    variant: 'sm'(사이드바 등 자주 다시 그려지는 작은 아이콘, ~5~9KB) /
             'md'(배지 획득 순간 큰 이미지, ~수십KB) / 'full'(원본, 수료증 등 1회성 용도)
    매 상호작용마다 스크립트가 다시 실행되며 화면이 재전송되는 Streamlit 특성상,
    자주 노출되는 작은 아이콘은 원본 대신 실제 표시 크기에 맞는 썸네일을 써야
    전송량이 크게 줄어든다(특히 여러 명이 같은 와이파이로 접속하는 컴퓨터실 환경에서 중요)."""
    emoji, name, filename, part, desc = ALL_BADGES[idx]
    if variant in ("sm", "md"):
        base, ext = os.path.splitext(filename)
        variant_filename = f"{base}_{variant}{ext}"
        if _img_b64(variant_filename) is not None:
            filename = variant_filename
    b64 = _img_b64(filename)
    if not b64:
        return f'<span style="font-size:{size}px;">{emoji}</span>'
    filt = "filter:grayscale(100%) opacity(0.35);" if grayscale else ""
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="height:{size}px; width:auto; {filt} vertical-align:middle;">')


def award_badge(idx):
    """배지를 세션에 기록하고, 획득한 로봇 부품 이미지를 화면에 예쁘게 보여준다."""
    emoji, name, filename, part, desc = ALL_BADGES[idx]
    is_new = name not in st.session_state["badges"]
    st.session_state["badges"].add(name)
    if is_new:
        st.balloons()
        play_sfx(SFX_BADGE_FILE)  # 부품 획득 효과음 (key)
    md_html(f"""
    <div style="text-align:center; background:linear-gradient(135deg,#FFF8E1,#FFFFFF);
                border:2px dashed #FFC93C; border-radius:20px; padding:18px; margin:10px 0;">
        {badge_img_tag(idx, size=110, variant="md")}
        <div style="font-family:'Jua',sans-serif; font-size:19px; color:#1B2A4A; margin-top:10px;">
            🔧 로봇 네오의 <b>{part}</b>을(를) 얻었어요!
        </div>
        <div style="font-size:13px; color:#3A4A6B; margin-top:4px;">{desc}</div>
    </div>
    """)
    _sync_progress_to_url()


# =========================================================
# 수료증 이미지 생성 (PIL) — 진짜 상장처럼 보이도록 직접 그린다
# =========================================================
FONT_DIR = os.path.join(MEDIA_DIR, "fonts")

_CERT_NAVY = (27, 42, 74)
_CERT_NAVY_SOFT = (58, 74, 107)
_CERT_GOLD = (196, 155, 40)
_CERT_GOLD_LIGHT = (255, 201, 60)
_CERT_CREAM = (255, 251, 240)
_CERT_CREAM2 = (255, 244, 214)
_CERT_CORAL = (255, 111, 89)
_CERT_SKY = (45, 156, 219)

_CERT_BADGE_LABELS = ["센서", "다리", "안테나", "하트", "팔", "뇌"]


@st.cache_data(show_spinner=False)
def _cert_font(name, size):
    path = os.path.join(FONT_DIR, name)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _cert_center_text(draw, cx, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((cx - w / 2, y), text, font=font, fill=fill)


def _cert_star(draw, cx, cy, r_out, r_in, fill, points=5, rotation=-90):
    pts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        ang = math.radians(rotation + i * 180 / points)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=fill)


def _cert_seal(draw, cx, cy):
    n = 20
    r_out, r_in = 95, 82
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        ang = math.radians(i * 360 / (n * 2))
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=_CERT_GOLD)
    draw.ellipse([cx - 72, cy - 72, cx + 72, cy + 72], fill=_CERT_GOLD_LIGHT, outline=_CERT_GOLD, width=3)
    draw.ellipse([cx - 58, cy - 58, cx + 58, cy + 58], outline=(255, 255, 255), width=2)
    _cert_star(draw, cx, cy - 10, 32, 14, fill=(255, 255, 255))
    _cert_center_text(draw, cx, cy + 24, "AI 탐험대", _cert_font("NotoSansKR-Bold.ttf", 15), _CERT_NAVY)
    draw.polygon([(cx - 42, cy + 68), (cx - 10, cy + 68), (cx - 20, cy + 148), (cx - 54, cy + 130)], fill=_CERT_CORAL)
    draw.polygon([(cx + 42, cy + 68), (cx + 10, cy + 68), (cx + 20, cy + 148), (cx + 54, cy + 130)], fill=_CERT_SKY)


def _cert_paste_badge(canvas, path, box):
    x, y, w, h = (int(v) for v in box)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=(255, 255, 255), outline=_CERT_GOLD, width=2)
    if path and os.path.exists(path):
        icon = Image.open(path).convert("RGBA")
        pad = 14
        max_w, max_h = w - 2 * pad, h - 2 * pad
        scale = min(max_w / icon.width, max_h / icon.height)
        icon = icon.resize((max(1, int(icon.width * scale)), max(1, int(icon.height * scale))), Image.LANCZOS)
        ix = x + (w - icon.width) // 2
        iy = y + (h - icon.height) // 2
        canvas.paste(icon, (int(ix), int(iy)), icon)


@st.cache_data(show_spinner=False)
def build_certificate_image(name, num_badges, num_bonus, date_str):
    """예쁜 수료증 PNG를 만들어 PIL Image로 반환 (이름/배지수/날짜가 같으면 캐시 재사용)."""
    W, H = 1600, 1131
    img = Image.new("RGB", (W, H), _CERT_CREAM)
    draw = ImageDraw.Draw(img)
    for cx, cy in [(0, 0), (W, 0), (0, H), (W, H)]:
        draw.ellipse([cx - 420, cy - 420, cx + 420, cy + 420], fill=_CERT_CREAM2)

    draw.rectangle([0, 0, W, 14], fill=_CERT_NAVY)
    draw.rectangle([0, H - 14, W, H], fill=_CERT_NAVY)

    margin = 42
    draw.rounded_rectangle([margin, margin, W - margin, H - margin], radius=26, outline=_CERT_GOLD, width=6)
    draw.rounded_rectangle([margin + 14, margin + 14, W - margin - 14, H - margin - 14], radius=18,
                            outline=_CERT_GOLD, width=2)

    deco_r = 46
    for (cx, cy, a0, a1) in [(margin + 14, margin + 14, 180, 270), (W - margin - 14, margin + 14, 270, 360),
                              (margin + 14, H - margin - 14, 90, 180), (W - margin - 14, H - margin - 14, 0, 90)]:
        draw.arc([cx - deco_r, cy - deco_r, cx + deco_r, cy + deco_r], a0, a1, fill=_CERT_GOLD, width=3)

    _cert_center_text(draw, W / 2, 92, "수 료 증", _cert_font("NotoSerifKR-Bold.ttf", 74), _CERT_NAVY)
    _cert_center_text(draw, W / 2, 188, "CERTIFICATE  OF  COMPLETION",
                       _cert_font("NotoSansKR-Regular.ttf", 21), _CERT_GOLD)

    line_y = 230
    draw.line([(W / 2 - 190, line_y), (W / 2 - 42, line_y)], fill=_CERT_GOLD, width=3)
    draw.ellipse([W / 2 - 9, line_y - 9, W / 2 + 9, line_y + 9], fill=_CERT_GOLD_LIGHT, outline=_CERT_GOLD, width=2)
    draw.line([(W / 2 + 42, line_y), (W / 2 + 190, line_y)], fill=_CERT_GOLD, width=3)

    _cert_center_text(draw, W / 2, 270, f"{name}  꼬마 탐험대장님",
                       _cert_font("NotoSerifKR-Bold.ttf", 44), _CERT_NAVY)

    body_font = _cert_font("NotoSansKR-Regular.ttf", 25)
    para = [
        "위 탐험대장님은 '꼬마 AI 탐험대'의 모든 과정을 훌륭히 완수하고,",
        "선형회귀 · 경사하강법 · 결정트리 · KNN · K-평균 군집화 · 인공신경망",
        "6가지 인공지능 원리를 직접 체험하며 로봇 네오의 부품을 모두 모아",
        "완벽하게 조립하였기에 이 수료증을 수여합니다.",
    ]
    y = 344
    for line in para:
        _cert_center_text(draw, W / 2, y, line, body_font, _CERT_NAVY_SOFT)
        y += 40

    badge_y = 538
    badge_size = 88
    gap = 26
    total_w = 6 * badge_size + 5 * gap
    start_x = W / 2 - total_w / 2
    for i in range(6):
        bx = start_x + i * (badge_size + gap)
        filename = ALL_BADGES[i][2]
        _cert_paste_badge(img, os.path.join(BADGE_ASSET_DIR, filename), (bx, badge_y, badge_size, badge_size))
        draw = ImageDraw.Draw(img)
        _cert_center_text(draw, bx + badge_size / 2, badge_y + badge_size + 8, _CERT_BADGE_LABELS[i],
                           _cert_font("NotoSansKR-Regular.ttf", 15), _CERT_NAVY_SOFT)

    caption = f"( 로봇 부품 {num_badges}/6개 획득" + (f" · 보너스 {num_bonus}개" if num_bonus else "") + " )"
    _cert_center_text(draw, W / 2, badge_y + badge_size + 34, caption, _cert_font("NotoSansKR-Regular.ttf", 18),
                       _CERT_GOLD)

    draw.text((margin + 70, H - 190), date_str, font=_cert_font("NotoSansKR-Regular.ttf", 24), fill=_CERT_NAVY)
    draw.line([(margin + 70, H - 140), (margin + 380, H - 140)], fill=_CERT_NAVY_SOFT, width=2)
    draw.text((margin + 70, H - 120), "꼬마 AI 탐험대  대장",
              font=_cert_font("NotoSansKR-Bold.ttf", 24), fill=_CERT_NAVY)

    _cert_seal(draw, W - 260, H - 260)
    return img


def render_share_buttons(png_bytes, filename, share_title):
    """다운로드 버튼 + (지원되는 브라우저에서) 웹 공유 버튼.
    모바일 크롬/삼성인터넷 등에서는 공유 시트가 떠서 구글 클래스룸, 카카오톡 등으로 바로 보낼 수 있고,
    지원하지 않는 브라우저(PC 등)에서는 다운로드 후 직접 첨부하도록 안내한다."""
    b64 = base64.b64encode(png_bytes).decode()
    st.download_button(
        "⬇️ 수료증 이미지 다운로드 (PNG)",
        data=png_bytes,
        file_name=filename,
        mime="image/png",
        **_STRETCH,
    )
    components.html(f"""
    <div style="font-family:sans-serif;">
      <button id="shareBtn" style="
          width:100%; padding:12px; margin-top:8px; border:none; border-radius:999px;
          background:#2D9CDB; color:white; font-size:15px; font-weight:bold; cursor:pointer;">
        📤 공유하기 (구글 클래스룸 · 카카오톡 등)
      </button>
      <p id="shareMsg" style="text-align:center; color:#888; font-size:12px; margin-top:6px;"></p>
    </div>
    <script>
      const b64 = "{b64}";
      const fname = "{filename}";
      document.getElementById("shareBtn").addEventListener("click", async () => {{
        const msg = document.getElementById("shareMsg");
        try {{
          const res = await fetch("data:image/png;base64," + b64);
          const blob = await res.blob();
          const file = new File([blob], fname, {{type: "image/png"}});
          if (navigator.canShare && navigator.canShare({{files: [file]}})) {{
            await navigator.share({{
              files: [file],
              title: "{share_title}",
              text: "{share_title}"
            }});
          }} else {{
            msg.textContent = "이 브라우저에서는 공유 시트를 지원하지 않아요. 위의 '다운로드' 버튼으로 저장한 뒤, 구글 클래스룸 등에 파일을 직접 첨부해주세요!";
          }}
        }} catch (e) {{
          if (e.name !== "AbortError") {{
            msg.textContent = "공유 창을 열지 못했어요. '다운로드' 버튼으로 저장 후 첨부해주세요!";
          }}
        }}
      }});
    </script>
    """, height=110)


# =========================================================
# 진행상황(배지) URL 저장 — 태블릿 대응
# =========================================================
# 태블릿에서 다른 앱으로 갔다 오거나 화면을 새로고침하면 session_state가 초기화될 수 있다.
# 배지 획득 상태를 짧은 코드로 압축해 주소창(쿼리 파라미터)에 저장해두면,
# 같은 브라우저 탭/주소가 살아있는 한 새로고침해도 진행상황이 복원된다.
BONUS_BADGE_NAMES = [
    "🌟 선형회귀 완벽 이해",
    "🌟 경사하강법 완벽 이해",
    "🌟 결정트리 완벽 이해",
    "🌟 KNN 완벽 이해",
    "🌟 K-Means 완벽 이해",
    "🌟 인공신경망 완벽 이해",
    "🌟 복습 미션 만점",
]


def _qp_get(key, default=None):
    try:
        return st.query_params.get(key, default)
    except Exception:
        try:
            v = st.experimental_get_query_params().get(key)
            return v[0] if v else default
        except Exception:
            return default


def _qp_set(key, value):
    try:
        st.query_params[key] = value
    except Exception:
        try:
            params = st.experimental_get_query_params()
            params[key] = value
            st.experimental_set_query_params(**params)
        except Exception:
            pass


def _encode_progress():
    b = 0
    for i, (_e, name, *_r) in enumerate(ALL_BADGES):
        if name in st.session_state.get("badges", set()):
            b |= (1 << i)
    bo = 0
    for i, name in enumerate(BONUS_BADGE_NAMES):
        if name in st.session_state.get("bonus_badges", set()):
            bo |= (1 << i)
    return f"{b:02x}{bo:02x}"


def _sync_progress_to_url():
    _qp_set("p", _encode_progress())


def _restore_progress_from_url():
    code = _qp_get("p")
    if not code or len(code) < 4:
        return
    try:
        b = int(code[0:2], 16)
        bo = int(code[2:4], 16)
    except ValueError:
        return
    for i, (_e, name, *_r) in enumerate(ALL_BADGES):
        if b & (1 << i):
            st.session_state["badges"].add(name)
    for i, name in enumerate(BONUS_BADGE_NAMES):
        if bo & (1 << i):
            st.session_state["bonus_badges"].add(name)


# =========================================================
# 무거운 계산 캐싱 — 컴퓨터실에서 여러 명이 동시에 슬라이더를 움직여도
# 서버가 매번 다시 계산하지 않도록 결과를 재사용한다.
# =========================================================
@st.cache_resource(show_spinner=False)
def _fit_knn_fruit(k_val):
    d = pd.DataFrame({
        '달콤한 정도': [8, 9, 7, 2, 3, 1, 8, 2, 4],
        '과일 크기': [7, 8, 9, 2, 1, 3, 6, 2, 4],
        '과일 종류': ['사과', '포도', '사과', '포도', '포도', '포도', '사과', '포도', '포도']
    })
    knn = KNeighborsClassifier(n_neighbors=k_val)
    knn.fit(d[['달콤한 정도', '과일 크기']], d['과일 종류'])
    return knn


@st.cache_data(show_spinner=False)
def _fit_kmeans_leaf(k):
    leaf = pd.DataFrame({
        '잎의 길이': [15, 14, 16, 5, 4, 6, 8, 9, 2, 3, 1],
        '잎의 넓이': [2, 1.5, 2.5, 5, 4.5, 6, 1, 1.5, 1, 0.5, 0.8],
    })
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(leaf)
    return labels, km.cluster_centers_


def build_review_quiz():
    """부품 6개마다 문제를 하나씩 랜덤으로 뽑아, 순서를 섞은 복습 퀴즈를 만든다."""
    quiz = []
    for part_idx in range(6):
        candidates = [q for q in REVIEW_QUESTIONS if q["part_idx"] == part_idx]
        quiz.append(random.choice(candidates))
    random.shuffle(quiz)
    return quiz


def start_review_game():
    # 이전 판의 오답 표시/라디오 선택 흔적을 깨끗이 지운다
    for k in list(st.session_state.keys()):
        if k.startswith("review_wrongmark_") or k.startswith("review_ans_"):
            del st.session_state[k]
    st.session_state["review_quiz"] = build_review_quiz()
    st.session_state["review_idx"] = 0
    st.session_state["review_fixed"] = []
    st.session_state["review_wrong"] = 0
    st.session_state["review_done"] = False


# =========================================================
# 게임용 세션 상태 초기화
# =========================================================
def init_game_state():
    defaults = {
        "badges": set(),
        "bonus_badges": set(),
        "fb_data": None,
        "gd_x": 8.0, "gd_steps": 0, "gd_history": [8.0], "gd_failed": False, "gd_success": False,
        "dt_mystery": None, "dt_candidates": None, "dt_asked": [], "dt_history": [],
        "knn_data": None,
        "knn_hatched": False,
        "km_data": None, "km_centroids": None, "km_labels": None, "km_iter": 0,
        "nn_rocket": False,
        "review_quiz": None, "review_idx": 0, "review_fixed": [], "review_wrong": 0, "review_done": False,
        "pre_score": None, "post_score": None, "student_name": "", "class_code": "",
        "pre_survey": {}, "post_survey": {},
        "km_phase": "assign", "km_class_run": False,
    }
    is_fresh_session = "badges" not in st.session_state
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if is_fresh_session:
        # 새 세션일 때만 URL의 진행상황을 복원한다(수동으로 초기화한 경우까지 되살리지 않도록).
        _restore_progress_from_url()

init_game_state()

# 2. 사이드바 메뉴
MENU_OPTIONS = [
    "🏠 탐험 본부 (홈)",
    "📝 나의 이름 & 사전 평가",
    "🌟 인공지능이 뭐예요?",
    "📈 1. 마법의 선 긋기",
    "⛰️ 2. 보물찾기 산",
    "🌳 3. 스무고개 탐정",
    "🤝 4. 가장 친한 친구",
    "🎨 5. 비슷한 친구끼리",
    "🧠 6. 똑똑한 생각 주머니",
    "⚖️ AI를 똑똑하게 쓰려면?",
    "🔧 로봇 네오 종합 점검 (복습)",
    "🏆 탐험 완료 (수료증)",
    "👩‍🏫 선생님 방 (학습 분석)",
]

if "nav_menu" not in st.session_state:
    st.session_state["nav_menu"] = MENU_OPTIONS[0]

# 버튼/이미지 클릭으로 예약된 이동이 있으면, 라디오 위젯이 만들어지기 '전에' 반영한다
if "_pending_nav" in st.session_state:
    st.session_state["nav_menu"] = st.session_state.pop("_pending_nav")
    st.session_state["_scroll_top"] = True  # 새 화면은 맨 위부터 보이도록 표시

# 수료증 메뉴는 항상 보인다.
# (사후 검사는 부품을 다 모으지 않아도 언제든 할 수 있고,
#  수료증 '이미지'는 부품 6개를 모두 모았을 때 나타난다.)
visible_menus = MENU_OPTIONS.copy()

def go_to(target_menu):
    """버튼/이미지 클릭으로 다른 섹션으로 이동시키는 헬퍼"""
    st.session_state["_pending_nav"] = target_menu
    st.rerun()


with st.sidebar:
    st.title("🧸 꼬마 AI 탐험대")
    st.write("안녕! 인공지능 세계로 떠나볼까?")
    st.divider()
    menu = st.radio(
        "어느 섬으로 탐험을 떠날까요?",
        visible_menus,
        key="nav_menu"
    )

    # 사이드바 메뉴로 화면을 바꾼 경우에도 '맨 위부터' 보이게 한다.
    # (이전에는 버튼 이동일 때만 맨 위로 갔고, 메뉴 클릭 시에는
    #  이전 스크롤 위치가 남아 화면 중간부터 보이는 문제가 있었다.)
    if st.session_state.get("_last_menu") != menu:
        st.session_state["_last_menu"] = menu
        st.session_state["_scroll_top"] = True

    st.divider()

    # 🗺️ 탐험 진행도
    collected_names = st.session_state["badges"]
    collected_count = len(collected_names)
    st.markdown("#### 🗺️ 로봇 네오 조립 현황")
    st.progress(collected_count / len(ALL_BADGES))
    st.caption(f"부품 {collected_count} / {len(ALL_BADGES)}개 모음")

    icon_row = " ".join(
        badge_img_tag(i, size=34, grayscale=(name not in collected_names))
        for i, (icon, name, filename, part, desc) in enumerate(ALL_BADGES)
    )
    st.markdown(f"<div style='display:flex; gap:6px; flex-wrap:wrap;'>{icon_row}</div>", unsafe_allow_html=True)

    st.divider()
    if collected_names:
        st.write("### 🔧 내가 모은 부품")
        for i, (icon, name, filename, part, desc) in enumerate(ALL_BADGES):
            if name in collected_names:
                md_html(f"""
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    {badge_img_tag(i, size=30)}
                    <span style="font-size:14px;">{part}</span>
                </div>
                """)
    else:
        st.caption("게임을 클리어하면 로봇 부품을 받아요! 🌟")

    # 6개 부품을 모두 모으면 복습 게임을 눈에 띄게 안내
    if collected_count >= 6:
        st.divider()
        md_html("""
        <div style="background:#EDE7F6; border:2px solid #7E57C2; border-radius:14px; padding:12px; text-align:center;">
            <div style="font-size:14px; color:#EAF6FF;">🎓 이제 배운 걸 정리해볼까요?</div>
        </div>
        """)
        if st.button("🔧 복습 미션 도전하기", key="side_go_review", **_STRETCH):
            go_to("🔧 로봇 네오 종합 점검 (복습)")

    bonus_names = st.session_state["bonus_badges"]
    if bonus_names:
        st.write(f"### 🌟 완벽 이해 보너스 ({len(bonus_names)}개)")
        for b in bonus_names:
            st.write(b)

    # 컴퓨터실처럼 한 대의 기기를 여러 학생이 이어서 쓸 때를 위한 전체 초기화 버튼
    st.divider()
    if st.session_state.get("_confirm_reset"):
        st.warning("정말요? 모은 로봇 부품과 **사전·사후 검사 결과**가 모두 사라져요! (다음 친구가 처음부터 다시 할 수 있어요.)")
        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("✅ 초기화", key="reset_yes", **_STRETCH):
                # 로봇 부품(배지)
                st.session_state["badges"] = set()
                st.session_state["bonus_badges"] = set()

                # 사전·사후 검사와 설문을 모두 초기화 → 다음 친구가 처음부터 다시 할 수 있음
                st.session_state["pre_score"] = None
                st.session_state["post_score"] = None
                st.session_state["pre_survey"] = {}
                st.session_state["post_survey"] = {}
                st.session_state["student_name"] = ""
                st.session_state["class_code"] = ""

                # 사전·사후 검사에 쓰인 개별 문항/설문 위젯 값도 함께 비운다
                for i in range(len(DIAGNOSTIC_QUESTIONS)):
                    st.session_state.pop(f"diag_pre_{i}", None)
                    st.session_state.pop(f"diag_post_{i}", None)
                for j in range(len(SURVEY_ITEMS)):
                    st.session_state.pop(f"survey_pre_{j}", None)
                    st.session_state.pop(f"survey_post_{j}", None)

                # 복습 미션 진행 상태도 초기화
                for k in ["review_quiz", "review_idx", "review_fixed",
                          "review_wrong", "review_done", "review_started"]:
                    st.session_state.pop(k, None)
                for k in list(st.session_state.keys()):
                    if k.startswith("review_wrongmark_") or k.startswith("review_ans_"):
                        st.session_state.pop(k, None)

                st.session_state["_confirm_reset"] = False
                _sync_progress_to_url()
                st.rerun()
        with rc2:
            if st.button("❌ 취소", key="reset_no", **_STRETCH):
                st.session_state["_confirm_reset"] = False
                st.rerun()
    else:
        if st.button("🔄 다음 친구를 위해 초기화", key="reset_all", **_STRETCH):
            st.session_state["_confirm_reset"] = True
            st.rerun()
        st.caption("다음 학생이 이어서 쓸 때 눌러주세요. (부품 + 사전·사후 검사 결과가 모두 초기화돼요.)")


NEXT_SECTION = {
    "📈 1. 마법의 선 긋기": "⛰️ 2. 보물찾기 산",
    "⛰️ 2. 보물찾기 산": "🌳 3. 스무고개 탐정",
    "🌳 3. 스무고개 탐정": "🤝 4. 가장 친한 친구",
    "🤝 4. 가장 친한 친구": "🎨 5. 비슷한 친구끼리",
    "🎨 5. 비슷한 친구끼리": "🧠 6. 똑똑한 생각 주머니",
    "🧠 6. 똑똑한 생각 주머니": "🏠 탐험 본부 (홈)",
}


def next_section_button(current_menu, key_suffix):
    """배지 획득 직후 다음 섬으로 바로 이동하는 버튼"""
    target = NEXT_SECTION.get(current_menu)
    if target:
        label = "🎉 6개 섬 완주! 탐험 본부로 돌아가기" if target == "🏠 탐험 본부 (홈)" else f"⛵ 다음 탐험지로 출발! ({target})"
        if st.button(label, key=f"next_{key_suffix}", **_STRETCH):
            go_to(target)



# 다른 화면으로 막 이동한 직후라면, 화면을 맨 위로 스크롤한다
# (탭/메뉴 이동 시 이전 스크롤 위치가 남아 중간부터 보이는 문제 해결)
if st.session_state.pop("_scroll_top", False):
    components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            function toTop() {
                const targets = [
                    doc.querySelector('section.main'),
                    doc.querySelector('[data-testid="stMain"]'),
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.querySelector('.main'),
                    doc.scrollingElement,
                    doc.documentElement,
                    doc.body
                ];
                targets.forEach(function (el) {
                    if (el && typeof el.scrollTo === 'function') {
                        el.scrollTo({top: 0, behavior: 'auto'});
                    }
                    if (el) { el.scrollTop = 0; }
                });
                window.parent.scrollTo({top: 0, behavior: 'auto'});
            }
            toTop();
            // 화면이 다 그려진 뒤에도 확실히 맨 위로
            setTimeout(toTop, 120);
            setTimeout(toTop, 400);
            setTimeout(toTop, 900);
        })();
        </script>
        """,
        height=0,
    )


def render_diagnostic_quiz(phase, score_key):
    """사전/사후 진단(지식 6문항 + 설문 4문항)을 렌더링하고, 제출하면 결과를 저장한다.
    phase: 'pre' 또는 'post' (위젯 key 구분 및 설문 문장 선택에 사용)"""
    survey_key = "pre_survey" if phase == "pre" else "post_survey"

    # ── 1부: 지식 문항 (채점됨) ─────────────────────────────
    st.markdown("##### 📚 1부. 인공지능 원리 알아보기 (6문항)")
    st.caption("잘 모르면 '아직 잘 몰라요'를 골라도 괜찮아요! (점수는 성적과 상관없어요 😊)")
    answers = []
    for i, q in enumerate(DIAGNOSTIC_QUESTIONS):
        choice = st.radio(
            f"**{i+1}.** {q['q']}",
            ["아직 잘 몰라요"] + AI_PRINCIPLES,
            key=f"diag_{phase}_{i}",
        )
        answers.append(choice)

    st.divider()

    # ── 2부: 설문 문항 (5점 척도, 채점 안 됨) ─────────────────
    st.markdown("##### 💬 2부. 나의 생각 이야기하기 (4문항)")
    st.caption("정답이 없는 질문이에요. 지금 내 마음과 가장 가까운 것을 솔직하게 골라주세요!")
    survey_choices = {}
    for j, item in enumerate(SURVEY_ITEMS):
        label = item["pre"] if phase == "pre" else item["post"]
        pick = st.radio(
            f"**{j+1}.** ({item['key']}) {label}",
            SURVEY_SCALE,
            index=2,  # 기본값 '보통이다'
            key=f"survey_{phase}_{j}",
        )
        survey_choices[item["key"]] = int(pick[0])  # 맨 앞 숫자가 점수

    if st.button("✅ 제출하기", key=f"diag_submit_{phase}", **_STRETCH):
        score = 0
        for i, q in enumerate(DIAGNOSTIC_QUESTIONS):
            if answers[i] != "아직 잘 몰라요" and AI_PRINCIPLES.index(answers[i]) == q["answer"]:
                score += 1
        st.session_state[score_key] = score
        st.session_state[survey_key] = survey_choices
        # 이름이 있으면 결과 저장
        nm = st.session_state.get("student_name", "").strip()
        if nm:
            save_student_record(nm)
        st.rerun()


# --- [메뉴 1: 홈] ---
if menu == "🏠 탐험 본부 (홈)":
    render_bgm_player(BGM_HOME_FILE, "home", compact=True)

    _mascot_b64 = _asset_img_b64("robot_mascot.png")
    _mascot_html = (f'<img src="data:image/png;base64,{_mascot_b64}" '
                    f'style="height:56px; width:auto; vertical-align:middle; margin-right:12px;">'
                    ) if _mascot_b64 else "🚀"
    md_html(f"""
    <h1 style="text-align:center; font-family:'Jua',sans-serif; font-size:clamp(24px, 4.2vw, 46px); color:#1B2A4A; margin-top:-6px; line-height:1.25; white-space:nowrap;">
        {_mascot_html}손끝에서 배우는 인공지능 원리 AI 탐험대
    </h1>
    <div style="text-align:center; margin-top:6px;">
        <span style="display:inline-block; background:#E3F2FD; border:2px solid #2D9CDB; border-radius:20px;
                     padding:6px 18px; font-family:'Jua',sans-serif; font-size:clamp(15px, 2vw, 20px); color:#1565C0;">
            🎯 대상 학년 : 초등학교 6학년
        </span>
    </div>
    """)
    st.divider()

    # 6개의 배지를 다 모았을 때 최종 섬 입장 유도 배너 출력
    if len(st.session_state["badges"]) >= 6:
        md_html("""
        <div style='background:#FFF3E0; border:2px solid #FFA726; border-radius:16px; padding:20px; text-align:center; margin-top:20px; margin-bottom:20px;'>
            <h2 style='color:#E65100; margin:0;'>🎉 축하합니다! 6개의 배지를 모두 모았습니다!</h2>
            <p style='font-size:18px; color:#E65100; margin-top:10px;'>배운 원리를 정리하는 <b>복습 미션</b>과 <b>최종 섬(수료증)</b>이 열렸어요!</p>
        </div>
        """)
        home_cta1, home_cta2 = st.columns(2)
        with home_cta1:
            if st.button("🔧 복습 미션 도전하기", key="home_go_review", **_STRETCH):
                go_to("🔧 로봇 네오 종합 점검 (복습)")
        with home_cta2:
            if st.button("🏆 수료증 받으러 가기 🌟", key="home_go_cert", **_STRETCH):
                go_to("🏆 탐험 완료 (수료증)")
        st.divider()

    VIDEO_PATH = os.path.join(MOVIE_DIR, "movie_intro.mp4")

    col_v1, col_v2, col_v3 = st.columns([1, 3, 1])
    with col_v2:
        if os.path.exists(VIDEO_PATH):
            st.video(VIDEO_PATH)
        else:
            st.info("📹 소개 영상이 없어요! `media/movie` 폴더 안에 `movie_intro.mp4` 파일을 넣어주세요.")

    st.divider()

    st.markdown("<h3 style='text-align:center;'>🎒 탐험대원 여러분, 준비가 끝났나요?</h3>", unsafe_allow_html=True)
    if st.button("네! 탐험을 시작하겠습니다! (클릭) 🎈", **_STRETCH):
        go_to("🌟 인공지능이 뭐예요?")

    st.divider()
    with st.expander("👩‍🏫 선생님을 위한 안내: 2022 개정 교육과정 성취기준 연계 (5~6학년군)"):
        md_html("""
        이 프로그램의 6개 활동은 모두 아래 **실과(정보) 영역** 성취기준을 공통 기반으로 합니다.

        - **[6실05-04]** 디지털 데이터와 아날로그 데이터의 특징을 이해하고, 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
        - **[6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.

        각 활동은 여기에 더해 아래 교과와도 연계할 수 있습니다.

        | 활동 | 핵심 AI 개념 | 추가 연계 교과·성취기준 |
        |---|---|---|
        | 🎯 붕어빵 가격 맞히기 | 선형회귀 | 수학 [6수04-01] 규칙과 대응 |
        | 🏂 눈꽃 골짜기 스노보드 | 경사하강법 | 실과 [6실05-04] 데이터 탐색 |
        | 🕵️‍♂️ 동물 스무고개 명탐정 | 결정트리 | 사회 [6사03-04] 정보화 시대의 변화상 |
        | 🍇 외계 알 구출작전 | KNN | 실과 [6실05-04] 데이터 탐색 |
        | 🧹 무인도 분리수거 로봇 | K-Means | 실과 [6실05-04] 데이터 탐색 |
        | 🤖 로봇 네오 발사 | 인공신경망 | 실과 [6실04-06] 로봇의 작동 원리 |

        각 게임 화면 상단의 "🎯 이 활동과 관련된 성취기준" 펼침 박스에서 자세한 내용을 확인할 수 있습니다.
        """)

# --- [메뉴: 이름 & 사전 평가] ---
elif menu == "📝 나의 이름 & 사전 평가":
    hero_card("📝", "탐험 시작 전, 나를 알려줘요!",
               "이름을 적고 간단한 사전 퀴즈를 풀면, 탐험이 끝난 뒤 얼마나 자랐는지 확인할 수 있어요.",
               "#5C6BC0")

    st.markdown("### 1️⃣ 학급 코드와 내 이름 적기")
    nc1, nc2 = st.columns(2)
    with nc1:
        class_in = st.text_input("학급 코드 (선생님이 알려주세요)", value=st.session_state.get("class_code", ""),
                                 placeholder="예: 6-3", key="class_input_pre")
    with nc2:
        name_in = st.text_input("이름", value=st.session_state.get("student_name", ""),
                                placeholder="예: 홍길동", key="name_input_pre")

    if class_in.strip():
        st.session_state["class_code"] = class_in.strip()
    if name_in.strip():
        st.session_state["student_name"] = name_in.strip()
        st.success(f"반가워요, **{name_in.strip()}** 탐험대원! 이제 사전 퀴즈를 풀어볼까요?")
    st.caption("💡 학급 코드는 우리 반 결과만 따로 모으기 위한 거예요. 선생님이 정해준 코드를 똑같이 적어주세요.")

    st.divider()
    st.markdown("### 2️⃣ 사전 퀴즈 (배우기 전, 지금 아는 만큼만!)")

    if not st.session_state.get("student_name", "").strip():
        st.info("먼저 위에 이름을 적어주세요! 😊")
    elif st.session_state.get("pre_score") is not None:
        st.success(f"✅ 사전 검사 완료! 지식 점수: **{st.session_state['pre_score']} / 6점**")
        sv = st.session_state.get("pre_survey", {})
        if sv:
            st.write("**내가 답한 생각 점수 (5점 만점)**")
            sc = st.columns(len(SURVEY_ITEMS))
            for c, item in zip(sc, SURVEY_ITEMS):
                c.metric(item["key"], f"{sv.get(item['key'], '-')}점")
        st.write("이제 탐험을 시작해봐요. 6개 섬을 모두 마친 뒤 '수료증' 방에서 사후 검사를 하면 얼마나 자랐는지 알 수 있어요!")
        if st.button("🚀 탐험 시작하기 (인공지능이 뭐예요?)", **_STRETCH):
            go_to("🌟 인공지능이 뭐예요?")
        with st.expander("다시 풀기 (사전 결과 초기화)"):
            if st.button("🔄 사전 검사 다시 하기", key="reset_pre"):
                st.session_state["pre_score"] = None
                st.session_state["pre_survey"] = {}
                for i in range(len(DIAGNOSTIC_QUESTIONS)):
                    st.session_state.pop(f"diag_pre_{i}", None)
                for j in range(len(SURVEY_ITEMS)):
                    st.session_state.pop(f"survey_pre_{j}", None)
                st.rerun()
    else:
        render_diagnostic_quiz("pre", "pre_score")

# --- [메뉴 0.5: 인공지능이 뭐예요?] ---
elif menu == "🌟 인공지능이 뭐예요?":
    hero_card("🌟", "인공지능이 뭐예요?",
               "6가지 원리를 만나기 전에, AI가 도대체 뭘 하는 건지 그림으로 차근차근 알아봐요!",
               "#546E7A")

    tab_w1, tab_w2, tab_w3, tab_w4, tab_w5, tab_w6 = st.tabs([
        "1️⃣ AI가 뭐예요?",
        "2️⃣ AI는 어떻게 배워요?",
        "3️⃣ 지도학습: 이름표 보고 배우기",
        "4️⃣ 비지도학습: 스스로 모둠 만들기",
        "5️⃣ 생활 속 AI 찾기",
        "6️⃣ 6가지 섬 미리보기",
    ])

    # ===== [1단계: AI가 뭐예요?] =========================================
    with tab_w1:
        st.markdown("### 🤔 인공지능(AI)이 뭘까요?")
        md_html("""
        <div style='font-size:19px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:20px 24px;'>
        인공지능은 <b>아주 똑똑한 로봇 두뇌</b>가 아니에요.<br>
        사실은, <b>아주 많은 예시(데이터)를 보고, 그 안에서 스스로 규칙을 찾아내는 컴퓨터 프로그램</b>이에요.<br><br>
        사람이 "이럴 땐 이렇게 해!"라고 하나하나 다 가르쳐주지 않아도, 예시를 많이 보여주면
        컴퓨터가 <b>"아, 이런 규칙이 있구나!"</b>하고 스스로 알아낸답니다.
        </div>
        """)

        st.write("")
        st.markdown("### 🆚 보통 프로그램과 AI, 뭐가 다를까요? (한눈에 비교!)")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            md_html("""
            <div style='background:#ECEFF1; border:2px solid #78909C; border-radius:16px; padding:16px; height:260px;'>
                <div style='text-align:center; font-size:36px;'>💻</div>
                <div style="text-align:center; font-family:'Jua',sans-serif; font-size:19px; color:#455A64;">보통 프로그램</div>
                <div style='font-size:14px; color:#3A4A6B; margin-top:10px; line-height:1.9; text-align:center;'>
                    사람: "규칙을 <b>전부 다</b> 알려줄게!"<br>
                    📏 <b>규칙</b>을 넣으면 →<br>
                    ⚙️ 컴퓨터가 시키는 대로만 해요<br><br>
                    <span style='color:#78909C;'>규칙에 없는 일은 하나도 못 해요 😥</span>
                </div>
            </div>
            """)
        with cmp2:
            md_html("""
            <div style='background:#E8F5E9; border:2px solid #2E7D32; border-radius:16px; padding:16px; height:260px;'>
                <div style='text-align:center; font-size:36px;'>🤖</div>
                <div style="text-align:center; font-family:'Jua',sans-serif; font-size:19px; color:#2E7D32;">인공지능(AI)</div>
                <div style='font-size:14px; color:#3A4A6B; margin-top:10px; line-height:1.9; text-align:center;'>
                    사람: "<b>예시</b>를 잔뜩 보여줄게!"<br>
                    📊 <b>예시(데이터)</b>를 넣으면 →<br>
                    🧠 컴퓨터가 <b>규칙을 스스로</b> 찾아요<br><br>
                    <span style='color:#2E7D32;'>새로운 상황도 짐작할 수 있어요 ✨</span>
                </div>
            </div>
            """)

        st.write("")
        st.markdown("### 🧩 AI는 이렇게 일해요 (공통 원리)")
        d1, d2, d3, d4, d5 = st.columns([1, 0.3, 1, 0.3, 1])
        with d1:
            st.markdown("<div style='text-align:center; background:#E3F2FD; border-radius:16px; padding:20px 10px;'><div style='font-size:36px;'>📊</div><b>① 예시(데이터)</b><br><span style='font-size:13px;'>과거에 있었던 일들</span></div>", unsafe_allow_html=True)
        with d2:
            st.markdown("<div style='text-align:center; font-size:30px; margin-top:30px;'>➡️</div>", unsafe_allow_html=True)
        with d3:
            st.markdown("<div style='text-align:center; background:#F3E5F5; border-radius:16px; padding:20px 10px;'><div style='font-size:36px;'>🧠</div><b>② AI가 규칙 찾기</b><br><span style='font-size:13px;'>스스로 패턴을 발견</span></div>", unsafe_allow_html=True)
        with d4:
            st.markdown("<div style='text-align:center; font-size:30px; margin-top:30px;'>➡️</div>", unsafe_allow_html=True)
        with d5:
            st.markdown("<div style='text-align:center; background:#E8F5E9; border-radius:16px; padding:20px 10px;'><div style='font-size:36px;'>🔮</div><b>③ 예측 / 판단</b><br><span style='font-size:13px;'>새로운 상황에 답하기</span></div>", unsafe_allow_html=True)

        st.write("")
        st.info("💡 **다음 단계!** 2번 탭에서 AI가 예시를 보고 **어떻게 점점 똑똑해지는지** 직접 눈으로 확인해봐요.")

    # ===== [2단계: AI는 어떻게 배워요?] ==================================
    with tab_w2:
        st.markdown("### 🐶🐱 AI에게 '강아지와 고양이 구분하기'를 가르쳐볼까요?")
        st.write("AI는 예시를 **많이 볼수록** 똑똑해져요. 예시(사진) 개수를 골라서, AI가 얼마나 잘 맞히는지 비교해봐요!")

        n_examples = st.select_slider(
            "🖼️ AI에게 보여줄 예시 사진 개수를 골라보세요!",
            options=[3, 30, 300, 3000], value=3, key="ai_learn_examples")

        if n_examples == 3:
            acc, face, msg, color, bg = 45, "😖", "겨우 3장 봤어요. 강아지랑 고양이가 아직 헷갈려요... 절반 정도밖에 못 맞혀요!", "#C62828", "#FFEBEE"
        elif n_examples == 30:
            acc, face, msg, color, bg = 70, "🙂", "30장을 보니 감이 조금 와요! '고양이는 귀가 뾰족하네?' 규칙이 보이기 시작해요.", "#F9A825", "#FFF8E1"
        elif n_examples == 300:
            acc, face, msg, color, bg = 90, "😄", "300장! 이제 웬만한 사진은 다 맞혀요. 규칙이 꽤 정확해졌어요!", "#558B2F", "#F1F8E9"
        else:
            acc, face, msg, color, bg = 98, "🤩", "3000장이나 봤더니 거의 다 맞혀요! 예시(데이터)가 많을수록 AI는 똑똑해져요!", "#2E7D32", "#E8F5E9"

        md_html(f"""
        <div style='background:{bg}; border-radius:16px; padding:18px 22px; text-align:center;'>
            <div style='font-size:44px;'>{face}</div>
            <div style="font-family:'Jua',sans-serif; font-size:26px; color:{color};">AI의 실력: {acc}점 / 100점</div>
            <div style='font-size:15px; color:#3A4A6B; margin-top:8px;'>{msg}</div>
        </div>
        """)
        st.progress(acc / 100)

        st.write("")
        md_html("""
        <div style="background:#E3F2FD; border-left:5px solid #2D9CDB; border-radius:14px; padding:16px 20px;">
            <b>🏷️ 방금 발견한 것</b><br>
            AI는 태어날 때부터 똑똑한 게 아니에요! <b>예시(데이터)를 보면서 조금씩 배우고</b>,
            예시가 많고 다양할수록 더 정확해져요. 이렇게 배우는 과정을 <b>'학습'</b>이라고 해요.
        </div>
        """)

        st.write("")
        st.info("💡 **다음 단계!** 그런데 AI가 배우는 방법에는 크게 **두 가지**가 있어요. 3번, 4번 탭에서 알아봐요!")

    # ===== [3단계: 지도학습] =============================================
    with tab_w3:
        st.markdown("### 🏷️ 지도학습: '이름표'를 보고 배워요!")
        md_html("""
        <div style='font-size:18px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:16px 22px;'>
        선생님이 낱말카드로 <b>"이건 사과 🍎, 이건 바나나 🍌"</b> 하고 <b>정답(이름표)을 알려주면서</b> 가르쳐주는 것처럼,<br>
        AI에게도 <b>정답 이름표가 붙은 예시</b>를 주면서 가르치는 방법을 <b>'지도학습'</b>이라고 해요.
        </div>
        """)

        st.write("")
        st.markdown("#### 👀 눈으로 봐요: 이름표 붙은 예시로 배우는 모습")
        _lab = "border-radius:12px; padding:10px 6px; text-align:center; font-size:13px;"
        md_html(f"""
        <div style='background:#FAFAFA; border-radius:16px; padding:16px 10px;'>
          <div style='display:flex; gap:8px; align-items:center;'>
            <div style='flex:2;'>
              <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>① 이름표 붙은 예시를 줘요</b></div>
              <div style='display:flex; gap:6px;'>
                <div style='flex:1; background:#FFEBEE; {_lab}'>🍎<br><b>"사과"</b></div>
                <div style='flex:1; background:#FFF9C4; {_lab}'>🍌<br><b>"바나나"</b></div>
                <div style='flex:1; background:#FFEBEE; {_lab}'>🍎<br><b>"사과"</b></div>
                <div style='flex:1; background:#FFF9C4; {_lab}'>🍌<br><b>"바나나"</b></div>
              </div>
            </div>
            <div style='font-size:24px;'>➡️</div>
            <div style='flex:1.2; background:#F3E5F5; {_lab}'>
              🧠<br><b>AI가 규칙 발견!</b><br>
              <span style='font-size:12px; color:#666;'>"빨갛고 동그라면 사과,<br>길고 노라면 바나나!"</span>
            </div>
            <div style='font-size:24px;'>➡️</div>
            <div style='flex:1.2; background:#E8F5E9; {_lab}'>
              ❓ 새 과일 등장!<br><span style='font-size:26px;'>🍎</span><br>
              <b style='color:#2E7D32;'>🤖 "사과예요!" ✅</b>
            </div>
          </div>
        </div>
        """)

        st.write("")
        st.markdown("#### 🎮 내가 AI가 되어볼까요? (지도학습 체험)")
        st.write("아래 이름표 붙은 예시들을 잘 보고, **내가 AI라면** 새 동물을 뭐라고 답할지 골라보세요!")
        md_html(f"""
        <div style='display:flex; gap:8px;'>
            <div style='flex:1; background:#E3F2FD; {_lab}'>🐦 <b>"하늘 친구"</b></div>
            <div style='flex:1; background:#E3F2FD; {_lab}'>🦅 <b>"하늘 친구"</b></div>
            <div style='flex:1; background:#E0F7FA; {_lab}'>🐟 <b>"물 친구"</b></div>
            <div style='flex:1; background:#E0F7FA; {_lab}'>🐙 <b>"물 친구"</b></div>
        </div>
        """)
        sup_pick = st.radio("❓ 새로 나타난 **🦆 오리(하늘을 날아요!)**는 뭐라고 답할까요?",
                            ["선택 안 함", "하늘 친구", "물 친구"],
                            horizontal=True, key="sup_learn_quiz")
        if sup_pick == "하늘 친구":
            st.success("🎉 정답! 이름표 붙은 예시에서 배운 규칙('날면 하늘 친구')으로 새 동물을 맞혔어요. "
                       "방금 여러분이 한 게 바로 **지도학습 AI**가 하는 일이에요!")
        elif sup_pick == "물 친구":
            st.warning("음, 오리는 헤엄도 치지만 **하늘을 난다**고 했죠? 예시를 보면 나는 친구들은 '하늘 친구'였어요. 다시 골라볼까요?")

        st.write("")
        st.info("💡 우리가 배울 섬 중에 **마법의 선 긋기, 스무고개 탐정, 가장 친한 친구**가 바로 이 지도학습 친구들이에요!")

    # ===== [4단계: 비지도학습] ===========================================
    with tab_w4:
        st.markdown("### 🧩 비지도학습: 이름표 없이 스스로 모둠을 만들어요!")
        md_html("""
        <div style='font-size:18px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:16px 22px;'>
        이번엔 <b>이름표(정답)가 하나도 없어요!</b><br>
        어질러진 장난감 상자를 정리할 때, 누가 알려주지 않아도
        <b>"비슷한 것끼리 모아볼까?"</b> 하고 스스로 나누죠?<br>
        AI가 이렇게 <b>정답 없이, 비슷한 것끼리 스스로 묶으며</b> 배우는 방법을 <b>'비지도학습'</b>이라고 해요.
        </div>
        """)

        st.write("")
        st.markdown("#### 👀 눈으로 봐요: 버튼을 눌러 AI에게 정리를 맡겨봐요!")
        unsup_run = st.radio("어질러진 장난감을 AI에게 맡기면?",
                             ["😵 정리 전 (뒤죽박죽)", "✨ AI가 정리한 후"],
                             horizontal=True, key="unsup_demo")
        _lab2 = "border-radius:12px; padding:12px 8px; text-align:center;"
        if unsup_run.startswith("😵"):
            md_html(f"""
            <div style='background:#FFEBEE; {_lab2}'>
                <div style='font-size:34px; letter-spacing:6px;'>🚗🧸⚽🧸🚕⚾🚙🏀🧸</div>
                <div style='font-size:14px; color:#C62828; margin-top:6px;'>이름표도 없고, 뒤죽박죽이에요! 😵</div>
            </div>
            """)
            st.caption("👆 위에서 '✨ AI가 정리한 후'를 눌러보세요!")
        else:
            md_html(f"""
            <div style='display:flex; gap:10px;'>
              <div style='flex:1; background:#E3F2FD; {_lab2}'>
                  <b>모둠 A</b><br><span style='font-size:30px;'>🚗🚕🚙</span><br>
                  <span style='font-size:12px; color:#666;'>바퀴 달린 친구들</span></div>
              <div style='flex:1; background:#FFF9C4; {_lab2}'>
                  <b>모둠 B</b><br><span style='font-size:30px;'>⚽⚾🏀</span><br>
                  <span style='font-size:12px; color:#666;'>동글동글 공 친구들</span></div>
              <div style='flex:1; background:#F3E5F5; {_lab2}'>
                  <b>모둠 C</b><br><span style='font-size:30px;'>🧸🧸🧸</span><br>
                  <span style='font-size:12px; color:#666;'>포근포근 인형 친구들</span></div>
            </div>
            """)
            st.success("✨ 아무도 정답을 알려주지 않았는데, AI가 **비슷한 것끼리 스스로** 모둠을 만들었어요! "
                       "이게 바로 **비지도학습**이에요.")

        st.write("")
        st.markdown("#### 🆚 지도학습 vs 비지도학습, 딱 한 줄로 정리!")
        v1, v2 = st.columns(2)
        with v1:
            md_html("""
            <div style='background:#E8F5E9; border:2px solid #2E7D32; border-radius:16px; padding:16px; text-align:center; height:190px;'>
                <div style='font-size:34px;'>🏷️</div>
                <div style="font-family:'Jua',sans-serif; font-size:19px; color:#2E7D32;">지도학습</div>
                <div style='font-size:14px; color:#3A4A6B; margin-top:8px; line-height:1.8;'>
                    <b>이름표(정답)가 있어요!</b><br>
                    "이건 사과 🍎, 이건 바나나 🍌"<br>
                    정답을 보고 규칙을 배워요</div>
            </div>
            """)
        with v2:
            md_html("""
            <div style='background:#F3E5F5; border:2px solid #8E24AA; border-radius:16px; padding:16px; text-align:center; height:190px;'>
                <div style='font-size:34px;'>🧩</div>
                <div style="font-family:'Jua',sans-serif; font-size:19px; color:#8E24AA;">비지도학습</div>
                <div style='font-size:14px; color:#3A4A6B; margin-top:8px; line-height:1.8;'>
                    <b>이름표가 없어요!</b><br>
                    "비슷한 것끼리 모아볼까?"<br>
                    스스로 모둠을 만들며 배워요</div>
            </div>
            """)

        st.write("")
        st.markdown("#### ✅ 진짜 이해했는지 확인해봐요!")
        chk = st.radio("사진 1000장을 주면서 **이름표 없이** '비슷한 사진끼리 알아서 앨범을 나눠줘!'라고 했어요. 이건 무슨 학습일까요?",
                       ["선택 안 함", "🏷️ 지도학습", "🧩 비지도학습"],
                       horizontal=True, key="sup_vs_unsup_quiz")
        if chk == "🧩 비지도학습":
            st.success("🎉 정답! **이름표(정답) 없이** 비슷한 것끼리 스스로 묶었으니 비지도학습이에요. 완벽해요!")
        elif chk == "🏷️ 지도학습":
            st.warning("힌트! 이름표(정답)를 **줬나요, 안 줬나요?** '이름표 없이'라고 했으니... 다시 골라볼까요?")

        st.write("")
        st.info("💡 우리가 배울 섬 중에 **비슷한 친구끼리(K-평균)**가 바로 이 비지도학습 친구예요!")

    # ===== [5단계: 생활 속 AI 찾기] ======================================
    with tab_w5:
        st.markdown("### 👀 우리도 이미 매일 AI를 만나고 있어요")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📱 **유튜브 추천 영상**\n\n내가 좋아했던 영상을 보고, '이런 영상도 좋아하겠지?' 하고 골라줘요.")
        with col2:
            st.info("🔓 **얼굴 인식 잠금 해제**\n\n내 얼굴을 많이 본 폰이, 다른 사람 얼굴과 내 얼굴을 구분해요.")
        with col3:
            st.info("📧 **스팸 메일 걸러내기**\n\n이상한 광고 메일들의 특징을 배워서, 광고 메일함으로 알아서 보내줘요.")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.info("🗣️ **AI 스피커 / 음성비서**\n\n수많은 목소리를 배워서, 내 말을 알아듣고 대답해줘요.")
        with col5:
            st.info("🗺️ **내비게이션 빠른 길 찾기**\n\n지금까지의 교통 데이터를 배워서, 가장 빠른 길을 예측해줘요.")
        with col6:
            st.info("🌍 **번역기**\n\n엄청나게 많은 문장을 배워서, 다른 나라 말로 바꿔줘요.")

        st.write("")
        st.markdown("#### 🔍 찾았다, AI! 다음 중 AI가 **아닌** 것은 무엇일까요?")
        life_q = st.radio("하나만 골라보세요!",
                          ["선택 안 함",
                           "① 내 취향을 배워서 영상을 골라주는 유튜브 추천",
                           "② 버튼을 누르면 정해진 시간만 도는 전자레인지 타이머",
                           "③ 내 목소리를 배워서 알아듣는 AI 스피커"],
                          key="life_ai_quiz")
        if life_q.startswith("②"):
            st.success("🎉 정답! 전자레인지 타이머는 **정해진 규칙대로만** 움직여요. "
                       "예시(데이터)를 보고 스스로 배우는 게 아니니까 AI가 아니에요!")
        elif life_q.startswith(("①", "③")):
            st.warning("그건 데이터를 보고 **스스로 배우는** AI가 맞아요! '배우지 않고 정해진 대로만' 움직이는 걸 찾아보세요.")

        st.write("")
        st.info("💡 그런데 AI가 '규칙을 찾는 방법'은 한 가지가 아니에요! 마지막 탭에서 앞으로 배울 **6가지 방법**을 미리 만나봐요.")

    # ===== [6단계: 6가지 섬 미리보기] ====================================
    with tab_w6:
        st.markdown("### 🗺️ 오늘 만날 6가지 방법 미리보기")
        st.write("앞으로 배울 6가지 섬은 전부 'AI가 규칙을 찾는 과정'을 **서로 다른 방법**으로 하는 거예요.")
        st.caption("👇 궁금한 방법을 눌러 바로 그 섬으로 갈 수 있어요!")

        preview = [
            ("📈", "마법의 선 긋기", "점들 사이 '선'으로 규칙 찾기", SECTION_COLORS["reg"], "📈 1. 마법의 선 긋기"),
            ("⛰️", "보물찾기 산", "한 걸음씩 정답에 다가가기", SECTION_COLORS["gd"], "⛰️ 2. 보물찾기 산"),
            ("🌳", "스무고개 탐정", "질문으로 좁혀가며 찾기", SECTION_COLORS["dt"], "🌳 3. 스무고개 탐정"),
            ("🤝", "가장 친한 친구", "가까운 이웃에게 물어보기", SECTION_COLORS["knn"], "🤝 4. 가장 친한 친구"),
            ("🎨", "비슷한 친구끼리", "닮은 것끼리 스스로 모으기", SECTION_COLORS["km"], "🎨 5. 비슷한 친구끼리"),
            ("🧠", "생각 주머니", "여러 요정의 판단을 합치기", SECTION_COLORS["nn"], "🧠 6. 똑똑한 생각 주머니"),
        ]
        pcols = st.columns(6)
        for i, (col, (emoji, name, desc, color, target)) in enumerate(zip(pcols, preview)):
            with col:
                md_html(f"""
                <div style="background:{color}18; border:2px solid {color}55; border-radius:16px; padding:14px 8px; text-align:center; height:172px;">
                    <div style="font-size:38px;">{emoji}</div>
                    <div style="font-family:'Jua',sans-serif; font-size:19px; color:{color}; margin-top:6px; line-height:1.25;">{name}</div>
                    <div style="font-size:12px; color:#3A4A6B; margin-top:6px;">{desc}</div>
                </div>
                """)
                if st.button("이동하기", key=f"preview_go_{i}", **_STRETCH):
                    go_to(target)

        st.write("")
        st.markdown("#### 🏷️ 어떤 섬이 무슨 학습일까요?")
        s1, s2 = st.columns(2)
        with s1:
            md_html("""
            <div style='background:#E8F5E9; border-radius:14px; padding:14px 18px; font-size:14px; line-height:2.0;'>
                <b>🏷️ 지도학습 (이름표 있음)</b><br>
                📈 마법의 선 긋기 · ⛰️ 보물찾기 산<br>
                🌳 스무고개 탐정 · 🤝 가장 친한 친구 · 🧠 생각 주머니
            </div>
            """)
        with s2:
            md_html("""
            <div style='background:#F3E5F5; border-radius:14px; padding:14px 18px; font-size:14px; line-height:2.0;'>
                <b>🧩 비지도학습 (이름표 없음)</b><br>
                🎨 비슷한 친구끼리<br>
                <span style='font-size:12px; color:#666;'>정답 없이 비슷한 것끼리 스스로 모둠을 만들어요!</span>
            </div>
            """)

        st.write("")
        if st.button("🚀 좋아요, 첫 번째 섬으로 출발!", **_STRETCH):
            go_to("📈 1. 마법의 선 긋기")

# --- [메뉴 2: 마법의 선 긋기 (선형회귀)] ---
elif menu == "📈 1. 마법의 선 긋기":
    hero_card("🎯", "마법의 선: 점들 사이로 선을 예쁘게 그려요!",
               "붕어빵 크기와 가격의 관계를 찾아 미래를 예측하는 AI 마법사가 되어봐요.",
               SECTION_COLORS["reg"], term="선형회귀 (Linear Regression)")

    with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
        md_html("""
        - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
          → 붕어빵 크기와 가격의 관계를 나타내는 '선'을 직접 찾아보며 **기계학습(선형회귀)의 기본 원리**를 체험해요.
        - **[실과] [6실05-04]** 디지털 데이터와 아날로그 데이터의 특징을 이해하고, 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
          → 크기·가격 같은 숫자 데이터가 어떻게 AI 예측에 쓰이는지 탐색해요.
        - **[수학] [6수04-01]** 한 양이 변할 때 다른 양이 그에 종속하여 변하는 대응 관계를 나타낸 표에서 규칙을 찾아 설명하고, □, △ 등을 사용하여 식으로 나타낼 수 있다.
          → 붕어빵 '크기'와 '가격'의 대응 관계를 표와 그래프로 나타내고 규칙(식)을 찾는 활동과 직접 연결돼요.
        """)

    tab_intro, tab_tutorial, tab_free, tab_game = st.tabs([
        "🌉 1단계: 먼저 알아보기",
        "🎯 2단계: 마법사 훈련소 (연습하기)",
        "🚀 3단계: 내 맘대로 마법진 (자유 놀이)",
        "🐟 4단계: 붕어빵 가게 게임 도전!"
    ])

    # [1단계: 먼저 알아보기 — 훅 → 미션 → 예시비교] ------------------------
    with tab_intro:
        st.markdown("### 🌉 어? 나 이거 이미 알아!")
        md_html("""
        <div style='font-size: 19px; line-height: 1.9; background:#FFF8E1; border-radius:14px; padding:18px 22px;'>
        떡볶이 1인분을 먹으면 조금 맵고, 2인분을 먹으면 더 맵겠죠?<br>
        그럼 <b>3인분을 먹으면 얼마나 매울지</b>도 미리 짐작할 수 있지 않을까요?<br><br>
        이렇게 <b>"이만큼 하면, 저만큼 된다"</b>는 규칙을 과거의 경험에서 찾아내는 것! 이게 바로 오늘 배울 AI 마법의 정체예요.
        </div>
        """)

        st.write("")
        st.markdown("### 🎯 오늘의 미션")
        st.success("**빨간 선을 파란 점들에 최대한 가깝게 붙여서, '틀린 정도(오차)'를 0에 가깝게 만들어보세요!**")

        st.write("")
        st.markdown("### 👀 조작하기 전에, 먼저 눈으로 봐요")

        ex_x = np.array([1, 2, 3, 4, 5])
        ex_y = np.array([20, 35, 55, 75, 95])

        # 왼쪽: 설명·선택 / 오른쪽: 그래프 → 그래프 비율이 알맞게 보인다
        col_lr_txt, col_lr_fig = st.columns([1, 1.4])

        with col_lr_txt:
            st.write("친구들이 '공부한 시간'과 '받은 시험 점수'를 파란 점으로 찍어놨어요. "
                     "아래 버튼을 눌러서 **나쁜 선**과 **좋은 선**이 어떻게 다른지 먼저 비교해봐요.")
            ex_choice = st.radio("어떤 선을 볼까요?", ["😵 엉뚱한 선", "😎 완벽한 선"], key="lr_example_choice")

        if ex_choice == "😵 엉뚱한 선":
            ex_w, ex_b = 3.0, 45.0
        else:
            ex_w, ex_b = np.polyfit(ex_x, ex_y, 1)

        ex_pred = ex_w * ex_x + ex_b
        ex_err = np.mean((ex_y - ex_pred) ** 2)

        with col_lr_txt:
            if ex_choice == "😵 엉뚱한 선":
                st.error(f"틀린 정도: {ex_err:.0f}점 → 점들과 선 사이 회색 틈(오차)이 크게 벌어져 있죠? 이러면 예측이 자꾸 틀려요.")
            else:
                st.success(f"틀린 정도: {ex_err:.0f}점 → 회색 틈이 거의 없어요! 이렇게 만드는 게 오늘의 목표예요.")

        with col_lr_fig:
            fig_ex = go.Figure()
            fig_ex.add_trace(go.Scatter(x=ex_x, y=ex_y, mode='markers', name='실제 점수', marker=dict(size=15, color='blue')))
            fig_ex.add_trace(go.Scatter(x=ex_x, y=ex_pred, mode='lines', name='선', line=dict(color='red', width=5)))
            for i in range(len(ex_x)):
                fig_ex.add_trace(go.Scatter(x=[ex_x[i], ex_x[i]], y=[ex_y[i], ex_pred[i]], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))
            fig_ex.update_layout(xaxis_title="공부시간(시간)", yaxis_title="시험점수(점)", height=330,
                                  margin=dict(l=20, r=20, t=30, b=20), dragmode=False, hovermode=False,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_ex.update_xaxes(fixedrange=True)
            fig_ex.update_yaxes(fixedrange=True)
            st.plotly_chart(fig_ex, **_STRETCH, config={'displayModeBar': False})

        st.write("")
        st.info("💡 **이제 2단계 '마법사 훈련소'에서 이 선을 직접 만들어볼까요?**")

    # [2단계: 마법사 훈련소 — 직접 조작 + 생각 질문 + 이름 배우기] ------------
    with tab_tutorial:
        md_html("""
        <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px;'>
            <h3 style='color: #1565C0; margin-top: 0;'>🕹️ 이제 네 차례! 빨간 선을 파란 점들에 딱 맞게 움직여라!</h3>
            <p style='font-size: 19px; line-height: 2.2; margin-bottom: 0;'>
            <b>1️⃣ 관찰하기 :</b> 파란 점들은 친구들이 '공부한 시간'과 '내일 받을 시험 점수'예요.<br>
            <b>2️⃣ 조종하기 :</b> 아래에 있는 2개의 동그란 버튼(조종기)을 마우스로 잡고 양옆으로 쓱쓱 밀어보세요.<br>
            <b>3️⃣ 목표 달성 :</b> 빨간 선이 파란 점들 한가운데를 딱! 지나가게 만들어보세요.
            </p>
        </div>
        <br>
        """)

        with st.expander("🤔 시작하기 전에 잠깐 생각해보기"):
            think1 = st.radio(
                "지금 빨간 선이 파란 점들보다 훨씬 **아래쪽**에 있다면, 기울기 조종기를 어느 쪽으로 밀어야 선이 점들에 가까워질까요?",
                ["선택 안 함", "① 기울기를 더 올린다 (조종기를 오른쪽으로)", "② 기울기를 더 내린다 (조종기를 왼쪽으로)"],
                key="lr_think1"
            )
            if think1.startswith("①"):
                st.success("맞아요! 선이 점들보다 아래에 있으면, 기울기를 올려서 선을 점들 쪽으로 끌어올려야 해요. 직접 확인해볼까요?")
            elif think1.startswith("②"):
                st.warning("한번 직접 해보면서 확인해봐요! 아래 조종기를 오른쪽으로 밀어서 기울기를 올려보면 어떻게 되는지 살펴보세요.")

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([20, 35, 55, 75, 95])

        st.markdown("### 🎛️ 마법의 선 조종기 (여기를 움직이세요!)")
        col_slider1, col_slider2 = st.columns(2)
        with col_slider1:
            w = st.slider("📐 선의 기울기 (위아래로 꺾기)", 0.0, 30.0, 10.0, 0.5, key="lr_w")
        with col_slider2:
            b = st.slider("⬆️ 선의 시작 위치 (위아래로 이동)", 0, 50, 10, 1, key="lr_b")

        st.divider()

        y_pred = w * x_data + b
        mse = np.mean((y_data - y_pred)**2)

        col_plot, col_loss = st.columns([1.5, 1])

        with col_plot:
            st.markdown("#### 🖍️ 내가 그리는 마법의 선")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='실제 점수', marker=dict(size=15, color='blue')))
            fig.add_trace(go.Scatter(x=x_data, y=y_pred, mode='lines', name='내가 그린 선', line=dict(color='red', width=5)))

            for i in range(len(x_data)):
                fig.add_trace(go.Scatter(x=[x_data[i], x_data[i]], y=[y_data[i], y_pred[i]], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))

            fig.update_layout(xaxis_title="공부시간(시간)", yaxis_title="시험점수(점)", dragmode=False, hovermode=False, height=260, margin=dict(l=20, r=20, t=30, b=20))
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            st.plotly_chart(fig, **_STRETCH, config={'displayModeBar': False})

            if mse < 20:
                st.success("🎉 와! 선을 완벽하게 그렸어요! 회색 선(틈)이 거의 다 사라졌네요!")
            elif mse < 100:
                st.info("👍 거의 다 왔어요! 조종기를 조금만 더 쓱쓱 밀어보세요.")
            else:
                st.warning("👀 파란 점들과 빨간 선 사이의 '회색 틈'이 짧아지도록 선을 움직여주세요!")

        with col_loss:
            st.markdown("#### 🎢 틀린 점수 미끄럼틀")
            st.write("오른쪽 주황색 공이 **내 위치**예요. 선을 잘 그릴수록 공이 골짜기 밑바닥으로 굴러가요!")

            w_range = np.linspace(0, 30, 100)
            loss_range = [np.mean((y_data - (wi * x_data + b))**2) for wi in w_range]
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=w_range, y=loss_range, mode='lines', name='틀린 점수 골짜기', line=dict(color='purple', width=4)))
            fig_loss.add_trace(go.Scatter(x=[w], y=[mse], mode='markers', name='현재 내 위치', marker=dict(size=25, color='orange', line=dict(width=2, color='white'))))

            fig_loss.update_layout(xaxis_title="선의 기울기", yaxis_title="틀린 점수(오차)", dragmode=False, hovermode=False, height=260, margin=dict(l=20, r=20, t=30, b=20))
            fig_loss.update_xaxes(fixedrange=True)
            fig_loss.update_yaxes(fixedrange=True)
            st.plotly_chart(fig_loss, **_STRETCH, config={'displayModeBar': False})

            st.metric("현재 틀린 점수 (0에 가까울수록 최고!)", f"{mse:.1f}점")

        if mse < 20:
            st.markdown("---")
            md_html("""
            <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:16px 20px;">
                <b>🏷️ 오늘 배운 것</b><br>
                방금 네가 한 것처럼, <b>점들 사이로 가장 잘 맞는 선을 찾는 것</b>을 인공지능에서는
                <b>'선형회귀(linear regression)'</b>라고 불러요. 어려운 이름이지만, 사실 너는 이미 할 줄 알게 됐어요!
            </div>
            """)

            with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                quiz_reg = st.radio(
                    "점들 사이로 가장 잘 맞는 선을 찾아 미래를 예측하는 방법을 뭐라고 배웠나요?",
                    ["선택 안 함", "① 선형회귀", "② 결정트리", "③ K-평균 군집화"],
                    key="quiz_reg"
                )
                if quiz_reg == "① 선형회귀":
                    st.success("🌟 정답이에요! '완벽 이해' 보너스 배지 획득!")
                    st.session_state["bonus_badges"].add("🌟 선형회귀 완벽 이해")
                elif quiz_reg != "선택 안 함":
                    st.warning("다시 한 번 생각해볼까요? 위 '오늘 배운 것' 카드를 다시 읽어보세요!")

            st.write("")
            next_section_button("📈 1. 마법의 선 긋기", "reg")


    # [3단계: 내 맘대로 마법진 (커스텀 데이터 + 예측 테스트)]
    with tab_free:
        st.markdown("<h2 style='color: #FF6B6B;'>🚀 내 맘대로 데이터 탐험!</h2>", unsafe_allow_html=True)
        st.write("이번엔 표에 숫자를 직접 마음대로 적어보고, 인공지능 요정이 미래를 얼마나 잘 맞히는지 테스트해봐요!")

        st.markdown("### 1️⃣ 어떤 마법을 부려볼까요? (나만의 주제 정하기)")
        col_x_name, col_y_name = st.columns(2)
        with col_x_name:
            custom_x = st.text_input("🟢 원인 (X축) 이름을 적어주세요", value="아이스크림 먹은 개수", key="free_xname")
        with col_y_name:
            custom_y = st.text_input("🔴 결과 (Y축) 이름을 적어주세요", value="배탈 난 횟수", key="free_yname")

        st.info("👇 아래 표의 숫자를 클릭해서 마음대로 바꿔보세요! 새로운 줄을 추가하거나 지울 수도 있어요.")

        default_free_data = pd.DataFrame({
            custom_x: [1, 2, 3, 4, 5],
            custom_y: [10, 25, 28, 45, 52]
        })

        edited_df = st.data_editor(default_free_data, num_rows="dynamic", **_STRETCH, key="free_editor")

        if len(edited_df) >= 2:
            edited_df = edited_df.dropna()
            try:
                free_x = pd.to_numeric(edited_df[custom_x]).values
                free_y = pd.to_numeric(edited_df[custom_y]).values
                parse_ok = True
            except (ValueError, TypeError):
                st.warning("앗! 표에는 숫자만 적어주세요, 마법사님! 🧙 글자나 이상한 기호가 섞여 있으면 마법을 부릴 수 없어요.")
                parse_ok = False

            if parse_ok:
                w_ai, b_ai = np.polyfit(free_x, free_y, 1)
                free_y_pred = w_ai * free_x + b_ai
                mse_ai = np.mean((free_y - free_y_pred)**2)

                st.divider()

                st.markdown("### 2️⃣ 인공지능 요정의 정답 확인하기")
                st.success(f"✨ **요정의 마법 성공!** 요정이 1칸 갈 때마다 **{w_ai:.1f}**씩 올라가는, "
                           f"점들에 **가장 잘 맞는 선**을 알아서 찾았어요! (모든 점을 정확히 지나가진 않아도, 오차가 가장 작은 선이에요.)")

                col_plot_free, col_loss_free = st.columns([1.5, 1])

                with col_plot_free:
                    st.markdown("#### 🖍️ 요정이 쫙! 그어준 마법의 선")
                    fig_free = go.Figure()
                    fig_free.add_trace(go.Scatter(x=free_x, y=free_y, mode='markers', name='내가 만든 점들', marker=dict(size=15, color='green')))
                    fig_free.add_trace(go.Scatter(x=free_x, y=free_y_pred, mode='lines', name='AI가 찾은 가장 잘 맞는 선', line=dict(color='gold', width=5)))

                    for i in range(len(free_x)):
                        fig_free.add_trace(go.Scatter(x=[free_x[i], free_x[i]], y=[free_y[i], free_y_pred[i]], mode='lines', line=dict(color='lightgray', dash='dash'), showlegend=False))

                    fig_free.update_layout(xaxis_title=custom_x, yaxis_title=custom_y, dragmode=False, hovermode="closest", height=300, margin=dict(l=20, r=20, t=30, b=20))
                    fig_free.update_xaxes(fixedrange=True)
                    fig_free.update_yaxes(fixedrange=True)
                    st.plotly_chart(fig_free, **_STRETCH, config={'displayModeBar': False})

                    if mse_ai < 20:
                        st.info("👍 **요정의 칭찬:** 점들이 선에 찰싹 붙어있네요! 규칙이 아주 뚜렷해서 예측이 잘 맞을 거예요!")
                    elif mse_ai < 100:
                        st.warning("🤔 **요정의 고민:** 점들이 선에서 조금 떨어져 있어요. 예측이 조금 틀릴 수도 있겠네요.")
                    else:
                        st.error("🌪️ **요정의 혼란:** 점들이 너무 뒤죽박죽 퍼져 있어요! 이런 데이터는 미래를 맞히기가 너무 어려워요.")

                with col_loss_free:
                    st.markdown("#### 🎢 요정의 미끄럼틀")
                    st.write("AI 요정은 틀린 점수가 **가장 0에 가까운 밑바닥(별 모양)**을 한 번에 찾아냈어요!")

                    w_range_free = np.linspace(w_ai - 10, w_ai + 10, 100)
                    loss_range_free = [np.mean((free_y - (wi * free_x + b_ai))**2) for wi in w_range_free]

                    fig_loss_free = go.Figure()
                    fig_loss_free.add_trace(go.Scatter(x=w_range_free, y=loss_range_free, mode='lines', name='틀린 점수 골짜기', line=dict(color='purple', width=4)))
                    fig_loss_free.add_trace(go.Scatter(x=[w_ai], y=[mse_ai], mode='markers', name='AI가 찾은 최하점', marker=dict(size=30, color='gold', symbol='star', line=dict(width=2, color='black'))))

                    fig_loss_free.update_layout(xaxis_title="선의 기울기", yaxis_title="틀린 점수(오차)", dragmode=False, hovermode=False, height=300, margin=dict(l=20, r=20, t=30, b=20))
                    fig_loss_free.update_xaxes(fixedrange=True)
                    fig_loss_free.update_yaxes(fixedrange=True)
                    st.plotly_chart(fig_loss_free, **_STRETCH, config={'displayModeBar': False})

                st.divider()

                st.markdown("### 3️⃣ 🔮 마법 구슬로 미래 예측하기 (테스트)")
                st.write("요정이 찾은 규칙(선)을 믿고, 표에 없는 새로운 숫자를 넣어 미래 파악해봐요!")

                col_test1, col_test2 = st.columns([1, 1.5])
                with col_test1:
                    test_val = st.number_input(f"궁금한 '{custom_x}'의 숫자를 입력하세요:", value=float(free_x.max() + 1), key="free_testval")
                    predicted_val = w_ai * test_val + b_ai

                with col_test2:
                    md_html(f"""
                    <div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px; border: 2px solid #FFB74D;'>
                        <h4 style='color: #E65100; margin-top: 0;'>🔮 마법 구슬의 대답</h4>
                        <p style='font-size: 18px;'>만약 <b>{custom_x}</b>(이)가 <b>{test_val}</b> 이라면,<br>
                        <b>{custom_y}</b>(은)는 약 <b style='color: #D84315; font-size: 24px;'>{predicted_val:.1f}</b> 일 것입니다!</p>
                    </div>
                    """)

        else:
            st.error("앗! 점이 최소 2개는 있어야 선을 이어볼 수 있어요. 표에 숫자를 더 적어주세요!")

    # [4단계: 붕어빵 가게 게임] ------------------------------------------------
    with tab_game:
        st.header("🎯 붕어빵 가게 사장님이 되어보자!")
        st.write("붕어빵 **크기(cm)**에 따라 손님들이 낸 **가격(원)**이 점으로 찍혀 있어요. "
                 "조종기를 움직여서 점들을 가장 잘 지나가는 **'마법의 가격 선'**을 그려봐요!")

        if st.session_state["fb_data"] is None:
            rng = np.random.default_rng(42)
            sizes = rng.uniform(5, 20, 12)
            true_slope, true_intercept = 45, 150
            prices = true_slope * sizes + true_intercept + rng.normal(0, 60, 12)
            st.session_state["fb_data"] = (sizes, prices, true_slope, true_intercept)

        sizes, prices, true_slope, true_intercept = st.session_state["fb_data"]

        colL, colR = st.columns([1, 2])
        with colL:
            st.write("#### 🕹️ 조종기")
            slope = st.slider("📐 기울기 (선이 얼마나 가파를까요?)", 0.0, 100.0, 30.0, step=1.0, key="fb_slope")
            intercept = st.slider("📍 시작 위치 (선이 어디서 출발할까요?)", 0.0, 400.0, 100.0, step=5.0, key="fb_intercept")

        pred = slope * sizes + intercept
        error = np.mean((pred - prices) ** 2)  # 1단계 훈련소와 동일하게 '틀린 점수(오차제곱평균, MSE)'로 통일

        with colR:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sizes, y=prices, mode="markers", name="손님이 낸 가격",
                                      marker=dict(size=14, color="#1E88E5")))
            x_line = np.linspace(0, 22, 50)
            fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                                      name="내가 그린 마법의 선", line=dict(color="red", width=4)))
            fig.update_layout(xaxis_title="붕어빵 크기 (cm)", yaxis_title="가격 (원)",
                               height=260, margin=dict(l=20, r=20, t=20, b=20),
                               xaxis_range=[0, 22], yaxis_range=[0, 1200])
            st.plotly_chart(fig, **_STRETCH)

        if error < 2500:
            stars, msg = "⭐⭐⭐", "완벽해요! 진짜 붕어빵 박사님이시네요!"
        elif error < 7000:
            stars, msg = "⭐⭐", "아주 좋아요! 조금만 더 다듬으면 완벽!"
        elif error < 23000:
            stars, msg = "⭐", "괜찮아요! 조종기를 조금 더 움직여봐요."
        else:
            stars, msg = "☁️", "음... 점들이랑 너무 멀어요. 다시 도전!"

        st.write(f"### 틀린 점수(오차): {error:.0f} {stars}")
        st.info(msg)

        if error < 2500:
            st.success("🏅 '붕어빵 가격 박사' 배지 획득!")
            award_badge(0)

        st.divider()
        st.write("#### 🐟 손님 등장! '왕왕 큼직 붕어빵(18cm)'이 왔어요. 얼마를 받을까요?")
        guess_price = st.number_input("내가 정한 가격(원)", min_value=0, max_value=1500,
                                       value=int(slope * 18 + intercept), step=10, key="fb_guess")
        if st.button("손님에게 가격 알려주기 💰", key="fb_submit"):
            real_price = true_slope * 18 + true_intercept
            diff = abs(guess_price - real_price)
            if diff < 80:
                st.balloons()
                play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
                st.success(f"손님: '정답! 딱 좋은 가격이에요!' 🎉 (참고 시세: 약 {real_price:.0f}원)")
            else:
                st.warning(f"손님: '음... 조금 비싸거나 싼 것 같은데요?' (참고 시세: 약 {real_price:.0f}원)")

# --- [메뉴 3: 보물찾기 산 (경사하강법)] ---
elif menu == "⛰️ 2. 보물찾기 산":
    hero_card("⛰️", "보물찾기 산: 넘어지지 않고 내려가요!",
               "발걸음을 잘 조절해서 골짜기 밑 보물상자(정답)를 찾아봐요.",
               SECTION_COLORS["gd"], term="경사하강법 (Gradient Descent)")

    with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
        md_html("""
        - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
          → AI는 한 번에 정답을 맞히지 않고, **틀린 정도(오차)를 조금씩 줄여가며 스스로 개선**한다는 기계학습 원리를 발걸음 조절 활동으로 체험해요.
        - **[실과] [6실05-04]** 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
          → 발걸음마다 달라지는 '위치 값'이 AI 학습에서 오차 데이터로 쓰이는 과정을 간접 체험해요.
        """)

    tab_intro, tab_practice, tab_game = st.tabs(["🌉 1단계: 먼저 알아보기", "🎒 2단계: 직접 내려가보기", "🏂 3단계: 게임: 눈꽃 골짜기 스노보드"])

    # [1단계: 먼저 알아보기] -----------------------------------------------
    with tab_intro:
        st.markdown("### 🌉 어? 나 이거 이미 알아!")
        md_html("""
        <div style='font-size:19px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:18px 22px;'>
        깜깜한 밤에 계단을 내려갈 때, 발로 <b>더듬더듬 한 걸음씩</b> 내려가 본 적 있나요?<br>
        발을 너무 크게 내디디면 <b>헛디뎌서 넘어질</b> 수 있고, 너무 조금씩 내디디면 <b>시간이 오래</b> 걸리죠.<br><br>
        AI도 정답을 찾을 때 똑같아요. <b>딱 알맞은 걸음 크기</b>로, 조금씩 조금씩 정답에 다가간답니다.
        </div>
        """)

        st.write("")
        st.markdown("### 🎯 오늘의 미션")
        st.success("**보폭을 잘 골라서, 산 밑바닥(가장 낮은 곳= 정답)까지 튕겨나가지 않고 도착해보세요!**")

        st.write("")
        st.markdown("### 👀 조작하기 전에, 먼저 눈으로 봐요")

        # 왼쪽: 설명·선택 / 오른쪽: 그래프 → 그래프가 납작해지지 않고 비율이 알맞게 보인다
        col_gd_txt, col_gd_fig = st.columns([1, 1.4])

        with col_gd_txt:
            st.write("아래 버튼으로 '보폭이 너무 큰 경우'와 '보폭이 딱 좋은 경우'를 비교해봐요.")
            ex_step = st.radio("어떤 경우를 볼까요?", ["😱 보폭이 너무 큼", "😊 보폭이 딱 좋음"], key="gd_example_choice")

            if ex_step == "😱 보폭이 너무 큼":
                st.error("보세요! 보폭이 너무 크니까 보물상자를 휙 지나쳐서 반대편으로, "
                         "또 반대편으로 튕겨나가요. 이러면 영영 도착 못 해요!")
            else:
                st.success("보폭이 알맞으니 한 걸음씩 착실하게 보물상자 쪽으로 다가가네요! "
                           "이렇게 되는 게 목표예요.")

        ex_lr = 0.95 if ex_step == "😱 보폭이 너무 큼" else 0.12

        ex_path_x = [-9.0]
        ex_x = -9.0
        for _ in range(8):
            ex_x = ex_x - ex_lr * 2 * ex_x
            if abs(ex_x) > 15:
                ex_x = np.sign(ex_x) * 15
                ex_path_x.append(ex_x)
                break
            ex_path_x.append(ex_x)
        ex_path_y = [v ** 2 for v in ex_path_x]

        with col_gd_fig:
            ex_xs = np.linspace(-15, 15, 100)
            fig_ex = go.Figure()
            fig_ex.add_trace(go.Scatter(x=ex_xs, y=ex_xs**2, mode='lines', line=dict(color='lightgray', width=5), name='골짜기'))
            fig_ex.add_trace(go.Scatter(x=ex_path_x, y=ex_path_y, mode='lines+markers',
                                         marker=dict(size=10, color='orange'), line=dict(color='red', width=2, dash='dot'), name='발자국'))
            fig_ex.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=16, color='gold', symbol='diamond'), name='보물상자'))
            fig_ex.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=20), yaxis_range=[-5, 230],
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ex, **_STRETCH, config={'displayModeBar': False})

        st.write("")
        st.info("🔗 **1번 활동과 연결돼요!** 아까 '마법의 선 긋기'에서 봤던 **'틀린 점수 미끄럼틀'** 그래프, 기억나나요? 그 U자 모양 골짜기가 바로 지금 이 산이에요. AI는 사람이 손으로 선을 움직이는 대신, 이렇게 **한 걸음씩 스스로 내려가면서** 가장 좋은 선을 찾아낸답니다!")

        st.write("")
        st.info("💡 **다음 단계!** 2단계 탭에서 직접 보폭을 조절하며 산을 내려가봐요.")

    # [2단계: 직접 내려가보기] -----------------------------------------------
    with tab_practice:
        col_control, col_map = st.columns([1, 2])

        with col_control:
            st.write("### 🎒 탐험 준비물 챙기기")

            with st.expander("🤔 시작하기 전에 잠깐 생각해보기"):
                think_gd = st.radio(
                    "보폭(발걸음 크기)을 너무 크게 하면 어떻게 될까요?",
                    ["선택 안 함", "① 목표를 지나쳐서 튕겨나간다", "② 천천히 안전하게 도착한다"],
                    key="gd_think1"
                )
                if think_gd.startswith("①"):
                    st.success("맞아요! 방금 위에서 본 것처럼, 보폭이 너무 크면 목표를 지나쳐 튕겨나가요.")
                elif think_gd.startswith("②"):
                    st.warning("한번 아래 슬라이더로 보폭을 크게 해서 직접 확인해봐요!")

            lr = st.slider("👣 발걸음 크기 (너무 크면 우주로 날아가요!)", 0.01, 1.10, 0.10, 0.05, key="gd_learn_lr")
            epochs = st.slider("⏱️ 탐험할 시간 (시간이 많을수록 멀리 가요)", 1, 30, 10, key="gd_learn_epochs")

            st.divider()

            optimizer = st.radio("👀 앞을 보는 방법",
                ["매의 눈 (아주 꼼꼼하지만 조금 느려요)",
                 "손전등 켜기 (조금 비틀거리며 내려가요)",
                 "안대 쓰고 뛰기 (완전 지그재그로 내려가요)"], key="gd_learn_opt")

            penalty = st.radio("🎒 마법 배낭의 규칙",
                ["규칙 없음 (기본)",
                 "가벼운 배낭 (무거운 짐은 버리고 가요!)",
                 "작은 배낭 (모든 짐을 조금씩 줄여요!)"], key="gd_learn_penalty")

        with col_map:
            st.write("### 🗺️ 꼬마 탐험가의 산 내려가기 지도")
            x_vals = np.linspace(-10, 10, 100)
            y_vals = x_vals**2

            path_x = [-9.0]
            path_y = [81.0]
            current_x = -9.0

            # 보폭/시간/방법을 안 바꿔도 화면이 매번 흔들리지 않도록,
            # 현재 설정값에서만 정해지는 고정 시드를 사용한다 (문자열 해시 대신 명시적 매핑 사용)
            opt_code = {"매의 눈 (아주 꼼꼼하지만 조금 느려요)": 0,
                        "손전등 켜기 (조금 비틀거리며 내려가요)": 1,
                        "안대 쓰고 뛰기 (완전 지그재그로 내려가요)": 2}[optimizer]
            pen_code = {"규칙 없음 (기본)": 0,
                        "가벼운 배낭 (무거운 짐은 버리고 가요!)": 1,
                        "작은 배낭 (모든 짐을 조금씩 줄여요!)": 2}[penalty]
            seed_key = int(round(lr, 3) * 1000) * 10000 + epochs * 100 + opt_code * 10 + pen_code
            noise_rng = np.random.default_rng(seed_key)

            for i in range(epochs):
                grad = 2 * current_x

                if "손전등" in optimizer:
                    grad += noise_rng.normal(0, 1.5)
                elif "안대" in optimizer:
                    grad += noise_rng.normal(0, 4.0)

                if "가벼운 배낭" in penalty:
                    grad += 1.0 * np.sign(current_x)
                elif "작은 배낭" in penalty:
                    grad += 0.5 * 2 * current_x

                current_x = current_x - lr * grad

                if abs(current_x) > 15:
                    st.error("앗! 발걸음이 너무 커서 탐험가가 산 밖으로 우주 끝까지 날아갔어요! 🚀 발걸음을 줄여주세요!")
                    path_x.append(current_x)
                    path_y.append(100)
                    break

                path_x.append(current_x)
                path_y.append(current_x**2)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='lightgray', width=5), name='골짜기 모양'))
            fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines+markers',
                                     marker=dict(size=12, color='orange', symbol='circle', line=dict(width=2, color='white')),
                                     line=dict(color='red', width=3, dash='dot'),
                                     name='탐험가의 발자국'))
            fig.add_trace(go.Scatter(x=[path_x[0]], y=[path_y[0]], mode='markers', marker=dict(size=25, color='blue', symbol='star'), name='출발점'))
            fig.add_trace(go.Scatter(x=[path_x[-1]], y=[path_y[-1]], mode='markers+text', text=["현재 내 위치"], textposition="bottom center", marker=dict(size=25, color='green', symbol='x'), name='현재 위치'))

            fig.update_layout(xaxis_title="탐험가의 위치", yaxis_title="산의 높이 (틀린 점수)", hovermode="closest", height=300)
            st.plotly_chart(fig, **_STRETCH)

            if abs(path_x[-1]) < 0.5:
                st.success("🎉 만세! 완벽한 발걸음으로 산 바닥에 있는 보물상자를 찾았어요!")
                st.balloons()
                play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
                md_html("""
                <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:16px 20px; margin-top:12px;">
                    <b>🏷️ 오늘 배운 것</b><br>
                    이렇게 <b>한 걸음씩 조금씩 이동하면서 가장 낮은 곳(정답)을 찾아가는 방법</b>을 인공지능에서는
                    <b>'경사하강법(gradient descent)'</b>이라고 불러요.
                </div>
                """)
            elif abs(path_x[-1]) > 10:
                st.warning("발걸음을 너무 크게 했나 봐요! 조금 더 총총걸음으로 줄여볼까요?")
            else:
                st.info("바닥까지 거의 다 왔어요! 탐험하는 시간을 조금 더 늘려주세요!")

    # [게임: 눈꽃 골짜기 스노보드] ---------------------------------------
    with tab_game:
        st.header("🏂 눈 덮인 골짜기를 내려가 보물상자를 찾아라!")
        st.write("이번엔 **한 걸음씩 직접** 내려가 봐요! 보폭을 잘 골라야 튕겨나가지 않아요.")

        def gd_f(x):
            return x ** 2

        def gd_grad(x):
            return 2 * x

        colL, colR = st.columns([1, 2])
        with colL:
            st.write("#### 🕹️ 보폭 고르기")
            step_choice = st.radio("이번 걸음은 얼마나 크게?",
                                    ["👶 아기 걸음 (0.05)", "🚶 보통 걸음 (0.3)", "🦵 성큼 걸음 (1.05)"], key="gd_step_choice")
            lr2 = {"👶 아기 걸음 (0.05)": 0.05, "🚶 보통 걸음 (0.3)": 0.3, "🦵 성큼 걸음 (1.05)": 1.05}[step_choice]

            c1, c2 = st.columns(2)
            with c1:
                step_btn = st.button("👣 한 걸음 내려가기", **_STRETCH, key="gd_step_btn")
            with c2:
                reset_btn = st.button("🔄 처음부터 다시", **_STRETCH, key="gd_reset_btn")

            st.metric("현재 걸음 수", st.session_state["gd_steps"])
            st.metric("현재 위치(x)", f'{st.session_state["gd_x"]:.2f}')

        if reset_btn:
            st.session_state["gd_x"] = 8.0
            st.session_state["gd_steps"] = 0
            st.session_state["gd_history"] = [8.0]
            st.session_state["gd_failed"] = False
            st.session_state["gd_success"] = False

        if step_btn and not st.session_state["gd_failed"] and not st.session_state["gd_success"]:
            x = st.session_state["gd_x"]
            new_x = x - lr2 * gd_grad(x)
            st.session_state["gd_x"] = new_x
            st.session_state["gd_steps"] += 1
            st.session_state["gd_history"].append(new_x)

            if abs(new_x) > 15:
                st.session_state["gd_failed"] = True
            elif abs(new_x) < 0.15:
                st.session_state["gd_success"] = True

        with colR:
            xs = np.linspace(-15, 15, 200)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=xs, y=gd_f(xs), mode="lines", name="골짜기", line=dict(color="#90CAF9", width=6)))
            hx = st.session_state["gd_history"]
            hy = [gd_f(v) for v in hx]
            fig2.add_trace(go.Scatter(x=hx, y=hy, mode="lines+markers", name="내가 지나온 길",
                                      line=dict(color="orange", width=2), marker=dict(size=8)))
            fig2.add_trace(go.Scatter(x=[hx[-1]], y=[hy[-1]], mode="markers", name="현재 위치 🏂",
                                      marker=dict(size=20, color="red", symbol="star")))
            fig2.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="보물상자 🎁",
                                      marker=dict(size=16, color="gold", symbol="diamond")))
            fig2.update_layout(height=315, yaxis_range=[-5, 230], xaxis_range=[-15, 15])
            st.plotly_chart(fig2, **_STRETCH)

        if st.session_state["gd_failed"]:
            st.error("💥 슝~! 보폭이 너무 커서 골짜기 밖으로 튕겨나갔어요! '처음부터 다시'를 눌러 더 작은 보폭으로 도전해봐요.")
            with st.expander("🤔 왜 우주로 날아갈까요?"):
                st.write("가파른 미끄럼틀에서 너무 세게 점프하면 반대편으로 튕겨나가는 것과 같아요! "
                         "보폭(발걸음)이 너무 크면, 목표 지점을 훌쩍 지나쳐서 반대편 산으로 넘어가 버려요. "
                         "그래서 AI도 '적당한 보폭'을 찾는 게 아주 중요하답니다.")
        elif st.session_state["gd_success"]:
            st.balloons()
            play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
            st.success(f"🎉 보물상자 발견! 총 {st.session_state['gd_steps']}걸음 만에 도착했어요!")
            award_badge(1)

            with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                quiz_gd = st.radio(
                    "한 걸음씩 조금씩 이동하면서 가장 낮은 곳(정답)을 찾아가는 방법을 뭐라고 배웠나요?",
                    ["선택 안 함", "① 경사하강법", "② KNN", "③ 인공신경망"],
                    key="quiz_gd"
                )
                if quiz_gd == "① 경사하강법":
                    st.success("🌟 정답이에요! '완벽 이해' 보너스 배지 획득!")
                    st.session_state["bonus_badges"].add("🌟 경사하강법 완벽 이해")
                elif quiz_gd != "선택 안 함":
                    st.warning("다시 한 번 생각해볼까요? '원리 알아보기' 탭 내용을 떠올려봐요!")

            st.write("")
            next_section_button("⛰️ 2. 보물찾기 산", "gd")
        else:
            st.info("한 걸음씩 내려가며 보물상자에 가까워져 봐요. 보폭이 너무 크면 튕겨나갈 수 있어요!")

# --- [메뉴 4: 스무고개 탐정 (결정트리)] ---
elif menu == "🌳 3. 스무고개 탐정":
    hero_card("🌳", "스무고개 탐정: 질문으로 정답 찾기",
               "예/아니오 질문을 던져서 인공지능처럼 정답을 쏙쏙 찾아내는 명탐정이 되어봐요!",
               SECTION_COLORS["dt"], term="결정트리 (Decision Tree)")

    with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
        md_html("""
        - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
          → 특징(질문)을 기준으로 대상을 좁혀가는 과정을 통해 **AI가 여러 조건을 조합해 정답을 찾아가는 원리(결정트리)**를 체험해요.
        - **[사회] [6사03-04]** 정보화 시대의 변화상을 이해하고, 생활에 미치는 영향을 탐구한다.
          → 스무고개처럼 조건을 좁혀 판단하는 방식이 실제 생활 속 AI 서비스(추천, 진단 등)에 어떻게 쓰이는지와 연결지어 생각해볼 수 있어요.
        """)

    tab_intro, tab1, tab2, tab3, tab4 = st.tabs([
        "🌉 1단계: 먼저 알아보기",
        "🌿 미션 1: 나뭇잎 이름 맞히기",
        "🦁 미션 2: 동물 가족 찾기",
        "💡 비밀: 섞임 점수 놀이",
        "🕵️‍♂️ 게임: 동물 탐정 챌린지"
    ])

    # [1단계: 먼저 알아보기] -----------------------------------------------
    with tab_intro:
        st.markdown("### 🌉 어? 나 이거 이미 알아!")
        md_html("""
        <div style='font-size:19px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:18px 22px;'>
        스무고개 놀이 해본 적 있죠? "동물이에요?" "다리가 4개예요?" 이렇게 <b>예/아니오 질문</b>만으로
        답을 척척 맞히는 놀이요.<br><br>
        사실 AI도 똑같은 방법을 써요! <b>좋은 질문을 골라서 후보를 확 줄여나가면</b>, 몇 번 안 물어보고도 정답을 찾을 수 있답니다.
        </div>
        """)

        st.write("")
        st.markdown("### 🎯 오늘의 미션")
        st.success("**아래 미션들을 풀면서, '좋은 질문'을 골라 정답을 빨리 찾는 방법을 알아봐요!**")

        st.write("")
        st.markdown("### 👀 조작하기 전에, 먼저 눈으로 봐요")
        st.write("스무고개 AI가 정답을 찾는 모습은 **거꾸로 자란 나무**처럼 생겼어요. "
                 "질문을 하나 할 때마다 길이 **두 갈래**로 갈라지고, 후보가 확확 줄어들어요!")

        # ── ① 결정트리 흐름도: 질문 → 예/아니오 갈래 → 정답 ──────────────
        _dt_q = "background:#E3F2FD; border:2px solid #2D9CDB; border-radius:14px; padding:10px 14px; text-align:center; font-size:15px;"
        _dt_leaf_ok = "background:#E8F5E9; border:2px solid #2E7D32; border-radius:14px; padding:10px 12px; text-align:center; font-size:15px;"
        _dt_arrow = "text-align:center; font-size:14px; color:#3A4A6B; font-weight:bold;"
        md_html(f"""
        <div style='background:#FAFAFA; border-radius:16px; padding:18px 10px;'>
          <div style='display:flex; justify-content:center;'>
            <div style='{_dt_q} min-width:280px;'>🤔 <b>질문 1.</b> 다리가 4개인가요?<br>
              <span style='font-size:12px; color:#666;'>후보: 🐶🐱🐦🦆 (4마리)</span></div>
          </div>
          <div style='display:flex; justify-content:center; gap:120px; margin:4px 0;'>
            <div style='{_dt_arrow}'>👈 네</div>
            <div style='{_dt_arrow}'>아니오 👉</div>
          </div>
          <div style='display:flex; justify-content:center; gap:40px;'>
            <div style='{_dt_q} min-width:220px;'>🤔 <b>질문 2.</b> 야옹 하고 우나요?<br>
              <span style='font-size:12px; color:#666;'>후보: 🐶🐱 (2마리)</span></div>
            <div style='{_dt_q} min-width:220px;'>🤔 <b>질문 2.</b> 헤엄을 잘 치나요?<br>
              <span style='font-size:12px; color:#666;'>후보: 🐦🦆 (2마리)</span></div>
          </div>
          <div style='display:flex; justify-content:center; gap:20px; margin:4px 0;'>
            <div style='{_dt_arrow} width:110px;'>네</div>
            <div style='{_dt_arrow} width:110px;'>아니오</div>
            <div style='{_dt_arrow} width:110px;'>네</div>
            <div style='{_dt_arrow} width:110px;'>아니오</div>
          </div>
          <div style='display:flex; justify-content:center; gap:14px; flex-wrap:wrap;'>
            <div style='{_dt_leaf_ok} width:110px;'>🐱 고양이!</div>
            <div style='{_dt_leaf_ok} width:110px;'>🐶 강아지!</div>
            <div style='{_dt_leaf_ok} width:110px;'>🦆 오리!</div>
            <div style='{_dt_leaf_ok} width:110px;'>🐦 참새!</div>
          </div>
          <div style='text-align:center; margin-top:12px; font-size:14px; color:#3A4A6B;'>
            질문 딱 <b>2번</b> 만에 4마리 중 정답을 찾았어요! 이렇게 갈래갈래 나뉜 모양이
            <b>나무(트리)</b> 같아서 <b>'결정트리'</b>라고 불러요. 🌳
          </div>
        </div>
        """)

        st.write("")
        # ── ② 좋은 질문 vs 별로인 질문: 후보가 줄어드는 걸 직접 비교 ──────
        st.markdown("#### 🔍 그런데, 아무 질문이나 해도 될까요? 버튼을 눌러 비교해봐요!")
        dt_ex_choice = st.radio(
            "동물 6마리(🐶🐱 털친구 2, 🐦🦆 날개친구 2, 🐟🐸 물친구 2) 중에서 찾을 때, 어떤 질문을 볼까요?",
            ["😵 별로인 질문: \"이름이 '오리'인가요?\"", "😎 좋은 질문: \"날개가 있나요?\""],
            horizontal=True, key="dt_intro_q_choice")

        _box_l = "background:#FFEBEE; border-radius:12px; padding:12px; text-align:center;"
        _box_r = "background:#E8F5E9; border-radius:12px; padding:12px; text-align:center;"
        if dt_ex_choice.startswith("😵"):
            md_html(f"""
            <div style='display:flex; gap:14px;'>
              <div style='{_box_l} flex:1;'>
                <b>"네" 상자</b><br><span style='font-size:30px;'>🦆</span><br>
                <span style='font-size:13px;'>딱 1마리만 찾음</span>
              </div>
              <div style='{_box_l} flex:3;'>
                <b>"아니오" 상자</b><br><span style='font-size:30px;'>🐶🐱🐦🐟🐸</span><br>
                <span style='font-size:13px;'>아직 5마리나 뒤죽박죽! 😵</span>
              </div>
            </div>
            """)
            st.error("한 마리씩 물어보면 최악의 경우 **5번**이나 질문해야 해요. 후보가 거의 줄지 않았죠?")
        else:
            md_html(f"""
            <div style='display:flex; gap:14px;'>
              <div style='{_box_r} flex:1;'>
                <b>"네" 상자</b><br><span style='font-size:30px;'>🐦🦆</span><br>
                <span style='font-size:13px;'>날개친구끼리 모임 ✨</span>
              </div>
              <div style='{_box_r} flex:1;'>
                <b>"아니오" 상자</b><br><span style='font-size:30px;'>🐶🐱🐟🐸</span><br>
                <span style='font-size:13px;'>절반으로 확 줄었어요!</span>
              </div>
            </div>
            """)
            st.success("질문 한 번에 후보가 **반으로 싹둑!** 이렇게 후보를 확 줄여주는 게 **좋은 질문**이에요. "
                       "AI는 매번 이런 좋은 질문부터 골라서 물어봐요.")

        st.write("")
        st.info("💡 **다음 단계!** 미션 1, 2 탭에서 직접 스무고개를 해보고, '섞임 점수' 탭에서 왜 그 질문이 좋은지 알아본 뒤, 마지막 게임 탭에서 실력을 확인해봐요.")

    with tab1:
        st.subheader("🌿 나뭇잎 스무고개")
        leaf_data = pd.DataFrame({
            '식물 이름': ['강아지풀', '벚나무', '단풍나무', '은행나무', '소나무'],
            '잎의 모양': ['길쭉하다', '달걀 모양', '손바닥 모양', '부채 모양', '바늘처럼 뾰족하다'],
            '가장자리': ['매끄럽다', '톱니 모양', '갈라짐', '매끄럽다', '매끄럽다']
        })
        st.table(leaf_data)

        st.write("#### 🕵️‍♂️ 탐정의 스무고개 시작!")
        ans1 = st.radio("**첫 번째 질문:** 잎의 모양이 바늘처럼 뾰족한가요?", ["선택안함", "네", "아니요"], key='leaf1')

        if ans1 == "네":
            st.success("🎯 삐빅! 정답은 **소나무** 입니다! (질문 한 번에 찾았어요!)")
        elif ans1 == "아니요":
            st.info("남은 친구들: 강아지풀, 벚나무, 단풍나무, 은행나무")
            ans2 = st.radio("**두 번째 질문:** 잎 가장자리가 톱니 모양인가요?", ["선택안함", "네", "아니요"], key='leaf2')
            if ans2 == "네":
                st.success("🎯 삐빅! 정답은 **벚나무** 입니다!")
            elif ans2 == "아니요":
                st.warning("스무고개 AI는 정답이 딱 1개 남을 때까지 계속 질문을 던져요!")

        st.caption("🔗 궁금하지 않나요? '왜 하필 이 질문을 먼저 물어봤을까요?' → **'💡 비밀: 섞임 점수 놀이'** 탭에서 그 답을 확인해봐요!")

    with tab2:
        st.subheader("🦁 동물 가족 스무고개")
        animal_data = pd.DataFrame({
            '동물': ['강아지(포유류)', '참새(조류)', '개구리(양서류)', '물고기(어류)'],
            '태어날 때': ['새끼를 낳아요', '알을 낳아요', '알을 낳아요', '알을 낳아요'],
            '사는 곳': ['땅', '하늘과 땅', '물과 땅', '물속']
        })
        st.table(animal_data)

        st.write("#### 🕵️‍♂️ 탐정의 스무고개 시작!")
        ans_a1 = st.radio("**첫 번째 질문:** 엄마가 알을 낳나요, 새끼를 낳나요?", ["선택안함", "새끼를 낳아요", "알을 낳아요"], key='ani1')

        if ans_a1 == "새끼를 낳아요":
            st.success("🎯 삐빅! 정답은 **강아지** 입니다!")
        elif ans_a1 == "알을 낳아요":
            st.info("남은 친구들: 참새, 개구리, 물고기")
            ans_a2 = st.radio("**두 번째 질문:** 평생 물속에서만 살면서 헤엄치나요?", ["선택안함", "네 (물속에서만)", "아니요 (땅에도 와요)"], key='ani2')
            if ans_a2 == "네 (물속에서만)":
                st.success("🎯 삐빅! 정답은 **물고기** 입니다!")
            elif ans_a2 == "아니요 (땅에도 와요)":
                st.warning("동물들의 특징을 보고 '네/아니오'로 길을 나누는 것! 이게 바로 AI의 방법이에요.")

        st.caption("🔗 이 질문들이 왜 '좋은 질문'인지는 **'💡 비밀: 섞임 점수 놀이'** 탭에서 알아볼 수 있어요!")

    with tab3:
        st.subheader("💡 좋은 질문의 비밀: '섞임 점수'")

        # ── 1) 섞임이 뭔지 눈으로 먼저 ──────────────────────────────
        st.markdown("#### 1️⃣ 먼저, '섞였다'는 게 뭘까요?")
        md_html("""
        <div style='font-size:18px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:16px 20px;'>
        상자 안에 <b>여러 종류가 뒤죽박죽</b> 있으면 → <b>많이 섞임</b> 😵<br>
        상자 안에 <b>한 종류만</b> 있으면 → <b>안 섞임 (정리 끝!)</b> ✨<br><br>
        AI는 이걸 <b>'섞임 점수'</b>로 재요. 점수가 <b>0점이면 완벽하게 정리</b>된 거예요!
        </div>
        """)

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            md_html("""
            <div style='text-align:center; background:#FFEBEE; border-radius:12px; padding:12px;'>
                <div style='font-size:26px;'>🐶🐦🐟</div>
                <div style='font-size:14px; margin-top:6px;'>다 달라요</div>
                <div style='font-family:Jua,sans-serif; color:#C62828;'>많이 섞임 😵</div>
            </div>
            """)
        with ex2:
            md_html("""
            <div style='text-align:center; background:#FFF8E1; border-radius:12px; padding:12px;'>
                <div style='font-size:26px;'>🐶🐶🐦</div>
                <div style='font-size:14px; margin-top:6px;'>거의 같아요</div>
                <div style='font-family:Jua,sans-serif; color:#F9A825;'>조금 섞임 🙂</div>
            </div>
            """)
        with ex3:
            md_html("""
            <div style='text-align:center; background:#E8F5E9; border-radius:12px; padding:12px;'>
                <div style='font-size:26px;'>🐶🐶🐶</div>
                <div style='font-size:14px; margin-top:6px;'>모두 같아요</div>
                <div style='font-family:Jua,sans-serif; color:#2E7D32;'>안 섞임 ✨ (0점)</div>
            </div>
            """)

        st.divider()

        # ── 2) 질문을 골라보며 섞임이 줄어드는 걸 직접 확인 ─────────────
        st.markdown("#### 2️⃣ 질문을 하면 섞임 점수가 어떻게 변할까요?")
        st.write("동물 6마리가 **한 상자에 뒤섞여** 있어요. 질문을 하나 고르면 **두 상자로 나뉘어요.** "
                 "어떤 질문이 가장 깔끔하게 정리해주는지 직접 확인해봐요!")

        gini_animals = pd.DataFrame([
            {"이름": "🐶 강아지", "무리": "털친구", "날개": 0, "헤엄": 0, "털": 1},
            {"이름": "🐱 고양이", "무리": "털친구", "날개": 0, "헤엄": 0, "털": 1},
            {"이름": "🐦 참새", "무리": "날개친구", "날개": 1, "헤엄": 0, "털": 0},
            {"이름": "🦆 오리", "무리": "날개친구", "날개": 1, "헤엄": 1, "털": 0},
            {"이름": "🐟 물고기", "무리": "물친구", "날개": 0, "헤엄": 1, "털": 0},
            {"이름": "🐸 개구리", "무리": "물친구", "날개": 0, "헤엄": 1, "털": 0},
        ])

        def _gini_of(series):
            """한 상자 안의 섞임 점수(지니 불순도). 0이면 한 종류만 있는 것."""
            n = len(series)
            if n == 0:
                return 0.0
            counts = series.value_counts()
            return float(1 - sum((c / n) ** 2 for c in counts))

        def _mix_bar(g, width=5):
            """섞임 점수를 색칠한 칸으로 표시 (최대 0.75 기준)."""
            lv = min(width, round(g / 0.75 * width))
            return "🟥" * lv + "⬜" * (width - lv)

        before_g = _gini_of(gini_animals["무리"])
        md_html(f"""
        <div style='background:#F5F7FF; border-radius:12px; padding:14px 18px; text-align:center;'>
            <div style='font-size:14px; color:#3A4A6B;'>지금 상자 (질문하기 전)</div>
            <div style='font-size:30px; letter-spacing:4px; margin:8px 0;'>🐶 🐱 🐦 🦆 🐟 🐸</div>
            <div style='font-size:20px;'>{_mix_bar(before_g)}</div>
            <div style='font-family:Jua,sans-serif; font-size:17px; color:#C62828; margin-top:4px;'>
                섞임 점수 {before_g:.2f} — 아주 많이 섞였어요!
            </div>
        </div>
        """)

        st.write("")
        q_choice = st.radio(
            "🃏 어떤 질문을 해볼까요?",
            ["🪽 날개가 있나요?", "🏊 헤엄을 칠 수 있나요?", "🧶 털이 있나요?"],
            key="gini_q_choice", horizontal=True,
        )
        col_map = {"🪽 날개가 있나요?": "날개", "🏊 헤엄을 칠 수 있나요?": "헤엄", "🧶 털이 있나요?": "털"}
        col_sel = col_map[q_choice]

        yes_df = gini_animals[gini_animals[col_sel] == 1]
        no_df = gini_animals[gini_animals[col_sel] == 0]
        g_yes, g_no = _gini_of(yes_df["무리"]), _gini_of(no_df["무리"])
        # 나눈 뒤의 평균 섞임 점수(각 상자 크기만큼 가중치)
        after_g = (len(yes_df) * g_yes + len(no_df) * g_no) / len(gini_animals)
        drop = before_g - after_g

        st.write("**질문에 답했더니 두 상자로 나뉘었어요!**")
        b1, b2 = st.columns(2)
        with b1:
            emo = " ".join(n.split()[0] for n in yes_df["이름"]) or "없음"
            md_html(f"""
            <div style='background:#E8F5E9; border:2px solid #66BB6A; border-radius:12px; padding:12px; text-align:center;'>
                <div style='font-family:Jua,sans-serif; color:#2E7D32;'>✅ "네" 상자</div>
                <div style='font-size:28px; letter-spacing:3px; margin:8px 0;'>{emo}</div>
                <div style='font-size:18px;'>{_mix_bar(g_yes)}</div>
                <div style='font-size:14px; color:#3A4A6B;'>섞임 {g_yes:.2f}</div>
            </div>
            """)
        with b2:
            emo = " ".join(n.split()[0] for n in no_df["이름"]) or "없음"
            md_html(f"""
            <div style='background:#FFEBEE; border:2px solid #EF5350; border-radius:12px; padding:12px; text-align:center;'>
                <div style='font-family:Jua,sans-serif; color:#C62828;'>❌ "아니오" 상자</div>
                <div style='font-size:28px; letter-spacing:3px; margin:8px 0;'>{emo}</div>
                <div style='font-size:18px;'>{_mix_bar(g_no)}</div>
                <div style='font-size:14px; color:#3A4A6B;'>섞임 {g_no:.2f}</div>
            </div>
            """)

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("질문 전 섞임", f"{before_g:.2f}")
        m2.metric("질문 후 섞임", f"{after_g:.2f}")
        m3.metric("줄어든 양", f"{drop:.2f}", delta=f"-{drop:.2f}")

        if drop >= 0.28:
            st.success(f"🎯 아주 좋은 질문이에요! 섞임 점수가 **{drop:.2f}만큼 확 줄었어요.** "
                       "이런 질문을 먼저 하면 정답을 빨리 찾을 수 있어요!")
        elif drop >= 0.15:
            st.info(f"🙂 괜찮은 질문이에요. 섞임이 {drop:.2f}만큼 줄었어요. 다른 질문과도 비교해볼까요?")
        else:
            st.warning(f"😅 이 질문은 별로 도움이 안 됐어요. 섞임이 {drop:.2f}밖에 안 줄었네요. "
                       "다른 질문을 골라보세요!")

        st.caption("💡 세 가지 질문을 모두 눌러보고, **어떤 질문이 섞임 점수를 가장 많이 줄이는지** 찾아보세요!")

        with st.expander("🤔 정리: 그래서 좋은 질문이란?"):
            think_dt = st.radio(
                "AI는 어떤 질문을 '좋은 질문'이라고 생각할까요?",
                ["선택 안 함",
                 "① 섞임 점수를 가장 많이 줄여주는 질문",
                 "② 섞임 점수를 더 높이는 질문"],
                key="dt_think1"
            )
            if think_dt.startswith("①"):
                st.success("맞아요! 섞임을 확 줄여서 **한 종류씩 깔끔하게 정리**되게 하는 질문이 좋은 질문이에요. "
                           "AI(결정트리)는 이런 질문을 스스로 찾아낸답니다!")
            elif think_dt.startswith("②"):
                st.warning("반대예요! 섞임 점수는 **낮을수록** 잘 정리된 거예요. 위에서 질문을 다시 눌러보며 확인해봐요.")

        st.caption("👉 이제 마지막 게임 탭에서 직접 좋은 질문을 골라 범인을 찾아봐요!")

    # [게임: 동물 탐정 챌린지] ----------------------------------------------
    with tab4:
        st.header("🕵️‍♂️ 사건 현장의 범인 동물을 찾아라!")
        st.markdown("""
        **🎯 게임 방법**
        1. 컴퓨터가 동물 6마리 중 하나를 몰래 **범인**으로 정했어요. (누군지는 비밀!)
        2. 아래 **질문 카드**를 누르면, 범인에 대한 예/아니오 답을 알려줘요.
        3. 답을 들을 때마다 **용의자가 줄어들어요.** 용의자가 1명만 남으면 그게 범인!
        4. **되도록 적은 질문(3번 이하)으로** 범인을 찾으면 명탐정 배지를 받아요. 🏅
        """)
        st.caption("💡 팁: 용의자를 '한쪽으로 몰아주는' 질문보다, **두 편으로 골고루 갈라주는 질문**일수록 후보가 확 줄어요!")

        mystery_animals = pd.DataFrame([
            {"이름": "🐶 강아지", "날개": 0, "털": 1, "발가락4개": 1, "헤엄": 0},
            {"이름": "🐱 고양이", "날개": 0, "털": 1, "발가락4개": 0, "헤엄": 0},
            {"이름": "🐦 참새",  "날개": 1, "털": 0, "발가락4개": 0, "헤엄": 0},
            {"이름": "🦆 오리",  "날개": 1, "털": 0, "발가락4개": 0, "헤엄": 1},
            {"이름": "🐟 물고기", "날개": 0, "털": 0, "발가락4개": 0, "헤엄": 1},
            {"이름": "🐸 개구리", "날개": 0, "털": 0, "발가락4개": 1, "헤엄": 1},
        ])
        dt_questions = {
            "날개가 있나요? 🪽": "날개",
            "털이 있나요? 🧶": "털",
            "발가락이 4개인가요? 🐾": "발가락4개",
            "헤엄을 칠 수 있나요? 🏊": "헤엄",
        }

        if st.session_state["dt_mystery"] is None:
            mystery = mystery_animals.sample(1, random_state=random.randint(0, 9999)).iloc[0]
            st.session_state["dt_mystery"] = mystery
            st.session_state["dt_candidates"] = mystery_animals.copy()
            st.session_state["dt_asked"] = []
            st.session_state["dt_history"] = []

        mystery = st.session_state["dt_mystery"]
        candidates = st.session_state["dt_candidates"]
        num_q = len(st.session_state["dt_asked"])

        st.divider()

        # 남은 용의자를 큰 이모지 줄로 표시
        suspect_emojis = " ".join(name.split()[0] for name in candidates["이름"].tolist())
        md_html(f"""
        <div style='background:#E8F5E9; border-radius:14px; padding:14px 18px; text-align:center; margin-bottom:6px;'>
            <div style='font-size:14px; color:#2E7D32;'>🔎 남은 용의자 {len(candidates)}명 · 지금까지 질문 {num_q}번</div>
            <div style='font-size:34px; letter-spacing:6px; margin-top:6px;'>{suspect_emojis}</div>
        </div>
        """)

        if st.button("🔄 다른 사건으로 다시 하기", key="dt_new_case"):
            mystery = mystery_animals.sample(1).iloc[0]
            st.session_state["dt_mystery"] = mystery
            st.session_state["dt_candidates"] = mystery_animals.copy()
            st.session_state["dt_asked"] = []
            st.session_state["dt_history"] = []
            st.rerun()

        if len(candidates) > 1:
            remaining_qs = {q: col for q, col in dt_questions.items()
                            if q not in st.session_state["dt_asked"]}

            # 각 질문이 용의자를 얼마나 골고루 나누는지 계산 → 가장 잘 나누는 질문 추천 표시
            def split_balance(col):
                yes_n = int((candidates[col] == 1).sum())
                no_n = int((candidates[col] == 0).sum())
                # 한쪽이 0이면 못 나눔(쓸모없는 질문), 반반에 가까울수록 좋음
                if yes_n == 0 or no_n == 0:
                    return -1
                return -abs(yes_n - no_n)  # 차이가 작을수록(=골고루) 점수 높음

            best_q = None
            if remaining_qs:
                best_q = max(remaining_qs, key=lambda q: split_balance(remaining_qs[q]))
                if split_balance(remaining_qs[best_q]) == -1:
                    best_q = None  # 모두 못 나누는 경우 추천 없음

            st.write("#### 🃏 질문 카드를 골라보세요")
            cols = st.columns(len(remaining_qs)) if remaining_qs else []
            for i, (q, col) in enumerate(remaining_qs.items()):
                yes_grp = [n.split()[0] for n in candidates[candidates[col] == 1]["이름"]]
                no_grp = [n.split()[0] for n in candidates[candidates[col] == 0]["이름"]]
                with cols[i]:
                    recommend = (q == best_q)
                    border = "#7E57C2" if recommend else "#C5CAE9"
                    tag = "<div style='font-size:11px; color:#7E57C2; font-weight:bold;'>👍 잘 갈라져요!</div>" if recommend else "<div style='font-size:11px; color:#aaa;'>&nbsp;</div>"
                    md_html(f"""
                    <div style='border:2px solid {border}; border-radius:12px; padding:10px; text-align:center; min-height:120px;'>
                        {tag}
                        <div style='font-size:13px; margin:6px 0; color:#1B2A4A;'>{q}</div>
                        <div style='font-size:12px; color:#2E7D32;'>✅ 네: {' '.join(yes_grp) if yes_grp else '없음'}</div>
                        <div style='font-size:12px; color:#C62828;'>❌ 아니오: {' '.join(no_grp) if no_grp else '없음'}</div>
                    </div>
                    """)
                    if st.button("이 질문 하기", key=f"dtq_{col}", **_STRETCH):
                        answer = int(mystery[col])
                        st.session_state["dt_candidates"] = candidates[candidates[col] == answer]
                        st.session_state["dt_asked"].append(q)
                        st.session_state["dt_history"].append((q, "네" if answer == 1 else "아니오"))
                        st.rerun()

            if st.session_state["dt_history"]:
                st.write("#### 📜 지금까지 질문 기록")
                for q, a in st.session_state["dt_history"]:
                    icon = "✅" if a == "네" else "❌"
                    st.write(f"- {q} → {icon} **{a}**")
        else:
            found = candidates.iloc[0]
            # 용의자가 1명으로 좁혀졌으면 그게 범인 (게임 규칙상 항상 정답과 일치)
            n_asked = len(st.session_state["dt_asked"])
            st.success(f"🎯 범인을 찾았어요! 범인은 바로 **{found['이름']}** 입니다!")

            if n_asked <= 3:
                st.balloons()
                play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
                st.markdown(f"### 🎉 명탐정! 단 **{n_asked}번**의 질문으로 범인을 찾았어요!")
                award_badge(2)

                md_html("""
                <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:16px 20px; margin-top:12px;">
                    <b>🏷️ 오늘 배운 것</b><br>
                    이렇게 <b>질문(조건)으로 후보를 계속 확 좁혀가며 답을 찾는 방법</b>을 인공지능에서는
                    <b>'결정트리(decision tree)'</b>라고 불러요. 그리고 용의자를 얼마나 잘 갈라놓는지 재는 점수를
                    <b>'섞임 점수(불순도)'</b>라고 한답니다. 잘 갈라주는 질문을 먼저 할수록 더 적은 질문으로 답을 찾아요!
                </div>
                """)

                with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                    quiz_dt = st.radio(
                        "질문(조건)으로 후보를 확 좁혀가며 답을 찾는 방법을 뭐라고 배웠나요?",
                        ["선택 안 함", "① 결정트리", "② 선형회귀", "③ K-평균 군집화"],
                        key="quiz_dt"
                    )
                    if quiz_dt == "① 결정트리":
                        st.success("🌟 정답이에요! '완벽 이해' 보너스 배지 획득!")
                        st.session_state["bonus_badges"].add("🌟 결정트리 완벽 이해")
                    elif quiz_dt != "선택 안 함":
                        st.warning("다시 한 번 생각해볼까요? '섞임 점수 놀이' 탭 내용을 떠올려봐요!")

                st.write("")
                next_section_button("🌳 3. 스무고개 탐정", "dt")
            else:
                st.info(f"범인은 찾았지만 질문을 {n_asked}번 했어요! 💡 '잘 갈라져요!' 표시가 있는 질문을 먼저 고르면 "
                        "3번 이하로 찾을 수 있어요. '다른 사건으로 다시 하기'로 도전해볼까요?")

# --- [메뉴 5: 가장 친한 친구 (KNN)] ---
elif menu == "🤝 4. 가장 친한 친구":
    hero_card("🤝", "가장 친한 친구: 내 주변에 누가 제일 많을까?",
               "새로운 데이터가 나타났을 때, 가까운 이웃들에게 물어보고 정체를 알아내요.",
               SECTION_COLORS["knn"], term="K-최근접이웃 (K-Nearest Neighbors, KNN)")

    with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
        md_html("""
        - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
          → 새로운 데이터가 **가까운 이웃 데이터들의 다수결로 분류**되는 AI(KNN)의 원리를 게임으로 체험해요.
        - **[실과] [6실05-04]** 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
          → 위치(가로·세로 값)라는 숫자 데이터가 AI 분류 판단에 어떻게 활용되는지 탐색해요.
        """)

    tab_intro, tab_practice, tab_game, tab_class = st.tabs([
        "🌉 1단계: 먼저 알아보기", "🕹️ 2단계: 직접 판정해보기",
        "🍇 3단계: 게임: 외계 알 구출작전", "🧑‍🤝‍🧑 4단계: 우리 반 데이터로 분류하기"])

    # [1단계: 먼저 알아보기] -----------------------------------------------
    with tab_intro:
        st.markdown("### 🌉 어? 나 이거 이미 알아!")
        md_html("""
        <div style='font-size:19px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:18px 22px;'>
        전학 온 친구가 쉬는 시간마다 <b>태권도복 입은 친구들 옆에</b> 있으면, "저 친구도 태권도 하는구나!"라고
        짐작하게 되죠?<br><br>
        AI도 새로운 데이터를 만나면, <b>가장 가까이 있는 이웃들이 누구인지</b> 보고 정체를 짐작해요.
        </div>
        """)

        st.write("")
        st.markdown("### 🎯 오늘의 미션")
        st.success("**새로운 과일이 사과인지 포도인지, 가까운 이웃 과일들에게 물어봐서 알아맞혀보세요!**")

        data = pd.DataFrame({
            '달콤한 정도': [8, 9, 7, 2, 3, 1, 8, 2, 4],
            '과일 크기': [7, 8, 9, 2, 1, 3, 6, 2, 4],
            '과일 종류': ['사과', '포도', '사과', '포도', '포도', '포도', '사과', '포도', '포도']
        })

        st.write("")
        st.markdown("### 👀 조작하기 전에, 먼저 눈으로 봐요")

        col_knn_txt, col_knn_fig = st.columns([1, 1.4])

        with col_knn_txt:
            cmp_choice = st.radio("새 과일을 어디에 놓아볼까요?", ["🍎 사과 무리 한가운데", "🍇 포도 무리 한가운데"], key="knn_cmp_choice")
            st.info(f"새 과일 주변에 **{cmp_choice.split()[1]}**들이 잔뜩 있으니, 이 과일도 그럴 확률이 높겠죠? "
                    f"이게 바로 KNN의 기본 생각이에요.")

        cmp_sweet, cmp_size = (8, 8) if cmp_choice == "🍎 사과 무리 한가운데" else (2, 2)

        with col_knn_fig:
            fig_cmp = go.Figure()
            for kind, color in zip(['사과', '포도'], ['red', 'green']):
                subset = data[data['과일 종류'] == kind]
                fig_cmp.add_trace(go.Scatter(x=subset['달콤한 정도'], y=subset['과일 크기'], mode='markers', name=kind, marker=dict(size=18, color=color)))
            fig_cmp.add_trace(go.Scatter(x=[cmp_sweet], y=[cmp_size], mode='markers', name='새 과일', marker=dict(size=28, color='blue', symbol='star')))
            fig_cmp.update_layout(xaxis_title="달콤한 정도", yaxis_title="과일 크기", height=330,
                                  margin=dict(l=20, r=20, t=20, b=20),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_cmp, **_STRETCH, config={'displayModeBar': False})

        st.write("")
        st.info("💡 **다음 단계!** 2단계 탭에서 직접 새 과일을 놓아보고 K값도 바꿔봐요.")

    # [2단계: 직접 판정해보기] -----------------------------------------------
    with tab_practice:
        st.markdown("### 🕹️ 이제 네 차례!")

        with st.expander("🤔 시작하기 전에 잠깐 생각해보기"):
            think_knn = st.radio(
                "K값을 1에서 5로 늘리면 무슨 일이 생길까요?",
                ["선택 안 함", "① 가까운 친구 5명의 다수결로 정한다", "② 그래도 가장 가까운 친구 1명만 본다"],
                key="knn_think1"
            )
            if think_knn.startswith("①"):
                st.success("맞아요! K가 커질수록 더 많은 이웃의 의견을 다수결로 반영해요.")
            elif think_knn.startswith("②"):
                st.warning("아래 K값을 1과 5로 바꿔가며 결과가 달라지는지 확인해봐요!")

        col1, col2 = st.columns([1, 2])
        with col1:
            new_sweet = st.slider("새 과일은 얼마나 달콤한가요?", 0, 10, 5, key="knn_learn_sweet")
            new_size = st.slider("새 과일은 얼마나 큰가요?", 0, 10, 5, key="knn_learn_size")
            k_val = st.number_input("주변 친구 몇 명에게 물어볼까요? (K)", 1, 5, 3, key="knn_learn_k")

            knn = _fit_knn_fruit(k_val)
            dist, ind = knn.kneighbors([[new_sweet, new_size]])
            st.subheader(f"🎯 인공지능의 생각: 이 과일은 **{knn.predict([[new_sweet, new_size]])[0]}** 입니다!")

        with col2:
            fig = go.Figure()
            for kind, color in zip(['사과', '포도'], ['red', 'green']):
                subset = data[data['과일 종류'] == kind]
                fig.add_trace(go.Scatter(x=subset['달콤한 정도'], y=subset['과일 크기'], mode='markers', name=kind, marker=dict(size=20, color=color)))
            fig.add_trace(go.Scatter(x=[new_sweet], y=[new_size], mode='markers', name='새 과일 (별모양)', marker=dict(size=30, color='blue', symbol='star')))

            fig.add_shape(type="circle", xref="x", yref="y", x0=new_sweet-dist[0][-1], y0=new_size-dist[0][-1], x1=new_sweet+dist[0][-1], y1=new_size+dist[0][-1], line_color="orange", opacity=0.2, fillcolor="orange")
            fig.update_layout(xaxis_title="달콤한 정도", yaxis_title="과일 크기", height=300, margin=dict(b=60, l=60))
            add_sweetness_axis_hints(fig)
            st.plotly_chart(fig, **_STRETCH)

    # [게임: 외계 알 구출작전] -----------------------------------------------
    with tab_game:
        st.header("🍇 포도밭에 떨어진 외계 알의 정체는?")
        st.write("외계 알을 원하는 곳에 놓고, 주변 친구들에게 물어봐서 사과인지 포도인지 알아내 봐요!")

        if st.session_state["knn_data"] is None:
            rng = np.random.default_rng(7)
            apple_x = rng.normal(3, 1.0, 6)
            apple_y = rng.normal(3, 1.0, 6)
            grape_x = rng.normal(8, 1.0, 6)
            grape_y = rng.normal(8, 1.0, 6)
            st.session_state["knn_data"] = (apple_x, apple_y, grape_x, grape_y)

        apple_x, apple_y, grape_x, grape_y = st.session_state["knn_data"]

        colL, colR = st.columns([1, 2])
        with colL:
            egg_x = st.slider("🍬 외계 알의 달콤한 정도", 0.0, 11.0, 5.5, step=0.2, key="knn_egg_x")
            egg_y = st.slider("📏 외계 알의 크기", 0.0, 11.0, 5.5, step=0.2, key="knn_egg_y")
            k = st.radio("🔭 몇 명한테 물어볼까요? (K)", [1, 3, 5], horizontal=True, key="knn_game_k")

        all_x = np.concatenate([apple_x, grape_x])
        all_y = np.concatenate([apple_y, grape_y])
        all_label = ["🍎"] * len(apple_x) + ["🍇"] * len(grape_x)
        dists = np.sqrt((all_x - egg_x) ** 2 + (all_y - egg_y) ** 2)
        order = np.argsort(dists)[:k]
        neighbor_labels = [all_label[i] for i in order]
        apple_votes = neighbor_labels.count("🍎")
        grape_votes = neighbor_labels.count("🍇")
        result = "🍎 사과" if apple_votes >= grape_votes else "🍇 포도"

        with colR:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=apple_x, y=apple_y, mode="markers", name="사과 친구들",
                                      marker=dict(size=16, color="#E53935")))
            fig2.add_trace(go.Scatter(x=grape_x, y=grape_y, mode="markers", name="포도 친구들",
                                      marker=dict(size=16, color="#8E24AA")))
            for i in order:
                fig2.add_trace(go.Scatter(x=[egg_x, all_x[i]], y=[egg_y, all_y[i]], mode="lines",
                                          line=dict(color="orange", width=2, dash="dot"), showlegend=False))
            fig2.add_trace(go.Scatter(x=[egg_x], y=[egg_y], mode="markers+text", name="👽 외계 알",
                                      marker=dict(size=26, color="pink", symbol="star"),
                                      text=["👽"], textposition="top center"))
            fig2.update_layout(height=315, xaxis_range=[0, 11], yaxis_range=[0, 11],
                               xaxis_title="달콤한 정도", yaxis_title="크기", margin=dict(b=60, l=60))
            add_sweetness_axis_hints(fig2)
            st.plotly_chart(fig2, **_STRETCH)

        st.write(f"### 📊 투표 결과: 🍎 {apple_votes}표 vs 🍇 {grape_votes}표")
        st.success(f"**{k}명의 가까운 친구들에게 물어본 결과, 이 알은 {result}로 판정되었어요!**")

        if st.button("👽 알 깨우기!", key="knn_hatch"):
            st.session_state["knn_hatched"] = True

        if st.session_state.get("knn_hatched"):
            award_badge(3)
            md_html("""
            <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:16px 20px; margin-top:12px;">
                <b>🏷️ 오늘 배운 것</b><br>
                이렇게 <b>가장 가까운 이웃들에게 물어보고 다수결로 정체를 정하는 방법</b>을 인공지능에서는
                <b>'K-최근접이웃(K-Nearest Neighbors, KNN)'</b>이라고 불러요.
            </div>
            """)

            with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                quiz_knn = st.radio(
                    "가까운 이웃들에게 물어보고 다수결로 정체를 정하는 방법을 뭐라고 배웠나요?",
                    ["선택 안 함", "① KNN (K-최근접이웃)", "② 경사하강법", "③ 인공신경망"],
                    key="quiz_knn"
                )
                if quiz_knn == "① KNN (K-최근접이웃)":
                    st.success("🌟 정답이에요! '완벽 이해' 보너스 배지 획득!")
                    st.session_state["bonus_badges"].add("🌟 KNN 완벽 이해")
                elif quiz_knn != "선택 안 함":
                    st.warning("다시 한 번 생각해볼까요? '원리 알아보기' 탭 내용을 떠올려봐요!")

            st.write("")
            next_section_button("🤝 4. 가장 친한 친구", "knn")

    # [4단계: 우리 반 데이터로 분류하기] -----------------------------------
    with tab_class:
        st.subheader("🧑‍🤝‍🧑 우리 반 친구들 데이터로 직접 분류해봐요!")
        md_html("""
        <div style='font-size:17px; line-height:1.8; background:#FFF8E1; border-radius:12px; padding:14px 18px;'>
        교실에서 <b>직접 잰 데이터</b>를 넣어봐요! 예를 들어 친구들의 <b>키</b>와 <b>발 크기</b>를 적고,
        운동을 좋아하는지(<b>구분</b>)를 적어보세요.<br>
        그다음 <b>새 친구</b>의 키·발 크기를 넣으면, AI가 <b>가장 가까운 이웃들</b>을 보고
        이 친구가 어느 쪽일지 맞혀줘요!
        </div>
        """)

        st.info("🎯 **미션**: 아래 ①②③을 차례로 하고, 마지막 **확인 문제**까지 맞히면 "
                "로봇 부품(하트)을 받고 다음 섬으로 갈 수 있어요! "
                "(3단계 게임에서 이미 부품을 받았다면, 복습으로 한 번 더 도전해봐요.)")

        # ── 1단계: 데이터 입력 ───────────────────────────────
        st.markdown("#### 1️⃣ 우리 반 데이터를 표에 넣어요")
        st.caption("💡 아래 표를 직접 고칠 수 있어요. 행을 추가하거나 숫자를 바꿔보세요. (구분은 두 종류로 적어주세요. 예: 운동파 / 독서파)")

        default_class_df = pd.DataFrame({
            "이름": ["가은", "나윤", "다온", "라율", "마루", "바다"],
            "키(cm)": [150, 142, 155, 138, 148, 160],
            "발크기(mm)": [230, 215, 240, 210, 225, 245],
            "구분": ["운동파", "독서파", "운동파", "독서파", "독서파", "운동파"],
        })
        class_df = st.data_editor(default_class_df, num_rows="dynamic", key="knn_class_editor", **_STRETCH)

        # 유효성 검사
        valid = True
        try:
            class_df = class_df.dropna(subset=["키(cm)", "발크기(mm)", "구분"])
            class_df = class_df[class_df["구분"].astype(str).str.strip() != ""]
            groups = class_df["구분"].astype(str).unique().tolist()
        except Exception:
            valid = False
            groups = []

        if not valid or len(class_df) < 3:
            st.warning("⚠️ 데이터를 **3줄 이상** 채워주세요! (키, 발크기, 구분을 모두 적어야 해요.)")
        elif len(groups) < 2:
            st.warning("⚠️ '구분'이 최소 두 종류는 있어야 AI가 나눌 수 있어요! (예: 운동파와 독서파)")
        else:
            st.success(f"✅ 데이터 {len(class_df)}개 준비 완료!")
            st.divider()
            st.markdown("#### 2️⃣ 새 친구의 숫자를 넣고, 이웃 수(K)를 정해요")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                new_h = st.number_input("새 친구 키(cm)", 100, 200, 150, key="knn_class_h")
            with cc2:
                new_f = st.number_input("새 친구 발크기(mm)", 150, 300, 228, key="knn_class_f")
            with cc3:
                max_k = min(7, len(class_df))
                k_class = st.slider("이웃 수 (K)", 1, max_k, min(3, max_k), key="knn_class_k")

            st.markdown("#### 3️⃣ AI의 판정 결과를 확인해요")
            X = class_df[["키(cm)", "발크기(mm)"]].astype(float).values
            y = class_df["구분"].astype(str).values
            knn_c = KNeighborsClassifier(n_neighbors=k_class)
            knn_c.fit(X, y)
            pred = knn_c.predict([[new_h, new_f]])[0]
            dist, ind = knn_c.kneighbors([[new_h, new_f]])

            # 시각화
            fig = go.Figure()
            for g in groups:
                sub = class_df[class_df["구분"].astype(str) == g]
                fig.add_trace(go.Scatter(
                    x=sub["키(cm)"], y=sub["발크기(mm)"], mode="markers+text",
                    text=sub["이름"], textposition="top center", name=str(g),
                    marker=dict(size=14)))
            fig.add_trace(go.Scatter(
                x=[new_h], y=[new_f], mode="markers+text", text=["새 친구"],
                textposition="bottom center", name="새 친구",
                marker=dict(size=20, color="red", symbol="star")))
            fig.update_layout(title="우리 반 친구들과 새 친구의 위치",
                              xaxis_title="키(cm)", yaxis_title="발크기(mm)",
                              height=294, margin=dict(l=20, r=20, t=48, b=20))
            st.plotly_chart(fig, **_STRETCH)

            neighbor_names = class_df.iloc[ind[0]]["이름"].tolist()
            st.success(f"🤖 AI 판정: 가장 가까운 이웃 {k_class}명({', '.join(map(str, neighbor_names))})을 보니, "
                       f"이 새 친구는 **'{pred}'** 같아요!")
            st.caption("K값(이웃 수)을 바꾸면 판정이 달라질 수 있어요. 왜 그런지 친구들과 이야기해봐요!")

            # ── 관찰 확인 → 부품 획득 ─────────────────────────
            st.divider()
            st.markdown("#### 🔍 마지막! 관찰한 것을 확인해요 (부품 획득 문제)")
            knn_obs = st.radio(
                "AI는 무엇을 보고 새 친구가 어느 쪽인지 정했을까요?",
                ["선택 안 함",
                 "① 새 친구와 가장 가까이 있는 이웃들을 보고 다수결로 정했다",
                 "② 이름이 예쁜 순서대로 정했다"],
                key="knn_class_obs"
            )
            if knn_obs.startswith("①"):
                st.success("🎉 정확해요! **가장 가까운 이웃들에게 물어보고 다수결로 정하는 것**, "
                           "이것이 바로 **K-최근접이웃(KNN)**이에요!")
                award_badge(3)
                st.write("")
                next_section_button("🤝 4. 가장 친한 친구", "knn_class")
            elif knn_obs.startswith("②"):
                st.warning("아니에요! AI는 이름이 아니라 **위치(키·발크기)가 가까운 이웃**을 보고 판단했어요. "
                           "그래프에서 새 친구 별표(⭐) 주변에 누가 있는지 다시 살펴볼까요?")

# --- [메뉴 6: 비슷한 친구끼리 (K-Means)] ---
elif menu == "🎨 5. 비슷한 친구끼리":
    hero_card("🎨", "비슷한 친구끼리: 이름표가 없어도 모일 수 있어요!",
               "정답 이름표가 없어도, 비슷한 것끼리 스스로 모둠을 나누는 AI를 체험해요.",
               SECTION_COLORS["km"], term="K-평균 군집화 (K-Means Clustering)")

    with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
        md_html("""
        - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
          → 정답 이름표 없이도 **비슷한 것끼리 스스로 모둠을 짓는 AI(K-Means)**의 원리를 체험해요.
        - **[실과] [6실05-04]** 디지털 데이터와 아날로그 데이터의 특징을 이해하고, 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
          → 이름표(정답)가 없는 데이터도 AI 학습에 활용될 수 있음을 탐색해요.
        """)

    tab_intro, tab_practice, tab_game, tab_class = st.tabs([
        "🌉 1단계: 먼저 알아보기", "🕹️ 2단계: 직접 나눠보기",
        "🧹 3단계: 게임: 무인도 분리수거 로봇", "🧑‍🤝‍🧑 4단계: 우리 반 데이터로 모둠 나누기"])

    # [1단계: 먼저 알아보기] -----------------------------------------------
    with tab_intro:
        st.markdown("### 🌉 어? 나 이거 이미 알아!")
        md_html("""
        <div style='font-size:19px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:18px 22px;'>
        장난감 블록을 정리할 때, 색깔별로 <b>저절로 눈에 모여 보이는</b> 것처럼 느낀 적 있나요?<br><br>
        AI도 이름표(정답)가 하나도 없어도, <b>생김새가 비슷한 것끼리</b> 스스로 무리를 지어줄 수 있어요.
        </div>
        """)

        st.write("")
        st.markdown("### 🎯 오늘의 미션")
        st.success("**이름표 없는 나뭇잎들을, AI가 생김새만 보고 몇 개의 모둠으로 알아서 나누게 해보세요!**")

        leaf_samples = pd.DataFrame({
            '잎의 길이': [15, 14, 16, 5, 4, 6, 8, 9, 2, 3, 1],
            '잎의 넓이': [2, 1.5, 2.5, 5, 4.5, 6, 1, 1.5, 1, 0.5, 0.8],
            '원래 이름': ['강아지풀', '강아지풀', '강아지풀', '단풍나무', '단풍나무', '단풍나무', '벚나무', '벚나무', '소나무', '소나무', '소나무']
        })

        st.write("")
        st.markdown("### 👀 조작하기 전에, 먼저 눈으로 봐요")

        col_km_txt, col_km_fig = st.columns([1, 1.4])

        with col_km_txt:
            cmp_km = st.radio("어떤 상태를 볼까요?", ["😵 이름표도 색깔도 없는 뒤죽박죽 상태", "😎 AI가 3모둠으로 나눈 후"], key="km_cmp_choice")

            if cmp_km.startswith("😵"):
                st.error("전부 회색이라 어떤 게 어떤 나뭇잎인지 전혀 모르겠죠?")
            else:
                st.success("색깔별로 비슷한 모양끼리 딱 모였죠? AI가 이름표 없이 이렇게 스스로 나눈 거예요!")

        with col_km_fig:
            fig_cmp = go.Figure()
            if cmp_km.startswith("😵"):
                fig_cmp.add_trace(go.Scatter(x=leaf_samples['잎의 길이'], y=leaf_samples['잎의 넓이'], mode='markers',
                                              marker=dict(size=18, color='gray'), name='이름표 없음'))
            else:
                _labels_demo, _ = _fit_kmeans_leaf(3)
                _colors_demo = ['#FF4B4B', '#1C83E1', '#00C781']
                for i in range(3):
                    mask = _labels_demo == i
                    fig_cmp.add_trace(go.Scatter(x=leaf_samples['잎의 길이'][mask], y=leaf_samples['잎의 넓이'][mask],
                                                  mode='markers', marker=dict(size=18, color=_colors_demo[i]), name=f'모둠 {i+1}'))
            fig_cmp.update_layout(xaxis_title="잎의 길이", yaxis_title="잎의 넓이", height=330,
                                  margin=dict(l=20, r=20, t=20, b=20),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_cmp, **_STRETCH, config={'displayModeBar': False})

        st.write("")
        st.info("💡 **다음 단계!** 2단계 탭에서 모둠 개수(K)를 직접 바꿔보며 결과가 어떻게 달라지는지 확인해봐요.")

    # [2단계: 직접 나눠보기] -----------------------------------------------
    with tab_practice:
        st.markdown("### 🕹️ 이제 네 차례!")

        with st.expander("🤔 시작하기 전에 잠깐 생각해보기"):
            think_km = st.radio(
                "모둠 개수(K)를 2개로 줄이면 어떻게 될까요?",
                ["선택 안 함", "① 원래 3종류였던 잎들이 억지로 2개 모둠에 나뉘어 담긴다", "② 항상 정확히 3종류로 알아서 나뉜다"],
                key="km_think1"
            )
            if think_km.startswith("①"):
                st.success("맞아요! AI는 내가 정해준 개수(K)에 맞춰 나누기 때문에, K를 잘못 정하면 원래 종류와 다르게 묶일 수도 있어요.")
            elif think_km.startswith("②"):
                st.warning("아래 슬라이더로 K를 2로 바꿔서 직접 확인해봐요!")

        col_ctrl, col_view = st.columns([1, 2])

        with col_ctrl:
            st.write("### 🔍 모둠 만들기 미션")
            k_clusters = st.slider("모둠을 몇 개로 나누어 볼까요?", 2, 4, 3, key="km_learn_k")

            X_leaf = leaf_samples[['잎의 길이', '잎의 넓이']]
            leaf_labels, leaf_centers = _fit_kmeans_leaf(k_clusters)
            leaf_samples['AI가 만든 모둠'] = leaf_labels

            st.divider()
            show_labels = st.checkbox("숨겨진 진짜 이름 확인하기", value=False, key="km_learn_labels")

        with col_view:
            fig = go.Figure()
            colors = ['#FF4B4B', '#1C83E1', '#00C781', '#FFAA00']
            symbols = ['circle', 'star', 'square', 'diamond']

            for i in range(k_clusters):
                cluster_subset = leaf_samples[leaf_samples['AI가 만든 모둠'] == i]
                fig.add_trace(go.Scatter(
                    x=cluster_subset['잎의 길이'],
                    y=cluster_subset['잎의 넓이'],
                    mode='markers+text',
                    name=f"모둠 {i+1}",
                    text=cluster_subset['원래 이름'] if show_labels else "",
                    textposition="top center",
                    marker=dict(size=20, color=colors[i], symbol=symbols[i], line=dict(width=2, color='white'))
                ))

            fig.add_trace(go.Scatter(
                x=leaf_centers[:, 0], y=leaf_centers[:, 1],
                mode='markers', name="모둠의 반장(가운데)",
                marker=dict(size=30, color='black', symbol='x', line=dict(width=4))
            ))

            fig.update_layout(xaxis_title="잎의 길이", yaxis_title="잎의 넓이", hovermode="closest", height=300)
            st.plotly_chart(fig, **_STRETCH)

    # [게임: 무인도 분리수거 로봇] --------------------------------------------
    with tab_game:
        st.header("🧹 무인도 분리수거 로봇")
        md_html("""
        <div style='font-size:17px; line-height:1.8; background:#E0F7FA; border-radius:12px; padding:14px 18px;'>
        무인도에 쓰레기가 흩어져 있어요. 그런데 <b>어떤 쓰레기인지 이름표가 하나도 없어요!</b> 😮<br>
        로봇 반장들이 <b>딱 두 가지 행동만 번갈아</b> 하면서 쓰레기를 모둠으로 나눠줄 거예요.
        </div>
        """)

        s1, s2 = st.columns(2)
        with s1:
            md_html("""
            <div style='background:#FFF8E1; border:2px solid #FFC93C; border-radius:12px; padding:12px; text-align:center; height:110px;'>
                <div style='font-family:Jua,sans-serif; font-size:17px; color:#F57C00;'>① 모이기</div>
                <div style='font-size:14px; color:#3A4A6B; margin-top:6px;'>
                    쓰레기가 <b>가장 가까운</b> 로봇 반장에게 모여요
                </div>
            </div>
            """)
        with s2:
            md_html("""
            <div style='background:#E8F5E9; border:2px solid #66BB6A; border-radius:12px; padding:12px; text-align:center; height:110px;'>
                <div style='font-family:Jua,sans-serif; font-size:17px; color:#2E7D32;'>② 이동하기</div>
                <div style='font-size:14px; color:#3A4A6B; margin-top:6px;'>
                    로봇 반장이 <b>자기 모둠 한가운데</b>로 옮겨가요
                </div>
            </div>
            """)
        st.caption("이 ①②를 여러 번 반복하면, 로봇 반장이 점점 완벽한 자리를 찾아가요!")
        st.divider()

        if st.session_state["km_data"] is None:
            rng = np.random.default_rng(3)
            cluster1 = rng.normal([2, 2], 0.8, (7, 2))
            cluster2 = rng.normal([8, 3], 0.8, (7, 2))
            cluster3 = rng.normal([5, 9], 0.8, (7, 2))
            pts = np.vstack([cluster1, cluster2, cluster3])
            st.session_state["km_data"] = pts

        pts = st.session_state["km_data"]
        if "km_phase" not in st.session_state:
            st.session_state["km_phase"] = "assign"   # 다음에 할 행동

        colL, colR = st.columns([1, 2])
        with colL:
            k = st.slider("🤖 로봇 반장 수 (K)", 2, 4, 3, key="km_game_k")

            if st.session_state["km_centroids"] is None:
                st.info("먼저 아래 버튼으로 로봇 반장을 무인도에 내려보내요!")
                place_btn = st.button("🚁 로봇 반장 내려보내기", **_STRETCH, key="km_place")
                step_btn = False
            else:
                place_btn = False
                if st.session_state["km_phase"] == "assign":
                    st.warning("다음 행동: **① 모이기**\n\n쓰레기들을 가장 가까운 반장에게 보내요.")
                    step_btn = st.button("① 가까운 반장에게 모이기 ▶️", **_STRETCH, key="km_step")
                else:
                    st.success("다음 행동: **② 이동하기**\n\n반장이 자기 모둠 한가운데로 갑니다.")
                    step_btn = st.button("② 반장이 가운데로 이동 ▶️", **_STRETCH, key="km_step")

            reset_km_btn = st.button("🔄 처음부터 다시", **_STRETCH, key="km_reset")
            st.metric("반복한 횟수", st.session_state["km_iter"])

        if reset_km_btn:
            st.session_state["km_data"] = None
            st.session_state["km_centroids"] = None
            st.session_state["km_labels"] = None
            st.session_state["km_iter"] = 0
            st.session_state["km_phase"] = "assign"
            st.rerun()

        if place_btn:
            rng2 = np.random.default_rng()
            idx = rng2.choice(len(pts), k, replace=False)
            st.session_state["km_centroids"] = pts[idx].copy()
            st.session_state["km_labels"] = None
            st.session_state["km_iter"] = 0
            st.session_state["km_phase"] = "assign"
            st.rerun()

        # 한 번에 한 행동씩만 수행 → 무슨 일이 일어나는지 눈으로 따라갈 수 있다
        if step_btn and st.session_state["km_centroids"] is not None:
            centroids = st.session_state["km_centroids"]
            if st.session_state["km_phase"] == "assign":
                dists = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
                st.session_state["km_labels"] = np.argmin(dists, axis=1)
                st.session_state["km_phase"] = "move"
            else:
                labels = st.session_state["km_labels"]
                new_centroids = centroids.copy()
                for j in range(len(centroids)):
                    if labels is not None and np.any(labels == j):
                        new_centroids[j] = pts[labels == j].mean(axis=0)
                st.session_state["km_centroids"] = new_centroids
                st.session_state["km_iter"] += 1
                st.session_state["km_phase"] = "assign"
            st.rerun()

        with colR:
            colors2 = ["#43A047", "#1E88E5", "#FB8C00", "#8E24AA"]
            fig2 = go.Figure()
            if st.session_state["km_labels"] is None:
                fig2.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                                          name="이름표 없는 쓰레기 🗑️",
                                          marker=dict(size=17, color="gray", symbol="circle",
                                                      line=dict(width=1, color="white"))))
            else:
                labels = st.session_state["km_labels"]
                cs_now = st.session_state["km_centroids"]
                for j in range(k):
                    mask = labels == j
                    if not np.any(mask):
                        continue
                    fig2.add_trace(go.Scatter(x=pts[mask, 0], y=pts[mask, 1], mode="markers",
                                              name=f"{chr(65+j)}모둠",
                                              marker=dict(size=17, color=colors2[j % 4],
                                                          line=dict(width=1, color="white"))))
                    # 쓰레기와 담당 반장을 선으로 이어 '누구에게 모였는지' 보이게 한다
                    for px, py in pts[mask]:
                        fig2.add_trace(go.Scatter(x=[px, cs_now[j][0]], y=[py, cs_now[j][1]],
                                                  mode="lines", showlegend=False,
                                                  line=dict(color=colors2[j % 4], width=1, dash="dot"),
                                                  hoverinfo="skip"))
            if st.session_state["km_centroids"] is not None:
                cs = st.session_state["km_centroids"]
                fig2.add_trace(go.Scatter(x=cs[:, 0], y=cs[:, 1], mode="markers+text",
                                          name="로봇 반장 🤖",
                                          text=[f"{chr(65+i)}" for i in range(len(cs))],
                                          textposition="top center",
                                          marker=dict(size=26, color="red", symbol="star",
                                                      line=dict(width=2, color="white"))))
            fig2.update_layout(height=315, xaxis_range=[-1, 11], yaxis_range=[-1, 11],
                               margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig2, **_STRETCH)

        if st.session_state["km_centroids"] is None:
            st.info("🚁 '로봇 반장 내려보내기'를 눌러 시작하세요! 반장들은 처음엔 아무 자리에나 서 있어요.")
        elif st.session_state["km_iter"] < 3:
            if st.session_state["km_phase"] == "assign":
                st.info("👀 반장이 가운데로 이동했어요! 이제 다시 **① 모이기**를 눌러 쓰레기를 다시 나눠봐요.")
            else:
                st.info("👀 쓰레기가 가장 가까운 반장에게 모였어요! (점선으로 연결) 이제 **② 이동하기**를 눌러보세요.")
        if st.session_state["km_centroids"] is not None and st.session_state["km_iter"] >= 3:
            st.success("🎉 쓰레기들이 아주 깔끔하게 모둠으로 나뉘었어요! 정돈 완료!")
            award_badge(4)
            md_html("""
            <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:16px 20px; margin-top:12px;">
                <b>🏷️ 오늘 배운 것</b><br>
                이렇게 <b>이름표 없이도 비슷한 것끼리 스스로 모둠을 짓는 방법</b>을 인공지능에서는
                <b>'K-평균 군집화(K-Means Clustering)'</b>라고 불러요.
            </div>
            """)

            with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                quiz_km = st.radio(
                    "이름표 없이도 비슷한 것끼리 스스로 모둠을 짓는 방법을 뭐라고 배웠나요?",
                    ["선택 안 함", "① K-평균 군집화", "② 결정트리", "③ 선형회귀"],
                    key="quiz_km"
                )
                if quiz_km == "① K-평균 군집화":
                    st.success("🌟 정답이에요! '완벽 이해' 보너스 배지 획득!")
                    st.session_state["bonus_badges"].add("🌟 K-Means 완벽 이해")
                elif quiz_km != "선택 안 함":
                    st.warning("다시 한 번 생각해볼까요? '원리 알아보기' 탭 내용을 떠올려봐요!")

            st.write("")
            next_section_button("🎨 5. 비슷한 친구끼리", "km")

    # [4단계: 우리 반 데이터로 모둠 나누기] --------------------------------
    with tab_class:
        st.subheader("🧑‍🤝‍🧑 우리 반 데이터로 AI가 모둠을 나눠줘요!")
        md_html("""
        <div style='font-size:17px; line-height:1.8; background:#FFF8E1; border-radius:12px; padding:14px 18px;'>
        이번엔 <b>정답(이름표)이 없는 진짜 데이터</b>로 해볼 거예요!<br>
        교실에서 잰 숫자(책상 크기, 친구들의 키·몸무게 등)를 넣으면
        AI가 <b>비슷한 것끼리 알아서 모둠</b>을 만들어준답니다.
        </div>
        """)

        st.info("🎯 **미션**: 아래 3단계를 모두 마치면 로봇 부품(팔)을 받을 수 있어요!")

        # ── 1단계: 데이터 입력 ───────────────────────────────
        st.markdown("#### 1️⃣ 데이터를 넣어요")
        st.caption("표의 숫자를 우리 반에서 직접 잰 값으로 바꿔보세요. 아래 ➕ 로 줄을 추가할 수도 있어요.")

        default_km_df = pd.DataFrame({
            "이름": ["책상A", "책상B", "책상C", "책상D", "책상E", "책상F"],
            "가로(cm)": [60, 62, 45, 44, 70, 43],
            "세로(cm)": [40, 42, 30, 29, 45, 31],
        })
        km_class_df = st.data_editor(default_km_df, num_rows="dynamic", key="km_class_editor", **_STRETCH)

        try:
            km_class_df = km_class_df.dropna(subset=["가로(cm)", "세로(cm)"])
        except Exception:
            km_class_df = pd.DataFrame()

        if len(km_class_df) < 3:
            st.warning("⚠️ 데이터를 **3줄 이상** 채워주세요! (가로·세로 숫자를 모두 적어야 해요.)")
        else:
            st.success(f"✅ 데이터 {len(km_class_df)}개 준비 완료!")

            # ── 2단계: 모둠 수 정하기 ──────────────────────────
            st.markdown("#### 2️⃣ 몇 개의 모둠으로 나눌지 정해요")
            max_groups = min(4, len(km_class_df))
            n_groups = st.slider("모둠 수 (K)", 2, max_groups, min(2, max_groups), key="km_class_k")

            # ── 3단계: AI에게 맡기기 ──────────────────────────
            st.markdown("#### 3️⃣ AI에게 모둠 나누기를 맡겨요")
            if st.button("🤖 AI에게 맡기기!", key="km_class_run_btn", **_STRETCH):
                st.session_state["km_class_run"] = True

            if not st.session_state.get("km_class_run"):
                st.info("👆 위 버튼을 누르면 AI가 비슷한 것끼리 모둠을 만들어줘요!")
            else:
                X = km_class_df[["가로(cm)", "세로(cm)"]].astype(float).values
                km_model = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
                labels = km_model.fit_predict(X)
                km_class_df = km_class_df.copy()
                km_class_df["AI 모둠"] = [f"{chr(65+l)}모둠" for l in labels]

                fig = go.Figure()
                palette = ["#2D9CDB", "#FF6F59", "#2EC4B6", "#FFC93C"]
                for g in sorted(km_class_df["AI 모둠"].unique()):
                    sub = km_class_df[km_class_df["AI 모둠"] == g]
                    idx = ord(g[0]) - 65
                    fig.add_trace(go.Scatter(
                        x=sub["가로(cm)"], y=sub["세로(cm)"], mode="markers+text",
                        text=sub["이름"], textposition="top center", name=g,
                        marker=dict(size=17, color=palette[idx % len(palette)],
                                    line=dict(width=1, color="white"))))
                for i, c in enumerate(km_model.cluster_centers_):
                    fig.add_trace(go.Scatter(
                        x=[c[0]], y=[c[1]], mode="markers", name=f"{chr(65+i)}모둠 중심 ⭐",
                        marker=dict(size=22, color=palette[i % len(palette)], symbol="x",
                                    line=dict(width=3))))
                fig.update_layout(title="AI가 비슷한 것끼리 나눈 모둠",
                                  xaxis_title="가로(cm)", yaxis_title="세로(cm)",
                                  height=294, margin=dict(l=20, r=20, t=48, b=20))
                st.plotly_chart(fig, **_STRETCH)

                st.success("🤖 완성! AI가 **정답을 하나도 알려주지 않았는데도** 비슷한 것끼리 묶었어요!")
                st.dataframe(km_class_df, **_STRETCH)

                # ── 관찰 확인 → 부품 획득 ─────────────────────
                st.divider()
                st.markdown("#### 🔍 마지막! 관찰한 것을 확인해요")
                obs = st.radio(
                    "AI는 무엇을 보고 모둠을 나눴을까요?",
                    ["선택 안 함",
                     "① 크기(가로·세로)가 비슷한 것끼리 묶었다",
                     "② 이름표(정답)를 미리 알고 있었다"],
                    key="km_class_obs"
                )
                if obs.startswith("①"):
                    st.success("🎉 정확해요! 이름표 없이 **숫자(특징)가 비슷한 것끼리** 스스로 묶는 것, "
                               "이것이 바로 **K-평균 군집화**예요!")
                    award_badge(4)
                    st.write("")
                    next_section_button("🎨 5. 비슷한 친구끼리", "km_class")
                elif obs.startswith("②"):
                    st.warning("아니에요! 우리는 '이름표(정답)'를 하나도 알려주지 않았어요. "
                               "AI는 오직 **숫자가 비슷한지**만 보고 나눴답니다. 다시 골라볼까요?")
                else:
                    st.caption("👆 위 질문에 답하면 로봇 부품을 받을 수 있어요!")

                st.caption("💡 모둠 수(K)를 바꾸고 다시 '맡기기'를 눌러보세요. 결과가 어떻게 달라지나요?")

# --- [메뉴 7: 똑똑한 생각 주머니 (인공신경망)] ---
elif menu == "🧠 6. 똑똑한 생각 주머니":
    hero_card("🧠", "똑똑한 생각 주머니: 내 몸속 요정들의 계산법!",
               "여러 요정이 힘을 합쳐 하나의 판단을 내리는 인공신경망의 원리를 로봇 네오로 체험해요.",
               SECTION_COLORS["nn"], term="인공신경망 (Artificial Neural Network)")

    with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
        md_html("""
        - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
          → 입력값에 가중치를 곱해 더하는 **인공 뉴런**들이 모여, 그 결과를 또 한 번 합쳐서 최종 판단을 내리는 **인공신경망(은닉층)의 기본 계산 원리**를 로봇 네오를 통해 체험해요.
        - **[실과] [6실04-06]** 생활 속에서 로봇 활용 사례를 통해 작동 원리와 활용 분야를 이해한다.
          → 여러 요정(뉴런)의 판단을 종합해 하나의 결론(파워 게이지)을 내리는 로봇의 작동 원리와 연결지어 이해해요.
        """)

    tab_intro, tab_practice, tab_game = st.tabs(["🌉 1단계: 먼저 알아보기", "🕹️ 2단계: 뉴런 1개 실습", "🤖 3단계: 게임: 로봇 네오 발사"])

    # [1단계: 먼저 알아보기] -----------------------------------------------
    with tab_intro:
        st.markdown("### 🌉 어? 나 이거 이미 알아!")
        md_html("""
        <div style='font-size:19px; line-height:1.9; background:#FFF8E1; border-radius:14px; padding:18px 22px;'>
        요리 점수를 매길 때, <b>맛 담당</b>은 맛에 점수를 주고, <b>영양 담당</b>은 영양에 점수를 주고,
        <b>모양 담당</b>은 모양에 점수를 주죠? 그리고 이 <b>점수들을 하나로 합쳐서</b> 최종 점수를 정해요.<br><br>
        인공신경망도 똑같아요. 여러 '판단 담당(뉴런)'이 <b>저마다 다른 부분</b>을 보고 점수를 매긴 다음,
        그 점수들을 <b>합쳐서</b> 하나의 결론을 내려요. (손드는 다수결 투표가 아니라, <b>점수를 곱하고 더해 합치는</b> 방식이에요!)
        </div>
        """)

        st.write("")
        st.markdown("### 🎯 오늘의 미션")
        st.success("**뉴런 요정 한 명의 계산법을 먼저 익히고, 나중엔 요정 3명을 함께 써서 로봇의 파워를 채워보세요!**")

        st.write("")
        st.markdown("### 👀 조작하기 전에, 먼저 눈으로 봐요")
        st.write("인공신경망은 아주 작은 **'인공 뉴런(요정)'**을 여러 개 이어 붙여 만든 **생각 그물망**이에요. "
                 "정보가 어떻게 흘러가는지 그림으로 먼저 볼까요?")

        # ── ① 인공신경망 흐름도: 입력 → 뉴런 요정들 → 합치기 → 결론 ──────
        _nn_in = "background:#E3F2FD; border:2px solid #2D9CDB; border-radius:14px; padding:12px 8px; text-align:center; font-size:14px; margin-bottom:10px;"
        _nn_hid = "background:#F3E5F5; border:2px solid #8E24AA; border-radius:50px; padding:14px 8px; text-align:center; font-size:14px; margin-bottom:10px;"
        _nn_out = "background:#E8F5E9; border:2px solid #2E7D32; border-radius:14px; padding:16px 8px; text-align:center; font-size:15px;"
        _nn_arrow = "text-align:center; font-size:26px; color:#3A4A6B; margin-top:52px;"
        md_html(f"""
        <div style='background:#FAFAFA; border-radius:16px; padding:18px 12px;'>
          <div style='display:flex; gap:8px; align-items:flex-start;'>
            <div style='flex:1.2;'>
              <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>① 입력</b><br>(내가 한 행동)</div>
              <div style='{_nn_in}'>🍕 피자<br>몇 조각?</div>
              <div style='{_nn_in}'>🏃 운동<br>몇 시간?</div>
            </div>
            <div style='flex:0.4;'><div style='{_nn_arrow}'>➡️</div></div>
            <div style='flex:1.2;'>
              <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>② 뉴런 요정들</b><br>(각자 계산)</div>
              <div style='{_nn_hid}'>🧚 요정1<br><span style='font-size:11px;'>×곱하고 +더하기</span></div>
              <div style='{_nn_hid}'>🧚‍♂️ 요정2<br><span style='font-size:11px;'>×곱하고 +더하기</span></div>
              <div style='{_nn_hid}'>🧚‍♀️ 요정3<br><span style='font-size:11px;'>×곱하고 +더하기</span></div>
            </div>
            <div style='flex:0.4;'><div style='{_nn_arrow}'>➡️</div></div>
            <div style='flex:1.2;'>
              <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>③ 의견 모으기</b></div>
              <div style='{_nn_hid} margin-top:40px;'>🗳️ 세 요정의<br>점수를 합쳐요</div>
            </div>
            <div style='flex:0.4;'><div style='{_nn_arrow}'>➡️</div></div>
            <div style='flex:1.2;'>
              <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>④ 최종 결론</b></div>
              <div style='{_nn_out} margin-top:34px;'>🤖 로봇 파워<br><b>87점!</b> 🚀</div>
            </div>
          </div>
          <div style='text-align:center; margin-top:12px; font-size:14px; color:#3A4A6B;'>
            요정(뉴런) 한 명 한 명은 <b>곱하고 더하는 간단한 계산</b>만 해요.<br>
            그런데 여러 명의 계산을 <b>모으면</b> 놀랍게 똑똑한 판단이 나와요! 이게 바로 <b>인공신경망</b>이에요.
          </div>
        </div>
        """)

        st.write("")
        # ── ② 혼자 판단 vs 여럿이 판단: 왜 여러 뉴런을 쓸까? ─────────────
        st.markdown("#### 🔍 요정이 왜 여러 명이나 필요할까요? 버튼을 눌러 비교해봐요!")
        st.caption("요정들은 손을 들어 '다수결 투표'를 하는 게 아니라, 각자 매긴 **점수를 하나로 합쳐서** 결론을 내요.")
        nn_ex_choice = st.radio(
            "튼튼 점수를 매길 때, 요정 몇 명이 볼까요?",
            ["🧚 요정 1명이 볼 때", "🧚‍♀️🧚🧚‍♂️ 요정 3명이 볼 때"],
            horizontal=True, key="nn_intro_compare")

        if nn_ex_choice.startswith("🧚 요정 1명"):
            md_html("""
            <div style='background:#FFEBEE; border-radius:12px; padding:14px 18px; text-align:center;'>
                <span style='font-size:30px;'>🧚</span> "운동만 보고 → <b style='color:#C62828;'>80점!</b>"<br>
                <span style='font-size:13px; color:#666;'>운동 하나만 보는 요정이라, 잠을 못 잔 건 놓쳐요.
                이 요정이 놓친 부분은 아무도 챙겨주지 못해요 😥</span>
            </div>
            """)
            st.error("요정이 **한 명**뿐이면, 그 요정이 보는 한 가지 관점만 반영돼요. 놓친 부분을 메워줄 사람이 없어요.")
        else:
            md_html("""
            <div style='background:#E8F5E9; border-radius:12px; padding:14px 18px; text-align:center;'>
                <span style='font-size:26px;'>🧚‍♀️</span> 운동 요정 "+40점" &nbsp;
                <span style='font-size:26px;'>🧚</span> 잠 요정 "+30점" &nbsp;
                <span style='font-size:26px;'>🧚‍♂️</span> 피자 요정 "−10점"<br>
                <b style='color:#2E7D32;'>세 점수를 합치면 → 60점!</b><br>
                <span style='font-size:13px; color:#666;'>요정마다 <b>보는 관점이 달라요.</b>
                각자의 점수를 <b>합치면</b> 운동·잠·피자를 골고루 따진 똑똑한 결론이 나와요 ✨</span>
            </div>
            """)
            st.success("여러 요정이 각자 다른 관점의 점수를 매기고, 그 점수를 **모두 합쳐서** 결론을 내요. "
                       "이렇게 **여러 관점을 곱하고 더해 합치는 것**이 인공신경망이 뉴런을 여러 개 쓰는 이유예요. "
                       "(손드는 다수결 투표가 아니에요!)")

        st.write("")
        st.info("💡 피자를 먹고 뛰어놀면, **뉴런 요정 한 명**이 '피자 점수'와 '운동 점수'를 자기만의 힘(가중치)으로 곱해서 더한 다음 건강 점수를 알려줘요.")
        st.info("💡 **다음 단계!** 2단계 탭에서 직접 슬라이더를 움직여 뉴런 요정 1명이 어떻게 계산하는지 확인해봐요.")

    # [2단계: 뉴런 1개 실습] -----------------------------------------------
    with tab_practice:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("### 📥 내가 한 행동 (입력)")
            pizza = st.slider("🍕 피자를 몇 조각 먹었나요?", 0, 10, 5, key="nn_learn_pizza")
            exercise = st.slider("🏃‍♂️ 밖에서 몇 시간 놀았나요?", 0, 10, 5, key="nn_learn_exercise")

        with col2:
            st.write("### ⚙️ 뉴런 요정 1명의 계산")
            weight_pizza = -1
            weight_exercise = 2

            hidden_score = (pizza * weight_pizza) + (exercise * weight_exercise)

            st.info(f"💡 피자 먹은 점수: {pizza}조각 × (-1점) = {pizza * weight_pizza}점")
            st.info(f"💡 뛰어논 점수: {exercise}시간 × (보너스 2점!) = {exercise * weight_exercise}점")
            st.warning(f"뉴런 요정이 계산한 점수: **{hidden_score}점**")

        with col3:
            st.write("### 📤 건강 점수 발표! (출력)")
            final_score = max(0, min(100, 50 + hidden_score))

            st.metric(label="나의 최종 건강 점수", value=f"{final_score}점")

            if final_score >= 80:
                st.success("튼튼 대장이에요! 최고! 🏆")
            elif final_score >= 50:
                st.warning("건강해요! 피자를 조금 줄이거나 더 뛰어놀면 완벽해요. 💪")
            else:
                st.error("아이고 배야! 피자를 너무 많이 먹었어요! 😭 밖에서 뛰어놀까요?")

        with st.expander("🤔 잠깐 생각해보기"):
            think_nn = st.radio(
                "뉴런 요정이 딱 1명뿐이면 어떤 점이 아쉬울까요?",
                ["선택 안 함", "① 한 가지 관점으로만 판단해서 다양한 상황을 놓칠 수 있다", "② 아쉬운 점이 전혀 없다"],
                key="nn_think1"
            )
            if think_nn.startswith("①"):
                st.success("맞아요! 그래서 진짜 인공신경망은 여러 뉴런을 함께 써서 다양한 관점을 모아요.")
            elif think_nn.startswith("②"):
                st.warning("게임 탭에서 요정 3명을 같이 써보면 왜 여러 명이 필요한지 느껴질 거예요!")

        st.success("🔗 **다음 단계!** 진짜 인공신경망은 이런 뉴런 요정을 **여러 명** 동시에 써요. 각 요정이 조금씩 다르게 판단한 걸 또 한 번 합쳐서 최종 결론을 내리죠. '게임' 탭에서 요정 3명이 힘을 합치는 걸 직접 만들어볼까요?")

    # [게임: 로봇 네오 발사 — 은닉층 요정 3명] -------------------------------
    with tab_game:
        st.header("🤖 뉴런 요정 3명의 힘을 모아 로봇 네오를 발사하자!")
        st.write("입력 3가지가 **은닉층 요정 3명**에게 각각 전달돼요. 요정들은 저마다 다른 힘(가중치)으로 판단하고, "
                 "그 3개의 판단을 **출력 요정**이 또 한 번 종합해서 최종 파워를 결정해요. 이게 진짜 인공신경망의 계산 방식이에요!")

        if st.button("🔄 처음부터 다시 하기", key="nn_reset"):
            for k in ["nn_game_pizza", "nn_game_exercise", "nn_game_sleep",
                      "nn_wo1", "nn_wo2", "nn_wo3", "nn_bias_out",
                      "nn_w1_pizza", "nn_w1_ex", "nn_w1_sleep",
                      "nn_w2_pizza", "nn_w2_ex", "nn_w2_sleep",
                      "nn_w3_pizza", "nn_w3_ex", "nn_w3_sleep"]:
                st.session_state.pop(k, None)
            st.rerun()

        colL, colR = st.columns([1, 1.3])
        with colL:
            st.write("#### 🎛️ 입력값 (오늘 네오가 한 일)")
            g_pizza = st.slider("🍕 피자 먹은 개수", 0, 5, 2, key="nn_game_pizza")
            g_exercise = st.slider("🏃 운동한 시간(분)", 0, 60, 20, key="nn_game_exercise")
            g_sleep = st.slider("😴 잠잔 시간(시간)", 0, 10, 6, key="nn_game_sleep")

            st.write("#### 🤖 출력 요정의 다이얼 (얼마나 믿고 반영할까?)")
            st.caption("은닉층 요정 3명은 이미 각자 '성격'(튼튼 담당·운동 담당·잠 담당)이 정해져 있어요. "
                       "너는 **출력 요정**이 되어, 그 3명의 의견을 각각 얼마나 반영할지만 정하면 돼요!")
            wo1 = st.slider("🧚 1번 요정(튼튼 담당) 의견 반영도", 0.0, 2.0, 0.5, step=0.1, key="nn_wo1")
            wo2 = st.slider("🧚 2번 요정(운동 담당) 의견 반영도", 0.0, 2.0, 0.4, step=0.1, key="nn_wo2")
            wo3 = st.slider("🧚 3번 요정(잠 담당) 의견 반영도", 0.0, 2.0, 0.3, step=0.1, key="nn_wo3")
            bias_out = st.slider("⚡ 기본 여유 힘 (bias)", 0.0, 50.0, 40.0, step=1.0, key="nn_bias_out")

            with st.expander("🔧 (선택) 은닉층 요정 3명의 성격도 직접 바꿔보고 싶다면"):
                st.caption("여기는 안 열어봐도 게임할 수 있어요. 궁금한 친구만 살짝 조절해봐요!")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("🧚 1번(튼튼)")
                    w1_pizza = st.slider("🍕", -5.0, 2.0, -3.0, step=0.1, key="nn_w1_pizza")
                    w1_ex = st.slider("🏃", 0.0, 2.0, 0.3, step=0.05, key="nn_w1_ex")
                    w1_sleep = st.slider("😴", 0.0, 2.0, 2.0, step=0.1, key="nn_w1_sleep")
                with c2:
                    st.write("🧚 2번(운동)")
                    w2_pizza = st.slider("🍕 ", -5.0, 2.0, -1.0, step=0.1, key="nn_w2_pizza")
                    w2_ex = st.slider("🏃 ", 0.0, 2.0, 0.6, step=0.05, key="nn_w2_ex")
                    w2_sleep = st.slider("😴 ", 0.0, 2.0, 0.5, step=0.1, key="nn_w2_sleep")
                with c3:
                    st.write("🧚 3번(잠)")
                    w3_pizza = st.slider("🍕  ", -5.0, 2.0, -1.5, step=0.1, key="nn_w3_pizza")
                    w3_ex = st.slider("🏃  ", 0.0, 2.0, 0.1, step=0.05, key="nn_w3_ex")
                    w3_sleep = st.slider("😴  ", 0.0, 2.0, 3.0, step=0.1, key="nn_w3_sleep")

        # 은닉층 요정 3명의 계산 (각자 입력 3개를 자기 힘으로 조합)
        h1 = g_pizza * w1_pizza + g_exercise * w1_ex + g_sleep * w1_sleep
        h2 = g_pizza * w2_pizza + g_exercise * w2_ex + g_sleep * w2_sleep
        h3 = g_pizza * w3_pizza + g_exercise * w3_ex + g_sleep * w3_sleep

        # 출력 요정이 은닉층 3명의 판단을 다시 한 번 종합
        raw = h1 * wo1 + h2 * wo2 + h3 * wo3 + bias_out
        power = max(0, min(100, raw))

        with colR:
            st.write("#### ⚡ 네오의 파워 게이지")
            st.progress(int(power) / 100)
            st.write(f"### {power:.0f} / 100")

            with st.expander("🔍 은닉층 요정 3명이 계산한 중간 점수 보기"):
                st.write(f"🧚 1번 요정: {h1:.1f}점 · 🧚 2번 요정: {h2:.1f}점 · 🧚 3번 요정: {h3:.1f}점")
                st.caption("출력 요정 = (1번×반영도) + (2번×반영도) + (3번×반영도) + 기본 힘")

            fig = go.Figure()
            node_x = [0, 0, 0, 1.6, 1.6, 1.6, 3.2]
            node_y = [2, 1, 0, 2, 1, 0, 1]
            labels = ["🍕", "🏃", "😴", "🧚", "🧚", "🧚", "🤖"]
            hidden_sizes = [28 + min(abs(h1), 40), 28 + min(abs(h2), 40), 28 + min(abs(h3), 40)]
            sizes = [26, 26, 26] + hidden_sizes + [40 + power * 0.3]
            colors = ["#FB8C00", "#43A047", "#1E88E5", "#BA68C8", "#AB47BC", "#8E24AA", "#E53935"]
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels,
                                      textfont=dict(size=22),
                                      marker=dict(size=sizes, color=colors)))
            for i in range(3):
                for j in range(3, 6):
                    fig.add_trace(go.Scatter(x=[node_x[i], node_x[j]], y=[node_y[i], node_y[j]],
                                              mode="lines", line=dict(color="lightgray", width=1), showlegend=False))
            for j in range(3, 6):
                fig.add_trace(go.Scatter(x=[node_x[j], node_x[6]], y=[node_y[j], node_y[6]],
                                          mode="lines", line=dict(color="lightgray", width=1), showlegend=False))
            fig.update_layout(height=400, showlegend=False, xaxis_visible=False, yaxis_visible=False,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, **_STRETCH)

        if power >= 100:
            st.balloons()
            play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
            st.success("🚀 파워 100 달성! 세 요정과 출력 요정이 힘을 합쳐 네오가 우주로 로켓 발사!! 🎉")
            award_badge(5)
            if len(st.session_state["badges"]) >= 6:
                st.balloons()
                st.success("🎊 6개 부품을 모두 모아 **로봇 네오가 완전히 조립됐어요!** 배운 걸 정리하는 복습 미션에 도전하거나, 바로 수료증을 받으러 갈 수 있어요.")
                cta1, cta2 = st.columns(2)
                with cta1:
                    if st.button("🔧 복습 미션 도전!", key="nn_to_review", **_STRETCH):
                        go_to("🔧 로봇 네오 종합 점검 (복습)")
                with cta2:
                    if st.button("🏆 수료증 받기", key="nn_to_cert", **_STRETCH):
                        go_to("🏆 탐험 완료 (수료증)")
            md_html("""
            <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:16px 20px; margin-top:12px;">
                <b>🏷️ 오늘 배운 것</b><br>
                이렇게 <b>여러 뉴런(요정)이 각자 입력에 힘(가중치)을 곱해 계산하고, 그 값들을 다시 힘(가중치)을 곱해 합쳐서
                최종 결론을 내리는 구조</b>를 인공지능에서는 <b>'인공신경망(Artificial Neural Network)'</b>이라고 불러요.
            </div>
            """)

            with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                quiz_nn = st.radio(
                    "여러 뉴런이 각자 판단하고 그 판단을 종합해서 결론을 내리는 구조를 뭐라고 배웠나요?",
                    ["선택 안 함", "① 인공신경망", "② KNN", "③ 경사하강법"],
                    key="quiz_nn"
                )
                if quiz_nn == "① 인공신경망":
                    st.success("🌟 정답이에요! '완벽 이해' 보너스 배지 획득! 6개 섬을 모두 클리어했어요! 🎉")
                    st.session_state["bonus_badges"].add("🌟 인공신경망 완벽 이해")
                elif quiz_nn != "선택 안 함":
                    st.warning("다시 한 번 생각해볼까요? '원리 알아보기' 탭 내용을 떠올려봐요!")

            st.write("")
            next_section_button("🧠 6. 똑똑한 생각 주머니", "nn")
        elif power >= 60:
            st.info("거의 다 왔어요! 요정들의 힘(가중치)이나 반영도를 조금 더 키워볼까요?")
        else:
            st.warning("아직 힘이 부족해요. 피자 담당 요정의 힘을 줄이거나, 운동/잠 담당 요정의 힘과 반영도를 늘려보세요!")

        with st.expander("🧚 요정 설명: 왜 요정이 여러 명 필요할까요?"):
            st.write("뉴런 요정 1명만 있으면 딱 한 가지 방식으로만 판단해요. 그런데 요정 3명이 **서로 다른 관점**(1번은 튼튼함 위주, "
                     "2번은 운동 위주, 3번은 잠 위주)으로 각자 판단하고, 출력 요정이 그 3개의 의견을 다시 한 번 종합하면 "
                     "훨씬 더 다양하고 정교한 판단을 내릴 수 있어요. 이렇게 여러 층의 뉴런이 쌓인 구조를 "
                     "**인공신경망(뉴런이 여러 층으로 쌓인 그물망)**이라고 불러요.")


# --- [AI 윤리: AI를 똑똑하게 쓰려면?] ---
elif menu == "⚖️ AI를 똑똑하게 쓰려면?":
    hero_card("⚖️", "AI를 똑똑하게 쓰려면?",
               "AI는 아주 편리하지만 항상 옳은 건 아니에요. AI를 현명하게 쓰는 법을 함께 생각해봐요.",
               "#00897B", term="AI 윤리 (AI Ethics)")

    with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
        md_html("""
        - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, **인공지능이 사회에 미치는 영향**을 탐색한다.
          → AI가 틀릴 수 있다는 점, 편향과 개인정보 문제 등을 생각하며 **비판적 사고**를 길러요.
        """)

    tab_e1, tab_e2, tab_e3, tab_e4 = st.tabs([
        "👀 1단계: AI의 비밀, 눈으로 보기",
        "🧐 2단계: 상황 판단 놀이",
        "⚡ 3단계: O/X 스피드 퀴즈",
        "✍️ 4단계: 나의 다짐 서약서"
    ])

    # [1단계: AI의 비밀, 눈으로 보기] --------------------------------------
    with tab_e1:
        st.markdown("### 👀 조작하기 전에, 먼저 눈으로 봐요")
        st.write("AI는 왜 실수를 할까요? 우리가 배운 **'AI가 배우는 과정'**을 다시 떠올리면 답이 보여요!")

        _e_box = "border-radius:14px; padding:16px 10px; text-align:center; font-size:14px;"
        md_html(f"""
        <div style='background:#FAFAFA; border-radius:16px; padding:18px 12px;'>
          <div style='display:flex; gap:8px; align-items:center;'>
            <div style='flex:1; background:#E3F2FD; {_e_box}'>
              <div style='font-size:34px;'>📊</div><b>① 사람이 예시를 줘요</b><br>
              <span style='font-size:12px; color:#666;'>사과 사진 1000장</span>
            </div>
            <div style='font-size:24px;'>➡️</div>
            <div style='flex:1; background:#F3E5F5; {_e_box}'>
              <div style='font-size:34px;'>🧠</div><b>② AI가 규칙을 배워요</b><br>
              <span style='font-size:12px; color:#666;'>"사과는 이렇게 생겼구나!"</span>
            </div>
            <div style='font-size:24px;'>➡️</div>
            <div style='flex:1; background:#E8F5E9; {_e_box}'>
              <div style='font-size:34px;'>🔮</div><b>③ AI가 판단해요</b><br>
              <span style='font-size:12px; color:#666;'>"이건 사과야!"</span>
            </div>
          </div>
          <div style='text-align:center; margin-top:12px; font-size:15px; color:#C62828;'>
            🚨 그런데! <b>①에서 준 예시가 이상하면 → ②의 규칙도 이상해지고 → ③의 판단도 틀려요!</b>
          </div>
        </div>
        """)

        st.write("")
        st.markdown("#### 🍎 직접 눈으로 확인해봐요: 어떤 예시를 주느냐에 따라 AI가 달라져요!")
        ethics_data_choice = st.radio(
            "AI 요리사에게 어떤 사과 예시를 보여줄까요?",
            ["😵 빨간 사과만 잔뜩 보여주기", "😎 여러 색 사과를 골고루 보여주기"],
            horizontal=True, key="ethics_data_choice")

        if ethics_data_choice.startswith("😵"):
            md_html("""
            <div style='display:flex; gap:14px;'>
              <div style='flex:1; background:#FFEBEE; border-radius:12px; padding:14px; text-align:center;'>
                <b>AI가 배운 예시</b><br><span style='font-size:32px;'>🍎🍎🍎🍎🍎</span><br>
                <span style='font-size:13px; color:#666;'>빨간 사과뿐!</span>
              </div>
              <div style='flex:1; background:#FFEBEE; border-radius:12px; padding:14px; text-align:center;'>
                <b>새로 만난 초록 사과</b><br><span style='font-size:32px;'>🍏</span><br>
                <span style='font-size:14px; color:#C62828;'>🤖 "이건 사과가 아니야!" ❌</span>
              </div>
            </div>
            """)
            st.error("한쪽으로 **치우친 예시(데이터)**로 배우면, AI도 **편견**을 갖게 돼요. 이걸 **'편향'**이라고 해요.")
        else:
            md_html("""
            <div style='display:flex; gap:14px;'>
              <div style='flex:1; background:#E8F5E9; border-radius:12px; padding:14px; text-align:center;'>
                <b>AI가 배운 예시</b><br><span style='font-size:32px;'>🍎🍏🍎🍏🍎</span><br>
                <span style='font-size:13px; color:#666;'>여러 색을 골고루!</span>
              </div>
              <div style='flex:1; background:#E8F5E9; border-radius:12px; padding:14px; text-align:center;'>
                <b>새로 만난 초록 사과</b><br><span style='font-size:32px;'>🍏</span><br>
                <span style='font-size:14px; color:#2E7D32;'>🤖 "이것도 사과야!" ⭕</span>
              </div>
            </div>
            """)
            st.success("**다양한 예시**를 골고루 보여주면 AI가 훨씬 공평하고 정확해져요!")

        st.write("")
        st.markdown("#### 🤖 AI의 약점 3가지, 카드로 기억해요!")
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            md_html("""
            <div style='background:#FFF3E0; border:2px solid #FB8C00; border-radius:16px; padding:16px 12px; text-align:center; height:200px;'>
                <div style='font-size:40px;'>🎭</div>
                <div style="font-family:'Jua',sans-serif; font-size:18px; color:#E65100;">치우쳐서 배워요</div>
                <div style='font-size:13px; color:#3A4A6B; margin-top:8px; line-height:1.6;'>
                한쪽 예시만 보면<br>편견(편향)이 생겨요</div>
            </div>
            """)
        with wc2:
            md_html("""
            <div style='background:#FCE4EC; border:2px solid #D81B60; border-radius:16px; padding:16px 12px; text-align:center; height:200px;'>
                <div style='font-size:40px;'>🤥</div>
                <div style="font-family:'Jua',sans-serif; font-size:18px; color:#AD1457;">그럴듯한 거짓말</div>
                <div style='font-size:13px; color:#3A4A6B; margin-top:8px; line-height:1.6;'>
                모르는 것도 아는 척<br>지어내서 말하기도 해요</div>
            </div>
            """)
        with wc3:
            md_html("""
            <div style='background:#E8EAF6; border:2px solid #3949AB; border-radius:16px; padding:16px 12px; text-align:center; height:200px;'>
                <div style='font-size:40px;'>🔓</div>
                <div style="font-family:'Jua',sans-serif; font-size:18px; color:#283593;">정보를 기억해요</div>
                <div style='font-size:13px; color:#3A4A6B; margin-top:8px; line-height:1.6;'>
                내 개인정보를 알려주면<br>어디로 갈지 몰라요</div>
            </div>
            """)

        st.write("")
        st.info("💡 **다음 단계!** 2단계 탭에서 진짜 있을 법한 상황들을 놓고 함께 판단해봐요.")

    # [2단계: 상황 판단 놀이] ----------------------------------------------
    with tab_e2:
        st.markdown("### 🧐 함께 생각해볼까요?")
        st.write("진짜 있을 법한 상황 6가지예요. 나라면 어떻게 할지 골라보세요!")

        with st.expander("🍎 상황 1. 치우친 데이터", expanded=True):
            st.write("어떤 AI에게 **빨간 사과 사진만 잔뜩** 보여주며 '이게 사과야'라고 가르쳤어요. "
                     "그런데 나중에 **초록색 사과**를 보여줬더니 '사과가 아니야!'라고 했어요.")
            e1 = st.radio("왜 이런 일이 생겼을까요?",
                          ["선택 안 함",
                           "① 빨간 사과만 봐서, 초록 사과는 배우지 못했다",
                           "② AI가 일부러 거짓말을 했다"],
                          key="ethics_q1")
            if e1.startswith("①"):
                st.success("맞아요! AI는 본 것만 배워요. 그래서 **다양한 예시**를 골고루 보여주는 게 중요해요. "
                           "한쪽으로 치우친 데이터로 배우면 AI도 편견을 갖게 돼요.")
            elif e1.startswith("②"):
                st.warning("AI는 거짓말을 하는 게 아니라, 배운 대로만 판단해요. 빨간 사과만 배웠으니 초록 사과를 모르는 거예요.")

        with st.expander("🔒 상황 2. 소중한 개인정보"):
            st.write("얼굴을 인식하는 AI를 만들려면 **내 얼굴 사진**이 많이 필요해요. "
                     "그런데 내 사진을 아무 데나 올리거나, 친구 사진을 몰래 쓰면 어떻게 될까요?")
            e2 = st.radio("가장 알맞은 생각은?",
                          ["선택 안 함",
                           "① 사진 같은 개인정보는 함부로 쓰면 안 되고, 허락을 받아야 한다",
                           "② AI를 위해서라면 누구 사진이든 마음대로 써도 된다"],
                          key="ethics_q2")
            if e2.startswith("①"):
                st.success("맞아요! 아무리 편리한 AI라도 **내 정보와 친구의 정보는 소중히** 지켜야 해요. 꼭 허락을 받아야 해요.")
            elif e2.startswith("②"):
                st.warning("다시 생각해볼까요? 다른 사람의 사진이나 정보를 허락 없이 쓰면 안 돼요.")

        with st.expander("🤔 상황 3. AI를 믿어도 될까?"):
            st.write("AI 스피커에게 '오늘 비 와?'라고 물었더니 '안 온다'고 했어요. 그런데 밖에 비가 쏟아지고 있어요!")
            e3 = st.radio("우리는 어떻게 해야 할까요?",
                          ["선택 안 함",
                           "① AI 말도 틀릴 수 있으니, 창밖도 보고 스스로 판단한다",
                           "② AI가 안 온다고 했으니 무조건 믿는다"],
                          key="ethics_q3")
            if e3.startswith("①"):
                st.success("맞아요! AI는 도움을 주는 도구일 뿐, **항상 옳은 건 아니에요.** "
                           "AI의 답을 참고하되, **스스로 생각하고 판단하는 힘**이 가장 중요해요.")
            elif e3.startswith("②"):
                st.warning("AI도 틀릴 수 있어요. AI 말만 100% 믿기보다 스스로도 확인하는 습관이 필요해요.")

        with st.expander("📝 상황 4. 숙제를 통째로 맡겨도 될까?"):
            st.write("독서 감상문 숙제가 있어요. AI 챗봇에게 부탁하면 감상문을 뚝딱 써줘요. "
                     "그대로 베껴서 내면 어떨까요?")
            e4 = st.radio("가장 알맞은 생각은?",
                          ["선택 안 함",
                           "① 내 생각을 먼저 쓰고, AI는 아이디어를 얻거나 다듬는 데만 도움받는다",
                           "② AI가 쓴 걸 그대로 베껴 내면 편하니까 그렇게 한다"],
                          key="ethics_q4")
            if e4.startswith("①"):
                st.success("맞아요! 숙제는 **내 생각을 키우는 연습**이에요. AI에게 통째로 맡기면 "
                           "내 실력은 하나도 자라지 않아요. AI는 **도우미**로만 써요!")
            elif e4.startswith("②"):
                st.warning("그럼 편하긴 하지만... 책을 읽고 느낀 건 나인데, 감상문엔 내 마음이 하나도 없겠죠? "
                           "게다가 정직하지 못한 행동이기도 해요.")

        with st.expander("🤥 상황 5. AI가 지어낸 이야기 (할루시네이션)"):
            st.write("AI 챗봇에게 '세종대왕이 만든 로봇 이름이 뭐야?'라고 물었더니, "
                     "**'세종로봇 1호입니다'**라고 아주 자신 있게 대답했어요. 그런 로봇은 세상에 없는데도요!")
            e5 = st.radio("왜 이런 일이 생길까요?",
                          ["선택 안 함",
                           "① AI는 모르는 것도 그럴듯하게 지어내서 말할 때가 있다",
                           "② AI가 말했으니 사실은 진짜 있었던 로봇일 것이다"],
                          key="ethics_q5")
            if e5.startswith("①"):
                st.success("맞아요! AI는 배운 글들을 바탕으로 **그럴듯한 말을 이어붙이는** 프로그램이라, "
                           "가끔 없는 사실을 진짜처럼 말해요. 중요한 내용은 꼭 **책이나 믿을 만한 곳에서 다시 확인**해요!")
            elif e5.startswith("②"):
                st.warning("아니에요! 세종대왕 시대에는 로봇이 없었죠? AI가 자신 있게 말해도 **틀린 정보**일 수 있어요.")

        with st.expander("🎭 상황 6. 진짜 같은 가짜 (딥페이크)"):
            st.write("인터넷에서 유명한 사람이 이상한 말을 하는 영상을 봤어요. 그런데 알고 보니 "
                     "AI로 얼굴과 목소리를 흉내 낸 **가짜 영상**이었대요!")
            e6 = st.radio("우리는 어떻게 해야 할까요?",
                          ["선택 안 함",
                           "① 영상도 가짜일 수 있으니, 출처를 확인하고 함부로 퍼뜨리지 않는다",
                           "② 영상으로 봤으니 무조건 진짜라고 믿고 친구들에게 알린다"],
                          key="ethics_q6")
            if e6.startswith("①"):
                st.success("맞아요! 이제는 AI로 **진짜 같은 가짜 사진·영상**을 만들 수 있어요. "
                           "누가 만들었는지 확인하고, 확실하지 않은 건 퍼뜨리지 않는 게 좋은 디지털 시민이에요. "
                           "그리고 남의 얼굴로 가짜를 만드는 건 절대 하면 안 돼요!")
            elif e6.startswith("②"):
                st.warning("조심! 눈으로 본 영상도 AI가 만든 가짜일 수 있는 시대예요. 출처를 꼭 확인해요.")

        st.write("")
        st.info("💡 **다음 단계!** 3단계 탭에서 O/X 스피드 퀴즈로 실력을 확인해봐요!")

    # [3단계: O/X 스피드 퀴즈] ---------------------------------------------
    with tab_e3:
        st.markdown("### ⚡ O/X 스피드 퀴즈!")
        st.write("배운 내용을 O/X로 빠르게 확인해봐요. 6문제 모두 맞히면 **AI 지킴이 인증**!")

        _OX_QUIZ = [
            ("AI는 항상 옳은 답만 말한다.", "X", "AI도 틀릴 수 있어요. 스스로 확인하는 습관이 필요해요."),
            ("AI에게 다양한 예시를 골고루 보여줘야 공평하게 배운다.", "O", "치우친 예시는 편견(편향)을 만들어요."),
            ("친구 사진을 허락 없이 AI에 올려도 된다.", "X", "개인정보는 꼭 허락을 받고 소중히 지켜야 해요."),
            ("AI가 자신 있게 말해도 지어낸 정보일 수 있다.", "O", "중요한 내용은 믿을 만한 곳에서 다시 확인해요."),
            ("숙제는 AI가 써준 걸 그대로 베껴 내는 게 좋다.", "X", "AI는 도우미! 내 생각을 키우는 게 진짜 공부예요."),
            ("영상으로 본 것도 AI가 만든 가짜일 수 있다.", "O", "딥페이크 시대! 출처를 확인하고 함부로 퍼뜨리지 않아요."),
        ]

        ox_correct = 0
        ox_answered = 0
        for qi, (q, ans, why) in enumerate(_OX_QUIZ):
            pick = st.radio(f"**Q{qi+1}. {q}**", ["선택 안 함", "⭕ 맞다", "❌ 아니다"],
                            horizontal=True, key=f"ethics_ox_{qi}")
            if pick != "선택 안 함":
                ox_answered += 1
                picked = "O" if pick.startswith("⭕") else "X"
                if picked == ans:
                    ox_correct += 1
                    st.success(f"정답! {why}")
                else:
                    st.warning(f"땡! 정답은 '{'⭕ 맞다' if ans == 'O' else '❌ 아니다'}'예요. {why}")

        st.divider()
        if ox_answered == len(_OX_QUIZ):
            st.progress(ox_correct / len(_OX_QUIZ))
            if ox_correct == len(_OX_QUIZ):
                st.balloons()
                play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
                md_html("""
                <div style='background:#E0F2F1; border:3px solid #00897B; border-radius:16px;
                            padding:18px; text-align:center;'>
                    <div style='font-size:40px;'>🛡️</div>
                    <div style="font-family:'Jua',sans-serif; font-size:22px; color:#00695C;">
                        6문제 모두 정답! 당신은 AI 지킴이! 🎉</div>
                </div>
                """)
            else:
                st.info(f"**{ox_correct} / {len(_OX_QUIZ)}문제** 맞혔어요! 틀린 문제의 설명을 읽고 다시 골라 만점에 도전해봐요.")
        else:
            st.caption(f"지금까지 {ox_answered} / {len(_OX_QUIZ)}문제 풀었어요. 모두 풀면 결과가 나와요!")

    # [4단계: 나의 다짐 서약서] --------------------------------------------
    with tab_e4:
        st.markdown("### ✍️ 슬기로운 AI 사용 서약서")
        st.write("아래 다짐을 하나씩 읽고, 지킬 수 있으면 체크해보세요!")

        p1 = st.checkbox("하나. AI에게 **다양하고 올바른 예시**를 보여주겠습니다.", key="ethics_pledge_1")
        p2 = st.checkbox("둘. **내 정보와 친구의 정보**를 소중히 지키겠습니다.", key="ethics_pledge_2")
        p3 = st.checkbox("셋. AI의 답을 참고하되, **스스로 생각하고 확인**하겠습니다.", key="ethics_pledge_3")
        p4 = st.checkbox("넷. AI를 **숙제 대신이 아니라 도우미**로 쓰고, 가짜를 만들거나 퍼뜨리지 않겠습니다.", key="ethics_pledge_4")

        if all([p1, p2, p3, p4]):
            st.balloons()
            play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
            _pledge_name = st.session_state.get("student_name", "").strip() or "꼬마 탐험대원"
            md_html(f"""
            <div style='background:#FFFDE7; border:3px double #00897B; border-radius:18px;
                        padding:24px; text-align:center;'>
                <div style='font-size:44px;'>🌟</div>
                <div style="font-family:'Jua',sans-serif; font-size:24px; color:#00695C;">슬기로운 AI 사용자 인증!</div>
                <div style='font-size:16px; color:#3A4A6B; margin-top:10px; line-height:1.8;'>
                    <b>{_pledge_name}</b> 님은 AI의 장점과 약점을 모두 알고<br>
                    AI를 바르고 똑똑하게 사용할 것을 서약했습니다. 🛡️
                </div>
            </div>
            """)
            st.write("")
            st.info("💡 AI 원리를 아는 것만큼, **AI를 바르게 쓰는 마음**도 중요해요. 훌륭한 탐험대원이 되었네요! 🌟")
        else:
            st.caption("네 가지를 모두 체크하면 '슬기로운 AI 사용자 인증'을 받을 수 있어요!")



# --- [복습 게임: 로봇 네오 종합 점검 미션] ---
elif menu == "🔧 로봇 네오 종합 점검 (복습)":
    all_done = len(st.session_state["badges"]) >= 6
    hero_card("🔧", "로봇 네오 종합 점검 미션",
               "6가지 인공지능 원리를 모두 떠올려 로봇 네오의 부품을 하나씩 고쳐주세요!",
               "#7E57C2")

    if not all_done:
        st.info("💡 이 복습 미션은 아무 때나 해볼 수 있어요! 6개 섬을 모두 마치고 나면 더 자신 있게 풀 수 있답니다. "
                f"(지금까지 모은 부품: {len(st.session_state['badges'])} / 6)")

    # 아직 퀴즈가 시작되지 않았거나 처음 진입한 경우: 시작 화면
    if st.session_state["review_quiz"] is None:
        md_html("""
        <div style='font-size:18px; line-height:1.9; background:#EDE7F6; border-radius:14px; padding:20px 24px;'>
        앗! 로봇 네오의 <b>부품 6개가 모두 점검이 필요한 상태</b>가 되었어요. 🛠️<br><br>
        각 부품마다 <b>실생활 이야기</b>가 하나씩 나와요. 그 이야기가 <b>어떤 인공지능 원리</b>인지
        맞히면 부품에 불이 반짝! 켜지면서 수리가 완료돼요.<br><br>
        6개 부품을 모두 고쳐서 네오를 다시 깨워볼까요?
        </div>
        """)
        st.write("")
        if st.button("🚀 점검 미션 시작하기!", key="review_start", **_STRETCH):
            start_review_game()
            st.rerun()

    # 미션 완료 화면
    elif st.session_state["review_done"]:
        total = len(st.session_state["review_quiz"])
        wrong = st.session_state["review_wrong"]
        correct_first_try = total - wrong if wrong <= total else 0

        st.balloons()
        play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)
        st.success("🎉 로봇 네오의 부품 6개를 모두 점검했어요! 네오가 다시 깨어났습니다!")

        # 점검 완료된 부품들을 이미지로 나란히 보여주기
        md_html("<div style='display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin:14px 0;'>"
                + " ".join(badge_img_tag(i, size=64, variant='sm') for i in range(6))
                + "</div>")

        # 결과 피드백 (틀린 횟수 기반)
        if wrong == 0:
            st.markdown("### 🏆 완벽해요! 한 번도 틀리지 않고 모든 원리를 정확히 기억했어요!")
            st.session_state["bonus_badges"].add("🌟 복습 미션 만점")
        elif wrong <= 2:
            st.markdown("### 👍 훌륭해요! 6가지 원리를 거의 완벽하게 기억하고 있네요.")
        else:
            st.markdown("### 💪 잘했어요! 헷갈린 원리는 해당 섬에 다시 들러 복습해봐요.")

        st.info(f"📊 이번 점검에서 **다시 시도한 횟수: {wrong}번** 이었어요.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 다시 도전하기", key="review_retry", **_STRETCH):
                start_review_game()
                st.rerun()
        with c2:
            if all_done:
                if st.button("🏆 수료증 받으러 가기", key="review_to_cert", **_STRETCH):
                    go_to("🏆 탐험 완료 (수료증)")
            else:
                if st.button("🏠 탐험 본부로", key="review_to_home", **_STRETCH):
                    go_to("🏠 탐험 본부 (홈)")

    # 진행 중 화면 (문제 풀이)
    else:
        quiz = st.session_state["review_quiz"]
        idx = st.session_state["review_idx"]
        total = len(quiz)
        q = quiz[idx]
        part_idx = q["part_idx"]
        part_name = ALL_BADGES[part_idx][3]

        # 상단 진행바 + 이미 고친 부품 표시
        st.progress(idx / total)
        fixed_row = " ".join(
            badge_img_tag(i, size=30, grayscale=(i not in st.session_state["review_fixed"]))
            for i in range(6)
        )
        md_html(f"<div style='display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px;'>{fixed_row}</div>")
        st.caption(f"부품 점검 {idx} / {total} 완료")

        st.divider()

        col_img, col_q = st.columns([1, 2.2])
        with col_img:
            md_html(f"""
            <div style="text-align:center; background:#F3E5F5; border-radius:18px; padding:16px;">
                {badge_img_tag(part_idx, size=120, variant='md')}
                <div style="font-family:'Jua',sans-serif; font-size:16px; color:#6A1B9A; margin-top:8px;">
                    ⚠️ {part_name} 점검 중
                </div>
            </div>
            """)
        with col_q:
            st.markdown(f"#### 🔍 {idx+1}번째 부품: 이건 어떤 인공지능 원리일까요?")
            md_html(f"""
            <div style='font-size:18px; line-height:1.8; background:#FFF8E1; border-radius:12px; padding:16px 20px;'>
            {q['scenario']}
            </div>
            """)

            answered_key = f"review_ans_{idx}"
            choice = st.radio(
                "알맞은 원리를 골라주세요:",
                ["(선택하세요)"] + AI_PRINCIPLES,
                key=answered_key,
            )

            with st.expander("🤔 힌트 보기"):
                st.write(q["hint"])

            if choice != "(선택하세요)":
                chosen_idx = AI_PRINCIPLES.index(choice)
                if chosen_idx == q["answer"]:
                    st.success(f"⚡ 정답! **{part_name}**에 불이 들어왔어요! 수리 완료! ✨")
                    if part_idx not in st.session_state["review_fixed"]:
                        st.session_state["review_fixed"].append(part_idx)
                    next_label = "다음 부품 점검하기 ▶️" if idx + 1 < total else "점검 마치고 결과 보기 🎉"
                    if st.button(next_label, key=f"review_next_{idx}", **_STRETCH):
                        if idx + 1 < total:
                            st.session_state["review_idx"] += 1
                        else:
                            st.session_state["review_done"] = True
                        st.rerun()
                else:
                    st.error("🔧 앗, 다른 원리예요! 힌트를 참고해서 다시 골라볼까요?")
                    wrong_key = f"review_wrongmark_{idx}_{chosen_idx}"
                    if wrong_key not in st.session_state:
                        st.session_state[wrong_key] = True
                        st.session_state["review_wrong"] += 1


# --- [새로 추가된 메뉴: 최종 섬 (수료증 발급소)] ---
elif menu == "🏆 탐험 완료 (수료증)":
    hero_card("🏆", "최종 섬 (수료증 발급소)",
               "사후 검사는 언제든지 할 수 있어요. 부품 6개를 모두 모으면 수료증도 짠! 하고 나타나요.",
               "#FFC93C")

    render_bgm_player(BGM_CERT_FILE, "cert", label="🎵 축하 음악")
    st.divider()

    # 사후 평가 (학습 효과 측정) — 부품을 다 모으지 않아도 언제든 참여할 수 있다.
    st.markdown("### 📊 사후 검사 — 얼마나 자랐는지 확인해요!")
    st.caption("💡 사후 검사는 로봇 부품을 다 모으지 않았어도 언제든지 할 수 있어요.")
    if st.session_state.get("post_score") is not None:
        pre = st.session_state.get("pre_score")
        post = st.session_state.get("post_score")
        c1, c2, c3 = st.columns(3)
        c1.metric("사전 지식점수", f"{pre if pre is not None else '-'} / 6")
        c2.metric("사후 지식점수", f"{post} / 6")
        if pre is not None:
            diff = post - pre
            c3.metric("향상", f"{diff:+d}점", delta=f"{diff:+d}")
            if diff > 0:
                st.success(f"🎉 대단해요! 탐험 전보다 **{diff}문제나 더** 맞혔어요. 인공지능 원리를 정말 잘 배웠네요!")
            elif diff == 0 and post == 6:
                st.success("🌟 처음부터 끝까지 완벽해요! 최고의 탐험대원이에요!")
            else:
                st.info("수고했어요! 헷갈리는 원리는 각 섬이나 복습 미션에서 다시 만나볼 수 있어요.")

        # 4개 영역 설문의 사전·사후 비교
        pre_sv = st.session_state.get("pre_survey", {}) or {}
        post_sv = st.session_state.get("post_survey", {}) or {}
        if post_sv:
            st.write("")
            st.markdown("**💬 나의 생각은 어떻게 달라졌을까요? (5점 만점)**")
            sc = st.columns(len(SURVEY_ITEMS))
            for c, item in zip(sc, SURVEY_ITEMS):
                k = item["key"]
                pv, qv = pre_sv.get(k), post_sv.get(k)
                if pv is not None and qv is not None:
                    c.metric(k, f"{qv}점", delta=f"{qv - pv:+d}")
                else:
                    c.metric(k, f"{qv if qv is not None else '-'}점")

        with st.expander("다시 하기 (사후 결과 초기화)"):
            if st.button("🔄 사후 검사 다시 하기", key="reset_post"):
                st.session_state["post_score"] = None
                st.session_state["post_survey"] = {}
                for i in range(len(DIAGNOSTIC_QUESTIONS)):
                    st.session_state.pop(f"diag_post_{i}", None)
                for j in range(len(SURVEY_ITEMS)):
                    st.session_state.pop(f"survey_post_{j}", None)
                st.rerun()
    else:
        if not st.session_state.get("student_name", "").strip():
            st.info("💡 '나의 이름 & 사전 평가' 방에서 먼저 이름을 적으면, 사전 결과와 비교할 수 있어요! (안 해도 사후 검사는 할 수 있어요.)")
        render_diagnostic_quiz("post", "post_score")

    st.divider()
    st.markdown("### 🎓 나만의 수료증 만들기")

    _cert_badge_count = len(st.session_state["badges"])
    if _cert_badge_count < 6:
        # ── 아직 부품을 다 모으지 못한 경우: 수료증 대신 진행 현황을 보여준다 ──
        md_html(f"""
        <div style='background:#FFF8E1; border:2px dashed #FFC93C; border-radius:16px;
                    padding:22px 24px; text-align:center;'>
            <div style='font-size:44px;'>🔒</div>
            <div style="font-family:'Jua',sans-serif; font-size:22px; color:#B8860B; margin-top:6px;">
                수료증은 로봇 부품 6개를 모두 모으면 나타나요!
            </div>
            <div style='font-size:17px; color:#3A4A6B; margin-top:8px;'>
                지금까지 모은 부품: <b>{_cert_badge_count} / 6개</b><br>
                각 섬의 게임을 클리어하면 부품을 하나씩 받을 수 있어요.
            </div>
        </div>
        """)
        st.write("")
        st.progress(_cert_badge_count / 6)

        # 아직 못 모은 부품과 그 부품을 주는 섬을 안내
        _badge_menu_map = [
            "📈 1. 마법의 선 긋기", "⛰️ 2. 보물찾기 산", "🌳 3. 스무고개 탐정",
            "🤝 4. 가장 친한 친구", "🎨 5. 비슷한 친구끼리", "🧠 6. 똑똑한 생각 주머니",
        ]
        missing = [(i, ALL_BADGES[i]) for i in range(6)
                   if ALL_BADGES[i][1] not in st.session_state["badges"]]
        if missing:
            st.markdown("#### 🧭 아직 못 모은 부품은 여기서 얻을 수 있어요")
            mcols = st.columns(min(3, len(missing)))
            for j, (i, (icon, bname, fname, part, desc)) in enumerate(missing):
                with mcols[j % len(mcols)]:
                    md_html(f"""
                    <div style='background:#F5F5F5; border-radius:12px; padding:10px 12px;
                                text-align:center; margin-bottom:6px;'>
                        <div style='font-size:26px;'>{icon}</div>
                        <div style='font-size:14px;'><b>{part}</b></div>
                        <div style='font-size:12px; color:#666;'>{_badge_menu_map[i]}</div>
                    </div>
                    """)
                    if st.button("이동하기", key=f"cert_go_missing_{i}", **_STRETCH):
                        go_to(_badge_menu_map[i])
        name = ""
    else:
        st.write("아래에 탐험대장님의 이름을 적으면, 로봇 부품 6개가 모두 새겨진 멋진 수료증이 짠! 하고 나타납니다.")
        name = st.text_input("나의 이름은?", value=st.session_state.get("student_name", ""),
                             placeholder="이름을 적어주세요 (예: 홍길동)", key="cert_name")

    if name:
        st.session_state["student_name"] = name.strip()
        save_student_record(name.strip())
        num_badges = len(st.session_state["badges"])
        num_bonus = len(st.session_state["bonus_badges"])
        now_kr = datetime.now(timezone(timedelta(hours=9)))
        today_kr = f"{now_kr.year} 년  {now_kr.month} 월  {now_kr.day} 일"

        with st.spinner("수료증을 만들고 있어요..."):
            cert_img = build_certificate_image(name, num_badges, num_bonus, today_kr)

        buf = io.BytesIO()
        cert_img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()

        st.image(cert_img, **_STRETCH)
        st.balloons()
        play_sfx(SFX_BADGE_FILE)  # 미션 성공 효과음 (key)

        st.write("")
        render_share_buttons(
            png_bytes,
            filename=f"AI탐험대_수료증_{name}.png",
            share_title=f"{name} 꼬마 탐험대장님의 꼬마 AI 탐험대 수료증",
        )
        st.caption("💡 다운로드한 이미지는 구글 클래스룸, 클래스팅, 카카오톡 등 어디에나 파일로 첨부해서 제출할 수 있어요!")

# --- [교사용: 선생님 방 (학습 분석)] ---
elif menu == "👩‍🏫 선생님 방 (학습 분석)":
    hero_card("👩‍🏫", "선생님 방 (학습 분석)",
               "학생들의 학습 데이터를 확인하고, 사전·사후 향상도를 분석할 수 있어요.",
               "#455A64")

    st.info("🔒 이 방은 선생님용이에요. 비밀번호를 입력하면 학급 전체의 학습 현황을 볼 수 있어요. "
            "기본 비밀번호는 teacher입니다.")

    pw = st.text_input("비밀번호", type="password", key="teacher_pw")
    if pw != TEACHER_PASSWORD:
        if pw:
            st.error("비밀번호가 맞지 않아요.")
        st.stop()

    st.success("환영합니다, 선생님! 👋")

    df = load_all_records()
    if df.empty:
        st.warning("아직 저장된 학습 기록이 없어요. 학생들이 사전 평가를 풀거나 수료증을 만들면 여기에 기록이 쌓입니다.")
    else:
        # 학급 코드로 걸러서, 우리 반 결과만 보이도록 한다
        if "학급코드" not in df.columns:
            df["학급코드"] = "미입력"
        all_classes = sorted(df["학급코드"].fillna("미입력").unique().tolist())
        pick = st.selectbox("📚 조회할 학급 코드를 선택하세요", ["(전체 보기)"] + all_classes, key="teacher_class_pick")
        if pick != "(전체 보기)":
            df = df[df["학급코드"] == pick]
        st.caption(f"현재 조회 중: **{pick}** · {len(df)}명")
        if df.empty:
            st.warning("이 학급의 기록이 아직 없어요.")
            st.stop()
        # 숫자형 변환 (분석용)
        num_df = df.copy()
        for col in ["사전점수", "사후점수", "향상점수", "획득배지수", "보너스배지수"]:
            if col in num_df.columns:
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce")

        st.markdown("### 📊 우리 반 요약")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("참여 학생 수", f"{len(df)}명")
        pre_mean = num_df["사전점수"].mean()
        post_mean = num_df["사후점수"].mean()
        c2.metric("사전 평균", f"{pre_mean:.1f} / 6" if pd.notna(pre_mean) else "-")
        c3.metric("사후 평균", f"{post_mean:.1f} / 6" if pd.notna(post_mean) else "-")
        if pd.notna(pre_mean) and pd.notna(post_mean):
            c4.metric("평균 향상", f"{post_mean - pre_mean:+.1f}점", delta=f"{post_mean - pre_mean:+.1f}")

        # 사전/사후 평균 비교 막대그래프
        if pd.notna(pre_mean) and pd.notna(post_mean):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["사전 평균", "사후 평균"], y=[pre_mean, post_mean],
                                 marker_color=["#90A4AE", "#2EC4B6"],
                                 text=[f"{pre_mean:.1f}", f"{post_mean:.1f}"], textposition="auto"))
            fig.update_layout(title="학급 사전·사후 평균 점수 비교", yaxis_range=[0, 6], height=238,
                              margin=dict(l=20, r=20, t=48, b=20))
            st.plotly_chart(fig, **_STRETCH)

        # ── 4개 영역 설문(5점 척도) 사전·사후 분석 ─────────────────
        st.markdown("### 💬 영역별 설문 결과 (5점 척도)")
        survey_rows = []
        for item in SURVEY_ITEMS:
            k = item["key"]
            pcol, qcol = f"{k}_사전", f"{k}_사후"
            pm = pd.to_numeric(num_df[pcol], errors="coerce").mean() if pcol in num_df.columns else float("nan")
            qm = pd.to_numeric(num_df[qcol], errors="coerce").mean() if qcol in num_df.columns else float("nan")
            survey_rows.append({
                "영역": k,
                "사전 평균": round(pm, 2) if pd.notna(pm) else None,
                "사후 평균": round(qm, 2) if pd.notna(qm) else None,
                "변화량": round(qm - pm, 2) if (pd.notna(pm) and pd.notna(qm)) else None,
            })
        survey_df = pd.DataFrame(survey_rows)

        if survey_df["사후 평균"].notna().any():
            figs = go.Figure()
            figs.add_trace(go.Bar(name="사전", x=survey_df["영역"], y=survey_df["사전 평균"],
                                  marker_color="#90A4AE",
                                  text=survey_df["사전 평균"], textposition="auto"))
            figs.add_trace(go.Bar(name="사후", x=survey_df["영역"], y=survey_df["사후 평균"],
                                  marker_color="#2D9CDB",
                                  text=survey_df["사후 평균"], textposition="auto"))
            figs.update_layout(barmode="group", title="영역별 사전·사후 평균 (5점 만점)",
                               yaxis_range=[0, 5], height=266,
                               margin=dict(l=20, r=20, t=48, b=20))
            st.plotly_chart(figs, **_STRETCH)
            st.dataframe(survey_df, **_STRETCH)
            st.caption("📄 이 표의 '사전 평균 / 사후 평균 / 변화량'을 보고서의 정량 분석 자료로 그대로 쓸 수 있어요.")
        else:
            st.info("아직 사후 설문 응답이 없어요. 학생들이 수료증 방에서 사후 검사를 마치면 여기에 표시됩니다.")

        st.markdown("### 📋 학생별 상세 기록")
        st.dataframe(df, **_STRETCH)

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        safe_pick = "전체" if pick == "(전체 보기)" else str(pick).replace(" ", "")
        st.download_button(f"⬇️ '{safe_pick}' 기록 CSV 다운로드", data=csv_bytes,
                           file_name=f"AI탐험대_학습기록_{safe_pick}.csv", mime="text/csv", **_STRETCH)
        st.caption("🔒 학생 기록은 이 앱이 실행 중인 컴퓨터의 data 폴더에만 저장돼요. "
                   "수업이 끝나면 CSV로 내려받아 보관하고, 공용 PC라면 아래에서 기록을 지워주세요.")

        with st.expander("⚠️ 기록 전체 삭제 (새 학급/새 수업 시작 시)"):
            st.caption("이 작업은 되돌릴 수 없어요. 먼저 CSV로 백업을 받아두세요.")
            confirm = st.text_input("삭제하려면 '삭제'라고 입력하세요", key="teacher_del_confirm")
            if st.button("🗑️ 모든 학습 기록 삭제", key="teacher_del_btn"):
                if confirm.strip() == "삭제":
                    try:
                        if os.path.exists(RESULTS_CSV):
                            os.remove(RESULTS_CSV)
                        st.success("모든 기록을 삭제했어요.")
                        st.rerun()
                    except Exception:
                        st.error("삭제 중 문제가 생겼어요.")
                else:
                    st.warning("'삭제'라고 정확히 입력해야 삭제됩니다.")

# 매 실행 끝에서 진행상황(배지)을 주소창에 동기화 — 태블릿 새로고침/탭 전환 대비
_sync_progress_to_url()