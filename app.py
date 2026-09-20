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
import re
import copy
import time
import logging
import inspect
import warnings
import traceback
import threading
import tempfile

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
    page_title="손끝에서 배우는 인공지능 원리 탐험대 (초등 6학년)",
    layout="wide",
    page_icon=_load_page_icon(),
    initial_sidebar_state="expanded",
)


def md_html(s):
    """들여쓰기된 여러 줄 HTML을 st.markdown으로 안전하게 렌더링.
    (들여쓰기가 있으면 마크다운이 코드블럭으로 착각해 HTML 태그가 그대로 노출되는 문제를 방지)"""
    st.markdown(textwrap.dedent(s), unsafe_allow_html=True)


def md_text(s, **kwargs):
    """여러 줄 '마크다운 글'(표·인용문 등)을 들여쓰기와 상관없이 올바르게 렌더링.
    (코드 안에서 예쁘게 들여쓴 표나 > 인용문이 마크다운 규칙상 '코드 블록'으로 바뀌어
    학생 화면에 회색 상자로 깨져 보이던 문제를 막는다)"""
    st.markdown(textwrap.dedent(s), **kwargs)


# =========================================================
# Streamlit 버전 호환 레이어
# =========================================================
# 최신 Streamlit은 width="stretch" 를, 구버전은 use_container_width=True 를 쓴다.
# 버전 번호를 짐작하는 대신, 실제로 st.button 이 width 인자를 받는지 '직접 확인'해서
# 학교 컴퓨터에 어떤 버전이 깔려 있어도 TypeError 가 나지 않게 한다.
def _get_stretch_kwargs():
    try:
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


_STRETCH = _get_stretch_kwargs()


def embed_html(html, height=0, **kwargs):
    """숨은 오디오·스크립트·공유 버튼 등 HTML 조각을 심는다.
    최신 Streamlit에서는 st.iframe 을(components.html 은 곧 사라져 노란 경고가 뜬다),
    구버전에서는 components.html 을 자동으로 골라 쓴다. 어느 쪽이 실패해도 앱은 계속 간다."""
    html = textwrap.dedent(html).strip()
    if hasattr(st, "iframe"):
        try:
            st.iframe(html, height=max(1, int(height or 0)))
            return
        except Exception:
            pass
    try:
        components.html(html, height=height, **kwargs)
    except Exception:
        pass


# =========================================================
# 🛡️ 안전장치(방어 코드) 모음
#   학생이 슬라이더를 난폭하게 움직이거나 버튼을 엉뚱한 순서로 눌러도,
#   또 태블릿 새로고침으로 세션 값이 이상해져도 앱이 멈추거나
#   빨간 오류 화면이 뜨지 않도록 하는 공통 도구들이다.
# =========================================================
# st.rerun()/st.stop() 은 Streamlit 이 화면을 다시 그리기 위해 쓰는 '신호'이지 오류가 아니다.
# 전역 방어막이 이것까지 가로채면 화면 이동이 멈추므로, 어떤 버전이든 확실히 구분해 둔다.
try:
    from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException as _CTRL_EXC
except Exception:  # 구버전 경로
    try:
        from streamlit.runtime.scriptrunner.script_runner import (RerunException as _RerunExc,
                                                                  StopException as _StopExc)
        _CTRL_EXC = (_RerunExc, _StopExc)
    except Exception:
        _CTRL_EXC = ()

_LOG = logging.getLogger("ai_explorer")


def safe_toast(msg, icon=None):
    """구석에 잠깐 떴다 사라지는 알림. 구버전(st.toast 없음)에서는 조용히 건너뛴다."""
    try:
        if icon:
            st.toast(msg, icon=icon)
        else:
            st.toast(msg)
    except Exception:
        pass


def celebrate_once(flag_key, sfx=True):
    """풍선·효과음 축하를 '한 번만' 한다.
    성공 상태에서 슬라이더를 조금만 건드려도 화면이 다시 그려지는데,
    그때마다 풍선이 또 터지면 정신없고 소리도 겹친다. flag_key 가 세션에 없을 때만
    축하하고 표시를 남긴다. 실패 상태로 돌아가면 clear_celebration() 으로 표시를 지운다."""
    if st.session_state.get(flag_key):
        return False
    st.session_state[flag_key] = True
    try:
        st.balloons()
    except Exception:
        pass
    if sfx:
        try:
            play_sfx(SFX_BADGE_FILE)
        except Exception:
            pass
    return True


def clear_celebration(*flag_keys):
    """성공 상태에서 벗어나면 축하 표시를 지워, 다음에 다시 성공할 때 또 축하할 수 있게 한다."""
    for k in flag_keys:
        st.session_state.pop(k, None)


def clamp_state(key, lo, hi, cast=None):
    """키가 있는 슬라이더/숫자입력 위젯을 만들기 '전에' 세션 값이 범위를 벗어났는지 확인한다.
    (표 줄 수가 줄어 K 의 최댓값이 작아졌는데 예전 값이 남아 있으면 Streamlit 이
    '값이 범위를 벗어났다'는 오류를 내며 멈춘다.)
    벗어났으면 키를 지워 위젯이 자기 기본값으로 다시 시작하게 한다. 값을 직접 고쳐 넣지 않는 이유는,
    기본값이 있는 위젯의 세션 값을 코드로 바꾸면 Streamlit 이 경고를 띄우기 때문이다."""
    if key not in st.session_state:
        return
    try:
        v = st.session_state[key]
        if cast is not None:
            v = cast(v)
        if not (lo <= v <= hi) or (isinstance(v, float) and not np.isfinite(v)):
            st.session_state.pop(key, None)
    except Exception:
        st.session_state.pop(key, None)


def finite_xy(df, xcol, ycol, max_rows=200):
    """표(data_editor)에서 (x, y) 숫자 쌍만 안전하게 뽑는다.
    글자·빈칸·무한대·NaN 은 버리고, 너무 많은 줄은 앞에서 max_rows 개만 쓴다."""
    try:
        x = pd.to_numeric(df[xcol], errors="coerce").astype(float)
        y = pd.to_numeric(df[ycol], errors="coerce").astype(float)
    except Exception:
        return np.array([]), np.array([])
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m].values[:max_rows], y[m].values[:max_rows]
    return x, y


def safe_polyfit_line(x, y):
    """np.polyfit 을 감싼 안전 버전. (점이 모두 같은 x 위에 있거나 값이 너무 크면
    계산이 실패하거나 경고가 난다.) 실패하면 None 을 돌려준다."""
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2 or len(np.unique(x)) < 2:
            return None
        if np.max(np.abs(x)) > 1e9 or np.max(np.abs(y)) > 1e9:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, b = np.polyfit(x, y, 1)
        if not (np.isfinite(w) and np.isfinite(b)):
            return None
        return float(w), float(b)
    except Exception:
        return None


def sanitize_filename(name, fallback="탐험대장"):
    """파일 이름에 쓸 수 없는 문자(/ \\ : * ? " < > |)를 지운다."""
    try:
        s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "", str(name)).strip()
        return s[:30] or fallback
    except Exception:
        return fallback


# 섬(메뉴)마다 '이 섬에서만 쓰는' 세션 키의 접두어. 오류 복구·다시 하기 버튼이 이 목록만 지운다.
# (배지·이름·사전/사후 검사 결과처럼 소중한 값은 절대 여기에 넣지 않는다.)
SECTION_STATE_PREFIXES = {
    "🤖 0. 내가 AI 선생님": ["neo_"],
    "📈 1. 마법의 선 긋기": ["lr_", "fb_", "free_", "quiz_reg", "rw_linear", "_celeb_reg"],
    "⛰️ 2. 보물찾기 산": ["gd_", "quiz_gd", "rw_gradient", "_celeb_gd"],
    "🌳 3. 스무고개 탐정": ["dt_", "leaf1", "leaf2", "ani1", "ani2", "gini_q_choice",
                          "quiz_dt", "rw_tree", "_celeb_dt"],
    "🤝 4. 가장 친한 친구": ["knn_", "quiz_knn", "rw_knn"],
    "🎨 5. 비슷한 친구끼리": ["km_", "quiz_km", "rw_kmeans"],
    "🧠 6. 똑똑한 생각 주머니": ["nn_", "quiz_nn", "rw_ann", "_celeb_nn"],
    "⚖️ AI를 똑똑하게 쓰려면?": ["ethics_", "_celeb_ethics"],
    "🔧 로봇 네오 종합 점검": ["review_", "_celeb_review"],
    "🏆 사후평가 (수료증)": ["cert_", "_celeb_cert"],
    "🌟 인공지능이 뭐예요?": ["ai_learn_examples", "sup_", "unsup_demo", "life_ai_quiz"],
}


def reset_section_state(menu):
    """해당 섬의 위젯·게임 상태만 깨끗이 지운다 (배지·이름·검사 결과는 그대로)."""
    prefixes = SECTION_STATE_PREFIXES.get(menu, [])
    if not prefixes:
        return
    for k in list(st.session_state.keys()):
        if any(str(k).startswith(p) for p in prefixes):
            st.session_state.pop(k, None)
    # 게임 데이터의 기본값도 되살린다
    for k in list(_GAME_DEFAULTS.keys()):
        if any(k.startswith(p) for p in prefixes):
            st.session_state[k] = copy.deepcopy(_GAME_DEFAULTS[k])


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
    "reg": "#2D9CDB",      # 1. 선 긋기 - 하늘색
    "gd": "#4FC3F7",       # 2. 한 걸음씩 가기 - 눈 덮인 얼음색
    "dt": "#43A047",       # 3. 스무고개 - 숲 초록
    "knn": "#8E24AA",      # 4. 이웃에게 물어보기 - 외계 보라
    "km": "#2EC4B6",       # 5. 비슷한 것끼리 모으기 - 무인도 민트
    "nn": "#FF6F59",       # 6. 신경망 - 로켓 코랄
}

# =========================================================
# 🈳 브라우저 '자동 번역' 차단
# =========================================================
# 크롬·엣지는 페이지 언어를 스스로 알아맞힌다. 스트림릿이 만드는 껍데기 화면은
# lang="en" 으로 되어 있어서, 브라우저가 이 화면을 '영어 쪽'으로 잘못 보고
# 한국어 글자를 한 번 더 번역해 버린다. 그러면
#   '탐험 완료' → '탐구심',  '저작권' → '제작자'  처럼 글자가 망가진다.
#
# 심사위원·참관자 컴퓨터의 설정을 우리가 정할 수는 없으므로,
# **페이지 쪽에서** "여기는 한국어이고 번역하지 말라"고 못을 박아 둔다.
#   ① <html lang="ko">          → 언어를 한국어로 명시
#   ② translate="no" / .notranslate → 번역 금지 (자식 요소까지 모두 상속된다)
#   ③ <meta name="google" content="notranslate"> → 구글 번역기 차단
# 화면이 새로 그려져도 유지되도록 잠깐 동안 여러 번 다시 표시한다.
#
# ⚠️ 이 조각은 '항상' 그려야 한다. 있다가 없어지면 아래 요소들의 자리가 밀려
#    보고 있던 탭이 ①번으로 튕긴다.
embed_html(
    """
    <script>
    (function () {
        function noTranslate() {
            try {
                var d = window.parent.document;
                var h = d.documentElement;
                if (!h) { return; }
                h.setAttribute('lang', 'ko');
                h.setAttribute('xml:lang', 'ko');
                h.setAttribute('translate', 'no');
                h.classList.add('notranslate');
                if (d.body) {
                    d.body.setAttribute('translate', 'no');
                    d.body.classList.add('notranslate');
                }
                if (!d.querySelector('meta[name="google"][content="notranslate"]')) {
                    var m = d.createElement('meta');
                    m.setAttribute('name', 'google');
                    m.setAttribute('content', 'notranslate');
                    (d.head || h).appendChild(m);
                }
                if (!d.querySelector('meta[http-equiv="content-language"]')) {
                    var m2 = d.createElement('meta');
                    m2.setAttribute('http-equiv', 'content-language');
                    m2.setAttribute('content', 'ko');
                    (d.head || h).appendChild(m2);
                }
                var root = d.querySelector('[data-testid="stAppViewContainer"]');
                if (root) {
                    root.setAttribute('translate', 'no');
                    root.classList.add('notranslate');
                }
            } catch (e) {}
        }
        noTranslate();
        // 스트림릿이 화면을 다시 그린 뒤에도 표시가 남아 있도록 몇 번 더
        [50, 200, 600, 1500, 3000].forEach(function (t) { setTimeout(noTranslate, t); });
    })();
    </script>
    """,
    height=0,
)

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

/* ── 사이드바 폭: 메뉴 글자가 잘리지도, 두 줄로 접히지도 않는 폭(248px)으로 고정 ──
   기본값 336px보다는 훨씬 좁아 본문(탐험 화면)이 넓어지고,
   가장 긴 메뉴("👩‍🏫 선생님 방 (학습 분석)")까지 잘리지 않고 한 줄에 들어간다.
   Streamlit 버전마다 폭을 잡는 요소가 달라서 여러 선택자를 함께 지정한다. */
:root, .stApp { --sidebar-width: 248px !important; }
section[data-testid="stSidebar"],
div[data-testid="stSidebar"] {
    width: 248px !important;
    min-width: 248px !important;
    max-width: 248px !important;
    flex: 0 0 248px !important;
}
div[data-testid="stSidebarContent"] {
    width: 248px !important;
    min-width: 248px !important;
    max-width: 248px !important;
}
/* ── ★ 폭 조절 손잡이 레이어 제거 ──
   이 레이어는 사이드바 전체를 덮는 투명 div라서, 폭을 건드리면
   메뉴 클릭과 스크롤을 통째로 가로챈다. 아예 없애 버린다. */
section[data-testid="stSidebar"] > div:not([data-testid="stSidebarContent"]) { display: none !important; }
[data-testid="stSidebarResizeHandle"] { display: none !important; }

/* ── ★ 잘림 방지: 사이드바 '안쪽' 컨테이너들도 모두 줄어든 폭을 따라가게 강제 ──
   사이드바 박스만 좁아지고 내용물 컨테이너가 예전 폭(336px)을 유지하면
   진행바·버튼·안내글이 오른쪽으로 삐져나가 잘려 보인다. 이를 막는다. */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="stElementContainer"],
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] [data-testid="stColumn"],
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .element-container {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}
/* 사이드바 밖으로 넘치기 쉬운 요소들만 콕 집어 제한한다.
   (예전처럼 * 전체에 걸면 내부 동작까지 건드릴 수 있어 위험하다.) */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] img,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { max-width: 100% !important; }

/* ── 스크롤·터치가 정상 동작하도록 명시적으로 보장 ── */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    -webkit-overflow-scrolling: touch !important;
    height: 100% !important;
}
section[data-testid="stSidebar"] { pointer-events: auto !important; }
section[data-testid="stSidebar"] .stRadio label { cursor: pointer !important; touch-action: manipulation !important; }

/* 진행바(로봇 조립 현황)를 사이드바 폭 안에 맞추고 살짝 얇게 */
section[data-testid="stSidebar"] .stProgress,
section[data-testid="stSidebar"] .stProgress > div,
section[data-testid="stSidebar"] .stProgress > div > div,
section[data-testid="stSidebar"] .stProgress > div > div > div {
    width: 100% !important; max-width: 100% !important; min-width: 0 !important;
}
section[data-testid="stSidebar"] .stProgress > div > div { height: 9px !important; border-radius: 999px !important; }

/* 버튼 글자가 잘리지 않도록: 폭에 맞춰 줄바꿈 허용 + 글자 축소 */
section[data-testid="stSidebar"] .stButton { width: 100% !important; }
section[data-testid="stSidebar"] .stButton>button {
    width: 100% !important; min-width: 0 !important;
    white-space: normal !important; word-break: keep-all;
    line-height: 1.25 !important; padding: 0.45em 0.4em !important;
}
section[data-testid="stSidebar"] .stButton>button p { font-size: 0.72rem !important; white-space: normal !important; }

/* 안내 문구(캡션)는 작게 + 줄바꿈해서 절대 잘리지 않게 */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    white-space: pre-line !important;
    word-break: keep-all !important;
    overflow-wrap: break-word !important;
    line-height: 1.35 !important;
}
/* 로봇 부품 아이콘 줄이 넘치지 않게 */
section[data-testid="stSidebar"] img { max-width: 100% !important; height: auto !important; }

/* 좁은 사이드바에서는 2칸 배치(초기화/취소 버튼)를 세로로 쌓아 글자가 안 잘리게 */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 4px !important; }
section[data-testid="stSidebar"] [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; }

/* 알림(경고) 문구도 폭 안에서 줄바꿈 */
section[data-testid="stSidebar"] .stAlert { padding: 8px 10px !important; }
section[data-testid="stSidebar"] .stAlert p {
    font-size: 0.7rem !important; line-height: 1.35 !important;
    white-space: pre-line !important; word-break: keep-all !important;
}

/* 밝은 배경 안내 카드의 글자는 흰색 강제 규칙에서 빼서 잘 보이게 한다 */
section[data-testid="stSidebar"] .review-tip,
section[data-testid="stSidebar"] .review-tip * { color: #4527A0 !important; }

/* 안내 카드가 바로 아래 버튼과 맞닿아(겹쳐) 보이던 문제를 막는다.
   카드에 아래 여백을 주고, 사이드바 버튼에도 위 여백을 줘서 확실히 떨어뜨린다. */
section[data-testid="stSidebar"] .review-tip {
    display: block !important;
    margin: 4px 0 14px 0 !important;
    overflow: hidden !important;
}
section[data-testid="stSidebar"] .stButton { margin-top: 6px !important; }

/* ══ 카드끼리, 또는 카드와 바로 아래 글자·버튼이 겹쳐 보이던 문제 ══
   직접 그린 안내 카드(<div style=…>)에 아래 여백이 없어서
   바로 다음 요소가 달라붙어 겹친 것처럼 보였다.
   마크다운으로 그려진 '맨 바깥 div' 에만 아래 여백을 준다.
   (카드 스스로 margin 을 적어 둔 곳은 그 값이 우선이라 그대로 유지된다) */
[data-testid="stMarkdownContainer"] > div { margin-bottom: 12px; }
/* 사이드바는 폭이 좁아 여백을 작게 (안내 카드 전용 규칙은 위에서 따로 지정) */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div { margin-bottom: 5px; }
/* 선택지·버튼이 바로 위 카드에 닿지 않도록 살짝 내려 준다 */
[data-testid="stAppViewContainer"] .stRadio { margin-top: 4px; }
[data-testid="stAppViewContainer"] .stButton { margin-top: 4px; }
section[data-testid="stSidebar"] .stRadio { margin-top: 0; }

/* ── 좁아진 폭에 맞춰 사이드바 글자 크기 조정 ── */
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] { padding: 0.5rem 0.5rem 2rem 0.5rem !important; }
section[data-testid="stSidebar"] h1 { font-size: 1.0rem !important; line-height: 1.25 !important; margin-bottom: 0.15rem !important; }
section[data-testid="stSidebar"] h2 { font-size: 0.9rem !important; }
section[data-testid="stSidebar"] h3 { font-size: 0.84rem !important; margin: 0.2rem 0 !important; }
section[data-testid="stSidebar"] h4 { font-size: 0.8rem !important; margin: 0.2rem 0 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span { font-size: 0.74rem !important; line-height: 1.35 !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: 0.68rem !important; }

/* ── 메뉴(라디오)는 무슨 일이 있어도 한 줄로 ──
   nowrap으로 줄바꿈 자체를 막고, 혹시 넘칠 경우에만 …으로 처리해 모양이 깨지지 않게 한다. */
section[data-testid="stSidebar"] .stRadio > div { gap: 3px !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 6px 6px !important;
    border-radius: 10px !important;
    font-size: 0.72rem !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    letter-spacing: -0.4px;
}
section[data-testid="stSidebar"] .stRadio label > div:last-child { min-width: 0 !important; overflow: hidden !important; }
section[data-testid="stSidebar"] .stRadio label div[data-testid="stMarkdownContainer"] { overflow: hidden !important; }
section[data-testid="stSidebar"] .stRadio label p {
    font-size: 0.72rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    letter-spacing: -0.4px;
}
/* 라디오 동그라미를 살짝 줄여 글자 공간을 벌어준다 */
section[data-testid="stSidebar"] .stRadio label > div:first-child { flex: 0 0 auto !important; transform: scale(0.82); transform-origin: center; margin-right: -2px; }

section[data-testid="stSidebar"] .stButton>button { padding: 0.4em 0.6em !important; font-size: 0.72rem !important; }
section[data-testid="stSidebar"] hr { margin: 0.5rem 0 !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }

/* ── 사이드바 접기 버튼에 'keyboard_double_arrow_left' 글자가 그대로 보이는 문제 해결 ──
   (아이콘 폰트가 늦게/못 불러와질 때 리거처 글자가 노출된다. 글자는 감추고 화살표 기호로 대체) */
[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] span[data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"],
[data-testid="stSidebarHeader"] span[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] span.material-symbols-rounded,
[data-testid="stSidebarCollapsedControl"] span.material-symbols-rounded,
[data-testid="stExpandSidebarButton"] span.material-symbols-rounded {
    font-size: 0 !important;
    color: transparent !important;
    line-height: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"]::after,
[data-testid="stSidebarHeader"] span[data-testid="stIconMaterial"]::after,
[data-testid="stSidebarCollapseButton"] span.material-symbols-rounded::after {
    content: "«";
    font-family: 'Gowun Dodum', sans-serif !important;
    font-size: 18px !important;
    line-height: 1 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebarCollapsedControl"] span[data-testid="stIconMaterial"]::after,
[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"]::after,
[data-testid="stSidebarCollapsedControl"] span.material-symbols-rounded::after,
[data-testid="stExpandSidebarButton"] span.material-symbols-rounded::after {
    content: "»";
    font-family: 'Gowun Dodum', sans-serif !important;
    font-size: 18px !important;
    line-height: 1 !important;
    color: #1B2A4A !important;
}

/* ── 상단 흰색 바(Fork / ⋮ / 배포 버튼 등) 숨기기 ──
   높이를 0으로 만들어 화면 위쪽 공간까지 되찾는다.
   단, 사이드바를 다시 펼치는 버튼은 남겨 둔다. */
header[data-testid="stHeader"],
header[data-testid="stAppHeader"],
[data-testid="stHeader"],
[data-testid="stAppHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stToolbar"],
[data-testid="stAppToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stToolbarActionButton"],
[data-testid="stHeaderActionElements"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
.stAppDeployButton,
#MainMenu,
footer,
[data-testid="stAppViewerBadge"],
[class*="viewerBadge"],
[data-testid="manage-app-button"] { display: none !important; visibility: hidden !important; }

/* ── 화면 오른쪽 아래 '앱 관리(Manage app)' 단추와 제작자 배지 숨기기 ──
   이 단추는 본래 **저장소 권한이 있는 사람이 로그인했을 때만** 보인다.
   (심사위원·학생에게는 원래 안 보인다.) 그래도 혹시 모르니 이름이 다른
   여러 판(version)을 한꺼번에 가려 둔다. 위쪽 툴바의 '메뉴 열기' 버튼은
   건드리지 않도록, 아래쪽 배지에 해당하는 것만 골라서 지정한다. */
[data-testid="stAppViewerBadge"],
[data-testid="stViewerBadge"],
[data-testid="manageAppButton"],
[class*="viewerBadge_container"],
[class*="viewerBadge_link"],
[class*="viewerBadge_text"],
[class*="profileContainer"],
[class*="stAppDeployButton"],
a[href*="streamlit.io/cloud"],
a[href*="share.streamlit.io"],
iframe[title="streamlitApp"] ~ div[class*="badge"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* 헤더는 높이 0의 '투명한 껍데기'로만 남기고, 터치를 절대 가로채지 못하게 한다.
   (이 처리를 빼먹으면 화면 위쪽을 덮어 클릭·스크롤이 먹히지 않는다.) */
header[data-testid="stHeader"],
header[data-testid="stAppHeader"],
[data-testid="stHeader"],
[data-testid="stAppHeader"] { pointer-events: none !important; }
header[data-testid="stHeader"] button,
header[data-testid="stAppHeader"] button,
[data-testid="stHeader"] button,
[data-testid="stAppHeader"] button { pointer-events: auto !important; }
/* ══════════════════════════════════════════════════════════════
   ★★ 태블릿에서 사이드바가 접히면 다시 못 여는 문제 해결 ★★

   스트림릿은 화면이 좁으면(태블릿·휴대전화) 사이드바를 스스로 접는다.
   그때 다시 여는 버튼은 [data-testid="stExpandSidebarButton"] 인데,
   이 버튼이 하필 **[data-testid="stToolbar"] 안에** 들어 있다.
   위에서 툴바를 통째로 숨겨 버렸기 때문에 **다시 열 방법이 사라졌다.**

   그래서 ① 툴바는 살리되(단, 빈 곳이 터치를 가로채지 않게)
        ② 그 안의 다른 것들(⋮ 메뉴·배포 버튼 등)은 계속 숨기고
        ③ '메뉴 열기' 버튼만 크고 눈에 띄게 되살린다.
   태블릿은 마우스를 올릴 수 없으므로(hover 없음) 항상 보이게 해야 한다.
   ══════════════════════════════════════════════════════════════ */
[data-testid="stToolbar"],
[data-testid="stAppToolbar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
    box-shadow: none !important;
    pointer-events: none !important;   /* 빈 영역은 터치를 통과시킨다 */
}
/* 툴바 안에서 '메뉴 열기'를 뺀 나머지는 계속 숨긴다 */
[data-testid="stToolbar"] [data-testid="stToolbarActions"],
[data-testid="stToolbar"] [data-testid="stMainMenu"],
[data-testid="stToolbar"] [data-testid="stAppDeployButton"],
[data-testid="stToolbar"] [data-testid="stStatusWidget"],
[data-testid="stToolbar"] [data-testid="stHeaderActionElements"],
[data-testid="stToolbar"] #MainMenu { display: none !important; }

/* ── '☰ 메뉴' 버튼: 항상 보이고, 손가락으로 누르기 좋은 크기(56px) ── */
[data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    width: 56px !important;
    height: 56px !important;
    min-width: 56px !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 16px !important;
    background: #FF6F59 !important;
    box-shadow: 0 3px 10px rgba(27,42,74,.32) !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 1000001 !important;
    touch-action: manipulation !important;
}
/* 아이콘 글꼴이 늦게 뜨면 'keyboard_double_arrow_right' 글자가 그대로 보인다.
   안쪽 글자는 완전히 지우고, 버튼 자체에 ☰ 를 그린다. */
[data-testid="stExpandSidebarButton"] span,
[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] span.material-symbols-rounded {
    font-size: 0 !important;
    color: transparent !important;
    line-height: 0 !important;
}
[data-testid="stExpandSidebarButton"] span::after,
[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"]::after,
[data-testid="stExpandSidebarButton"] span.material-symbols-rounded::after { content: none !important; }
[data-testid="stExpandSidebarButton"]::after {
    content: "☰";
    font-family: 'Gowun Dodum', sans-serif !important;
    font-size: 27px !important;
    line-height: 1 !important;
    color: #FFFFFF !important;
}
/* 눌렀을 때 살짝 들어가는 느낌 */
[data-testid="stExpandSidebarButton"]:active { transform: translateY(2px) !important; }

/* ── 접기(«) 버튼도 항상 보이게 ──
   스트림릿 기본값은 '마우스를 올렸을 때만' 보이는데(visibility:hidden),
   태블릿에는 마우스가 없어서 영영 안 보인다. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* ══ 아이들이 손대고 싶어지게: 버튼·탭·제목 손보기 ══ */

/* 본문 제목에 둥근 글꼴(Jua)을 써서 딱딱함을 덜어준다 */
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4 {
    font-family: 'Jua', sans-serif !important;
    letter-spacing: -0.3px;
}

/* 버튼: 크고 둥글게, 누르면 살짝 눌리는 느낌 */
[data-testid="stMain"] .stButton > button {
    font-family: 'Jua', sans-serif !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 0.6em 1.1em !important;
    box-shadow: 0 3px 0 rgba(0,0,0,0.13) !important;
    transition: transform .07s ease, box-shadow .07s ease !important;
}
[data-testid="stMain"] .stButton > button:hover { transform: translateY(-1px); }
[data-testid="stMain"] .stButton > button:active {
    transform: translateY(2px) !important;
    box-shadow: 0 1px 0 rgba(0,0,0,0.13) !important;
}
/* 핵심 행동 버튼(게임 도전·가르치기 등)은 더 크고 눈에 띄게 */
[data-testid="stMain"] .stButton > button[kind="primary"] {
    font-size: 1.05rem !important;
    padding: 0.75em 1.2em !important;
}

/* 탭도 둥근 글꼴 + 선택된 탭이 확실히 보이게 */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Jua', sans-serif !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 6px 12px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #2D9CDB !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {
    color: #FFFFFF !important;
}
/* 탭 아래 빨간 밑줄은 숨긴다 (선택된 탭 자체가 파랗게 채워지므로) */
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

/* 슬라이더 손잡이를 크게 — 태블릿에서 손가락으로 잡기 쉽게 */
[data-testid="stMain"] [data-testid="stSlider"] [role="slider"] {
    width: 22px !important; height: 22px !important;
}

/* 성공·경고·오류 메시지를 카드처럼 또렷하게 */
[data-testid="stMain"] .stAlert {
    border-radius: 14px !important;
    border-left-width: 6px !important;
}

/* ── 0번 섬의 '단계 고르기' 버튼을 탭처럼 보이게 ── */
[data-testid="stMain"] .stRadio [role="radiogroup"] { gap: 6px !important; flex-wrap: wrap !important; }
[data-testid="stMain"] div[data-testid="stElementContainer"]:has(.stRadio) [role="radiogroup"] label {
    background: #EEF4FC; border: 2px solid #D3E2F5; border-radius: 12px;
    padding: 7px 13px !important; cursor: pointer;
}
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked) {
    background: #2D9CDB !important; border-color: #1B6FA8 !important;
}
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked) p { color: #FFFFFF !important; }

/* ── 탭이 화면 밖으로 밀려 잘리지 않고 줄바꿈되게 한다 ──
   태블릿에서 마지막 단계 탭이 안 보여 학생·심사위원이 놓치는 것을 막는다. */
[data-testid="stTabs"] [role="tablist"] {
    flex-wrap: wrap !important;
    gap: 2px 6px !important;
    overflow-x: visible !important;
}
[data-testid="stTabs"] [role="tablist"] button { white-space: nowrap !important; }

/* ── 화면을 컴팩트하게: 여백을 줄여 한 화면에 더 많이 담기게 한다 ── */
.block-container { padding-top: 0.8rem !important; padding-bottom: 1.5rem !important; }

/* ── 각 탭 제목 글자 크기 축소 (태블릿에서 스크롤을 줄이기 위함) ── */
.block-container h1 { font-size: 1.5rem !important; }
.block-container h2 { font-size: 1.22rem !important; }
.block-container h3 { font-size: 1.05rem !important; }
.block-container h4 { font-size: 0.95rem !important; }
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
    /* 사이드바는 좁게 유지하되, 터치 영역만 '세로로' 넉넉하게 (가로 여백은 그대로 → 한 줄 유지) */
    section[data-testid="stSidebar"] .stRadio label { padding: 9px 6px !important; }
    section[data-testid="stSidebar"] .stButton>button { font-size: 0.72rem !important; padding: 0.5em 0.6em !important; }
}
/* =========================================================
   ✨ 아이들 눈높이 UI 부품 (고도화 추가분)
   ========================================================= */
/* 본문 기본 글자를 조금 키우고 줄 간격을 넉넉하게 — 초등학생 가독성 */
[data-testid="stMain"] p, [data-testid="stMain"] li { font-size: 15.5px; line-height: 1.7; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { font-size: 13px; line-height: 1.55; }
[data-testid="stMain"] .stRadio label p, [data-testid="stMain"] .stCheckbox label p { font-size: 15px; }
[data-testid="stMain"] [data-testid="stSlider"] label p { font-size: 15px; font-family: 'Jua', sans-serif; }

/* 🤖 네오의 말풍선 */
.neo-bubble { display:flex; align-items:flex-start; gap:10px; margin:6px 0 10px 0; }
.neo-bubble .neo-face { font-size:34px; line-height:1; flex:0 0 auto; margin-top:2px; }
.neo-bubble .neo-face img { height:46px; width:auto; }
.neo-bubble .neo-text {
    position:relative; background:#FFFFFF; border:2px solid #CFE3F7; border-radius:16px;
    padding:10px 14px; font-family:'Gowun Dodum', sans-serif; font-size:15px; color:#1B2A4A;
    line-height:1.7; box-shadow:0 2px 8px rgba(27,42,74,0.07);
}
.neo-bubble .neo-text::before {
    content:""; position:absolute; left:-10px; top:14px; border:8px solid transparent;
    border-right-color:#CFE3F7; border-left:0;
}
.neo-bubble.happy .neo-text { background:#F1FBF4; border-color:#A5D6A7; }
.neo-bubble.happy .neo-text::before { border-right-color:#A5D6A7; }
.neo-bubble.think .neo-text { background:#FFF8E1; border-color:#FFE082; }
.neo-bubble.think .neo-text::before { border-right-color:#FFE082; }
.neo-bubble.oops .neo-text { background:#FDECEF; border-color:#F8BBD0; }
.neo-bubble.oops .neo-text::before { border-right-color:#F8BBD0; }

/* 🏝️ 탐험 지도(홈)의 섬 카드 */
.island-card {
    border-radius:18px; padding:12px 8px 10px 8px; text-align:center; border:2px solid;
    background:#FFFFFF; transition: transform .12s ease, box-shadow .12s ease; min-height:150px;
    box-shadow:0 2px 8px rgba(27,42,74,0.06);
}
.island-card:hover { transform: translateY(-3px); box-shadow:0 8px 18px rgba(27,42,74,0.14); }
.island-card .island-emoji { font-size:34px; line-height:1; }
.island-card .island-title { font-family:'Jua',sans-serif; font-size:16px; margin-top:6px; line-height:1.2; }
.island-card .island-desc { font-family:'Gowun Dodum',sans-serif; font-size:12px; color:#3A4A6B; margin-top:4px; line-height:1.4; }
.island-card .island-badge {
    display:inline-block; margin-top:7px; border-radius:999px; padding:2px 10px;
    font-family:'Jua',sans-serif; font-size:12px; color:#fff;
}
.island-card.done { background:linear-gradient(180deg,#FFFFFF,#F1FBF4); }

/* 🔤 이름 풀이 카드 (어려운 AI 용어를 낱말 단위로 풀어 준다) */
.name-decoder {
    margin-top:10px; background:#FFFFFFAA; border:1.5px dashed #9FB3C8; border-radius:12px;
    padding:8px 12px; font-family:'Gowun Dodum',sans-serif; font-size:14px; color:#28406B; line-height:1.7;
}
.name-decoder b { font-family:'Jua',sans-serif; color:#1B2A4A; }
.name-decoder .chip {
    display:inline-block; background:#E3F2FD; border-radius:8px; padding:0 8px; margin:0 2px;
    font-family:'Jua',sans-serif; color:#1565C0;
}

/* 🎯 오늘의 미션 카드 */
.mission-card {
    display:flex; gap:12px; align-items:center; background:linear-gradient(135deg,#FFF8E1,#FFFFFF);
    border:2px solid #FFC93C; border-radius:16px; padding:12px 16px; margin:4px 0 10px 0;
}
.mission-card .m-icon { font-size:32px; line-height:1; }
.mission-card .m-title { font-family:'Jua',sans-serif; font-size:15px; color:#B26A00; }
.mission-card .m-body { font-family:'Gowun Dodum',sans-serif; font-size:15.5px; color:#1B2A4A; line-height:1.6; }

/* 알림 토스트를 둥글고 또렷하게 */
[data-testid="stToast"] { border-radius:14px !important; font-family:'Gowun Dodum', sans-serif !important; }

/* 표(data_editor/dataframe)도 카드처럼 */
[data-testid="stDataFrame"], [data-testid="stDataEditor"], [data-testid="stTable"] {
    border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(27,42,74,0.06);
}

/* 눌러 달라고 손짓하는 핵심 버튼(primary): 은은한 두근두근 효과 */
@keyframes neo-pulse {
  0%   { box-shadow:0 3px 0 rgba(0,0,0,0.13), 0 0 0 0 rgba(255,111,89,0.45); }
  70%  { box-shadow:0 3px 0 rgba(0,0,0,0.13), 0 0 0 12px rgba(255,111,89,0); }
  100% { box-shadow:0 3px 0 rgba(0,0,0,0.13), 0 0 0 0 rgba(255,111,89,0); }
}
[data-testid="stMain"] .stButton > button[kind="primary"]:not(:disabled) { animation: neo-pulse 2.2s ease-out infinite; }
</style>
""")


def neo_says(text, mood="normal"):
    """🤖 로봇 네오가 말풍선으로 안내한다. mood: normal / happy / think / oops
    (같은 말도 마스코트가 하면 아이들이 훨씬 잘 읽는다.)"""
    face = {"normal": "🤖", "happy": "🤖✨", "think": "🤔", "oops": "😵"}.get(mood, "🤖")
    b64 = _asset_img_b64("robot_mascot.png") if "_asset_img_b64" in globals() else None
    face_html = (f'<img src="data:image/png;base64,{b64}" alt="네오">'
                 if (b64 and mood == "normal") else face)
    md_html(f"""
    <div class="neo-bubble {mood}">
      <div class="neo-face">{face_html}</div>
      <div class="neo-text">{text}</div>
    </div>
    """)


def mission_card(body, title="🎯 오늘의 미션 — 이것만 하면 돼요!"):
    """섬마다 '무엇을 하면 되는지'를 한 문장으로 또렷하게 보여주는 카드."""
    md_html(f"""
    <div class="mission-card">
      <div class="m-icon">🎯</div>
      <div><div class="m-title">{title}</div><div class="m-body">{body}</div></div>
    </div>
    """)


def name_decoder(parts, meaning):
    """어려운 용어를 낱말 단위로 풀어주는 '이름 풀이' 카드 HTML을 돌려준다.
    parts: [(낱말, 뜻), ...]  meaning: 합쳐서 무슨 뜻인지 한 줄"""
    chips = " + ".join(f"<span class='chip'>{w}</span>{m}" for w, m in parts)
    return (f"<div class='name-decoder'>🔤 <b>이름 풀이</b> &nbsp; {chips}"
            f"<br>➡️ {meaning}</div>")


def hero_card(emoji, title, subtitle, color="#2D9CDB", term=None):
    """섹션 상단에 표시되는 카드형 히어로 헤더"""
    term_html = ""
    if term:
        term_html = (f'<div style="display:inline-block; margin-top:7px; background:rgba(255,255,255,0.28); '
                     f'border:2px solid rgba(255,255,255,0.75); border-radius:999px; padding:3px 12px; '
                     f'font-family:\'Jua\', sans-serif; font-size:13px; color:white;">'
                     f'🏷️ 오늘 해볼 일: <b>{term}</b></div>')
    # 이모지와 제목을 한 줄에 나란히 배치해 카드 높이를 크게 줄인다.
    card_html = (f'<div style="background: linear-gradient(135deg, {color}, {color}CC); '
                 f'border-radius: 16px; padding: 12px 16px; margin-bottom: 10px; '
                 f'box-shadow: 0 4px 12px rgba(27,42,74,0.14);">'
                 f'<div style="display:flex; align-items:center; gap:9px;">'
                 f'<span style="font-size: 24px; line-height: 1;">{emoji}</span>'
                 f'<span style="font-family: \'Jua\', sans-serif; font-size: 19px; color: white; line-height: 1.25;">{title}</span>'
                 f'</div>'
                 f'<div style="font-family: \'Gowun Dodum\', sans-serif; font-size: 12.5px; color: white; opacity: 0.92; margin-top: 4px; line-height: 1.45;">{subtitle}</div>'
                 f'{term_html}'
                 f'</div>')
    st.markdown(card_html, unsafe_allow_html=True)


def real_world(key, items):
    """'이 원리가 진짜로 어디에 쓰이는지'를 눌러서 확인하는 짧은 활동.
    설명글을 길게 붙이면 아이들이 읽지 않으므로, 궁금한 것만 눌러 보게 한다.
    (탭 위쪽이 너무 길어져 정작 활동을 못 찾는 일이 없도록, 접힌 상자 안에 둔다.)"""
    with st.expander("🌍 이 원리, 우리 생활 어디에 숨어 있을까요? (궁금하면 눌러보세요)", expanded=False):
        st.caption("👆 아래 중 궁금한 것을 눌러보세요!")
        cols = st.columns(len(items))
        for i, (emoji, name, _desc) in enumerate(items):
            with cols[i]:
                if st.button(f"{emoji} {name}", key=f"rw_{key}_{i}", **_STRETCH):
                    st.session_state[f"rw_{key}"] = i
        picked = st.session_state.get(f"rw_{key}")
        if isinstance(picked, int) and 0 <= picked < len(items):
            e, n, d = items[picked]
            md_html(f"<div style='background:#E8F5E9; border-left:6px solid #43A047; "
                    f"border-radius:12px; padding:12px 16px; margin-top:8px; "
                    f"font-family:\"Gowun Dodum\", sans-serif; font-size:15px; "
                    f"color:#1B3A24; line-height:1.75;'>"
                    f"<b style=\"font-family:'Jua',sans-serif; font-size:17px;\">{e} {n}</b><br>{d}</div>")


def level_guide(has_free=True, minutes=15):
    """섬마다 '어디까지가 기본이고 어디부터가 선택인지'를 딱 두 줄로 안내한다.

    모든 섬이 같은 4단계 위계를 쓴다.
        ① 알아보기 → ② 연습하기 → ③ 게임 도전 🏅 → ④ 자유 탐구(선택)
    2022 개정 실과 [6실05-05]가 요구하는 '기계학습의 기본 원리 이해'는
    ①②③ 만으로 달성되도록 설계했고, ④는 시간이 남거나 더 궁금한 학생을 위한 선택 활동이다.
    (원본은 섬마다 별 개수가 어긋나 '몇 단계까지 해야 하는지'를 알기 어려웠다.)
    """
    free_html = ('<br><span style="background:#C7A4DF; color:#fff; border-radius:999px; padding:2px 10px;'
                 ' font-family:\'Jua\',sans-serif; white-space:nowrap;">④ 선택</span> '
                 '시간이 남거나 더 궁금한 친구만 해보세요.') if has_free else ""
    md_html(f"""
    <div style="background:#F1F6FF; border-left:5px solid #2D9CDB; border-radius:10px;
                padding:9px 13px; margin-bottom:10px;
                font-family:'Gowun Dodum', sans-serif; font-size:13.5px; color:#28406B; line-height:2.0;">
      <span style="background:#2D9CDB; color:#fff; border-radius:999px; padding:2px 10px;
                   font-family:'Jua',sans-serif; white-space:nowrap;">①②③ 기본</span>
      여기까지 하면 <b>로봇 부품 🏅</b>을 받고 <b>[6실05-05] 기계학습의 기본 원리</b>도 다 배워요.
      &nbsp;<b style="color:#1565C0;">⏰ 약 {minutes}분</b>{free_html}
    </div>
    """)


def predict_first(key, question, options, answer_idx, why):
    """① 알아보기 맨 위에 두는 '먼저 예상해 보기' 한 문항.

    아이들은 읽어도 되고 안 읽어도 되는 글은 잘 안 읽는다.
    그래서 설명을 읽히기 전에 **먼저 한 번 생각하게** 만든다.
    맞히는 것이 목적이 아니므로 **점수를 매기지 않고, 틀렸다고 말하지도 않는다.**
    (탐구 학습의 '예측 → 조작·관찰 → 원리 발견'에서 '예측'에 해당한다.)

    화면에 그려지는 요소 개수는 항상 같게 유지한다.
    (개수가 바뀌면 보고 있던 탭이 1번으로 튕기기 때문)
    """
    md_html(f"""
    <div style="background:#FFF3E0; border:3px solid #FB8C00; border-radius:16px;
                padding:12px 18px; margin:0 0 12px 0;">
      <div style="font-family:'Jua',sans-serif; font-size:17px; color:#E65100;">
        🤔 먼저 예상해 볼까요? <span style="font-size:14px; color:#8D6E63;">(점수 없어요!)</span></div>
      <div style="font-family:'Gowun Dodum',sans-serif; font-size:18px;
                  color:#4E342E; line-height:1.7; margin-top:4px;">{question}</div>
    </div>
    """)
    pick = st.radio(question, ["🤷 아직 모르겠어"] + list(options),
                    horizontal=True, key=key, label_visibility="collapsed")

    if pick == "🤷 아직 모르겠어":
        msg = ("<span style='color:#6D4C41;'>괜찮아요! 하나 골라 보고 "
               "<b>아래에서 직접 확인</b>해 봐요. 틀려도 하나도 안 창피해요 😊</span>")
        bg, bd = "#FAFAFA", "#E0E0E0"
    elif pick == options[answer_idx]:
        msg = f"<b style='color:#2E7D32;'>👍 그렇게 생각했군요!</b> {why}"
        bg, bd = "#E8F5E9", "#66BB6A"
    else:
        msg = (f"<b style='color:#1565C0;'>🤔 그럴 것 같죠?</b> 아래에서 직접 해 보면 "
               f"깜짝 놀랄지도 몰라요! {why}")
        bg, bd = "#E3F2FD", "#42A5F5"

    md_html(f"""
    <div style="background:{bg}; border-left:6px solid {bd}; border-radius:12px;
                padding:10px 15px; margin:6px 0 14px 0;
                font-family:'Gowun Dodum',sans-serif; font-size:16px;
                color:#3A4A6B; line-height:1.7;">{msg}</div>
    """)
    return pick


# ══════════════════════════════════════════════════════════════════
#  [0번 섬] 내가 AI 선생님 — 기계학습의 기본 원리 체험용 도구들
#  물건 하나 = (색, 모양, 큼?, 반짝?)  ※ 모두 눈에 보이는 특징
#  숨은 진짜 규칙 = "✨반짝이면 네오의 부품"
# ══════════════════════════════════════════════════════════════════
NEO_FEATURE_NAMES = ["색깔", "모양", "크기", "반짝임"]


def neo_features(item):
    """물건을 AI가 읽을 수 있는 숫자 4개로 바꾼다. (색, 모양, 크기, 반짝임)"""
    color, shape, big, shiny = item[0], item[1], item[2], item[3]
    return [1 if color == "주황" else 0,
            1 if shape == "네모" else 0,
            1 if big else 0,
            1 if shiny else 0]


def neo_item_html(color, shape, big, shiny, size=72, caption=None):
    """우주 물건 카드 한 장을 그린다 (색·모양·크기·반짝임이 눈에 보이게)."""
    fill = "#F9A03F" if color == "주황" else "#3D8BFD"
    radius = "50%" if shape == "동그라미" else "16%"
    box = int(size * (0.72 if big else 0.5))
    spark = "<div style='position:absolute; top:-2px; right:-2px; font-size:19px;'>✨</div>" if shiny else ""
    cap = (f"<div style='font-size:11px; color:#3A4A6B; margin-top:3px; line-height:1.3;'>{caption}</div>"
           if caption else "")
    # 들여쓰기가 있으면 마크다운이 '코드 블록'으로 오해하므로 한 줄로 만든다.
    return (
        f"<div style=\"display:inline-block; text-align:center; margin:3px; vertical-align:top;\">"
        f"<div style=\"position:relative; width:{size}px; height:{size}px; background:#F4F8FF; "
        f"border:2px solid #D3E2F5; border-radius:14px; display:flex; align-items:center; "
        f"justify-content:center;\">{spark}"
        f"<div style=\"width:{box}px; height:{box}px; background:{fill}; border-radius:{radius}; "
        f"box-shadow:0 2px 5px rgba(0,0,0,0.16);\"></div></div>{cap}</div>"
    )


# 가르칠 때 쓰는 물건 8개 (색·모양·크기를 골고루 섞어, '반짝임'만 정답과 연결되게 함)
NEO_TRAIN = [
    ("파랑", "동그라미", True,  True),
    ("주황", "네모",     False, True),
    ("파랑", "네모",     False, False),
    ("주황", "동그라미", True,  False),
    ("주황", "동그라미", False, True),
    ("파랑", "네모",     True,  True),
    ("주황", "네모",     True,  False),
    ("파랑", "동그라미", False, False),
]

# 시험 문제 4개 — 위 8개에 없던 새로운 조합만 골랐다
NEO_TEST = [
    ("주황", "동그라미", True,  True),
    ("파랑", "네모",     False, True),
    ("파랑", "동그라미", True,  False),
    ("주황", "네모",     False, False),
]

# [4단계] '두 번째 상자' — 1~3단계와는 다른 상자다.
# 이 상자에서 네오가 본 부품은 하필 전부 파란색, 쓰레기는 전부 주황색이었다.
# (반짝임은 양쪽에 섞여 있어 단서가 되지 않으므로, 네오는 '색깔'을 규칙으로 배운다.)
# 앞 단계의 '반짝이면 부품' 규칙을 이 상자에 적용하지 않으므로 설명이 서로 어긋나지 않는다.
NEO_BIAS_TRAIN = [
    (("파랑", "동그라미", True,  True),  1),
    (("파랑", "네모",     False, True),  1),
    (("파랑", "네모",     True,  False), 1),
    (("파랑", "동그라미", False, False), 1),
    (("주황", "동그라미", True,  True),  0),
    (("주황", "네모",     False, True),  0),
    (("주황", "네모",     True,  False), 0),
    (("주황", "동그라미", False, False), 0),
]

# 시험 문제와 '진짜 정답'.
# 이 상자에는 주황색 부품도 분명히 있었지만, 네오에게 한 번도 보여주지 않았다.
NEO_BIAS_TEST = [
    ("주황", "동그라미", False, True),
    ("파랑", "네모",     True,  True),
]
NEO_BIAS_TRUTH = [1, 1]   # 둘 다 진짜 부품


def neo_fit(X, y):
    """이름표(y)로 작은 스무고개를 학습시키고, 찾아낸 규칙을 우리말로 돌려준다."""
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)

    if model.tree_.feature[0] < 0:          # 한쪽 답만 배운 경우
        only = "부품" if model.predict([[0, 0, 0, 0]])[0] == 1 else "쓰레기"
        return model, f"“전부 다 {only}이야!”"

    fi = int(model.tree_.feature[0])
    left_is_part = model.tree_.value[1][0][1] > model.tree_.value[1][0][0]
    yes = {0: "주황색", 1: "네모난", 2: "큰", 3: "✨반짝이는"}[fi]
    no = {0: "파란색", 1: "동그란", 2: "작은", 3: "안 반짝이는"}[fi]
    word = no if left_is_part else yes
    return model, f"“{word} 것이 부품이야!”"


def neo_exam(model, items, key_prefix="", truths=None):
    """네오가 처음 보는 물건을 판정하게 하고, 결과를 카드로 보여준다."""
    cards, correct = [], 0
    for idx, it in enumerate(items):
        pred = int(model.predict([neo_features(it)])[0])
        # 기본 규칙은 '반짝이면 부품'이지만, 상자가 다르면 정답을 따로 넘겨받는다.
        truth = truths[idx] if truths else (1 if it[3] else 0)
        ok = (pred == truth)
        correct += ok
        said = "🔧 부품" if pred else "🗑️ 쓰레기"
        mark = "⭕ 맞음" if ok else "❌ 틀림"
        color = "#2E7D32" if ok else "#C62828"
        cards.append(neo_item_html(
            *it, size=64,
            caption=f"네오: <b>{said}</b><br><span style='color:{color}'>{mark}</span>"))
    md_html("<div style='display:flex; gap:8px; flex-wrap:wrap; justify-content:center;'>"
            + "".join(cards) + "</div>")
    total = len(items)
    st.markdown(f"#### 🎯 네오의 점수: **{correct} / {total}**")
    if correct == total:
        st.success("💯 네오가 처음 보는 물건도 전부 맞혔어요! 잘 가르쳤네요 👏")
    elif correct == 0:
        st.error("😵 네오가 하나도 못 맞혔어요!")
    else:
        st.warning("🤔 네오가 몇 개는 틀렸어요.")
    return correct, total


def section_card(content_html, color="#2D9CDB"):
    """옅은 색 배경의 보조 카드 (성취기준 등)"""
    md_html(f"""
    <div style="background: {color}14; border-left: 5px solid {color};
                border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;">
        {content_html}
    </div>
    """)


def add_sweetness_axis_hints(fig):
    """이웃에게 물어보기 그래프에 축 의미를 직관적으로 알려주는 안내 문구 추가"""
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
        embed_html(f"""
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
    embed_html(f"""
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
# AI가 규칙을 찾는 6가지 방법 — '학생이 읽는 이름'이다.
# 학술 명칭(선형회귀·경사하강법…)은 교사용 안내에만 두고, 여기에는 쓰지 않는다.
AI_PRINCIPLES = [
    "📈 선 긋기 방법",
    "⛰️ 한 걸음씩 방법",
    "🌳 스무고개 방법",
    "🤝 이웃에게 물어보기",
    "🎨 비슷한 것끼리 모으기",
    "🧠 생각 그물 방법",
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
     "scenario": "🧠 여러 '판단 담당'이 각자 의견을 내고, 그 의견들을 다시 한 번 모아서 최종 결론을 내려요.",
     "hint": "요정 여러 명의 판단을 합쳐 로봇 파워를 정하던 원리예요."},
    {"part_idx": 5, "answer": 5,
     "scenario": "🤖 입력을 여러 층의 요정이 조금씩 다르게 계산하고 종합해서 똑똑한 결론을 만들어요.",
     "hint": "요정을 여러 층으로 쌓아 만든 '생각 그물망'이에요."},
]


# =========================================================
# 사전/사후 진단 평가 문항 — 학습 효과 측정용
# =========================================================
# 학습 전(사전)과 후(사후)에 똑같은 문항을 풀게 해서 점수 변화를 비교한다.
# answer는 AI_PRINCIPLES의 인덱스. (0선형회귀 1경사하강 2결정트리 3KNN 4K평균 5신경망)
# 문항마다 보기(options)를 따로 두고, answer 는 그 보기 안에서의 번호다.
# ① AI가 배우는 방법 ② 자료의 양 ③ 정답 없이 배우기 ④ 치우친 자료(편향)
# ⑤ AI의 한계 ⑥ 사회와 직업의 변화  —  [6실05-05]의 두 축을 고르게 담았다.
DIAGNOSTIC_QUESTIONS = [
    {"q": "🤖 인공지능(AI)은 어떻게 똑똑해질까요?",
     "options": ["사람이 규칙을 하나하나 다 알려줘서",
                 "많은 예시(자료)를 보고 스스로 규칙을 찾아서",
                 "만들 때부터 이미 다 알고 있어서"],
     "answer": 1},
    {"q": "🐶 강아지 사진을 3장만 본 AI와 3000장 본 AI, 누가 더 잘 맞힐까요?",
     "options": ["3장만 본 AI", "3000장 본 AI", "둘이 똑같아요"],
     "answer": 1},
    {"q": "🧩 AI에게 사진을 잔뜩 주면서 “비슷한 것끼리 알아서 나눠줘”라고 했어요. 이때 AI에게 정답(이름표)을 알려줬나요?",
     "options": ["네, 정답을 알려줬어요",
                 "아니요, 정답 없이 스스로 나눴어요",
                 "정답이 있어야만 나눌 수 있어요"],
     "answer": 1},
    {"q": "🍎 빨간 사과 사진만 잔뜩 보고 배운 AI에게 초록 사과를 보여주면 어떻게 될까요?",
     "options": ["“사과가 아니야”라고 잘못 말할 수 있어요",
                 "처음 보는 것도 무조건 맞혀요",
                 "사과를 빨갛게 바꿔 버려요"],
     "answer": 0},
    {"q": "💬 AI가 아주 자신 있게 대답하면, 그 말은 항상 맞을까요?",
     "options": ["항상 맞으니까 그대로 믿어요",
                 "틀릴 수도 있으니 다시 확인해요",
                 "AI는 원래 대답을 못 해요"],
     "answer": 1},
    {"q": "🏙️ AI가 많아지면 사람들의 일(직업)은 어떻게 될까요?",
     "options": ["모든 직업이 사라져요",
                 "일하는 방법이 바뀌고, 새로운 직업도 생겨요",
                 "아무것도 바뀌지 않아요"],
     "answer": 1},
]


# =========================================================
# 사전/사후 설문 문항 (5점 척도) — 보고서 정량 분석용
# =========================================================
# 사전·사후에 **똑같은 문장**으로 물어야 점수 차이를 '향상도'로 해석할 수 있다.
#   (예전처럼 사후에만 "얼마나 도움이 되었나요?"로 물으면 향상을 전제한 유도 질문이 되고,
#    사전 점수와 뺄셈해서 향상도라고 말할 수 없다.)
# 그래서 1~3번은 pre 문장을 그대로 post 에도 쓴다.
# 4번만 예외다. 이 문항은 '학습 전에 쓰던 도구'와 '이 프로그램'을 비교하는 것이 목적이므로
#   사전에는 기존 도구를, 사후에는 이 프로그램을 묻는다. (시판 소프트웨어와의 차별성 근거)
# key: CSV 저장 및 대시보드 표시에 쓰이는 영역 이름
SURVEY_ITEMS = [
    {"key": "흥미도",
     "pre":  "인공지능(AI)이 작동하는 원리를 배우는 것에 흥미가 얼마나 있나요?",
     "post": "인공지능(AI)이 작동하는 원리를 배우는 것에 흥미가 얼마나 있나요?"},
    {"key": "AI이해력",
     "pre":  "인공지능이 자료(데이터)를 보고 스스로 규칙을 찾아내는 원리를 얼마나 잘 이해하고 있나요?",
     "post": "인공지능이 자료(데이터)를 보고 스스로 규칙을 찾아내는 원리를 얼마나 잘 이해하고 있나요?"},
    {"key": "문제해결력",
     "pre":  "변수를 직접 조작하며 결과를 스스로 예측하고 발견하는 학습 활동에 얼마나 자신이 있나요?",
     "post": "변수를 직접 조작하며 결과를 스스로 예측하고 발견하는 학습 활동에 얼마나 자신이 있나요?"},
    # 4번만 사전·사후 대상이 다르다 (기존 도구 → 이 프로그램). 차별성 비교용.
    {"key": "프로그램만족도",
     "pre":  "지금까지 써 본 AI 교육용 소프트웨어(프로그램)에 얼마나 만족하나요?",
     "post": "이 프로그램(AI 원리 탐험대)에 얼마나 만족하나요?"},
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
def _load_teacher_password():
    """선생님 방 비밀번호를 소스 코드 밖에서 읽어 온다.
       ① .streamlit/secrets.toml 의 TEACHER_PASSWORD
       ② 환경 변수 TEACHER_PASSWORD
       ③ 둘 다 없으면 기본값 "teacher"
    학급마다 비밀번호를 바꾸려면 secrets.toml 만 고치면 되고, 코드는 건드리지 않아도 된다."""
    try:
        v = st.secrets.get("TEACHER_PASSWORD")      # secrets 파일이 없으면 예외가 날 수 있다
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get("TEACHER_PASSWORD") or "teacher"


TEACHER_PASSWORD = _load_teacher_password()
# 기본 비밀번호를 그대로 쓰고 있는지 (선생님 방에서 안내하기 위해)
TEACHER_PASSWORD_IS_DEFAULT = (TEACHER_PASSWORD == "teacher")


# 학습 기록 파일을 한 번에 한 명씩만 고치도록 하는 잠금.
# (Streamlit 은 학생마다 '스레드'를 만들어 같은 프로세스에서 돌리므로 스레드 잠금이면 충분하다.)
_RECORD_LOCK = threading.RLock()


def _ensure_data_dir():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except OSError:
        pass


def _atomic_write_csv(df, path, retries=5):
    """같은 폴더의 임시 파일에 먼저 다 쓴 뒤, 완성된 파일을 '한 번에' 바꿔치기한다.
    - 쓰는 도중에 다른 학생이 읽어도 '반쯤 쓰인 파일'을 보지 않는다.
    - 저장 중에 프로그램이 멈춰도 원래 파일이 그대로 남는다.
    - 선생님이 엑셀로 CSV 를 열어 둔 경우(윈도우) 교체가 잠시 막히므로 몇 번 다시 시도한다."""
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp_result_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        last = None
        for i in range(retries):
            try:
                os.replace(tmp, path)     # 원자적 교체
                return True
            except PermissionError as e:  # 윈도우에서 파일이 열려 있을 때
                last = e
                time.sleep(0.2 * (i + 1))
        raise last if last else OSError("파일을 바꿔치기하지 못했습니다")
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_student_record(name):
    """현재 학생의 학습 결과 한 줄을 CSV에 저장/갱신한다.
    '학급코드 + 이름'을 기준으로 같은 학생이면 최신 값으로 덮어쓴다(중복 방지).
    학급코드로 구분해 저장하므로, 여러 학급·학교가 같은 프로그램을 써도 서로 섞이지 않는다."""
    _ensure_data_dir()
    now_kr = datetime.now(timezone(timedelta(hours=9)))
    record = {
        "학급코드": st.session_state.get("class_code", "").strip() or "미입력",
        "이름": name.strip(),
        "사전점수": st.session_state.get("pre_score", ""),
        "사후점수": st.session_state.get("post_score", ""),
        "향상점수": (st.session_state.get("post_score", 0) - st.session_state.get("pre_score", 0))
                     if (isinstance(st.session_state.get("pre_score"), int)
                         and isinstance(st.session_state.get("post_score"), int)) else "",
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
    # ★ '읽고 → 고치고 → 쓰기' 전체를 잠가야 한다.
    #   읽기만 잠그면, 두 학생이 각자 읽은 옛 내용을 바탕으로 써서 한 명 기록이 사라진다.
    try:
        with _RECORD_LOCK:
            if os.path.exists(RESULTS_CSV):
                df = pd.read_csv(RESULTS_CSV, dtype=str)
            else:
                df = pd.DataFrame(columns=list(record.keys()))
            # 같은 학급 + 같은 이름의 기존 행만 제거 후 추가 (최신값 유지)
            if "학급코드" not in df.columns:
                df["학급코드"] = "미입력"
            if "이름" not in df.columns:
                df["이름"] = ""
            df = df[~((df["이름"].astype(str) == record["이름"])
                      & (df["학급코드"].astype(str) == record["학급코드"]))]
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            _atomic_write_csv(df, RESULTS_CSV)
        return True
    except Exception:
        _LOG.exception("학습 기록 저장 실패")
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
    """배지를 세션에 기록하고, 획득한 로봇 부품 이미지를 화면에 예쁘게 보여준다.
    (처음 얻는 순간에만 풍선·효과음·토스트로 축하하고, 이후엔 조용히 카드만 보여준다.)"""
    if not isinstance(idx, int) or not (0 <= idx < len(ALL_BADGES)):
        return
    emoji, name, filename, part, desc = ALL_BADGES[idx]
    is_new = name not in st.session_state["badges"]
    st.session_state["badges"].add(name)
    if is_new:
        try:
            st.balloons()
        except Exception:
            pass
        play_sfx(SFX_BADGE_FILE)  # 부품 획득 효과음 (key)
        safe_toast(f"로봇 네오의 {part} 획득! ({len(st.session_state['badges'])}/6)", icon="🔧")
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


@st.cache_resource(show_spinner=False)
def _cert_font(name, size):
    """수료증 글꼴. (cache_data는 글꼴 객체를 피클로 저장하다 실패할 수 있어 cache_resource를 쓴다.
    글꼴 파일이 없거나 깨져 있어도 기본 글꼴로 안전하게 대체한다.)"""
    path = os.path.join(FONT_DIR, name)
    try:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
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
    _cert_center_text(draw, cx, cy + 24, "탐험대", _cert_font("NotoSansKR-Bold.ttf", 15), _CERT_NAVY)
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

    _cert_center_text(draw, W / 2, 270, f"{name}  탐험대장님",
                       _cert_font("NotoSerifKR-Bold.ttf", 44), _CERT_NAVY)

    body_font = _cert_font("NotoSansKR-Regular.ttf", 25)
    para = [
        "위 탐험대장님은 '손끝에서 배우는 인공지능 원리 탐험대'의 모든 과정을 완수하고,",
        "선 긋기 · 한 걸음씩 · 스무고개 · 이웃에게 묻기 · 비슷한 것끼리 · 생각 그물",
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
    draw.text((margin + 70, H - 120), "손끝에서 배우는 인공지능 원리 탐험대",
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
    embed_html(f"""
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
    "🌟 선 긋기 완벽 이해",
    "🌟 한 걸음씩 완벽 이해",
    "🌟 스무고개 완벽 이해",
    "🌟 이웃에게 묻기 완벽 이해",
    "🌟 비슷한 것끼리 완벽 이해",
    "🌟 생각 그물 완벽 이해",
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
    if isinstance(code, (list, tuple)):   # 구버전 API는 리스트로 돌려준다
        code = code[0] if code else None
    if not isinstance(code, str) or len(code) < 4:
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
        '과일 크기': [7, 3, 9, 2, 1, 3, 6, 2, 4],
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
    clear_celebration("_celeb_review_done")


# =========================================================
# 게임용 세션 상태 초기화
# =========================================================
# 게임 상태의 기본값과 '올바른 모양'(자료형). 세션 값이 이 모양과 다르면 — 태블릿 새로고침,
# 브라우저 탭 복원, 뒤엉킨 클릭 등 어떤 이유로든 — 기본값으로 조용히 되돌린다.
_GAME_DEFAULTS = {
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


def _state_ok(key, value, default):
    """세션에 든 값이 기본값과 같은 '종류'인지 검사한다. (None 기본값은 비어 있어도 되는 값)"""
    try:
        if key in ("pre_score", "post_score"):
            return value is None or (isinstance(value, int) and not isinstance(value, bool))
        if key == "km_phase":
            return value in ("assign", "move")
        if key == "gd_history":
            return (isinstance(value, list) and len(value) > 0
                    and all(isinstance(v, (int, float)) and np.isfinite(v) for v in value))
        if key in ("badges", "bonus_badges"):
            return isinstance(value, set) and all(isinstance(v, str) for v in value)
        if key in ("review_fixed", "dt_asked", "dt_history"):
            return isinstance(value, list)
        if key == "review_quiz":
            return value is None or (isinstance(value, list) and len(value) > 0
                                     and all(isinstance(q, dict) and "answer" in q and "part_idx" in q
                                             for q in value))
        if default is None:
            return True
        if isinstance(default, bool):
            return isinstance(value, bool)
        if isinstance(default, int):
            return isinstance(value, int) and not isinstance(value, bool)
        if isinstance(default, float):
            return isinstance(value, (int, float)) and np.isfinite(value)
        return isinstance(value, type(default))
    except Exception:
        return False


def init_game_state():
    """세션 상태를 준비한다. 없는 키는 채우고, 모양이 이상한 키는 기본값으로 고친다.
    (태블릿에서 새로고침하거나 브라우저가 탭을 복원하면 값이 깨진 채 남을 수 있다.)"""
    is_fresh_session = "badges" not in st.session_state
    for k, v in _GAME_DEFAULTS.items():
        if k not in st.session_state or not _state_ok(k, st.session_state[k], v):
            st.session_state[k] = copy.deepcopy(v)
    # 진행 인덱스가 범위를 벗어나면 복습 게임을 처음으로 되돌린다
    rq = st.session_state.get("review_quiz")
    if rq is not None and not (0 <= st.session_state.get("review_idx", 0) < len(rq)):
        st.session_state["review_quiz"] = None
        st.session_state["review_idx"] = 0
        st.session_state["review_done"] = False
    if is_fresh_session:
        # 새 세션일 때만 URL의 진행상황을 복원한다(수동으로 초기화한 경우까지 되살리지 않도록).
        try:
            _restore_progress_from_url()
        except Exception:
            pass


init_game_state()

# 2. 사이드바 메뉴
MENU_OPTIONS = [
    "🏠 탐험 본부 (홈)",
    "📝 나의 이름 & 사전 평가",
    "🌟 인공지능이 뭐예요?",
    "🤖 0. 내가 AI 선생님",
    "📈 1. 마법의 선 긋기",
    "⛰️ 2. 보물찾기 산",
    "🌳 3. 스무고개 탐정",
    "🤝 4. 가장 친한 친구",
    "🎨 5. 비슷한 친구끼리",
    "🧠 6. 똑똑한 생각 주머니",
    "⚖️ AI를 똑똑하게 쓰려면?",
    "🔧 로봇 네오 종합 점검",
    "🏆 사후평가 (수료증)",
    "👩‍🏫 선생님 방 (학습 분석)",
    "📜 자료 출처 · 저작권",
]

if "nav_menu" not in st.session_state:
    st.session_state["nav_menu"] = MENU_OPTIONS[0]

# 버튼/이미지 클릭으로 예약된 이동이 있으면, 라디오 위젯이 만들어지기 '전에' 반영한다
if "_pending_nav" in st.session_state:
    st.session_state["nav_menu"] = st.session_state.pop("_pending_nav")
    st.session_state["_scroll_top"] = True  # 새 화면은 맨 위부터 보이도록 표시

# 메뉴 값이 목록에 없으면(예전 버전 주소로 접속, 세션 복원 오류 등) 홈으로 안전하게 되돌린다.
# 라디오 위젯은 목록에 없는 값을 받으면 오류를 내며 멈추므로, 위젯을 만들기 전에 확인한다.
if st.session_state.get("nav_menu") not in MENU_OPTIONS:
    st.session_state["nav_menu"] = MENU_OPTIONS[0]

# 수료증 메뉴는 항상 보인다.
# (사후 검사는 부품을 다 모으지 않아도 언제든 할 수 있고,
#  수료증 '이미지'는 부품 6개를 모두 모았을 때 나타난다.)
visible_menus = MENU_OPTIONS.copy()

def go_to(target_menu):
    """버튼/이미지 클릭으로 다른 섹션으로 이동시키는 헬퍼"""
    if target_menu not in MENU_OPTIONS:   # 잘못된 목적지면 홈으로
        target_menu = MENU_OPTIONS[0]
    st.session_state["_pending_nav"] = target_menu
    st.rerun()


# =========================================================
# 🛡️ 전역 오류 방어막 — 어떤 섬에서 무슨 일이 나도 빨간 오류 화면 대신
#    아이들 눈높이의 안내 카드와 '다시 시작' 버튼을 보여준다.
# =========================================================
def render_error_boundary(slot, exc, menu):
    """오류가 난 자리(slot)에 친절한 복구 카드를 그린다."""
    _LOG.exception("페이지 렌더링 중 오류 (%s)", menu)
    with slot.container():
        md_html("""
        <div style="background:linear-gradient(135deg,#FFF3E0,#FFFFFF); border:3px dashed #FFA726;
                    border-radius:18px; padding:18px 22px; margin:6px 0 10px 0; text-align:center;">
          <div style="font-size:42px; line-height:1;">🤖💤</div>
          <div style="font-family:'Jua',sans-serif; font-size:21px; color:#E65100; margin-top:6px;">
            앗! 로봇 네오가 잠깐 딸꾹질을 했어요</div>
          <div style="font-family:'Gowun Dodum',sans-serif; font-size:15px; color:#5D4037;
                      line-height:1.7; margin-top:6px;">
            걱정 마세요, <b>모은 부품과 검사 결과는 그대로</b>예요.<br>
            아래 버튼을 눌러 이 섬을 처음부터 다시 시작하면 돼요!</div>
        </div>
        """)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 이 섬만 처음부터 다시", key="_err_reset_section", type="primary", **_STRETCH):
                reset_section_state(menu)
                st.rerun()
        with c2:
            if st.button("🏠 탐험 본부로 돌아가기", key="_err_go_home", **_STRETCH):
                reset_section_state(menu)
                go_to(MENU_OPTIONS[0])
        with st.expander("🛠️ 선생님용: 무슨 일이 있었나요? (기술 정보)"):
            st.caption("아래 내용은 프로그램을 고칠 때 참고하는 정보예요. 학생은 신경 쓰지 않아도 돼요.")
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-3000:],
                    language="text")


def guarded_render(menu):
    """render_page(menu) 를 실행하되, 예외가 나면 오류 화면 대신 복구 카드를 보여준다.
    st.rerun()/st.stop() 이 내는 제어용 예외는 그대로 통과시켜 화면 이동이 정상 동작하게 한다."""
    slot = st.empty()            # 오류 카드가 화면 '맨 위'에 나타나도록 자리를 먼저 잡아 둔다
    try:
        render_page(menu)
    except _CTRL_EXC:
        raise
    except Exception as exc:     # noqa: BLE001 — 아이들 화면을 지키는 것이 최우선
        render_error_boundary(slot, exc, menu)
    finally:
        try:
            _sync_progress_to_url()   # 어떤 경우에도 진행상황(배지)은 주소창에 보존
        except Exception:
            pass


with st.sidebar:
    # 제목은 두 줄 모두 '같은 글꼴(Jua) + 같은 크기'로 맞추고 이모지는 넣지 않는다.
    # 각 줄이 사이드바 폭 안에 한 줄로 딱 들어가도록 끊어 배치.
    md_html("""
    <div style="padding: 2px 0 4px 0;">
        <div style="font-family:'Jua', sans-serif; font-size:0.95rem; color:#FFFFFF;
                    white-space:nowrap; letter-spacing:-0.5px; line-height:1.35;">손끝에서 배우는</div>
        <div style="font-family:'Jua', sans-serif; font-size:0.95rem; color:#FFFFFF;
                    white-space:nowrap; letter-spacing:-0.5px; line-height:1.35;">인공지능 원리 탐험대</div>
    </div>
    """)
    # 👋 누가 탐험 중인지 한눈에 (이름을 적은 학생만). 컴퓨터실에서 '내 자리가 맞나' 확인용.
    _sb_name = str(st.session_state.get("student_name", "") or "").strip()
    _sb_class = str(st.session_state.get("class_code", "") or "").strip()
    if _sb_name:
        md_html(f"""
        <div style="background:#24365C; border-radius:10px; padding:5px 8px; margin:2px 0 4px 0;
                    font-size:0.72rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            👋 <b>{_sb_name}</b> 탐험대원{(' · ' + _sb_class) if _sb_class else ''}
        </div>
        """)
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

    # 👉 다음 단계로 — 어느 화면에 있든 사이드바에서 바로 다음 순서로 넘어간다.
    #    (화면 안 '다음' 버튼은 탭 속에 묻혀 못 보고 지나치는 일이 있어서 따로 둔다.)
    _idx = visible_menus.index(menu) if menu in visible_menus else 0
    if _idx < len(visible_menus) - 3:          # 선생님 방·자료 출처는 제외
        _nxt = visible_menus[_idx + 1]
        _short = _nxt.split(" ", 1)[-1]
        if len(_short) > 11:
            _short = _short[:11] + "…"
        if st.button(f"👉 다음: {_short}", key="side_next", **_STRETCH):
            go_to(_nxt)

    st.divider()

    # 🗺️ 탐험 진행도
    collected_names = st.session_state["badges"]
    collected_count = len(collected_names)
    st.markdown("#### 🗺️ 로봇 네오 조립 현황")
    st.progress(collected_count / len(ALL_BADGES))
    st.caption(f"부품 {collected_count} / {len(ALL_BADGES)}개 모음")

    icon_row = " ".join(
        badge_img_tag(i, size=27, grayscale=(name not in collected_names))
        for i, (icon, name, filename, part, desc) in enumerate(ALL_BADGES)
    )
    st.markdown(f"<div style='display:flex; gap:4px; flex-wrap:wrap; max-width:100%;'>{icon_row}</div>", unsafe_allow_html=True)

    st.divider()
    if collected_names:
        st.write("### 🔧 내가 모은 부품")
        for i, (icon, name, filename, part, desc) in enumerate(ALL_BADGES):
            if name in collected_names:
                md_html(f"""
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:5px;">
                    {badge_img_tag(i, size=22)}
                    <span style="font-size:0.72rem; word-break:keep-all;">{part}</span>
                </div>
                """)
    else:
        st.caption("게임을 깨면 부품을 받아요! 🌟")

    # 6개 부품을 모두 모으면 복습 게임을 눈에 띄게 안내
    if collected_count >= 6:
        st.divider()
        md_html("""
        <div class="review-tip" style="background:#EDE7F6; border:2px solid #7E57C2; border-radius:12px; padding:8px; text-align:center;">
            <div style="font-size:0.72rem; line-height:1.3; word-break:keep-all;">🎓 이제 배운 걸<br>정리해볼까요?</div>
        </div>
        """)
        if st.button("🔧 복습 미션 도전하기", key="side_go_review", **_STRETCH):
            go_to("🔧 로봇 네오 종합 점검")

    bonus_names = st.session_state["bonus_badges"]
    if bonus_names:
        st.write(f"### 🌟 완벽 이해 보너스 ({len(bonus_names)}개)")
        for b in bonus_names:
            st.write(b)

    # 처음 오신 선생님·참관자가 학습 분석 대시보드를 놓치지 않도록 안내한다
    st.divider()
    md_html("""
    <div class="review-tip" style="background:#E8F0FE; border:2px solid #5C6BC0; border-radius:12px;
                                   padding:8px 9px 10px 9px;">
      <div style="font-size:0.68rem; line-height:1.45; word-break:keep-all;">
        👩‍🏫 <b>선생님·참관자님께</b><br>
        맨 아래 <b>‘선생님 방’</b>에서 우리 반 사전·사후 향상도를 볼 수 있어요.
        <b>비밀번호 없이</b> 바로 열어보는 버튼도 있어요!
      </div>
    </div>
    """)
    if st.button("👩‍🏫 선생님 방 열기 (체험용)", key="side_go_teacher", **_STRETCH):
        st.session_state["teacher_unlocked"] = True
        go_to("👩‍🏫 선생님 방 (학습 분석)")

    # 컴퓨터실처럼 한 대의 기기를 여러 학생이 이어서 쓸 때를 위한 전체 초기화 버튼
    st.divider()
    if st.session_state.get("_confirm_reset"):
        st.warning("정말요?\n모은 부품과 검사 결과가\n모두 사라져요!")
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
                st.session_state.pop("_diag_part1_done_pre", None)
                st.session_state.pop("_diag_part1_done_post", None)

                # 복습 미션 진행 상태도 초기화
                for k in ["review_quiz", "review_idx", "review_fixed",
                          "review_wrong", "review_done", "review_started"]:
                    st.session_state.pop(k, None)
                for k in list(st.session_state.keys()):
                    if k.startswith("review_wrongmark_") or k.startswith("review_ans_"):
                        st.session_state.pop(k, None)

                # 각 섬의 게임 진행·퀴즈 답·축하 표시까지 모두 비워 '완전히 새 친구' 상태로 만든다
                for _m in SECTION_STATE_PREFIXES:
                    reset_section_state(_m)
                for k in list(st.session_state.keys()):
                    if str(k).startswith("_celeb_") or str(k).startswith("quiz_"):
                        st.session_state.pop(k, None)
                for k, v in _GAME_DEFAULTS.items():
                    st.session_state[k] = copy.deepcopy(v)

                st.session_state["_confirm_reset"] = False
                _sync_progress_to_url()
                st.rerun()
        with rc2:
            if st.button("❌ 취소", key="reset_no", **_STRETCH):
                st.session_state["_confirm_reset"] = False
                st.rerun()
    else:
        if st.button("🔄 처음부터 다시", key="reset_all", **_STRETCH):
            st.session_state["_confirm_reset"] = True
            st.rerun()
        st.caption("다음 친구가 쓸 때 눌러요.\n부품·검사 결과가 모두 지워져요.")


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


# 🗺️ 탐험 지도: 6개 섬의 (메뉴, 이모지, 이름, 한 줄 설명, 색, 부품 번호)
ISLANDS = [
    ("📈 1. 마법의 선 긋기",    "📈", "마법의 선 긋기",  "점들 사이로 '선'을 그어 예측", SECTION_COLORS["reg"], 0),
    ("⛰️ 2. 보물찾기 산",      "⛰️", "보물찾기 산",     "한 걸음씩 정답에 다가가기",   SECTION_COLORS["gd"],  1),
    ("🌳 3. 스무고개 탐정",     "🌳", "스무고개 탐정",   "질문으로 후보를 좁혀 찾기",   SECTION_COLORS["dt"],  2),
    ("🤝 4. 가장 친한 친구",    "🤝", "가장 친한 친구",  "가까운 이웃에게 물어보기",    SECTION_COLORS["knn"], 3),
    ("🎨 5. 비슷한 친구끼리",   "🎨", "비슷한 친구끼리", "닮은 것끼리 스스로 모으기",   SECTION_COLORS["km"],  4),
    ("🧠 6. 똑똑한 생각 주머니", "🧠", "생각 주머니",     "여러 요정의 판단을 합치기",   SECTION_COLORS["nn"],  5),
]


def island_map(key_prefix="map"):
    """홈 화면의 '탐험 지도'. 섬마다 부품을 얻었는지(✅) 아직인지(🔒) 보여주고 바로 이동할 수 있다.
    (아이들이 '내가 어디까지 했지?'를 한눈에 알 수 있게 한다.)"""
    got = st.session_state.get("badges", set())
    cols = st.columns(6)
    for i, (menu_name, emoji, title, desc, color, bidx) in enumerate(ISLANDS):
        done = ALL_BADGES[bidx][1] in got
        part = ALL_BADGES[bidx][3]
        badge = (f"<span class='island-badge' style='background:#43A047;'>✅ {part} 획득</span>" if done
                 else f"<span class='island-badge' style='background:#9FB3C8;'>🔒 {part}</span>")
        with cols[i]:
            md_html(f"""
            <div class="island-card {'done' if done else ''}" style="border-color:{color}66;">
              <div class="island-emoji">{emoji}</div>
              <div class="island-title" style="color:{color};">{title}</div>
              <div class="island-desc">{desc}</div>
              {badge}
            </div>
            """)
            if st.button("복습하기 🔁" if done else "가기 ⛵", key=f"{key_prefix}_go_{i}", **_STRETCH):
                go_to(menu_name)



# 다른 화면으로 막 이동한 직후라면, 화면을 맨 위로 스크롤한다
# (탭/메뉴 이동 시 이전 스크롤 위치가 남아 중간부터 보이는 문제 해결)
#
# ⚠️ 이 조각은 '있다가 없어지면' 안 된다.
#    화면 맨 위에 있던 요소 하나가 사라지면 그 아래 탭 묶음의 자리가 한 칸씩 밀리고,
#    그러면 보고 있던 탭이 ① 번으로 튕겨 버린다.
#    그래서 **언제나 똑같이 한 번 그리고**, 스크롤할 때만 안쪽 스크립트가 동작하게 한다.
#    (스크롤이 필요 없을 때는 내용이 하나도 안 바뀌므로 다시 실행되지도 않는다.)
_scroll_now = bool(st.session_state.pop("_scroll_top", False))
if _scroll_now:
    st.session_state["_scroll_tick"] = int(st.session_state.get("_scroll_tick", 0)) + 1
_scroll_tick = int(st.session_state.get("_scroll_tick", 0))

if True:
    embed_html(
        """
        <script>
        (function () {
            const GO = __GO__;              /* 이동 횟수: __TICK__ */
            if (!GO) { return; }            /* 평소에는 아무 일도 하지 않는다 */
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
        """.replace("__GO__", "true" if _scroll_now else "false")
           .replace("__TICK__", str(_scroll_tick)),
        height=0,
    )


def render_diagnostic_quiz(phase, score_key):
    """사전/사후 진단(지식 6문항 + 설문 4문항)을 렌더링하고, 제출하면 결과를 저장한다.
    phase: 'pre' 또는 'post' (위젯 key 구분 및 설문 문장 선택에 사용)"""
    survey_key = "pre_survey" if phase == "pre" else "post_survey"

    # 경고문이 '나타났다 사라지면' 아래 탭이 한 칸 밀려 ① 번으로 튕긴다.
    # 자리(컨테이너)는 늘 하나 잡아두고, 그 안에서만 내용이 바뀌게 한다.
    _warn_slot = st.container()
    with _warn_slot:
        if st.session_state.pop("_save_warn", False):
            st.warning("⚠️ 결과를 파일에 저장하지 못했어요. 선생님께 알려주세요! "
                       "(화면의 내 점수는 그대로 볼 수 있어요.)")

    # 문항이 10개라 한 화면에 다 쌓으면 스크롤이 길어진다. 1부·2부를 탭으로 나눈다.
    tab_know, tab_think = st.tabs(["📚 인공지능 알아보기 (6문항)",
                                   "💬 나의 생각 (4문항)"])

    # 제출 버튼이 탭 '바깥'에 하나뿐이면, 1부만 풀고 눌렀을 때 2부를 건너뛴 채
    # 곧바로 결과로 넘어가 버린다. 그래서 탭마다 버튼을 따로 두고,
    # 2부의 최종 제출은 1부를 마친 뒤에만 눌리게 한다.
    part1_flag = f"_diag_part1_done_{phase}"
    part1_done = bool(st.session_state.get(part1_flag))

    # ── 1부: 지식 문항 (채점됨) ─────────────────────────────
    with tab_know:
        st.caption("잘 모르면 '아직 잘 몰라요'를 골라도 괜찮아요! (점수는 성적과 상관없어요 😊)")
        answers = []
        for i, q in enumerate(DIAGNOSTIC_QUESTIONS):
            choice = st.radio(
                f"**{i+1}.** {q['q']}",
                ["아직 잘 몰라요"] + list(q["options"]),
                key=f"diag_{phase}_{i}",
            )
            answers.append(choice)

        st.write("")
        _unsure = sum(1 for a in answers if a == "아직 잘 몰라요")
        if st.button("✅ 6문항 다 풀었어요! (다음으로 가기)", key=f"diag_part1_{phase}",
                     type="primary", **_STRETCH):
            st.session_state[part1_flag] = True
            part1_done = True
        # 화면에 그려지는 요소 개수가 바뀌면 보고 있던 탭이 튕기므로,
        # 안내문은 '항상 한 줄' 그리되 내용만 바꾼다.
        st.caption("✅ 잘했어요! 이제 위쪽 **💬 나의 생각 (4문항)** 탭을 눌러주세요."
                   if part1_done else
                   (f"👆 6문항을 모두 고른 뒤 위 버튼을 눌러주세요. "
                    f"(지금 '아직 잘 몰라요' {_unsure}개)"))

    # ── 2부: 설문 문항 (5점 척도, 채점 안 됨) ─────────────────
    with tab_think:
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

        st.write("")
        submitted = st.button("📨 최종 제출하기", key=f"diag_submit_{phase}",
                              type="primary", disabled=not part1_done, **_STRETCH)
        st.caption("👆 4개를 모두 고른 뒤 최종 제출을 눌러주세요."
                   if part1_done else
                   "🔒 먼저 **📚 인공지능 알아보기 (6문항)** 탭을 끝내야 제출할 수 있어요.")

    if submitted:
        score = 0
        for i, q in enumerate(DIAGNOSTIC_QUESTIONS):
            _opts = list(q["options"])
            if answers[i] in _opts and _opts.index(answers[i]) == q["answer"]:
                score += 1
        st.session_state[score_key] = score
        st.session_state[survey_key] = survey_choices
        st.session_state.pop(part1_flag, None)     # 다시 풀 때를 위해 초기화
        # 이름이 있으면 결과 저장
        nm = st.session_state.get("student_name", "").strip()
        if nm and not save_student_record(nm):
            # 저장이 안 됐는데 아무 말도 없으면 선생님이 결과를 잃는다
            st.session_state["_save_warn"] = True
        st.rerun()


# =========================================================
# 페이지 본문 — 전부 render_page() 안에서 실행되어 전역 오류 방어막의 보호를 받는다
# =========================================================
def render_page(menu):
    """선택된 메뉴(섬) 화면을 그린다. 어떤 예외가 나더라도 아래 guarded_render()가 받아 준다."""
    # --- [메뉴 1: 홈] ---
    if menu == "🏠 탐험 본부 (홈)":
        # ─────────────────────────────────────────────────────────────
        #  홈 화면은 '위쪽 고정 영역 + 탭 3개'로 구성해 스크롤을 짧게 유지한다.
        #    위쪽(항상 보임) : 제목 · 네오의 안내 한 마디 · 시작 버튼
        #    🗺️ 탐험 지도    : 6개 섬 카드 (학생이 가장 많이 쓰는 화면 → 기본 탭)
        #    🎬 소개 영상    : 도입 영상
        #    👩‍🏫 선생님 안내 : 성취기준 지도 · 차시별 코스 · 교육과정 연계표
        # ─────────────────────────────────────────────────────────────
        render_bgm_player(BGM_HOME_FILE, "home", compact=True)

        _mascot_b64 = _asset_img_b64("robot_mascot.png")
        _mascot_html = (f'<img src="data:image/png;base64,{_mascot_b64}" '
                        f'style="height:46px; width:auto; vertical-align:middle; margin-right:10px;">'
                        ) if _mascot_b64 else "🚀"
        md_html(f"""
        <h1 style="text-align:center; font-family:'Jua',sans-serif; font-size:clamp(21px, 3.6vw, 38px);
                   color:#1B2A4A; margin:-8px 0 2px 0; line-height:1.2; white-space:nowrap;">
            {_mascot_html}손끝에서 배우는 인공지능 원리 탐험대
        </h1>
        <div style="text-align:center; margin-bottom:8px;">
            <span style="display:inline-block; background:#E3F2FD; border:2px solid #2D9CDB; border-radius:20px;
                         padding:3px 15px; font-family:'Jua',sans-serif; font-size:14px; color:#1565C0;">
                🎯 초등학교 6학년 · 실과 [6실05-05]
            </span>
        </div>
        """)

        # ── 네오의 안내 한 마디 + 시작 버튼 (스크롤 없이 바로 보이게) ──
        _got = len(st.session_state["badges"])
        _named = bool(str(st.session_state.get("student_name", "") or "").strip())
        if _got >= 6:
            neo_says("와, 부품 6개를 모두 모았어요! 이제 <b>복습 미션</b>으로 배운 걸 정리하거나 "
                     "<b>수료증</b>을 받으러 가요. 🎉", mood="happy")
        elif _got > 0:
            neo_says(f"지금까지 부품 <b>{_got}개</b>를 모았어요. 아래 지도에서 🔒 표시가 있는 섬으로 가면 "
                     "나머지 부품을 찾을 수 있어요!", mood="happy")
        elif _named:
            neo_says("이름도 적었으니 준비 끝! 먼저 <b>인공지능이 뭐예요?</b>에서 AI가 뭔지 알아본 뒤, "
                     "<b>0번 섬</b>에서 나를 직접 가르쳐 주세요.")
        else:
            neo_says("안녕! 나는 로봇 <b>네오</b>예요. 부품이 우주에 흩어져서 몸이 다 없어졌어요. 😢<br>"
                     "6개 섬에서 AI 원리를 배우면 부품을 하나씩 찾을 수 있대요!")

        if _got >= 6:
            _cta1, _cta2 = st.columns(2)
            with _cta1:
                if st.button("🔧 복습 미션 도전하기", key="home_go_review", type="primary", **_STRETCH):
                    go_to("🔧 로봇 네오 종합 점검")
            with _cta2:
                if st.button("🏆 수료증 받으러 가기 🌟", key="home_go_cert", type="primary", **_STRETCH):
                    go_to("🏆 사후평가 (수료증)")
        elif _named:
            if st.button("🎈 탐험을 시작하겠습니다!", key="home_start", type="primary", **_STRETCH):
                go_to("🌟 인공지능이 뭐예요?")
        else:
            if st.button("✏️ 이름 적고 탐험 시작하기! 🎈", key="home_start", type="primary", **_STRETCH):
                go_to("📝 나의 이름 & 사전 평가")

        tab_map, tab_video, tab_teacher = st.tabs(["🗺️ 탐험 지도", "🎬 소개 영상", "👩‍🏫 선생님 안내"])

        # ── 🗺️ 탐험 지도 ──────────────────────────────────────────────
        with tab_map:
            st.caption("✅ 부품을 얻은 섬 · 🔒 아직 부품이 남은 섬 — 순서대로 하는 게 가장 좋아요!")
            island_map("home")
            if _got >= 6:
                md_html("""
                <div style='background:#FFF3E0; border:2px solid #FFA726; border-radius:14px;
                            padding:12px 16px; text-align:center; margin-top:10px;'>
                    <span style="font-family:'Jua',sans-serif; font-size:17px; color:#E65100;">
                    🎉 6개 부품을 모두 모았어요! 위 버튼으로 복습 미션과 수료증을 만나보세요.</span>
                </div>
                """)

        # ── 🎬 소개 영상 ──────────────────────────────────────────────
        with tab_video:
            VIDEO_PATH = os.path.join(MOVIE_DIR, "movie_intro.mp4")
            col_v1, col_v2, col_v3 = st.columns([1, 3, 1])
            with col_v2:
                if os.path.exists(VIDEO_PATH):
                    st.video(VIDEO_PATH)
                else:
                    st.info("📹 소개 영상이 없어요! `media/movie` 폴더 안에 `movie_intro.mp4` 파일을 넣어주세요.")
            st.caption("🎧 컴퓨터실에서는 헤드폰을 쓰거나 소리를 줄여주세요.")

        # ── 👩‍🏫 선생님 안내 ───────────────────────────────────────────
        with tab_teacher:
            md_html("""
            <div style="background:#EAF4FF; border:2px solid #2D9CDB; border-radius:16px; padding:14px 18px;">
              <div style="font-family:'Jua',sans-serif; font-size:17px; color:#1565C0; margin-bottom:6px;">
                🎯 [6실05-05] 성취기준 달성 지도
              </div>
              <div style="font-family:'Gowun Dodum',sans-serif; font-size:13.5px; color:#22354F; line-height:1.7;">
                <b>대표 성취기준 [6실05-05]</b> 인공지능이 만들어지는 과정을 체험하고,
                인공지능이 사회에 미치는 영향을 탐색한다.
                <div style="margin-top:7px; background:#FFFFFFAA; border-radius:10px; padding:8px 11px;">
                  <b style="color:#1565C0;">① 기계학습의 기본 원리 체험</b><br>
                  🤖 <b>0. 내가 AI 선생님</b> — 이름표 붙이기 → 학습 → 시험 → 편향까지
                  <b>AI가 만들어지는 전 과정</b>을 직접 수행<br>
                  📈⛰️🌳🤝🎨🧠 <b>6개 섬</b> — AI가 규칙을 찾는 방법을 손으로 조작하며 체험
                  <span style="color:#5C7599;">(각 섬 ①②③까지가 기본)</span>
                </div>
                <div style="margin-top:6px; background:#FFFFFFAA; border-radius:10px; padding:8px 11px;">
                  <b style="color:#00695C;">② 사회에 미치는 영향 탐색</b><br>
                  ⚖️ <b>AI를 똑똑하게 쓰려면?</b> — 편향 · 개인정보 · 가짜 정보와 함께
                  <b>사회의 발전과 직업의 변화</b>를 탐색
                </div>
              </div>
            </div>
            <div style="background:#F3F7FB; border-radius:14px; padding:10px 16px; margin-top:8px;
                        font-family:'Gowun Dodum',sans-serif; font-size:13.5px; color:#28406B; text-align:center;">
              🧭 <b>모든 섬은 같은 4단계로 배워요</b> &nbsp;
              <span style="font-family:'Jua',sans-serif; color:#1B2A4A;">
                ① 알아보기 → ② 연습하기 → ③ 게임 도전 🏅 → ④ 자유 탐구(선택)</span>
              <br><span style="font-size:12.5px; color:#5C7599;">
                ①②③(약 15분)까지가 성취기준 도달 지점이고, ③에서 로봇 부품이 나옵니다.</span>
            </div>
            """)

            st.write("")
            md_text("""
            **📌 학습 목표는 '명칭 암기'가 아니라 '원리 체험'입니다.**
            [6실05-05] 해설이 요구하는 것은 *기계학습이 적용된 간단한 인공지능 도구의 체험을 통해
            기계학습의 기본 원리를 이해*하는 것입니다. 학생이 도달해야 할 지점은
            **각 섬의 ①②③을 직접 조작해 보는 것**이며, 선형회귀·경사하강법 같은
            **학술 명칭은 평가 대상이 아닌 참고 정보**로만 제시합니다.
            (학생 화면에는 *"🏷️ 오늘 해볼 일: 한 걸음씩 내려가기"* 처럼 **우리말 이름만** 나오고,
            학술 명칭은 각 섬의 *"🎯 이 활동과 관련된 성취기준"* 펼침 박스와 이 선생님 안내에만 있습니다.)
            """)

            with st.expander("⏰ 수업 시간이 부족하세요? — 차시별 추천 코스", expanded=False):
                st.caption("학급 사정에 맞게 **필요한 섬만 골라** 쓰셔도 됩니다. "
                           "어떤 코스를 골라도 [6실05-05] 성취기준은 달성됩니다.")
                md_text("""
                | 코스 | 차시 | 무엇을 하나요 | 이렇게 도착합니다 |
                |---|---|---|---|
                | **맛보기** | **2차시** | `0. 내가 AI 선생님` → `1. 마법의 선 긋기` → `AI를 똑똑하게 쓰려면?` | AI가 데이터로 배운다는 것과, 편향을 조심해야 한다는 것 |
                | **기본** | **4차시** | 맛보기 + `3. 스무고개 탐정` + `6. 똑똑한 생각 주머니` | 위 내용 + 분류와 신경망까지 |
                | **전체** | **6차시** | 6개 섬 전부 + 복습 + 수료증 | 인공지능 핵심 원리 6가지를 모두 |
                """)
                st.caption("⏱️ **1차시 = 섬 1개**가 기본 속도예요. 시간이 빠듯하면 각 섬의 "
                           "**①②③까지만** 해도 성취기준은 달성됩니다. (④는 선택 활동)")
                st.info("💡 **어느 코스든 이렇게 하세요** ①`나의 이름 & 사전 평가`로 시작 → "
                        "②고른 섬들 진행 → ③`사후평가 (수료증)`에서 사후 검사와 수료증. "
                        "`선생님 방`에서 우리 반 향상도를 바로 확인할 수 있어요.")
                c_a, c_b = st.columns(2)
                with c_a:
                    if st.button("🥉 맛보기 코스로 시작 (2차시)", key="course_short", **_STRETCH):
                        go_to("🤖 0. 내가 AI 선생님")
                with c_b:
                    if st.button("🥇 전체 코스로 시작 (6차시)", key="course_full", **_STRETCH):
                        go_to("📝 나의 이름 & 사전 평가")

            with st.expander("📚 교과 연계 상세 (2022 개정 교육과정 5~6학년군)", expanded=False):
                md_text("""
                이 프로그램의 활동은 아래 **실과(정보) 영역** 성취기준을 공통 기반으로 합니다.

                - **[6실05-04]** 디지털 데이터와 아날로그 데이터의 특징을 이해하고, 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
                - **[6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다. *(대표 성취기준)*

                각 활동은 여기에 더해 아래 교과와도 연계할 수 있습니다.

                | 활동 | 핵심 AI 개념 | 추가 연계 교과·성취기준 |
                |---|---|---|
                | 🎯 붕어빵 가격 맞히기 | 선형회귀 | 수학 [6수04-01] 규칙과 대응 |
                | 🏂 눈꽃 골짜기 스노보드 | 경사하강법 | 실과 [6실05-04] 데이터 탐색 |
                | 🕵️‍♂️ 동물 스무고개 명탐정 | 결정트리 | 사회 [6사03-04] 정보화 시대의 변화상 |
                | 🍇 외계 알 구출작전 | KNN | 실과 [6실05-04] 데이터 탐색 |
                | 🧹 무인도 분리수거 로봇 | K-Means | 실과 [6실05-04] 데이터 탐색 |
                | 🤖 로봇 네오 발사 | 인공신경망 | 실과 [6실04-06] 로봇의 작동 원리 |
                | 🏙️ AI와 직업의 변화 | 사회 영향 | 사회 [6사03-04] 정보화 시대의 변화상 |

                각 섬 상단의 "🎯 이 활동과 관련된 성취기준" 펼침 박스에서 자세한 내용을 확인할 수 있습니다.
                """)

            if st.button("👩‍🏫 선생님 방 열기 (학습 분석 대시보드)", key="home_go_teacher", **_STRETCH):
                st.session_state["teacher_unlocked"] = True
                go_to("👩‍🏫 선생님 방 (학습 분석)")

    # --- [메뉴: 이름 & 사전 평가] ---
    elif menu == "📝 나의 이름 & 사전 평가":
        hero_card("📝", "탐험 시작 전, 나를 알려줘요!",
                   "이름을 적고 간단한 사전 퀴즈를 풀면, 탐험이 끝난 뒤 얼마나 자랐는지 확인할 수 있어요.",
                   "#5C6BC0")

        neo_says("안녕! 나는 로봇 <b>네오</b>예요. 이름을 적어 주면 탐험이 끝난 뒤 <b>얼마나 자랐는지</b> "
                 "함께 확인할 수 있어요. 사전 퀴즈는 <b>점수랑 상관없으니</b> 편하게 풀어요!")
        st.markdown("### 1️⃣ 학급 코드와 내 이름 적기")
        nc1, nc2 = st.columns(2)
        with nc1:
            class_in = st.text_input("학급 코드 (선생님이 알려주세요)", value=st.session_state.get("class_code", ""),
                                     placeholder="예: 6-3", key="class_input_pre", max_chars=20)
        with nc2:
            name_in = st.text_input("이름", value=st.session_state.get("student_name", ""),
                                    placeholder="예: 홍길동", key="name_input_pre", max_chars=20)

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
            st.write("이제 탐험을 시작해봐요. 6개 섬을 모두 마친 뒤 '🏆 사후평가 (수료증)' 방에서 사후 검사를 하면 얼마나 자랐는지 알 수 있어요!")
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
                    st.session_state.pop("_diag_part1_done_pre", None)
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
            "3️⃣ 이름표 보고 배우기",
            "4️⃣ 이름표 없이 스스로",
            "5️⃣ 생활 속 AI 찾기",
            "6️⃣ 6가지 섬 미리보기",
        ])

        # ===== [1단계: AI가 뭐예요?] =========================================
        with tab_w1:
            st.markdown("### 🤔 인공지능(AI)이 뭘까요?")
            md_html("""
            <div style='font-size:16px; line-height:1.75; background:#FFF8E1; border-radius:14px; padding:13px 18px;'>
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
            st.caption("2번 탭에서 AI가 예시를 보고 어떻게 점점 똑똑해지는지 직접 눈으로 확인해봐요.")

        # ===== [2단계: AI는 어떻게 배워요?] ==================================
        with tab_w2:
            st.markdown("### 🐶🐱 AI에게 '강아지와 고양이 구분하기'를 가르쳐볼까요?")
            st.write("AI는 예시를 **많이 볼수록** 똑똑해져요. 예시(사진) 개수를 골라서, AI가 얼마나 잘 맞히는지 비교해봐요!")

            n_examples = st.select_slider(
                "🖼️ AI에게 보여줄 예시 사진 개수를 골라보세요!",
                options=[3, 30, 300, 3000], value=3, key="ai_learn_examples")

            # ⚠️ 사실 확인: 예시를 아무리 많이 줘도 AI가 100%를 맞히지는 않는다.
            #    여기서 10/10을 보여 주면 "AI는 완벽하다"는 잘못된 인상을 주고,
            #    사전·사후 진단 5번("AI가 자신 있게 답해도 틀릴 수 있다")과도 어긋난다.
            #    그래서 맞힌 개수를 단계마다 직접 정하고, 최고 단계도 9/10에서 멈춘다.
            # 강아지/고양이 둘 중 하나를 고르는 문제라 '찍으면' 10개 중 5개다.
            # 그래서 3장일 때를 5개로 두어야 "찍는 수준"이라는 말이 사실이 된다.
            # 진행바도 글자와 같은 값을 쓰도록 acc = n_ok * 10 으로 맞춘다.
            if n_examples == 3:
                n_ok, face, msg, color, bg = (
                    5, "😖", "3장으론 감이 안 와요. <b>동전 던지기</b>랑 똑같은 수준이에요!", "#C62828", "#FFEBEE")
            elif n_examples == 30:
                n_ok, face, msg, color, bg = (
                    7, "🙂", "“고양이는 귀가 뾰족하네?” 규칙이 보이기 시작해요!", "#F9A825", "#FFF8E1")
            elif n_examples == 300:
                n_ok, face, msg, color, bg = (
                    8, "😄", "이제 웬만한 사진은 맞혀요. 규칙이 꽤 정확해졌어요!", "#558B2F", "#F1F8E9")
            else:
                n_ok, face, msg, color, bg = (
                    9, "🤩",
                    "거의 다 맞혀요! 그래도 <b>가끔은 틀려요.</b> 아무리 많이 배워도 AI는 100%가 아니랍니다.",
                    "#2E7D32", "#E8F5E9")
            acc = n_ok * 10

            # ── ① AI가 '본' 사진을 실제로 보여준다 ──────────────────────
            _pool = ["🐶", "🐱", "🐱", "🐶", "🐶", "🐱", "🐶", "🐱", "🐱", "🐶"]
            _shown = 3 if n_examples == 3 else 10
            _pic = "border:2px solid #D3E2F5; border-radius:10px; background:#F4F8FF; padding:3px 6px; font-size:22px; margin:2px;"
            _train = "".join(f"<span style='display:inline-block; {_pic}'>{_pool[i % 10]}</span>"
                             for i in range(_shown))
            _more = (f"<span style='display:inline-block; {_pic} color:#5C7599; font-size:13px; "
                     f"font-family:\'Jua\',sans-serif;'>…+{n_examples - _shown}장</span>"
                     if n_examples > _shown else "")

            # ── ② 시험 10문제 채점표 — 실력이 그대로 ⭕❌ 개수로 보인다 ──
            _n_ok = n_ok
            _order = [0, 5, 2, 8, 1, 7, 4, 9, 3, 6]      # 매번 흔들리지 않게 고정 패턴
            _ok = set(_order[:_n_ok])
            _test = "".join(
                f"<span style='display:inline-block; border:2px solid {'#66BB6A' if i in _ok else '#EF5350'}; "
                f"border-radius:10px; background:{'#E8F5E9' if i in _ok else '#FFEBEE'}; "
                f"padding:2px 5px; margin:2px; font-size:19px; line-height:1.2;'>"
                f"{_pool[i]}<br><span style='font-size:13px;'>{'⭕' if i in _ok else '❌'}</span></span>"
                for i in range(10))

            md_html(f"""
            <div style='background:{bg}; border-radius:16px; padding:14px 16px;'>
              <div style='font-size:13.5px; color:#3A4A6B; margin-bottom:3px;'>
                📚 <b>AI가 공부한 사진 {n_examples}장</b></div>
              <div>{_train}{_more}</div>
              <div style='font-size:13.5px; color:#3A4A6B; margin:9px 0 3px 0;'>
                📝 <b>시험! 처음 보는 사진 10장을 맞혀보래요</b></div>
              <div>{_test}</div>
              <div style='text-align:center; margin-top:9px;'>
                <span style='font-size:34px;'>{face}</span>
                <span style="font-family:'Jua',sans-serif; font-size:23px; color:{color}; margin-left:8px;">
                  10개 중 {_n_ok}개 정답!</span>
                <div style='font-size:14px; color:#3A4A6B; margin-top:3px;'>{msg}</div>
              </div>
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
            st.caption("그런데 AI가 배우는 방법에는 크게 두 가지가 있어요. 3번, 4번 탭에서 알아봐요!")

        # ===== [3단계: 지도학습] =============================================
        with tab_w3:
            st.markdown("### 🏷️ 이름표를 보고 배워요!")
            md_html("""
            <div style='font-size:19px; line-height:1.8; background:#FFF8E1;
                        border-radius:14px; padding:14px 20px;'>
            <b>정답(이름표)을 알려주면서</b> 가르치는 방법이에요.<br>
            선생님이 낱말카드로 <b>"이건 사과 🍎"</b> 하고 알려주시는 것과 똑같아요.
            </div>
            """)

            st.write("")
            # 글자를 줄이고 그림을 크게 — 세 칸이 한눈에 들어오게 한다
            _big = ("border-radius:16px; padding:14px 8px; text-align:center;"
                    "font-size:15px; line-height:1.5;")
            md_html(f"""
            <div style='display:flex; gap:8px; align-items:stretch;'>
              <div style='flex:1.4; background:#FFF3E0; {_big}'>
                <div style='font-size:14px; color:#8D6E63;'><b>① 이름표를 붙여 줘요</b></div>
                <div style='font-size:46px; letter-spacing:4px; margin:8px 0;'>🍎🍌🍎🍌</div>
                <div style='font-size:14px; color:#5D4037;'>"사과" "바나나"<br>"사과" "바나나"</div>
              </div>
              <div style='display:flex; align-items:center; font-size:30px;'>➡️</div>
              <div style='flex:1; background:#F3E5F5; {_big}'>
                <div style='font-size:46px; margin:8px 0;'>🧠</div>
                <div style='color:#6A1B9A;'><b>규칙 찾기!</b><br>
                <span style='font-size:14px;'>"빨갛고 동그라면<br>사과구나!"</span></div>
              </div>
              <div style='display:flex; align-items:center; font-size:30px;'>➡️</div>
              <div style='flex:1; background:#E8F5E9; {_big}'>
                <div style='font-size:14px; color:#558B2F;'><b>③ 새 과일도 척척</b></div>
                <div style='font-size:46px; margin:8px 0;'>🍎</div>
                <div style='color:#2E7D32;'><b>"사과예요!" ✅</b></div>
              </div>
            </div>
            """)

            st.write("")
            st.markdown("#### 🎮 내가 AI라면?")
            md_html(f"""
            <div style='background:#FAFAFA; border-radius:16px; padding:14px 10px;'>
              <div style='display:flex; gap:8px;'>
                <div style='flex:1; background:#E3F2FD; {_big}'>
                    <div style='font-size:44px;'>🐦</div><b>하늘 친구</b></div>
                <div style='flex:1; background:#E3F2FD; {_big}'>
                    <div style='font-size:44px;'>🦅</div><b>하늘 친구</b></div>
                <div style='flex:1; background:#E0F7FA; {_big}'>
                    <div style='font-size:44px;'>🐟</div><b>물 친구</b></div>
                <div style='flex:1; background:#E0F7FA; {_big}'>
                    <div style='font-size:44px;'>🐙</div><b>물 친구</b></div>
              </div>
            </div>
            """)
            sup_pick = st.radio("❓ 하늘을 나는 **🦆 오리**는 뭐라고 답할까요?",
                                ["선택 안 함", "하늘 친구", "물 친구"],
                                horizontal=True, key="sup_learn_quiz")
            if sup_pick == "하늘 친구":
                st.success("🎉 정답! 예시에서 배운 규칙(**날면 하늘 친구**)으로 맞혔어요. "
                           "방금 여러분이 한 일이 바로 AI가 하는 일이에요!")
            elif sup_pick == "물 친구":
                st.warning("오리는 **하늘을 난다**고 했죠? 예시에서 나는 친구들은 '하늘 친구'였어요.")

            st.caption("💡 **선 긋기 · 스무고개 · 가장 친한 친구** 섬이 이렇게 배우는 곳이에요.")

        # ===== [4단계: 비지도학습] ===========================================
        with tab_w4:
            st.markdown("### 🧩 이름표 없이 스스로 모둠을 만들어요!")
            md_html("""
            <div style='font-size:19px; line-height:1.8; background:#FFF8E1;
                        border-radius:14px; padding:14px 20px;'>
            이번엔 <b>이름표(정답)가 하나도 없어요!</b><br>
            어질러진 장난감을 <b>"비슷한 것끼리 모아볼까?"</b> 하고 스스로 나누는 것과 똑같아요.
            </div>
            """)

            st.write("")
            unsup_run = st.radio("어질러진 장난감을 AI에게 맡기면?",
                                 ["😵 정리 전 (뒤죽박죽)", "✨ AI가 정리한 후"],
                                 horizontal=True, key="unsup_demo")
            _lab2 = "border-radius:16px; padding:16px 8px; text-align:center;"
            if unsup_run.startswith("😵"):
                md_html(f"""
                <div style='background:#FFEBEE; {_lab2}'>
                    <div style='font-size:52px; letter-spacing:8px; line-height:1.4;'>🚗🧸⚽🧸🚕⚾🚙🏀🧸</div>
                    <div style='font-size:17px; color:#C62828; margin-top:8px;'>이름표도 없고, 뒤죽박죽! 😵</div>
                </div>
                """)
                st.caption("👆 위에서 **'✨ AI가 정리한 후'** 를 눌러보세요!")
            else:
                md_html(f"""
                <div style='display:flex; gap:10px;'>
                  <div style='flex:1; background:#E3F2FD; {_lab2}'>
                      <div style='font-size:48px; letter-spacing:4px;'>🚗🚕🚙</div>
                      <div style='font-size:16px; color:#1565C0; margin-top:6px;'><b>바퀴 친구들</b></div></div>
                  <div style='flex:1; background:#FFF9C4; {_lab2}'>
                      <div style='font-size:48px; letter-spacing:4px;'>⚽⚾🏀</div>
                      <div style='font-size:16px; color:#F57F17; margin-top:6px;'><b>동글동글 공</b></div></div>
                  <div style='flex:1; background:#F3E5F5; {_lab2}'>
                      <div style='font-size:48px; letter-spacing:4px;'>🧸🧸🧸</div>
                      <div style='font-size:16px; color:#6A1B9A; margin-top:6px;'><b>포근한 인형</b></div></div>
                </div>
                """)
                st.success("✨ 정답을 아무도 안 알려줬는데 **비슷한 것끼리 스스로** 모였어요!")

            st.write("")
            v1, v2 = st.columns(2)
            with v1:
                md_html("""
                <div style='background:#E8F5E9; border:2px solid #2E7D32; border-radius:16px;
                            padding:14px; text-align:center;'>
                    <div style='font-size:44px;'>🏷️</div>
                    <div style="font-family:'Jua',sans-serif; font-size:20px; color:#2E7D32;">이름표 보고 배우기</div>
                    <div style='font-size:16px; color:#3A4A6B; margin-top:6px;'>
                        정답을 <b>알려줘요</b></div>
                </div>
                """)
            with v2:
                md_html("""
                <div style='background:#F3E5F5; border:2px solid #8E24AA; border-radius:16px;
                            padding:14px; text-align:center;'>
                    <div style='font-size:44px;'>🧩</div>
                    <div style="font-family:'Jua',sans-serif; font-size:20px; color:#8E24AA;">이름표 없이 배우기</div>
                    <div style='font-size:16px; color:#3A4A6B; margin-top:6px;'>
                        정답을 <b>안 알려줘요</b></div>
                </div>
                """)

            st.write("")
            chk = st.radio("❓ **이름표 없이** '비슷한 사진끼리 앨범을 나눠줘!' — 어느 쪽일까요?",
                           ["선택 안 함", "🏷️ 이름표 보고 배우기", "🧩 이름표 없이 스스로 배우기"],
                           horizontal=True, key="sup_vs_unsup_quiz")
            if chk == "🧩 이름표 없이 스스로 배우기":
                st.success("🎉 정답! 정답을 안 줬으니까요. 완벽해요!")
            elif chk == "🏷️ 이름표 보고 배우기":
                st.warning("힌트! 정답(이름표)을 **줬나요, 안 줬나요?**")

            st.caption("💡 **🎨 비슷한 친구끼리** 섬이 이렇게 배우는 곳이에요.")

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
                    <b>🏷️ 이름표를 보고 배우기</b><br>
                    📈 마법의 선 긋기 · 🌳 스무고개 탐정<br>
                    🤝 가장 친한 친구 · 🧠 생각 주머니<br>
                    <span style='font-size:12.5px; color:#555;'>
                    (⛰️ 보물찾기 산은 배우는 <b>방법</b>이 아니라,
                    <b>더 잘 맞히려고 고쳐 나가는 요령</b>이라 따로예요)</span>
                </div>
                """)
            with s2:
                md_html("""
                <div style='background:#F3E5F5; border-radius:14px; padding:14px 18px; font-size:14px; line-height:2.0;'>
                    <b>🧩 이름표 없이 스스로 배우기</b><br>
                    🎨 비슷한 친구끼리<br>
                    <span style='font-size:12px; color:#666;'>정답 없이 비슷한 것끼리 스스로 모둠을 만들어요!</span>
                </div>
                """)

            st.write("")
            st.info("🤖 먼저 **내가 직접 AI를 가르쳐보는** 체험부터 해볼까요? "
                    "AI가 어떻게 만들어지는지 손으로 겪어본 뒤에 6개 섬으로 떠나요!")
            if st.button("🤖 좋아요, 내가 AI 선생님이 되어볼래요!", **_STRETCH):
                go_to("🤖 0. 내가 AI 선생님")

    # --- [메뉴 1.5: 내가 AI 선생님 (기계학습 기본 원리 체험)] ---
    elif menu == "🤖 0. 내가 AI 선생님":
        hero_card("🤖", "내가 AI 선생님: 네오를 가르쳐요!",
                  "AI는 어떻게 만들어질까요? 내가 직접 이름표를 붙여서 네오를 똑똑하게 만들어봐요.",
                  "#F4A259", term="내가 AI 가르치기")

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 **만들어지는 과정을 체험**하고, 인공지능이 사회에 미치는 영향을 탐색한다.
              → 이름표(정답)를 직접 붙여 AI를 학습시키고, 시험까지 보게 하며 **인공지능이 만들어지는 전체 과정**을 체험해요.
            - **[실과] [6실05-04]** 인공지능에 활용할 수 있는 **데이터의 유형이나 형태**를 탐색한다.
              → 색깔·모양·크기·반짝임 같은 **특징(데이터)**을 보고 AI가 판단한다는 것을 알게 돼요.
            """)

        level_guide(has_free=False, minutes=20)

        # 탭(st.tabs)은 화면 안 요소가 바뀌면 선택이 1단계로 되돌아가는 문제가 있어,
        # 상태가 확실히 유지되는 '단계 고르기' 버튼으로 만든다. 태블릿 터치에도 더 편하다.
        # 이 섬은 네 단계 모두가 [6실05-05]의 핵심 체험(자료 → 이름표 → 학습 → 시험 → 편향)이라
        # ④까지 전부 '기본'이다.
        NEO_STEPS = [
            "① 문제 만나기",
            "② 네오 가르치기",
            "③ 거꾸로 가르치기",
            "④ 한쪽만 가르치기",
        ]
        # 예전 버전의 단계 이름이 세션에 남아 있으면 라디오가 오류를 내므로 미리 비운다
        if st.session_state.get("neo_step") not in NEO_STEPS:
            st.session_state.pop("neo_step", None)
        step = st.radio("단계 고르기", NEO_STEPS, key="neo_step",
                        horizontal=True, label_visibility="collapsed")
        st.divider()

        # ===== [1단계: 문제 상황] =========================================
        if step == NEO_STEPS[0]:
            st.markdown("### 🛸 우주에 물건들이 둥둥 떠다녀요")
            md_html("""
            <div style='font-size:17px; line-height:1.9; background:#FFF6E5; border-radius:14px; padding:16px 22px;'>
            로봇 <b>네오</b>의 부품이 우주에 흩어졌어요. 그런데 우주에는 <b>쓰레기</b>도 같이 떠다녀요.<br>
            네오가 스스로 <b>부품과 쓰레기를 구분</b>할 수 있으면 좋겠죠?<br><br>
            그런데 네오는 <b>아직 아무것도 몰라요.</b> 누군가 가르쳐 줘야 해요.<br>
            오늘은 <b>여러분이 네오의 선생님</b>이에요! 👩‍🏫
            </div>
            """)

            st.write("")
            st.markdown("#### 🔍 물건에는 이런 특징이 있어요")
            _a = neo_item_html("파랑", "동그라미", True, True, caption="파랑·동그라미<br>큼·✨반짝")
            _b = neo_item_html("주황", "네모", False, False, caption="주황·네모<br>작음·안 반짝")
            md_html("<div style='display:flex; gap:10px; flex-wrap:wrap; justify-content:center;'>"
                    + _a + _b + "</div>")
            st.caption("👀 색깔 / 모양 / 크기 / 반짝임 — 이렇게 눈에 보이는 것들을 **특징(데이터)** 이라고 해요.")

            st.write("")
            st.info("🤫 **비밀!** 사실 네오의 진짜 부품은 **✨반짝이는 것**이에요. "
                    "색깔·모양·크기는 상관없어요. 하지만 **네오는 이 비밀을 몰라요!** "
                    "여러분이 이름표를 붙여주면, 네오가 **스스로 규칙을 찾아낼** 거예요.")

            st.write("")
            md_html("""
            <div style='background:#EAF4FF; border-radius:14px; padding:14px 20px; font-size:16px; line-height:1.9;'>
            🧭 <b>AI는 이렇게 만들어져요</b><br>
            ① 자료 모으기 → ② <b>이름표 붙이기</b> → ③ <b>가르치기(학습)</b> → ④ <b>시험 보기</b>
            </div>
            """)
            st.caption("우리가 앞으로 만날 6개의 섬은, 이 중 ③ '가르치기' 안에서 기계가 쓰는 **6가지 방법**이에요!")

        # ===== [2단계: 라벨링 → 학습 → 규칙 공개 → 시험] ==================
        elif step == NEO_STEPS[1]:
            st.markdown("### 🏷️ 물건 8개에 이름표를 붙여주세요")
            st.write("✨**반짝이면 부품**, 안 반짝이면 쓰레기예요. 아래에서 골라주세요!")
            st.caption("🔧 = 네오의 부품 　 🗑️ = 우주 쓰레기 　 ❓ = 아직 안 정함")

            # 도움 버튼은 라디오보다 '먼저' 그려야 한다.
            # 그래야 눌렀을 때 라디오가 만들어지기 전에 값을 넣을 수 있고,
            # st.rerun() 없이 처리되어 보고 있던 탭이 1단계로 튕기지 않는다.
            hc1, hc2 = st.columns(2)
            with hc1:
                fill = st.button("✍️ 한 번에 알맞게 붙이기 (도움)", key="neo_autofill", **_STRETCH)
            with hc2:
                clear = st.button("🧹 다시 고르기", key="neo_clear", **_STRETCH)
            if fill:
                for _i, _item in enumerate(NEO_TRAIN):
                    st.session_state[f"neo_lab_{_i}"] = "🔧" if _item[3] else "🗑️"
            if clear:
                for _i in range(len(NEO_TRAIN)):
                    st.session_state[f"neo_lab_{_i}"] = "❓"
                st.session_state.pop("neo_trained", None)

            st.write("")
            cols = st.columns(4)
            labels = []
            for i, item in enumerate(NEO_TRAIN):
                with cols[i % 4]:
                    md_html(neo_item_html(*item, size=66))
                    # 태블릿에서 세로로 길어지지 않도록 가로 배치 + 짧은 이모지 라벨
                    pick = st.radio(
                        f"{i+1}번",
                        ["❓", "🔧", "🗑️"],
                        key=f"neo_lab_{i}", label_visibility="collapsed",
                        horizontal=True,
                    )
                    labels.append(pick)

            done = [l for l in labels if l != "❓"]
            ready = len(done) == len(NEO_TRAIN)
            st.progress(len(done) / len(NEO_TRAIN))
            # 화면에 그려지는 요소의 '개수'가 바뀌면 보고 있던 탭이 1단계로 튕긴다.
            # 그래서 조건에 따라 나타났다 사라지게 하지 않고, 항상 같은 개수를 그리되
            # 내용과 활성화 여부만 바꾼다.
            st.caption(f"이름표 {len(done)} / {len(NEO_TRAIN)}개 붙임"
                       + ("  ✅ 다 붙였어요! 아래 버튼을 눌러보세요."
                          if ready else "  👆 8개를 모두 붙여야 네오를 가르칠 수 있어요."))

            st.write("")
            # st.rerun()도 쓰지 않는다. 버튼을 누른 그 순간 바로 아래에서 결과를 보여준다.
            if st.button("🤖 네오야, 배워라!", key="neo_train_btn", type="primary",
                         disabled=not ready, **_STRETCH):
                st.session_state["neo_trained"] = [1 if l == "🔧" else 0 for l in labels]


            if st.session_state.get("neo_trained"):
                y = st.session_state["neo_trained"]
                X = [neo_features(it) for it in NEO_TRAIN]
                model, rule = neo_fit(X, y)

                st.success("🎉 네오가 여러분이 붙인 이름표 8개로 공부를 마쳤어요!")
                md_html("<div style='background:#FFF8E1; border:3px solid #FFB300; border-radius:16px; "
                        "padding:16px 20px; text-align:center;'>"
                        "<div style=\"font-family:'Jua',sans-serif; font-size:17px; color:#E65100;\">"
                        "🧠 네오가 스스로 찾아낸 규칙</div>"
                        f"<div style=\"font-family:'Jua',sans-serif; font-size:22px; color:#1B2A4A; "
                        f"margin-top:8px;\">{rule}</div></div>")
                st.caption("👏 아무도 규칙을 알려주지 않았는데, 네오가 **예시만 보고 스스로** 찾아냈어요. "
                           "이게 바로 **기계학습**이에요!")

                st.write("")
                st.markdown("#### 📝 시험 시간! 네오가 처음 보는 물건이에요")
                neo_exam(model, NEO_TEST, key_prefix="ok")

        # ===== [3단계: 반대로 가르치기] ===================================
        elif step == NEO_STEPS[2]:
            st.markdown("### 😈 일부러 반대로 가르치면 어떻게 될까요?")
            md_html("""
            <div style='font-size:18px; line-height:1.8; background:#FDECEF; border-radius:14px; padding:14px 20px;'>
            이번엔 <b>일부러 거짓말로</b> 가르쳐 봐요.<br>
            ✨<b>반짝이면 🗑️ 쓰레기</b>, 안 반짝이면 <b>🔧 부품</b> — <b>반대로</b> 붙여주세요!
            </div>
            """)

            st.write("")
            # 학생이 직접 반대 이름표를 붙이게 한다. (예전에는 버튼 하나로 정답이 바로 나왔다)
            bc1, bc2 = st.columns(2)
            with bc1:
                bfill = st.button("😈 반대로 한 번에 붙이기 (도움)", key="neo_bad_autofill", **_STRETCH)
            with bc2:
                bclear = st.button("🧹 다시 고르기", key="neo_bad_clear", **_STRETCH)
            if bfill:
                for _i, _item in enumerate(NEO_TRAIN):
                    st.session_state[f"neo_bad_lab_{_i}"] = "🗑️" if _item[3] else "🔧"
            if bclear:
                for _i in range(len(NEO_TRAIN)):
                    st.session_state[f"neo_bad_lab_{_i}"] = "❓"
                st.session_state.pop("neo_flipped", None)

            st.write("")
            bcols = st.columns(4)
            bad_labels = []
            for i, item in enumerate(NEO_TRAIN):
                with bcols[i % 4]:
                    md_html(neo_item_html(*item, size=66))
                    bad_labels.append(st.radio(
                        f"{i+1}번", ["❓", "🔧", "🗑️"],
                        key=f"neo_bad_lab_{i}", label_visibility="collapsed", horizontal=True))

            bdone = [l for l in bad_labels if l != "❓"]
            bready = len(bdone) == len(NEO_TRAIN)
            st.progress(len(bdone) / len(NEO_TRAIN))
            st.caption(f"이름표 {len(bdone)} / {len(NEO_TRAIN)}개 붙임"
                       + ("  ✅ 다 붙였어요! 아래 버튼을 눌러보세요."
                          if bready else "  👆 8개를 모두 붙여야 네오를 가르칠 수 있어요."))

            st.write("")
            if st.button("😈 네오야, 이대로 배워라!", key="neo_flip_btn", type="primary",
                         disabled=not bready, **_STRETCH):
                st.session_state["neo_flipped"] = [1 if l == "🔧" else 0 for l in bad_labels]

            if st.session_state.get("neo_flipped"):
                y_bad = st.session_state["neo_flipped"]      # 학생이 직접 붙인 이름표
                X = [neo_features(it) for it in NEO_TRAIN]
                model_bad, rule_bad = neo_fit(X, y_bad)

                md_html("<div style='background:#FDECEF; border:3px solid #E57373; border-radius:16px; "
                        "padding:16px 20px; text-align:center;'>"
                        "<div style=\"font-family:'Jua',sans-serif; font-size:17px; color:#C62828;\">"
                        "🧠 네오가 찾아낸 규칙</div>"
                        f"<div style=\"font-family:'Jua',sans-serif; font-size:22px; color:#1B2A4A; "
                        f"margin-top:8px;\">{rule_bad}</div></div>")
                st.write("")
                st.markdown("#### 📝 같은 시험을 다시 보면?")
                _bad_ok, _bad_total = neo_exam(model_bad, NEO_TEST, key_prefix="bad")

                st.write("")
                # 학생이 이름표를 어떻게 붙였는지에 따라 결과가 달라지므로,
                # '반대로 대답했다'고 단정하지 않고 실제 채점 결과에 맞춰 말한다.
                if _bad_ok == 0:
                    st.error("😮 네오가 **하나도 못 맞혔죠?** 네오가 바보라서가 아니라 "
                             "**여러분이 그렇게 가르쳤기 때문**이에요.")
                elif _bad_ok < _bad_total:
                    st.error(f"😮 네오가 {_bad_total - _bad_ok}개나 틀렸죠? 네오가 바보라서가 아니라 "
                             "**여러분이 붙인 이름표대로 배웠기 때문**이에요.")
                else:
                    st.info("🤔 이번엔 네오가 다 맞혔네요! 이름표를 **반대로** 붙였는지 다시 확인해 볼까요? "
                            "(✨반짝이면 🗑️ 쓰레기로 붙여야 해요)")
                md_text("""
                > ### 💡 오늘의 큰 깨달음
                > **AI는 사람이 가르친 대로만 배워요.** 네오는 '반짝임'이 뭔지 몰라요.
                > 우리가 붙인 이름표만 보고 규칙을 찾을 뿐이에요.
                """)

        # ===== [4단계: 치우친 데이터 = 편향] ==============================
        elif step == NEO_STEPS[3]:
            st.markdown("### 📦 두 번째 상자를 열어봐요")
            md_html("""
            <div style='font-size:18px; line-height:1.8; background:#EAF4FF; border-radius:14px; padding:14px 20px;'>
            <b>새로운 상자</b>예요. 이번엔 <b>파란 것이 🔧 부품</b>, <b>주황 것이 🗑️ 쓰레기</b>랍니다.<br>
            보이는 대로 이름표를 붙여 주세요!
            </div>
            """)

            st.write("")
            # 학생이 직접 이름표를 붙이게 한다. (예전에는 버튼 하나로 결과가 바로 나왔다)
            zc1, zc2 = st.columns(2)
            with zc1:
                zfill = st.button("✍️ 한 번에 알맞게 붙이기 (도움)", key="neo_bias_autofill", **_STRETCH)
            with zc2:
                zclear = st.button("🧹 다시 고르기", key="neo_bias_clear", **_STRETCH)
            if zfill:
                for _i, (_it, _lab) in enumerate(NEO_BIAS_TRAIN):
                    st.session_state[f"neo_bias_lab_{_i}"] = "🔧" if _lab else "🗑️"
            if zclear:
                for _i in range(len(NEO_BIAS_TRAIN)):
                    st.session_state[f"neo_bias_lab_{_i}"] = "❓"
                st.session_state.pop("neo_biased", None)

            st.write("")
            zcols = st.columns(4)
            bias_labels = []
            for i, (item, _lab) in enumerate(NEO_BIAS_TRAIN):
                with zcols[i % 4]:
                    md_html(neo_item_html(*item, size=62))
                    bias_labels.append(st.radio(
                        f"{i+1}번", ["❓", "🔧", "🗑️"],
                        key=f"neo_bias_lab_{i}", label_visibility="collapsed", horizontal=True))

            zdone = [l for l in bias_labels if l != "❓"]
            zready = len(zdone) == len(NEO_BIAS_TRAIN)
            st.progress(len(zdone) / len(NEO_BIAS_TRAIN))
            st.caption(f"이름표 {len(zdone)} / {len(NEO_BIAS_TRAIN)}개 붙임"
                       + ("  ✅ 다 붙였어요! 아래 버튼을 눌러보세요."
                          if zready else "  👆 8개를 모두 붙여야 네오를 가르칠 수 있어요."))

            st.write("")
            if st.button("🤖 이 자료로 네오 가르치기", key="neo_bias_btn", type="primary",
                         disabled=not zready, **_STRETCH):
                st.session_state["neo_biased"] = [1 if l == "🔧" else 0 for l in bias_labels]

            if st.session_state.get("neo_biased"):
                X = [neo_features(it) for it, _ in NEO_BIAS_TRAIN]
                y = st.session_state["neo_biased"]           # 학생이 직접 붙인 이름표
                model_b, rule_b = neo_fit(X, y)

                md_html("<div style='background:#EAF4FF; border:3px solid #2D9CDB; border-radius:16px; "
                        "padding:16px 20px; text-align:center;'>"
                        "<div style=\"font-family:'Jua',sans-serif; font-size:17px; color:#1565C0;\">"
                        "🧠 네오가 찾아낸 규칙</div>"
                        f"<div style=\"font-family:'Jua',sans-serif; font-size:22px; color:#1B2A4A; "
                        f"margin-top:8px;\">{rule_b}</div></div>")
                st.caption("네오가 본 물건들 안에서는 '색깔'만 봐도 100% 맞출 수 있었거든요. "
                           "그래서 네오는 **색깔이 정답**이라고 믿게 됐어요.")

                st.write("")
                st.markdown("#### 📝 그런데 이 상자에는 **주황색 부품**도 있었어요!")
                _bias_ok, _bias_total = neo_exam(model_b, NEO_BIAS_TEST, key_prefix="bias",
                                                 truths=NEO_BIAS_TRUTH)

                st.write("")
                # 학생이 다르게 붙였을 수도 있으니 실제 채점 결과에 맞춰 말한다.
                if _bias_ok < _bias_total:
                    st.error("😮 네오가 **주황색 진짜 부품**을 쓰레기라고 버렸어요! "
                             "**주황색 부품을 한 번도 못 봤기 때문**이에요.")
                else:
                    st.info("🤔 이번엔 네오가 다 맞혔네요! 보이는 대로 "
                            "**파란 것 = 🔧 부품, 주황 것 = 🗑️ 쓰레기**로 붙였는지 확인해 볼까요?")
                md_text("""
                > ### 💡 이것을 **'치우친 자료'** 라고 해요
                > 평생 빨간 사과만 본 사람이 초록 사과를 보고 "이건 사과가 아니야!" 하는 것과 같아요. 🍏
                > 그래서 AI를 가르칠 때는 **골고루, 다양하게** 보여줘야 해요!
                """)

                st.write("")
                st.success("🏅 축하해요! 여러분은 이제 **AI 선생님**이에요. "
                           "AI가 어떻게 만들어지는지 직접 겪어봤어요!")
                if not st.session_state.get("neo_done"):
                    st.session_state["neo_done"] = True
                    st.balloons()

            st.write("")
            if st.button("🚀 이제 진짜 탐험을 시작해볼까요?", key="neo_go_next", **_STRETCH):
                go_to("📈 1. 마법의 선 긋기")


    # --- [메뉴 2: 마법의 선 긋기 (선형회귀)] ---
    elif menu == "📈 1. 마법의 선 긋기":
        hero_card("🎯", "마법의 선: 점들 사이로 선을 예쁘게 그려요!",
                   "붕어빵 크기와 가격의 관계를 찾아 미래를 예측하는 AI 마법사가 되어봐요.",
                   SECTION_COLORS["reg"], term="점들 사이로 선 긋기")

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
              → 붕어빵 크기와 가격의 관계를 나타내는 '선'을 직접 찾아보며 **기계학습(선형회귀)의 기본 원리**를 체험해요.
            - **[실과] [6실05-04]** 디지털 데이터와 아날로그 데이터의 특징을 이해하고, 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
              → 크기·가격 같은 숫자 데이터가 어떻게 AI 예측에 쓰이는지 탐색해요.
            - **[수학] [6수04-01]** 한 양이 변할 때 다른 양이 그에 종속하여 변하는 대응 관계를 나타낸 표에서 규칙을 찾아 설명하고, □, △ 등을 사용하여 식으로 나타낼 수 있다.
              → 붕어빵 '크기'와 '가격'의 대응 관계를 표와 그래프로 나타내고 규칙(식)을 찾는 활동과 직접 연결돼요.
            """)

        level_guide(minutes=15)
        real_world("linear", [
            ("📺", "유튜브", "여러분이 이 영상을 <b>몇 분이나 볼지</b> 미리 예측해서 추천 영상을 골라요. "
                            "'많이 볼 영상'과 '금방 끌 영상' 사이에 선을 긋는 거예요!"),
            ("🛵", "배달앱", "“30분 뒤 도착!” — 가게까지의 <b>거리와 걸린 시간</b>의 관계에 선을 그어, "
                            "오늘 우리 집까지 몇 분 걸릴지 맞히는 거예요."),
            ("🔋", "휴대폰 배터리", "지금 쓰는 속도와 남은 배터리의 관계에 선을 그어 "
                                  "<b>몇 시간 더 쓸 수 있는지</b> 알려줘요. "
                                  "여러분이 붕어빵 가격을 맞힌 것과 똑같은 방법이에요!"),
        ])
        tab_intro, tab_tutorial, tab_game, tab_free = st.tabs([
            "① 알아보기",
            "② 연습하기 · 마법사 훈련소",
            "③ 게임 도전 🏅 붕어빵 가게",
            "④ 자유 탐구 · 내 맘대로 마법진 (선택)",
        ])

        # [1단계: 먼저 알아보기 — 훅 → 미션 → 예시비교] ------------------------
        with tab_intro:
            predict_first(
                "predict_lr",
                "점들 가운데를 지나는 선과, 점들보다 한참 위에 있는 선 —<br><b>어느 쪽이 덜 틀릴까요?</b>",
                ['가운데를 지나는 선', '위에 있는 선', '똑같아요'],
                0,
                "아래에서 슬라이더를 움직이면 <b>틀린 정도</b>가 숫자로 바로 보여요.",
            )

            st.markdown("### 🌉 어? 나 이거 이미 알아!")
            md_html("""
            <div style='font-size: 16px; line-height: 1.75; background:#FFF8E1; border-radius:14px; padding:13px 18px;'>
            수도꼭지를 <b>1분</b> 틀면 컵에 물이 <b>반</b>, <b>2분</b> 틀면 <b>가득</b> 차요.
            그럼 <b>3분</b>이면? 안 해 봐도 "넘치겠다!" 하고 알 수 있죠? 🚰<br>
            이렇게 <b>지난 경험으로 아직 안 해본 일을 미리 맞히는 것</b>이 오늘 배울 AI 마법이에요.
            점들 사이에 <b>자를 대고 선을 한 줄 긋는 것</b>과 똑같답니다. 📏
            </div>
            """)

            st.write("")
            mission_card("빨간 선을 파란 점들에 최대한 가깝게 붙여서, '틀린 정도(오차)'를 0에 가깝게 만들어보세요!")

            st.write("")

            ex_x = np.array([1, 2, 3, 4, 5])
            ex_y = np.array([20, 35, 55, 75, 95])

            # 왼쪽: 설명·선택 / 오른쪽: 그래프 → 그래프 비율이 알맞게 보인다
            col_lr_txt, col_lr_fig = st.columns([1, 1.4])

            with col_lr_txt:
                st.write("친구들이 '공부한 시간'과 '받은 시험 점수'를 파란 점으로 찍어놨어요. "
                         "아래 버튼을 눌러서 **나쁜 선**과 **좋은 선**이 어떻게 다른지 먼저 비교해봐요.")
                ex_choice = st.radio("어떤 선을 볼까요?", ["😵 엉뚱한 선", "😎 가장 잘 맞는 선"], key="lr_example_choice")

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
                <p style='font-size: 16.5px; line-height: 1.9; margin-bottom: 0;'>
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
                    st.success("맞아요! 지금처럼 **가로축 값이 모두 양수**일 때는, 기울기를 올리면 "
                               "선이 점들 쪽으로 올라가요. 직접 확인해볼까요?")
                elif think1.startswith("②"):
                    st.warning("한번 직접 해보면서 확인해봐요! 아래 조종기를 오른쪽으로 밀어서 기울기를 올려보면 어떻게 되는지 살펴보세요.")

            x_data = np.array([1, 2, 3, 4, 5])
            y_data = np.array([20, 35, 55, 75, 95])

            st.markdown("### 🎛️ 마법의 선 조종기 (여기를 움직이세요!)")
            col_slider1, col_slider2 = st.columns(2)
            with col_slider1:
                w = st.slider("📐 선의 기울기 (시소처럼 기울여요)", 0.0, 30.0, 10.0, 0.5, key="lr_w",
                              help="오른쪽으로 밀수록 선이 가파르게 서고, 왼쪽으로 밀수록 눕는 시소예요.")
            with col_slider2:
                b = st.slider("⬆️ 선의 시작 높이 (엘리베이터처럼 위아래)", 0, 50, 10, 1, key="lr_b",
                              help="선 전체를 통째로 위아래로 태워 옮기는 엘리베이터예요.")

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

            if mse >= 20:
                clear_celebration("_celeb_reg_tutorial")   # 다시 흐트러지면 다음 성공 때 또 축하
            if mse < 20:
                celebrate_once("_celeb_reg_tutorial", sfx=False)
                st.markdown("---")
                md_html("""
                <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:14px 18px;">
                    <b>🏷️ 오늘 배운 것</b> 📏<br>
                    흩어진 점들 사이로 <b>가장 잘 맞는 선 한 줄</b>을 찾으면,
                    아직 해보지 않은 일도 <b>미리 맞힐 수</b> 있어요.
                    AI는 이 선을 <b>틀린 점수가 가장 작아질 때까지</b> 스스로 움직여서 찾아낸답니다.
                </div>
                """)

                with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                    quiz_reg = st.radio(
                        "점들에서 **멀리 떨어진 선**과 **가까이 붙은 선** 중에서, 미래를 더 잘 맞히는 선은?",
                        ["선택 안 함", "① 가까이 붙은 선", "② 멀리 떨어진 선", "③ 아무 선이나 똑같다"],
                        key="quiz_reg"
                    )
                    if quiz_reg.startswith("①"):
                        st.success("🌟 맞아요! 점들에 가까울수록 '틀린 점수'가 작아서 예측이 잘 맞아요. 보너스 배지 획득!")
                        st.session_state["bonus_badges"].add("🌟 선 긋기 완벽 이해")
                    elif quiz_reg != "선택 안 함":
                        st.warning("위 그래프에서 회색 틈(점과 선 사이)이 짧을 때와 길 때를 다시 비교해볼까요?")

                st.write("")
                next_section_button("📈 1. 마법의 선 긋기", "reg")


        # [3단계: 내 맘대로 마법진 (커스텀 데이터 + 예측 테스트)]
        with tab_free:
            st.markdown("<h2 style='color: #FF6B6B;'>🚀 내 맘대로 데이터 탐험!</h2>", unsafe_allow_html=True)
            st.write("이번엔 표에 숫자를 직접 마음대로 적어보고, 인공지능 요정이 미래를 얼마나 잘 맞히는지 테스트해봐요!")

            st.markdown("### 1️⃣ 어떤 마법을 부려볼까요? (나만의 주제 정하기)")
            col_x_name, col_y_name = st.columns(2)
            with col_x_name:
                # ⚠️ '원인 → 결과'로 이름 붙이면 상관관계를 인과관계로 오해하게 된다.
                #    선 긋기는 두 값의 '대응 관계'를 찾는 것이지 원인을 밝히는 게 아니다.
                custom_x = st.text_input("🟢 먼저 아는 값 (X축) 이름을 적어주세요", value="키 (cm)", key="free_xname")
            with col_y_name:
                custom_y = st.text_input("🔴 알아맞힐 값 (Y축) 이름을 적어주세요", value="신발 크기 (mm)", key="free_yname")

            st.info("👇 아래 표의 숫자를 클릭해서 마음대로 바꿔보세요! 새로운 줄을 추가하거나 지울 수도 있어요.")

            # 축 이름은 '겉에 보이는 이름표'(column_config)로만 쓰고, 표의 진짜 열 이름은 x·y로 고정한다.
            # (두 축 이름을 똑같이 적거나 비워도 표가 꼬이지 않고, 이름을 바꿔도 적어 둔 숫자가 사라지지 않는다.)
            _xl = (custom_x or "").strip() or "먼저 아는 값"
            _yl = (custom_y or "").strip() or "알아맞힐 값"
            if _xl == _yl:
                _yl = _yl + " (Y축)"
                st.caption("💡 두 축 이름이 똑같아서 아래쪽에 '(Y축)'을 붙여 구분했어요.")
            default_free_data = pd.DataFrame({"x": [130.0, 140.0, 145.0, 155.0, 165.0],
                                              "y": [210.0, 225.0, 235.0, 250.0, 265.0]})
            try:
                edited_df = st.data_editor(
                    default_free_data, num_rows="dynamic", key="free_editor", **_STRETCH,
                    column_config={
                        "x": st.column_config.NumberColumn(f"🟢 {_xl}", format="%.1f"),
                        "y": st.column_config.NumberColumn(f"🔴 {_yl}", format="%.1f"),
                    })
            except Exception:
                # 아주 오래된 Streamlit(column_config 없음)에서도 표는 뜨게 한다
                edited_df = st.data_editor(default_free_data, num_rows="dynamic", key="free_editor", **_STRETCH)
            custom_x, custom_y = _xl, _yl

            free_x, free_y = finite_xy(edited_df, "x", "y")   # 글자·빈칸·무한대는 자동으로 걸러진다
            parse_ok = False
            if len(free_x) < 2:
                st.error("앗! 숫자가 적힌 점이 최소 2개는 있어야 선을 이어볼 수 있어요. 표에 숫자를 더 적어주세요!")
            elif len(np.unique(free_x)) < 2:
                st.warning(f"음, '{custom_x}' 숫자가 전부 똑같아요! 원인이 변해야 규칙을 찾을 수 있어요. "
                           "왼쪽 열의 숫자를 서로 다르게 적어볼까요? 🧙")
            else:
                _fit = safe_polyfit_line(free_x, free_y)
                if _fit is None:
                    st.warning("앗! 이 숫자들로는 선을 그을 수 없어요. 너무 큰 수나 이상한 값이 섞여 있는지 확인해봐요, 마법사님! 🧙")
                else:
                    w_ai, b_ai = _fit
                    parse_ok = True

            if parse_ok:
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
                    # 예측값이 무한대로 튀지 않도록 입력 범위를 넉넉하되 유한하게 제한한다
                    _default_test = float(min(1e6, max(-1e6, free_x.max() + 1)))
                    clamp_state("free_testval", -1e6, 1e6, cast=float)
                    test_val = st.number_input(f"궁금한 '{custom_x}'의 숫자를 입력하세요:",
                                               min_value=-1e6, max_value=1e6, value=_default_test,
                                               key="free_testval",
                                               help="표에 없는 새 숫자를 넣으면 요정이 결과를 미리 맞혀 줘요.")
                    predicted_val = w_ai * float(test_val) + b_ai
                    if not np.isfinite(predicted_val):
                        predicted_val = 0.0

                with col_test2:
                    md_html(f"""
                    <div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px; border: 2px solid #FFB74D;'>
                        <h4 style='color: #E65100; margin-top: 0;'>🔮 마법 구슬의 대답</h4>
                        <p style='font-size: 18px;'>만약 <b>{custom_x}</b>(이)가 <b>{test_val}</b> 이라면,<br>
                        <b>{custom_y}</b>(은)는 약 <b style='color: #D84315; font-size: 24px;'>{predicted_val:.1f}</b> 일 것입니다!</p>
                    </div>
                    """)


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
                stars, msg = "⭐⭐", "아주 좋아요! 이 정도면 훌륭한 붕어빵 사장님이에요!"
            elif error < 23000:
                stars, msg = "⭐", "괜찮아요! 아래 힌트를 보고 조종기를 움직여봐요."
            else:
                stars, msg = "☁️", "음... 점들이랑 너무 멀어요. 아래 힌트를 따라가 볼까요?"

            st.write(f"### 틀린 점수(오차): {error:.0f} {stars}")
            st.info(msg)

            # ── 🧭 방향 힌트 ─────────────────────────────────────────
            # 아이가 "어느 쪽으로 밀어야 하는지"를 알 수 없으면 8천 개가 넘는 눈금 조합을
            # 헤매게 된다. 선이 점들보다 위/아래인지, 기울기가 급한지 완만한지 계산해서
            # 다음에 할 행동 '하나'만 콕 집어 알려준다. (스스로 찾아가되, 길을 잃지는 않게)
            _resid = pred - prices                      # +면 선이 점보다 위
            _g_slope = float(np.mean(_resid * sizes))   # 기울기를 어느 쪽으로 고칠지
            _g_inter = float(np.mean(_resid))           # 시작 위치를 어느 쪽으로 고칠지
            # 두 값은 단위가 달라서 그냥 비교하면 안 된다.
            # (_g_slope 는 크기(평균 약 12cm)가 곱해져 있어 늘 _g_inter 보다 크다.
            #  예전에는 여기에 12를 한 번 더 곱해서 '시작 위치' 힌트가 한 번도 나오지 않았고,
            #  기울기가 이미 맞는데도 "선이 너무 누웠어요"라고 잘못 안내했다.)
            # 그래서 '그 조종기 하나만 가장 잘 맞췄을 때 오차가 얼마나 줄어드는가'로 공정하게 견준다.
            # (시작 위치만 최적으로 고치면 오차가 _g_inter² 만큼,
            #  기울기만 최적으로 고치면 _g_slope² / 크기제곱평균 만큼 줄어든다.)
            _mean_s2 = float(np.mean(sizes ** 2)) or 1.0
            _gain_slope = _g_slope * _g_slope / _mean_s2
            _gain_inter = _g_inter * _g_inter
            if error >= 2500:
                if _gain_slope >= _gain_inter:            # 기울기를 고치는 쪽이 더 효과가 클 때
                    # 얼마나 밀어야 하는지도 알려준다 (한 칸씩만 밀다 지치지 않도록)
                    _far = abs(_g_slope / _mean_s2) >= 8 * 1.0      # 기울기 한 칸 = 1.0
                    _how = "<b>쭉</b>" if _far else "<b>조금</b>"
                    _tip = (f"📐 <b>기울기</b> 조종기를 <b>왼쪽</b>으로 {_how} 밀어보세요 (선이 너무 가팔라요)"
                            if _g_slope > 0 else
                            f"📐 <b>기울기</b> 조종기를 <b>오른쪽</b>으로 {_how} 밀어보세요 (선이 너무 누웠어요)")
                else:
                    _far = abs(_g_inter) >= 8 * 5.0                  # 시작 위치 한 칸 = 5.0
                    _how = "<b>쭉</b>" if _far else "<b>조금</b>"
                    _tip = (f"📍 <b>시작 위치</b> 조종기를 <b>왼쪽</b>으로 {_how} 밀어보세요 (선이 점들보다 위에 있어요)"
                            if _g_inter > 0 else
                            f"📍 <b>시작 위치</b> 조종기를 <b>오른쪽</b>으로 {_how} 밀어보세요 (선이 점들보다 아래에 있어요)")

                # 🔥 뜨거워 / 🧊 차가워 — 직전보다 나아졌는지 바로 알려준다
                _prev = st.session_state.get("_fb_prev_err")
                st.session_state["_fb_prev_err"] = error
                if _prev is None:
                    _temp = ""
                elif error < _prev * 0.98:
                    _temp = "<span style='color:#D84315;'>🔥 <b>뜨거워요!</b> 잘 가고 있어요</span><br>"
                elif error > _prev * 1.02:
                    _temp = "<span style='color:#1565C0;'>🧊 <b>차가워요!</b> 반대로 가고 있어요</span><br>"
                else:
                    _temp = ""
                md_html(f"""
                <div style="background:#FFF8E1; border:2px dashed #FFB300; border-radius:14px;
                            padding:12px 16px; margin:4px 0 8px 0;
                            font-family:'Gowun Dodum',sans-serif; font-size:15.5px; color:#5D4037; line-height:1.7;">
                  {_temp}🧭 <b>다음에 할 일</b> — {_tip}
                </div>
                """)
            else:
                st.session_state["_fb_prev_err"] = error

            # 별 2개(아주 좋아요)부터 부품을 준다.
            # 원본은 '완벽(별 3개)'이어야만 부품이 나와서, 첫 섬에서 막히는 아이가 많았다.
            # 원리를 이해했다면 부품을 받고 다음 섬으로 나아가게 하고, 별 3개는 '도전 과제'로 남긴다.
            if error < 7000:
                if error < 2500:
                    st.success("🏅 완벽합니다! '붕어빵 가격 박사' 부품 획득! (별 3개 달성 🎉)")
                else:
                    st.success("🏅 '붕어빵 가격 박사' 부품 획득! "
                               "⭐⭐⭐에 도전하고 싶으면 선을 조금 더 다듬어 보세요!")
                award_badge(0)

            st.divider()
            st.write("#### 🐟 손님 등장! '왕왕 큼직 붕어빵(18cm)'이 왔어요. 얼마를 받을까요?")
            # ★ 기울기·시작 위치를 둘 다 최대로 밀면 '내 선'이 예측한 값(최대 2,200원)이 입력칸의 최댓값을
            #   넘어 앱이 멈추던 문제 → 기본값을 항상 0~1500 사이로 잘라서 넣는다.
            _suggest = int(min(1500, max(0, round((slope * 18 + intercept) / 10) * 10)))
            clamp_state("fb_guess", 0, 1500, cast=int)
            guess_price = st.number_input("내가 정한 가격(원)", min_value=0, max_value=1500,
                                           value=_suggest, step=10, key="fb_guess",
                                           help="내가 그린 선이 18cm에서 가리키는 값을 참고해 정해 보세요.")
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
                   SECTION_COLORS["gd"], term="한 걸음씩 내려가기")

        # '이 섬을 왜 하는지' 가 안 보인다는 의견이 있어, 앞 섬과 이어 주는 한 칸을 맨 위에 둔다.
        md_html("""
        <div style='background:#FFF8E1; border:3px solid #FFB300; border-radius:16px;
                    padding:14px 20px; font-size:18px; line-height:1.8; color:#4E342E;'>
          <b>🤔 이 섬은 왜 할까요?</b><br>
          앞 섬에서는 <b>여러분이 손으로</b> 선을 움직여 덜 틀리게 만들었죠.<br>
          그런데 <b>AI는 혼자 있을 때 어느 쪽으로 움직일지 어떻게 알까요?</b><br>
          👉 답은 <b>“발밑이 기울어진 쪽으로 한 걸음”</b> 이에요. 그걸 여기서 해 봐요!
        </div>
        """)

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
              → AI는 한 번에 정답을 맞히지 않고, **틀린 정도(오차)를 조금씩 줄여가며 스스로 개선**한다는 기계학습 원리를 발걸음 조절 활동으로 체험해요.
            - **[실과] [6실05-04]** 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
              → 발걸음마다 달라지는 '위치 값'이 AI 학습에서 오차 데이터로 쓰이는 과정을 간접 체험해요.
            """)

        level_guide(has_free=False, minutes=15)
        real_world("gradient", [
            ("🤖", "요즘 AI 대부분", "챗봇·번역기·얼굴 인식처럼 <b>숫자를 조금씩 고쳐 가며 배우는 AI</b>는 "
                                  "거의 다 이 방법을 써요. (스무고개나 이웃에게 물어보기는 다른 방법이랍니다!)"),
            ("🎮", "게임 AI", "바둑·게임 AI는 <b>수백만 번 져 보면서</b> 한 걸음씩 더 나은 수를 찾아가요. "
                             "여러분이 골짜기를 내려간 것과 같아요."),
            ("🗣️", "번역기", "처음엔 엉뚱하게 번역하다가, 틀린 만큼을 <b>조금씩 고쳐가며</b> "
                            "점점 자연스러운 말이 돼요."),
        ])
        tab_intro, tab_practice, tab_game = st.tabs([
            "① 알아보기", "② 연습하기 · 직접 내려가보기", "③ 게임 도전 🏅 눈꽃 골짜기"])

        # [1단계: 먼저 알아보기] -----------------------------------------------
        with tab_intro:
            predict_first(
                "predict_gd",
                "산을 내려갈 때 <b>보폭을 아주 크게</b> 하면 어떻게 될까요?",
                ['더 빨리 도착해요', '지나쳐서 튕겨 나가요', '제자리에 있어요'],
                1,
                "아래에서 보폭을 크게·작게 바꿔 보면 바로 알 수 있어요.",
            )

            st.markdown("### 🌉 어? 나 이거 이미 알아!")
            md_html("""
            <div style='font-size:17px; line-height:1.8; background:#FFF8E1; border-radius:14px; padding:13px 18px;'>
            안개 낀 산에서도 <b>발밑이 기울어진 쪽</b>은 알 수 있죠? 그쪽으로 한 걸음, 또 한 걸음! 🥾<br>
            단, <b>너무 크게 디디면 지나쳐 튕기고</b> 너무 작게 디디면 해가 져요.
            </div>
            """)

            st.write("")
            mission_card("보폭을 잘 골라서, 산 밑바닥(가장 낮은 곳= 정답)까지 튕겨나가지 않고 도착해보세요!")

            st.write("")

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

            # ⚠️ 0.95 로는 x ← -0.9x 가 되어 '진동하면서 결국 도착'한다.
            #    화면 설명("튕겨나가서 영영 도착 못 해요")과 어긋나므로, 실제로 발산하는
            #    1.05( x ← -1.1x )를 쓴다. 이래야 그림과 글이 같은 말을 한다.
            ex_lr = 1.05 if ex_step == "😱 보폭이 너무 큼" else 0.12

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
            st.caption("2단계 탭에서 직접 보폭을 조절하며 산을 내려가봐요.")

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

                # 한 번에 만질 것이 많으면 아이들이 무엇을 조절해야 할지 몰라 헤맨다.
                # 핵심인 '발걸음 크기'를 먼저 시키고, 나머지는 접어 둔다.
                st.info("👉 먼저 **발걸음 크기**만 움직여 보세요! 나머지는 안 건드려도 돼요.")

                lr = st.slider("👣 발걸음 크기 (너무 크면 우주로 날아가요!)", 0.01, 1.10, 0.10, 0.05, key="gd_learn_lr",
                               help="AI에서는 '학습률'이라고 불러요. 한 번에 얼마나 크게 움직일지 정하는 값이에요.")
                epochs = st.slider("⏱️ 탐험할 시간 (시간이 많을수록 멀리 가요)", 1, 30, 10, key="gd_learn_epochs",
                                   help="몇 걸음까지 걸을지 정해요. AI에서는 '반복 횟수'라고 해요.")

                with st.expander("🔬 ⭐⭐⭐ 더 해보고 싶은 친구만 열어보세요"):
                    st.caption("안 열어봐도 괜찮아요! 이 섬의 학습목표는 위 두 가지만으로 충분해요.")
                    optimizer = st.radio("👀 앞을 보는 방법",
                        ["매의 눈 (아주 꼼꼼하지만 조금 느려요)",
                         "손전등 켜기 (조금 비틀거리며 내려가요)",
                         "안대 쓰고 뛰기 (완전 지그재그로 내려가요)"], key="gd_learn_opt")

                    penalty = st.radio("🎒 마법 배낭의 규칙",
                        ["규칙 없음 (기본)",
                         "가벼운 배낭 (쓸모없는 짐은 아예 버리고 가요!)",
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
                            "가벼운 배낭 (쓸모없는 짐은 아예 버리고 가요!)": 1,
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
                    celebrate_once("_celeb_gd_practice")   # 슬라이더를 살짝 건드릴 때마다 풍선이 또 터지지 않게
                    md_html("""
                    <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:14px 18px; margin-top:10px;">
                        <b>🏷️ 오늘 배운 것</b> 🥾<br>
                        비탈을 따라 <b>한 걸음씩</b> 내려가면 가장 낮은 곳(정답)에 닿아요.
                        걸음이 너무 크면 <b>지나쳐서 튕기고</b>, 너무 작으면 <b>하염없이 걸려요.</b>
                        AI도 이렇게 <b>딱 알맞은 보폭</b>으로 조금씩 정답에 다가간답니다.
                    </div>
                    """)
                elif abs(path_x[-1]) > 10:
                    clear_celebration("_celeb_gd_practice")
                    st.warning("발걸음을 너무 크게 했나 봐요! 조금 더 총총걸음으로 줄여볼까요?")
                else:
                    clear_celebration("_celeb_gd_practice")
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
                # 보폭마다 '몇 번 눌러야 하는지'를 미리 알려준다.
                # (아기 걸음은 보물까지 38번을 눌러야 해서, 모르고 고르면 지쳐서 포기한다)
                st.caption({"👶 아기 걸음 (0.05)": "👶 아기 걸음은 안전하지만 **약 38번**이나 눌러야 도착해요. 팔 아파요!",
                            "🚶 보통 걸음 (0.3)": "🚶 보통 걸음이면 **5번**만에 도착해요. 딱 알맞은 보폭이에요! 👍",
                            "🦵 성큼 걸음 (1.05)": "🦵 성큼 걸음은 목표를 지나쳐서 **튕겨나가요.** 정말 그런지 확인해볼까요? 😆"}[step_choice])

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
                clear_celebration("_celeb_gd_game")

            if step_btn and not st.session_state["gd_failed"] and not st.session_state["gd_success"]:
                x = float(st.session_state["gd_x"])
                new_x = float(x - lr2 * gd_grad(x))
                if not np.isfinite(new_x):          # 혹시라도 값이 폭발하면 '튕겨나감'으로 처리
                    new_x = 999.0
                st.session_state["gd_x"] = new_x
                st.session_state["gd_steps"] += 1
                # 기록은 최근 300걸음만 남겨, 오래 눌러도 메모리가 계속 늘지 않게 한다
                st.session_state["gd_history"] = (st.session_state["gd_history"] + [new_x])[-300:]

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
                celebrate_once("_celeb_gd_game")
                st.success(f"🎉 보물상자 발견! 총 {st.session_state['gd_steps']}걸음 만에 도착했어요!")
                award_badge(1)

                with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                    quiz_gd = st.radio(
                        "발걸음(보폭)을 **너무 크게** 하면 어떻게 될까요?",
                        ["선택 안 함", "① 목표를 지나쳐서 반대편으로 튕겨나간다",
                         "② 언제나 더 빨리, 더 정확히 도착한다", "③ 아무 일도 생기지 않는다"],
                        key="quiz_gd"
                    )
                    if quiz_gd.startswith("①"):
                        st.success("🌟 맞아요! 그래서 '딱 알맞은 보폭'을 고르는 게 중요해요. 보너스 배지 획득!")
                        st.session_state["bonus_badges"].add("🌟 한 걸음씩 완벽 이해")
                    elif quiz_gd != "선택 안 함":
                        st.warning("성큼 걸음으로 눌렀을 때 무슨 일이 있었는지 떠올려볼까요?")

                st.write("")
                next_section_button("⛰️ 2. 보물찾기 산", "gd")
            else:
                st.info("한 걸음씩 내려가며 보물상자에 가까워져 봐요. 보폭이 너무 크면 튕겨나갈 수 있어요!")

    # --- [메뉴 4: 스무고개 탐정 (결정트리)] ---
    elif menu == "🌳 3. 스무고개 탐정":
        hero_card("🌳", "스무고개 탐정: 질문으로 정답 찾기",
                   "예/아니오 질문을 던져서 인공지능처럼 정답을 쏙쏙 찾아내는 명탐정이 되어봐요!",
                   SECTION_COLORS["dt"], term="질문으로 좁혀가기")

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
              → 특징(질문)을 기준으로 대상을 좁혀가는 과정을 통해 **AI가 여러 조건을 조합해 정답을 찾아가는 원리(결정트리)**를 체험해요.
            - **[사회] [6사03-04]** 정보화 시대의 변화상을 이해하고, 생활에 미치는 영향을 탐구한다.
              → 스무고개처럼 조건을 좁혀 판단하는 방식이 실제 생활 속 AI 서비스(추천, 진단 등)에 어떻게 쓰이는지와 연결지어 생각해볼 수 있어요.
            """)

        level_guide(minutes=15)
        real_world("tree", [
            ("🏥", "병원", "의사 선생님처럼 AI도 <b>“열이 나나요?” “기침을 하나요?”</b> 하고 "
                          "하나씩 물어보며 무슨 병인지 좁혀가요."),
            ("📧", "스팸 메일함", "“광고라는 말이 있나?” “모르는 사람인가?” <b>질문을 이어가며</b> "
                                 "나쁜 메일을 자동으로 걸러내요."),
            ("💳", "카드 회사", "“평소와 다른 곳인가?” “금액이 큰가?” 질문으로 <b>수상한 결제</b>를 "
                               "찾아내서 문자를 보내줘요."),
        ])
        tab_intro, tab1, tab2, tab4, tab3 = st.tabs([
            "① 알아보기",
            "②-1 연습하기 · 나뭇잎",
            "②-2 연습하기 · 동물 가족",
            "③ 게임 도전 🏅 동물 탐정",
            "④ 자유 탐구 · 섞임 점수의 비밀 (선택)",
        ])

        # [1단계: 먼저 알아보기] -----------------------------------------------
        with tab_intro:
            predict_first(
                "predict_dt",
                "동물 6마리 중 범인을 빨리 찾으려면 <b>어떤 질문</b>이 좋을까요?",
                ['6마리를 반씩 갈라 주는 질문', '한 마리만 콕 집는 질문', '아무 질문이나 똑같아요'],
                0,
                "아래에서 질문을 눌러 보면 용의자가 얼마나 줄어드는지 보여요.",
            )

            st.markdown("### 🌉 어? 나 이거 이미 알아!")
            md_html("""
            <div style='font-size:16px; line-height:1.75; background:#FFF8E1; border-radius:14px; padding:13px 18px;'>
            스무고개 해봤죠? 그런데 "이름이 코끼리예요?"처럼 <b>하나씩 찍는 질문</b>은 운에 맡기는 것,
            "다리가 4개예요?"처럼 <b>후보를 반으로 뚝 자르는 질문</b>은 몇 번 만에 답이 나와요. ✂️<br>
            AI도 <b>잘 갈라 주는 질문부터</b> 골라서 물어본답니다.
            </div>
            """)

            st.write("")
            mission_card("아래 미션들을 풀면서, '좋은 질문'을 골라 정답을 빨리 찾는 방법을 알아봐요!")

            st.write("")
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
                <b>나무</b>를 닮아서 어른들은 <b>'나무(트리)'</b>라고 불러요.
                예/아니오 질문으로 좁혀 가는 모습이 <b>스무고개 놀이</b>와 꼭 닮았죠? 🌳
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
            st.caption("미션 1, 2 탭에서 직접 스무고개를 해보고, '섞임 점수' 탭에서 왜 그 질문이 좋은지 알아본 뒤, 마지막 게임 탭에서 실력을 확인해봐요.")

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
                """한 상자 안의 섞임 점수(섞임 점수). 0이면 한 종류만 있는 것."""
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
                               "AI(스무고개)는 이런 질문을 스스로 찾아낸답니다!")
                elif think_dt.startswith("②"):
                    st.warning("반대예요! 섞임 점수는 **낮을수록** 잘 정리된 거예요. 위에서 질문을 다시 눌러보며 확인해봐요.")

            st.caption("👉 이제 마지막 게임 탭에서 직접 좋은 질문을 골라 범인을 찾아봐요!")

        # [게임: 동물 탐정 챌린지] ----------------------------------------------
        with tab4:
            st.markdown("### 🕵️‍♂️ 범인 동물을 찾아라!")
            md_html("""
            <div style='background:#FFF8E1; border-radius:14px; padding:12px 18px;
                        font-size:18px; line-height:1.75; color:#4E342E;'>
              동물 6마리 중 <b>한 마리가 범인</b>이에요. 질문 카드를 눌러 범인을 찾으세요!<br>
              🏅 <b>질문 3번 안에</b> 찾으면 명탐정 배지를 받아요.
            </div>
            """)

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

            def _dt_new_case():
                """새 사건: 범인을 몰래 정하고 용의자·질문 기록을 비운다."""
                m = mystery_animals.sample(1, random_state=random.randint(0, 9999)).iloc[0]
                st.session_state["dt_mystery"] = m
                st.session_state["dt_candidates"] = mystery_animals.copy()
                st.session_state["dt_asked"] = []
                st.session_state["dt_history"] = []
                clear_celebration("_celeb_dt")

            # 범인·용의자 정보가 없거나 깨져 있으면(새로고침·엉뚱한 클릭 등) 조용히 새 사건으로 시작한다
            _m = st.session_state.get("dt_mystery")
            _c = st.session_state.get("dt_candidates")
            if (_m is None or not isinstance(_m, pd.Series)
                    or not isinstance(_c, pd.DataFrame) or len(_c) == 0
                    or not all(col in _c.columns for col in dt_questions.values())
                    or not all(col in _m.index for col in dt_questions.values())):
                _dt_new_case()

            mystery = st.session_state["dt_mystery"]
            candidates = st.session_state["dt_candidates"]
            num_q = len(st.session_state["dt_asked"])

            # 남은 용의자를 큰 이모지 줄로 표시 (질문 카드가 화면 위쪽에 오도록 위아래 여백을 줄였다)
            suspect_emojis = " ".join(name.split()[0] for name in candidates["이름"].tolist())
            md_html(f"""
            <div style='background:#E8F5E9; border-radius:14px; padding:10px 18px; text-align:center;
                        margin:10px 0 12px 0;'>
                <div style='font-size:16px; color:#2E7D32;'>🔎 남은 용의자 <b>{len(candidates)}명</b> · 질문 {num_q}번</div>
                <div style='font-size:48px; letter-spacing:8px; margin-top:4px;'>{suspect_emojis}</div>
            </div>
            """)

            if len(candidates) > 1:
                asked = st.session_state["dt_asked"]
                remaining_qs = {q: col for q, col in dt_questions.items() if q not in asked}

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

                st.markdown("#### 🃏 질문 카드를 눌러보세요")
                # ⚠️ 이미 물어본 카드를 화면에서 '없애면' 그려지는 요소 개수가 달라져서
                #    보고 있던 탭이 ①번으로 튕긴다(질문할 때마다 게임 탭을 다시 눌러야 했다).
                #    그래서 카드 4장을 항상 그리고, 이미 물어본 카드만 흐리게 + 버튼 비활성으로 둔다.
                cols = st.columns(len(dt_questions))
                for i, (q, col) in enumerate(dt_questions.items()):
                    used = q in asked
                    yes_grp = [n.split()[0] for n in candidates[candidates[col] == 1]["이름"]]
                    no_grp = [n.split()[0] for n in candidates[candidates[col] == 0]["이름"]]
                    with cols[i]:
                        recommend = (not used) and (q == best_q)
                        border = "#7E57C2" if recommend else ("#E0E0E0" if used else "#C5CAE9")
                        bg = "#F5F5F5" if used else "#FFFFFF"
                        if used:
                            _ans = next((a for _q, a in st.session_state["dt_history"] if _q == q), "")
                            tag = (f"<div style='font-size:13px; color:#9E9E9E; font-weight:bold;'>"
                                   f"✔️ 물어봤어요 · {_ans}</div>")
                        elif recommend:
                            tag = "<div style='font-size:13px; color:#7E57C2; font-weight:bold;'>👍 잘 갈라져요!</div>"
                        else:
                            tag = "<div style='font-size:13px; color:#aaa;'>&nbsp;</div>"
                        _dim = "opacity:0.45;" if used else ""
                        md_html(f"""
                        <div style='border:2px solid {border}; background:{bg}; border-radius:12px;
                                    padding:10px 8px; text-align:center; min-height:132px; {_dim}'>
                            {tag}
                            <div style='font-size:16px; margin:6px 0; color:#1B2A4A;'><b>{q}</b></div>
                            <div style='font-size:22px; letter-spacing:2px;'>✅ {' '.join(yes_grp) if yes_grp else '—'}</div>
                            <div style='font-size:22px; letter-spacing:2px;'>❌ {' '.join(no_grp) if no_grp else '—'}</div>
                        </div>
                        """)
                        if st.button("이 질문 하기", key=f"dtq_{col}", disabled=used, **_STRETCH):
                            answer = int(mystery[col])
                            narrowed = candidates[candidates[col] == answer]
                            if len(narrowed) == 0:   # 있을 수 없는 일이지만, 0명이 되면 사건을 새로 시작
                                _dt_new_case()
                            else:
                                st.session_state["dt_candidates"] = narrowed
                                if q not in st.session_state["dt_asked"]:
                                    st.session_state["dt_asked"].append(q)
                                st.session_state["dt_history"].append((q, "네" if answer == 1 else "아니오"))
                            st.rerun()

                st.caption("💡 ✅와 ❌에 동물이 **골고루 갈라지는 질문**일수록 용의자가 확 줄어요!")

                if st.session_state["dt_history"]:
                    with st.expander(f"📜 지금까지 한 질문 {len(st.session_state['dt_history'])}개", expanded=False):
                        for q, a in st.session_state["dt_history"]:
                            icon = "✅" if a == "네" else "❌"
                            st.write(f"- {q} → {icon} **{a}**")

            else:
                found = candidates.iloc[0]
                # 용의자가 1명으로 좁혀졌으면 그게 범인 (게임 규칙상 항상 정답과 일치)
                n_asked = len(st.session_state["dt_asked"])
                st.success(f"🎯 범인을 찾았어요! 범인은 바로 **{found['이름']}** 입니다!")

                if n_asked <= 3:
                    celebrate_once("_celeb_dt")
                    st.markdown(f"### 🎉 명탐정! 단 **{n_asked}번**의 질문으로 범인을 찾았어요!")
                    award_badge(2)

                    md_html("""
                    <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:14px 18px; margin-top:10px;">
                        <b>🏷️ 오늘 배운 것</b> 🌳<br>
                        <b>예/아니오 질문</b>으로 후보를 반씩 줄여가면 몇 번 만에 답을 찾아요.
                        비결은 <b>후보를 깔끔하게 갈라 주는 질문부터</b> 고르는 것!
                        AI도 그런 질문을 스스로 찾아서 먼저 물어본답니다.
                    </div>
                    """)

                    with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                        quiz_dt = st.radio(
                            "어떤 질문이 **좋은 질문**일까요?",
                            ["선택 안 함", "① 남은 후보를 반으로 확 갈라 주는 질문",
                             "② 한 마리씩 콕 찍어 물어보는 질문", "③ 어떤 질문이든 똑같다"],
                            key="quiz_dt"
                        )
                        if quiz_dt.startswith("①"):
                            st.success("🌟 맞아요! 후보를 확 줄여 주는 질문부터 하면 빨리 찾아요. 보너스 배지 획득!")
                            st.session_state["bonus_badges"].add("🌟 스무고개 완벽 이해")
                        elif quiz_dt != "선택 안 함":
                            st.warning("'이름이 오리인가요?'로 물었을 때와 '날개가 있나요?'로 물었을 때를 비교해봐요!")

                    st.write("")
                    next_section_button("🌳 3. 스무고개 탐정", "dt")
                else:
                    st.info(f"범인은 찾았지만 질문을 {n_asked}번 했어요! 💡 '잘 갈라져요!' 표시가 있는 질문을 먼저 고르면 "
                            "3번 이하로 찾을 수 있어요. 아래 버튼으로 다시 도전해볼까요?")

            # 다시 하기 버튼은 한 번만 그린다 (질문 중에도, 범인을 찾은 뒤에도 같은 자리)
            st.write("")
            if st.button("🔄 다른 사건으로 다시 하기", key="dt_new_case"):
                _dt_new_case()
                st.rerun()

    # --- [메뉴 5: 가장 친한 친구 (KNN)] ---
    elif menu == "🤝 4. 가장 친한 친구":
        hero_card("🤝", "가장 친한 친구: 내 주변에 누가 제일 많을까?",
                   "새로운 데이터가 나타났을 때, 가까운 이웃들에게 물어보고 정체를 알아내요.",
                   SECTION_COLORS["knn"], term="가까운 이웃에게 물어보기")

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
              → 새로운 데이터가 **가까운 이웃 데이터들의 다수결로 분류**되는 AI(KNN)의 원리를 게임으로 체험해요.
            - **[실과] [6실05-04]** 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
              → 위치(가로·세로 값)라는 숫자 데이터가 AI 분류 판단에 어떻게 활용되는지 탐색해요.
            """)

        level_guide(minutes=15)
        real_world("knn", [
            ("🎵", "음악 앱", "<b>나와 취향이 가장 비슷한 사람들</b>이 즐겨 들은 노래를 찾아서 "
                            "“이 노래 어때요?” 하고 추천해줘요."),
            ("🛒", "쇼핑몰", "“이 상품을 산 사람들이 <b>함께 본 상품</b>” — 나와 가까운 손님들에게 "
                            "물어보고 추천하는 거예요."),
            ("🎬", "영화 추천", "나와 비슷한 영화를 좋아한 사람들이 <b>다음에 뭘 봤는지</b> 보고 "
                               "다수결로 추천해줘요."),
        ])
        tab_intro, tab_practice, tab_game, tab_class = st.tabs([
            "① 알아보기", "② 연습하기 · 직접 판정해보기",
            "③ 게임 도전 🏅 외계 알 구출작전", "④ 자유 탐구 · 우리 반 데이터 (선택)"])

        # [1단계: 먼저 알아보기] -----------------------------------------------
        with tab_intro:
            predict_first(
                "predict_knn",
                "처음 보는 친구가 어떤 무리인지 알고 싶어요.<br><b>누구에게 물어보면 좋을까요?</b>",
                ['가까이 있는 친구들', '제일 멀리 있는 친구', '아무도 몰라요'],
                0,
                "아래에서 <b>몇 명에게 물어볼지</b>를 바꿔 보면 답이 달라져요.",
            )

            st.markdown("### 🌉 어? 나 이거 이미 알아!")
            md_html("""
            <div style='font-size:16px; line-height:1.75; background:#FFF8E1; border-radius:14px; padding:13px 18px;'>
            전학 온 친구가 늘 <b>태권도복 입은 친구들 옆에</b> 있으면 "저 친구도 태권도 하나 봐!" 싶죠? 🥋<br>
            AI도 처음 보는 것을 만나면 <b>가장 가까이 있는 이웃</b>을 세어 보고,
            "3명 중 2명이 사과니까 너도 사과!" 하고 <b>다수결</b>로 정한답니다.
            </div>
            """)

            st.write("")
            mission_card("새로운 과일이 사과인지 포도인지, 가까운 이웃 과일들에게 물어봐서 알아맞혀보세요!")

            data = pd.DataFrame({
                '달콤한 정도': [8, 9, 7, 2, 3, 1, 8, 2, 4],
                '과일 크기': [7, 3, 9, 2, 1, 3, 6, 2, 4],
                '과일 종류': ['사과', '포도', '사과', '포도', '포도', '포도', '사과', '포도', '포도']
            })

            st.write("")

            col_knn_txt, col_knn_fig = st.columns([1, 1.4])

            with col_knn_txt:
                cmp_choice = st.radio("새 과일을 어디에 놓아볼까요?", ["🍎 사과 무리 한가운데", "🍇 포도 무리 한가운데"], key="knn_cmp_choice")
                st.info(f"새 과일 주변에 **{cmp_choice.split()[1]}**들이 잔뜩 있으니, 이 과일도 그럴 확률이 높겠죠? "
                        f"이게 바로 이웃에게 물어보기의 기본 생각이에요.")

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
            st.caption("2단계 탭에서 직접 새 과일을 놓아보고 K값도 바꿔봐요.")

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

            if st.button("👽 알 깨우기!", key="knn_hatch", type="primary"):
                st.session_state["knn_hatched"] = True

            if st.session_state.get("knn_hatched"):
                award_badge(3)
                md_html("""
                <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:14px 18px; margin-top:10px;">
                    <b>🏷️ 오늘 배운 것</b> 🥋<br>
                    처음 보는 것도 <b>가까운 이웃 몇 명</b>에게 물어보고 <b>다수결</b>로 정하면 정체를 알 수 있어요.
                    한 명한테만 물으면 우연히 옆에 있던 친구 말만 듣게 되니,
                    <b>서너 명</b>에게 물어보는 게 더 믿음직해요.
                </div>
                """)

                with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                    quiz_knn = st.radio(
                        "물어볼 이웃을 **딱 1명**만 정하면 어떤 점이 걱정될까요?",
                        ["선택 안 함", "① 우연히 옆에 있던 한 명 말만 듣게 된다",
                         "② 1명한테만 물으면 언제나 더 정확하다", "③ 아무 차이도 없다"],
                        key="quiz_knn"
                    )
                    if quiz_knn.startswith("①"):
                        st.success("🌟 맞아요! 그래서 서너 명에게 물어보는 게 더 믿음직해요. 보너스 배지 획득!")
                        st.session_state["bonus_badges"].add("🌟 이웃에게 묻기 완벽 이해")
                    elif quiz_knn != "선택 안 함":
                        st.warning("K를 1로 했을 때와 5로 했을 때 판정이 달라졌는지 다시 확인해볼까요?")

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
            try:
                class_df = st.data_editor(
                    default_class_df, num_rows="dynamic", key="knn_class_editor", **_STRETCH,
                    column_config={
                        "키(cm)": st.column_config.NumberColumn("키(cm)", min_value=50, max_value=250, step=1),
                        "발크기(mm)": st.column_config.NumberColumn("발크기(mm)", min_value=100, max_value=350, step=1),
                        "구분": st.column_config.TextColumn("구분 (두 종류만)", max_chars=10),
                    })
            except Exception:
                class_df = st.data_editor(default_class_df, num_rows="dynamic", key="knn_class_editor", **_STRETCH)

            # 유효성 검사: 글자·빈칸·무한대는 버리고, 너무 긴 표는 앞 100줄만 쓴다
            valid = True
            try:
                class_df = class_df.copy()
                class_df["키(cm)"] = pd.to_numeric(class_df["키(cm)"], errors="coerce")
                class_df["발크기(mm)"] = pd.to_numeric(class_df["발크기(mm)"], errors="coerce")
                class_df = class_df.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=["키(cm)", "발크기(mm)", "구분"])
                class_df["구분"] = class_df["구분"].astype(str).str.strip()
                class_df = class_df[class_df["구분"] != ""].head(100).reset_index(drop=True)
                class_df["이름"] = class_df["이름"].fillna("친구").astype(str)
                groups = class_df["구분"].unique().tolist()
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
                    clamp_state("knn_class_h", 100, 200, cast=int)
                    new_h = st.number_input("새 친구 키(cm)", 100, 200, 150, key="knn_class_h")
                with cc2:
                    clamp_state("knn_class_f", 150, 300, cast=int)
                    new_f = st.number_input("새 친구 발크기(mm)", 150, 300, 228, key="knn_class_f")
                with cc3:
                    max_k = max(1, min(7, len(class_df)))
                    clamp_state("knn_class_k", 1, max_k, cast=int)   # 표 줄 수가 줄어도 K가 범위를 넘지 않게
                    k_class = st.slider("이웃 수 (K)", 1, max_k, min(3, max_k), key="knn_class_k",
                                        help="새 친구 주변에서 몇 명에게 물어볼지 정해요.")

                st.markdown("#### 3️⃣ AI의 판정 결과를 확인해요")
                try:
                    X = class_df[["키(cm)", "발크기(mm)"]].astype(float).values
                    y = class_df["구분"].astype(str).values
                    knn_c = KNeighborsClassifier(n_neighbors=int(min(k_class, len(X))))
                    knn_c.fit(X, y)
                    pred = knn_c.predict([[new_h, new_f]])[0]
                    dist, ind = knn_c.kneighbors([[new_h, new_f]])
                except Exception:
                    st.warning("🤖 이 숫자들로는 AI가 판단하기 어려워요. 표의 숫자를 다시 한 번 확인해 볼까요?")
                    st.stop()

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
                               "이것이 바로 **이웃에게 물어보기**예요!")
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
                   SECTION_COLORS["km"], term="비슷한 것끼리 모으기")

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
              → 정답 이름표 없이도 **비슷한 것끼리 스스로 모둠을 짓는 AI(K-Means)**의 원리를 체험해요.
            - **[실과] [6실05-04]** 디지털 데이터와 아날로그 데이터의 특징을 이해하고, 인공지능에 활용할 수 있는 데이터의 유형이나 형태를 탐색한다.
              → 이름표(정답)가 없는 데이터도 AI 학습에 활용될 수 있음을 탐색해요.
            """)

        level_guide(minutes=15)
        real_world("kmeans", [
            ("📸", "사진 앱", "누가 누구인지 알려주지 않아도, <b>비슷한 얼굴끼리 저절로</b> 앨범을 "
                            "만들어줘요. 이름표 없는 분류죠!"),
            ("🏪", "마트·쇼핑몰", "손님들을 <b>비슷한 무리로 나눠서</b> 각 무리에 딱 맞는 할인 쿠폰을 "
                                 "보내줘요."),
            ("🚚", "택배·배달", "<b>가까운 주문끼리 묶어서</b> 한 번에 돌면 훨씬 빨라요. "
                               "로봇 반장이 자리를 찾아간 것과 같아요."),
        ])
        tab_intro, tab_practice, tab_game, tab_class = st.tabs([
            "① 알아보기", "② 연습하기 · 직접 나눠보기",
            "③ 게임 도전 🏅 무인도 분리수거", "④ 자유 탐구 · 우리 반 데이터 (선택)"])

        # [1단계: 먼저 알아보기] -----------------------------------------------
        with tab_intro:
            predict_first(
                "predict_km",
                "<b>이름표(정답)가 하나도 없어도</b> 비슷한 것끼리 나눌 수 있을까요?",
                ['나눌 수 있어요', '정답이 있어야만 나눠요', '반반이에요'],
                0,
                "아래에서 모둠 수만 정해 주면 스스로 모이는 걸 볼 수 있어요.",
            )

            st.markdown("### 🌉 어? 나 이거 이미 알아!")
            md_html("""
            <div style='font-size:16px; line-height:1.75; background:#FFF8E1; border-radius:14px; padding:13px 18px;'>
            블록이 바닥에 흩어져 있으면, 누가 시키지 않아도
            <b>"빨간 건 여기, 파란 건 저기"</b> 하고 모으죠? 블록에 이름표가 없는데도요! 🧱<br>
            AI도 <b>이름표(정답) 없이</b> 비슷한 것끼리 스스로 무리를 지을 수 있답니다.
            </div>
            """)

            st.write("")
            mission_card("이름표 없는 나뭇잎들을, AI가 생김새만 보고 몇 개의 모둠으로 알아서 나누게 해보세요!")

            leaf_samples = pd.DataFrame({
                '잎의 길이': [15, 14, 16, 5, 4, 6, 8, 9, 2, 3, 1],
                '잎의 넓이': [2, 1.5, 2.5, 5, 4.5, 6, 1, 1.5, 1, 0.5, 0.8],
                '원래 이름': ['강아지풀', '강아지풀', '강아지풀', '단풍나무', '단풍나무', '단풍나무', '벚나무', '벚나무', '소나무', '소나무', '소나무']
            })

            st.write("")

            col_km_txt, col_km_fig = st.columns([1, 1.4])

            with col_km_txt:
                cmp_km = st.radio("어떤 상태를 볼까요?", ["😵 이름표도 색깔도 없는 뒤죽박죽 상태", "😎 AI가 4모둠으로 나눈 후"], key="km_cmp_choice")

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
                    # 잎 자료는 4종류(강아지풀·단풍나무·벚나무·소나무)다.
                    # K=3으로 시연하면 단풍나무와 벚나무가 한 모둠으로 묶여,
                    # "비슷한 것끼리 딱 모였다"는 설명과 화면이 어긋난다. 그래서 K=4로 보여준다.
                    _labels_demo, _ = _fit_kmeans_leaf(4)
                    _colors_demo = ['#FF4B4B', '#1C83E1', '#00C781', '#FFA500']
                    for i in range(4):
                        mask = _labels_demo == i
                        fig_cmp.add_trace(go.Scatter(x=leaf_samples['잎의 길이'][mask], y=leaf_samples['잎의 넓이'][mask],
                                                      mode='markers', marker=dict(size=18, color=_colors_demo[i]), name=f'모둠 {i+1}'))
                fig_cmp.update_layout(xaxis_title="잎의 길이", yaxis_title="잎의 넓이", height=330,
                                      margin=dict(l=20, r=20, t=20, b=20),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_cmp, **_STRETCH, config={'displayModeBar': False})

            st.write("")
            st.caption("2단계 탭에서 모둠 개수(K)를 직접 바꿔보며 결과가 어떻게 달라지는지 확인해봐요.")

        # [2단계: 직접 나눠보기] -----------------------------------------------
        with tab_practice:
            st.markdown("### 🕹️ 이제 네 차례!")

            with st.expander("🤔 시작하기 전에 잠깐 생각해보기"):
                think_km = st.radio(
                    "모둠 개수(K)를 2개로 줄이면 어떻게 될까요?",
                    ["선택 안 함", "① 원래 4종류였던 잎들이 억지로 2개 모둠에 나뉘어 담긴다", "② 항상 정확히 4종류로 알아서 나뉜다"],
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
            if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 4):
                st.session_state["km_data"] = None          # 데이터가 깨졌으면 다음 실행 때 새로 만든다
                st.session_state["km_centroids"] = None
                st.session_state["km_labels"] = None
                st.session_state["km_iter"] = 0
                st.rerun()
            if "km_phase" not in st.session_state:
                st.session_state["km_phase"] = "assign"   # 다음에 할 행동

            colL, colR = st.columns([1, 2])
            with colL:
                k = st.slider("🤖 로봇 반장 수 (K)", 2, 4, 3, key="km_game_k",
                              help="반장 수 = 만들 모둠 수예요. 바꾸면 반장을 새로 내려보내요.")

                # 반장을 내려보낸 뒤 K를 바꾸면 반장 수와 모둠 수가 어긋나므로, 조용히 처음 상태로 되돌린다
                _cs = st.session_state.get("km_centroids")
                _lb = st.session_state.get("km_labels")
                _cs_bad = _cs is not None and not (isinstance(_cs, np.ndarray)
                                                   and _cs.ndim == 2 and _cs.shape[1] == 2)
                _lb_bad = _lb is not None and not (isinstance(_lb, np.ndarray) and len(_lb) == len(pts))
                if _cs is not None and (_cs_bad or _lb_bad or len(_cs) != k):
                    st.session_state["km_centroids"] = None
                    st.session_state["km_labels"] = None
                    st.session_state["km_iter"] = 0
                    st.session_state["km_phase"] = "assign"
                    if not _cs_bad and not _lb_bad:
                        safe_toast("반장 수가 바뀌어서 처음부터 다시 시작해요!", icon="🤖")
                    st.rerun()

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
                idx = rng2.choice(len(pts), min(k, len(pts)), replace=False)
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
                    for j in range(min(k, len(cs_now))):
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
                    # 막 내려보낸 직후(아직 한 번도 모이기를 안 누른 상태)에
                    # "이동했어요"라고 말하면 사실과 다르다.
                    if st.session_state.get("km_labels") is None:
                        st.info("👀 반장들이 자리를 잡았어요! 먼저 **① 모이기**를 눌러보세요.")
                    else:
                        st.info("👀 반장이 가운데로 이동했어요! 이제 다시 **① 모이기**를 눌러 쓰레기를 다시 나눠봐요.")
                else:
                    st.info("👀 쓰레기가 가장 가까운 반장에게 모였어요! (점선으로 연결) 이제 **② 이동하기**를 눌러보세요.")
            if st.session_state["km_centroids"] is not None and st.session_state["km_iter"] >= 3:
                # K를 2나 4로 바꾸면 덩어리가 쪼개지거나 합쳐질 수 있으므로
                # "깔끔하게 나뉘었다"고 단정하지 않는다.
                st.success("🎉 반장들이 자기 자리를 찾았어요! 모둠 나누기 완료! "
                           "**반장 수(K)를 바꾸면 어떻게 달라지는지도 해보세요.**")
                award_badge(4)
                md_html("""
                <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:14px 18px; margin-top:10px;">
                    <b>🏷️ 오늘 배운 것</b> 🧩<br>
                    <b>정답(이름표)이 하나도 없어도</b> 비슷한 것끼리 모을 수 있어요.
                    로봇 반장이 <b>① 모이기 → ② 가운데로 이동</b>만 되풀이했는데 저절로 정리됐죠?
                    이 '반장 자리'가 바로 모둠의 <b>한가운데</b>랍니다.
                </div>
                """)

                with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                    quiz_km = st.radio(
                        "로봇 반장은 **무엇을 보고** 쓰레기를 모둠으로 나눴을까요?",
                        ["선택 안 함", "① 누가 더 가까이 있는지(자리)를 보고",
                         "② 쓰레기에 붙은 이름표(정답)를 보고", "③ 쓰레기 이름의 첫 글자를 보고"],
                        key="quiz_km"
                    )
                    if quiz_km.startswith("①"):
                        st.success("🌟 맞아요! 이름표 없이 '가까운 정도'만 보고 스스로 나눴어요. 보너스 배지 획득!")
                        st.session_state["bonus_badges"].add("🌟 비슷한 것끼리 완벽 이해")
                    elif quiz_km != "선택 안 함":
                        st.warning("쓰레기에는 이름표가 하나도 없었어요! 반장이 무엇을 보고 나눴는지 다시 볼까요?")

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
            try:
                km_class_df = st.data_editor(
                    default_km_df, num_rows="dynamic", key="km_class_editor", **_STRETCH,
                    column_config={
                        "가로(cm)": st.column_config.NumberColumn("가로(cm)", min_value=0, max_value=1000, step=1),
                        "세로(cm)": st.column_config.NumberColumn("세로(cm)", min_value=0, max_value=1000, step=1),
                    })
            except Exception:
                km_class_df = st.data_editor(default_km_df, num_rows="dynamic", key="km_class_editor", **_STRETCH)

            try:
                km_class_df = km_class_df.copy()
                km_class_df["가로(cm)"] = pd.to_numeric(km_class_df["가로(cm)"], errors="coerce")
                km_class_df["세로(cm)"] = pd.to_numeric(km_class_df["세로(cm)"], errors="coerce")
                km_class_df = (km_class_df.replace([np.inf, -np.inf], np.nan)
                               .dropna(subset=["가로(cm)", "세로(cm)"]).head(100).reset_index(drop=True))
                km_class_df["이름"] = km_class_df["이름"].fillna("이름없음").astype(str)
            except Exception:
                km_class_df = pd.DataFrame()

            if len(km_class_df) < 3:
                st.warning("⚠️ 데이터를 **3줄 이상** 채워주세요! (가로·세로 숫자를 모두 적어야 해요.)")
            else:
                st.success(f"✅ 데이터 {len(km_class_df)}개 준비 완료!")

                # ── 2단계: 모둠 수 정하기 ──────────────────────────
                st.markdown("#### 2️⃣ 몇 개의 모둠으로 나눌지 정해요")
                max_groups = max(2, min(4, len(km_class_df)))
                clamp_state("km_class_k", 2, max_groups, cast=int)   # 표 줄 수가 줄어도 K가 범위를 넘지 않게
                n_groups = st.slider("모둠 수 (K)", 2, max_groups, min(2, max_groups), key="km_class_k")

                # ── 3단계: AI에게 맡기기 ──────────────────────────
                st.markdown("#### 3️⃣ AI에게 모둠 나누기를 맡겨요")
                if st.button("🤖 AI에게 맡기기!", key="km_class_run_btn", **_STRETCH):
                    st.session_state["km_class_run"] = True

                if not st.session_state.get("km_class_run"):
                    st.info("👆 위 버튼을 누르면 AI가 비슷한 것끼리 모둠을 만들어줘요!")
                else:
                    X = km_class_df[["가로(cm)", "세로(cm)"]].astype(float).values
                    _n_distinct = len(np.unique(X, axis=0))
                    _km_ok = True
                    if _n_distinct < n_groups:
                        st.warning(f"⚠️ 서로 다른 값이 {_n_distinct}개뿐이라 {n_groups}개 모둠으로 나눌 수 없어요. "
                                   "표의 숫자를 조금씩 다르게 적어볼까요?")
                        _km_ok = False
                    if _km_ok:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                km_model = KMeans(n_clusters=int(n_groups), random_state=42, n_init=10)
                                labels = km_model.fit_predict(X)
                        except Exception:
                            st.warning("🤖 이 숫자들로는 AI가 모둠을 나누기 어려워요. 표를 다시 한 번 확인해 볼까요?")
                            _km_ok = False
                if st.session_state.get("km_class_run") and _km_ok:
                    km_class_df = km_class_df.copy()
                    km_class_df["AI 모둠"] = [f"{chr(65+int(l))}모둠" for l in labels]

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
                                   "이것이 바로 **비슷한 것끼리 모으기**예요!")
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
                   "여러 요정이 힘을 합쳐 하나의 판단을 내리는 생각 그물의 원리를 로봇 네오로 체험해요.",
                   SECTION_COLORS["nn"], term="여러 판단을 합치기")

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, 인공지능이 사회에 미치는 영향을 탐색한다.
              → 입력값에 가중치를 곱해 더하는 **인공 뉴런**들이 모여, 그 결과를 또 한 번 합쳐서 최종 판단을 내리는 **인공신경망(은닉층)의 기본 계산 원리**를 로봇 네오를 통해 체험해요.
            - **[실과] [6실04-06]** 생활 속에서 로봇 활용 사례를 통해 작동 원리와 활용 분야를 이해한다.
              → 여러 요정(뉴런)의 판단을 종합해 하나의 결론(파워 게이지)을 내리는 로봇의 작동 원리와 연결지어 이해해요.
            """)

        level_guide(has_free=False, minutes=15)
        real_world("ann", [
            ("📱", "얼굴로 폰 열기", "눈·코·입 같은 <b>여러 특징을 요정들이 나눠 보고</b>, "
                                    "그 판단을 합쳐서 “주인 맞아!”를 결정해요."),
            ("🗣️", "음성 비서", "“내일 날씨 알려줘” — 소리를 잘게 나눠 <b>여러 요정이 함께 판단</b>해서 "
                               "무슨 말인지 알아들어요."),
            ("🎨", "AI 그림·챗봇", "요즘 이야기하는 AI는 대부분 이 <b>생각 그물</b>로 만들어져요. "
                                  "여러분이 오늘 그 원리를 배운 거예요!"),
        ])
        tab_intro, tab_practice, tab_game = st.tabs([
            "① 알아보기", "② 연습하기 · 뉴런 1개", "③ 게임 도전 🏅 로봇 네오 발사"])

        # [1단계: 먼저 알아보기] -----------------------------------------------
        with tab_intro:
            predict_first(
                "predict_nn",
                "친구 셋이 의견을 냈어요. 그중 <b>한 명의 말을 더 크게</b> 들으면?",
                ['결과가 달라져요', '결과는 똑같아요', '아무 말도 못 해요'],
                0,
                "아래에서 <b>반영도 다이얼</b>을 돌려 보면 결과가 바뀌는 게 보여요.",
            )

            st.markdown("### 🌉 어? 나 이거 이미 알아!")
            md_html("""
            <div style='font-size:16px; line-height:1.75; background:#FFF8E1; border-radius:14px; padding:13px 18px;'>
            요리 대회 심사위원을 떠올려 봐요. 🍳 <b>맛 담당 · 영양 담당 · 모양 담당</b>이 각자 점수를 주고,
            그 점수를 <b>하나로 합쳐</b> 최종 점수를 정하죠. 보는 곳이 달라서 혼자보다 공정해요.<br>
            AI의 생각 그물도 똑같아요. 여러 '판단 담당'이 점수를 매긴 뒤
            <b>곱하고 더해 합친답니다.</b> (손 드는 투표가 아니에요!)
            </div>
            """)

            st.write("")
            mission_card("판단 요정 한 명의 계산법을 먼저 익히고, 나중엔 요정 3명을 함께 써서 로봇의 파워를 채워보세요!")

            st.write("")
            st.write("생각 그물은 아주 작은 **'작은 판단 요정'**을 여러 개 이어 붙여 만든 **생각 그물망**이에요. "
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
                  <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>② 판단 요정들</b><br>(각자 계산)</div>
                  <div style='{_nn_hid}'>🧚 요정1<br><span style='font-size:11px;'>×곱하고 +더하기</span></div>
                  <div style='{_nn_hid}'>🧚‍♂️ 요정2<br><span style='font-size:11px;'>×곱하고 +더하기</span></div>
                  <div style='{_nn_hid}'>🧚‍♀️ 요정3<br><span style='font-size:11px;'>×곱하고 +더하기</span></div>
                </div>
                <div style='flex:0.4;'><div style='{_nn_arrow}'>➡️</div></div>
                <div style='flex:1.2;'>
                  <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>③ 의견 모으기</b></div>
                  <div style='{_nn_hid} margin-top:40px;'>➕ 세 요정의<br>점수를 합쳐요</div>
                </div>
                <div style='flex:0.4;'><div style='{_nn_arrow}'>➡️</div></div>
                <div style='flex:1.2;'>
                  <div style='text-align:center; font-size:13px; color:#666; margin-bottom:6px;'><b>④ 최종 결론</b></div>
                  <div style='{_nn_out} margin-top:34px;'>🤖 로봇 파워<br><b>87점!</b> 🚀</div>
                </div>
              </div>
              <div style='text-align:center; margin-top:12px; font-size:14px; color:#3A4A6B;'>
                판단 요정 한 명 한 명은 <b>곱하고 더하는 간단한 계산</b>을 한 뒤,<br>
                <b>점수가 0보다 낮으면 아예 신호를 보내지 않아요</b>(신호를 보낼지 말지 스스로 정하는 거예요).<br>
                바로 이 '보낼까 말까' 규칙 덕분에, 여러 명의 계산을 <b>모으면</b>
                혼자서는 못 하던 똑똑한 판단이 나온답니다! 이게 바로 <b>생각 그물</b>이에요.
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
                           "이렇게 **여러 관점을 곱하고 더해 합치는 것**이 생각 그물이 뉴런을 여러 개 쓰는 이유예요. "
                           "(손드는 다수결 투표가 아니에요!)")

            st.write("")
            st.info("💡 피자를 먹고 뛰어놀면, **판단 요정 한 명**이 '피자 점수'와 '운동 점수'를 자기만의 힘으로 곱해서 더한 다음 건강 점수를 알려줘요.")
            st.caption("2단계 탭에서 직접 슬라이더를 움직여 판단 요정 1명이 어떻게 계산하는지 확인해봐요.")

        # [2단계: 뉴런 1개 실습] -----------------------------------------------
        with tab_practice:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### 📥 내가 한 행동 (입력)")
                pizza = st.slider("🍕 피자를 몇 조각 먹었나요?", 0, 10, 5, key="nn_learn_pizza")
                exercise = st.slider("🏃‍♂️ 밖에서 몇 시간 놀았나요?", 0, 10, 5, key="nn_learn_exercise")

            with col2:
                st.write("### ⚙️ 판단 요정 1명의 계산")
                weight_pizza = -1
                weight_exercise = 2

                hidden_score = (pizza * weight_pizza) + (exercise * weight_exercise)

                st.info(f"💡 피자 먹은 점수: {pizza}조각 × (-1점) = {pizza * weight_pizza}점")
                st.info(f"💡 뛰어논 점수: {exercise}시간 × (보너스 2점!) = {exercise * weight_exercise}점")
                st.warning(f"판단 요정이 계산한 점수: **{hidden_score}점**")
                # 오른쪽 칸의 최종 점수는 '기본 점수 50 + 요정 점수'인데
                # 예전에는 이 50점을 아무 데서도 설명하지 않아 두 숫자가 연결되지 않았다.
                st.caption(f"➕ 여기에 **누구나 갖고 시작하는 기본 점수 50점**을 더해요. "
                           f"→ 50 + ({hidden_score}) = **{50 + hidden_score}점** "
                           f"(0점보다 낮거나 100점보다 높으면 0·100으로 맞춰요)")

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
                    "판단 요정이 딱 1명뿐이면 어떤 점이 아쉬울까요?",
                    ["선택 안 함", "① 한 가지 관점으로만 판단해서 다양한 상황을 놓칠 수 있다", "② 아쉬운 점이 전혀 없다"],
                    key="nn_think1"
                )
                if think_nn.startswith("①"):
                    st.success("맞아요! 그래서 진짜 생각 그물은 여러 요정을 함께 써서 다양한 관점을 모아요.")
                elif think_nn.startswith("②"):
                    st.warning("게임 탭에서 요정 3명을 같이 써보면 왜 여러 명이 필요한지 느껴질 거예요!")

            st.success("🔗 **다음 단계!** 진짜 생각 그물은 이런 판단 요정을 **여러 명** 동시에 써요. 각 요정이 조금씩 다르게 판단한 걸 또 한 번 합쳐서 최종 결론을 내리죠. '게임' 탭에서 요정 3명이 힘을 합치는 걸 직접 만들어볼까요?")

        # [게임: 로봇 네오 발사 — 은닉층 요정 3명] -------------------------------
        with tab_game:
            st.header("🤖 판단 요정 3명의 힘을 모아 로봇 네오를 발사하자!")
            st.write("입력 3가지가 **가운데 요정 3명**에게 각각 전달돼요. 요정들은 저마다 다른 힘으로 판단하고, "
                     "그 3개의 판단을 **출력 요정**이 또 한 번 종합해서 최종 파워를 결정해요. 이게 진짜 생각 그물의 계산 방식이에요!")

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
                st.caption("가운데 요정 3명은 이미 각자 '성격'(튼튼 담당·운동 담당·잠 담당)이 정해져 있어요. "
                           "너는 **출력 요정**이 되어, 그 3명의 의견을 각각 얼마나 반영할지만 정하면 돼요!")
                wo1 = st.slider("🧚 1번 요정(튼튼 담당) 의견 반영도", 0.0, 2.0, 0.5, step=0.1, key="nn_wo1")
                wo2 = st.slider("🧚 2번 요정(운동 담당) 의견 반영도", 0.0, 2.0, 0.4, step=0.1, key="nn_wo2")
                wo3 = st.slider("🧚 3번 요정(잠 담당) 의견 반영도", 0.0, 2.0, 0.3, step=0.1, key="nn_wo3")
                bias_out = st.slider("⚡ 기본 여유 힘", 0.0, 50.0, 40.0, step=1.0, key="nn_bias_out")

                with st.expander("🔧 (선택) 가운데 요정 3명의 성격도 직접 바꿔보고 싶다면"):
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
            z1 = g_pizza * w1_pizza + g_exercise * w1_ex + g_sleep * w1_sleep
            z2 = g_pizza * w2_pizza + g_exercise * w2_ex + g_sleep * w2_sleep
            z3 = g_pizza * w3_pizza + g_exercise * w3_ex + g_sleep * w3_sleep

            # ★ 활성화 함수(ReLU): 계산값이 0보다 낮으면 신호를 보내지 않는다(0으로 만든다).
            #   이 단계가 없으면 '곱하고 더하기'를 두 번 반복하는 것뿐이라, 요정을 몇 명을 쓰든
            #   결국 요정 1명으로 만들 수 있는 결과와 똑같아진다(선형 결합을 합치면 다시 선형).
            #   즉 "여러 뉴런을 모으면 더 똑똑해진다"는 이 활동의 설명이 성립하려면 반드시 필요하다.
            h1, h2, h3 = max(0.0, z1), max(0.0, z2), max(0.0, z3)

            # 출력 요정이 은닉층 3명의 판단을 다시 한 번 종합
            raw = h1 * wo1 + h2 * wo2 + h3 * wo3 + bias_out
            power = max(0, min(100, raw))

            with colR:
                st.write("#### ⚡ 네오의 파워 게이지")
                st.progress(int(power) / 100)
                st.write(f"### {power:.0f} / 100")

                with st.expander("🔍 가운데 요정 3명이 계산한 중간 점수 보기"):
                    for _i, (_z, _h) in enumerate([(z1, h1), (z2, h2), (z3, h3)], start=1):
                        if _h == 0:
                            st.write(f"🧚 {_i}번 요정: 계산값 {_z:.1f}점 → 😴 **신호 없음(0점)**")
                        else:
                            st.write(f"🧚 {_i}번 요정: **{_h:.1f}점** 전달!")
                    st.caption("요정은 계산값이 0보다 낮으면 신호를 보내지 않아요(0점). "
                               "이렇게 신호를 보낼지 말지 스스로 정한답니다. "
                               "출력 요정 = (1번×반영도) + (2번×반영도) + (3번×반영도) + 기본 힘")

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
                celebrate_once("_celeb_nn_power")   # 슬라이더를 건드릴 때마다 풍선이 또 터지지 않게
                st.success("🚀 파워 100 달성! 세 요정과 출력 요정이 힘을 합쳐 네오가 우주로 로켓 발사!! 🎉")
                award_badge(5)
                if len(st.session_state["badges"]) >= 6:
                    st.success("🎊 6개 부품을 모두 모아 **로봇 네오가 완전히 조립됐어요!** 배운 걸 정리하는 복습 미션에 도전하거나, 바로 수료증을 받으러 갈 수 있어요.")
                    cta1, cta2 = st.columns(2)
                    with cta1:
                        if st.button("🔧 복습 미션 도전!", key="nn_to_review", **_STRETCH):
                            go_to("🔧 로봇 네오 종합 점검")
                    with cta2:
                        if st.button("🏆 수료증 받기", key="nn_to_cert", **_STRETCH):
                            go_to("🏆 사후평가 (수료증)")
                md_html("""
                <div style="background:#EDE7F6; border-left:5px solid #7E57C2; border-radius:14px; padding:14px 18px; margin-top:10px;">
                    <b>🏷️ 오늘 배운 것</b> 🧠<br>
                    여러 <b>판단 담당(요정)</b>이 각자 다른 것을 보고 점수를 매긴 뒤 <b>합치면</b>,
                    혼자서는 못 하던 똑똑한 판단이 나와요. 우리 뇌가 일하는 모습을 흉내 내어 만든 거예요.<br>
                    이 요정을 <b>아주 많이</b> 늘리고 여러 겹으로 쌓으면
                    <b>챗봇·번역기·자율주행</b>도 만들 수 있어요. 오늘 그 씨앗을 만들었어요!
                </div>
                """)

                with st.expander("🌟 보너스 퀴즈: 완벽 이해 배지 도전!"):
                    quiz_nn = st.radio(
                        "판단하는 요정이 **한 명**보다 **여러 명**일 때 좋은 점은?",
                        ["선택 안 함", "① 서로 다른 것을 보고 매긴 점수를 합쳐 더 똑똑해진다",
                         "② 요정이 많으면 그냥 시끄럽기만 하다", "③ 한 명일 때와 똑같다"],
                        key="quiz_nn"
                    )
                    if quiz_nn.startswith("①"):
                        st.success("🌟 맞아요! 여러 관점을 합치면 혼자서는 못 하던 판단이 나와요. 6개 섬 완주! 🎉")
                        st.session_state["bonus_badges"].add("🌟 생각 그물 완벽 이해")
                    elif quiz_nn != "선택 안 함":
                        st.warning("요정 1명일 때와 3명일 때를 비교한 화면을 다시 떠올려볼까요?")

                st.write("")
                next_section_button("🧠 6. 똑똑한 생각 주머니", "nn")
            elif power >= 60:
                clear_celebration("_celeb_nn_power")
                st.info("거의 다 왔어요! 요정들의 의견 반영도를 조금 더 키우거나, 네오를 더 재우고 운동시켜 볼까요?")
            else:
                clear_celebration("_celeb_nn_power")
                st.warning("아직 힘이 부족해요. 피자를 줄이고 잠·운동을 늘리거나, 요정들의 의견 반영도를 키워보세요!")

            with st.expander("🧚 요정 설명: 왜 요정이 여러 명 필요할까요?"):
                st.write("판단 요정 1명만 있으면 딱 한 가지 방식으로만 판단해요. 그런데 요정 3명이 **서로 다른 관점**(1번은 튼튼함 위주, "
                         "2번은 운동 위주, 3번은 잠 위주)으로 각자 판단하고, 출력 요정이 그 3개의 의견을 다시 한 번 종합하면 "
                         "훨씬 더 다양하고 정교한 판단을 내릴 수 있어요. 이렇게 여러 층의 요정이 쌓인 구조를 "
                         "**생각 그물(요정이 여러 층으로 쌓인 그물망)**이라고 불러요.")


    # --- [AI 윤리: AI를 똑똑하게 쓰려면?] ---
    elif menu == "⚖️ AI를 똑똑하게 쓰려면?":
        hero_card("⚖️", "AI를 똑똑하게 쓰려면?",
                   "AI는 아주 편리하지만 항상 옳은 건 아니에요. AI를 현명하게 쓰는 법을 함께 생각해봐요.",
                   "#00897B", term="AI를 바르게 쓰기")

        with st.expander("🎯 이 활동과 관련된 성취기준 (2022 개정 교육과정)", expanded=False):
            md_html("""
            - **[실과] [6실05-05]** 인공지능이 만들어지는 과정을 체험하고, **인공지능이 사회에 미치는 영향**을 탐색한다.
              → AI가 틀릴 수 있다는 점, 편향과 개인정보 문제 등을 생각하며 **비판적 사고**를 길러요.
            """)

        tab_e1, tab_e2, tab_e5, tab_e3, tab_e4 = st.tabs([
            "① 알아보기 · AI의 약점",
            "② 생각해보기 · 상황 판단",
            "③ 달라지는 세상 · AI와 직업",
            "④ 확인하기 · O/X 퀴즈",
            "⑤ 다짐하기 · 나의 서약서",
        ])

        # [1단계: AI의 비밀, 눈으로 보기] --------------------------------------
        with tab_e1:
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
            st.caption("② 탭에서 진짜 있을 법한 상황들을 놓고 함께 판단해봐요.")

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
            st.caption("③ 탭에서 AI 때문에 세상이 어떻게 달라지는지 살펴봐요.")

        # [③ 달라지는 세상 · AI와 직업] ----------------------------------------
        #   [6실05-05] "인공지능으로 인한 사회의 발전과 직업의 변화를 이해하여
        #   인공지능이 사회에 미치는 영향을 탐색한다" — 이 탭이 그 축을 담당한다.
        with tab_e5:
            st.markdown("### 🏙️ AI가 오면 세상은 어떻게 달라질까요?")
            neo_says("AI는 사람의 일을 <b>빼앗기만 하는 게 아니에요.</b> 많은 경우 "
                     "<b>일하는 방법을 바꾸고</b>, <b>새로운 일</b>도 만들어 낸답니다.<br>"
                     "아래 직업들이 앞으로 어떻게 바뀔지 <b>내 생각</b>을 골라보세요. "
                     "<b>정답은 없어요!</b> 친구와 이야기 나눠봐도 좋아요. 💬")

            # (이모지, 직업, 눌렀을 때 보여줄 이야기) — 분야별로 5개씩, 모두 15개
            # 한 화면에 15개를 세로로 쌓으면 스크롤이 너무 길어져서 분야 탭으로 나눈다.
            _JOB_GROUPS = [
                ("🏠 생활 · 서비스", [
                    ("🚚", "트럭 운전사",
                     "스스로 달리는 <b>자율주행 트럭</b>이 시험 중이에요.<br>"
                     "대신 <b>트럭을 정비하고, 잘 가는지 지켜보는 일</b>이 새로 생겼어요."),
                    ("👨‍🍳", "요리사",
                     "볶음밥을 볶는 <b>조리 로봇</b>이 벌써 식당에서 일해요.<br>"
                     "그래도 <b>새로운 맛을 상상하는 일</b>은 사람이 훨씬 잘해요."),
                    ("🧑‍🌾", "농부",
                     "드론이 <b>밭을 살펴보고</b> AI가 물 줄 때를 알려줘요.<br>"
                     "무엇을 심을지, 날씨를 어떻게 이겨낼지는 <b>농부의 경험</b>이 필요해요."),
                    ("🚒", "소방관",
                     "AI가 <b>불이 번질 방향</b>을 예측하고 로봇이 먼저 들어가기도 해요.<br>"
                     "하지만 <b>사람을 찾아 구하는 판단</b>은 소방관이 해요."),
                    ("🛒", "마트 계산원",
                     "계산대 없이 <b>그냥 들고 나가는 가게</b>가 생겼어요.<br>"
                     "대신 <b>손님을 돕고 매장을 살피는 일</b>이 더 중요해졌어요."),
                ]),
                ("🏥 전문 · 공공", [
                    ("👩‍⚕️", "의사",
                     "AI가 <b>엑스레이 사진</b>에서 이상한 곳을 빨리 찾아 줘요.<br>"
                     "하지만 <b>환자의 마음을 달래고 치료를 정하는 일</b>은 의사의 몫이에요."),
                    ("👩‍🏫", "선생님",
                     "AI가 <b>문제를 골라 주고 채점</b>을 도와줘요.<br>"
                     "친구와 싸운 마음을 알아주고 <b>함께 자라게 돕는 일</b>은 선생님만 할 수 있어요."),
                    ("⚖️", "변호사",
                     "AI가 <b>수만 장의 법 문서</b>를 몇 초 만에 찾아 줘요.<br>"
                     "어떤 이야기로 <b>사람을 설득할지</b>는 변호사가 정해요."),
                    ("👮", "경찰관",
                     "AI가 <b>사고가 잦은 곳</b>을 알려줘 순찰을 도와줘요.<br>"
                     "다만 <b>사람을 의심하는 판단</b>을 AI에게 맡기면 억울한 일이 생길 수 있어요."),
                    ("🧑‍💼", "은행원",
                     "간단한 일은 <b>앱과 AI 상담</b>이 대신해요.<br>"
                     "큰돈에 대해 <b>같이 고민해 주는 일</b>은 사람을 찾게 돼요."),
                ]),
                ("🎨 예술 · 미디어", [
                    ("🎨", "웹툰 작가",
                     "AI가 <b>배경 그림</b>을 빨리 그려 줘 이야기에 집중할 수 있어요.<br>"
                     "대신 “누구의 그림을 따라 한 걸까?” 하는 <b>새로운 고민</b>도 생겼어요."),
                    ("🎬", "영화감독",
                     "AI가 <b>특수 효과</b>를 훨씬 싸고 빠르게 만들어 줘요.<br>"
                     "어떤 이야기로 <b>사람을 울고 웃게 할지</b>는 감독이 정해요."),
                    ("📰", "기자",
                     "AI가 <b>경기 결과 기사</b> 정도는 혼자 써요.<br>"
                     "현장에 가서 <b>직접 묻고 사실을 확인하는 일</b>이 더 중요해졌어요."),
                    ("🎼", "작곡가",
                     "AI가 <b>비슷한 느낌의 곡</b>을 순식간에 만들어 줘요.<br>"
                     "내 마음을 담아 <b>처음 듣는 느낌</b>을 만드는 건 사람이에요."),
                    ("🗣️", "통역사",
                     "번역 앱이 <b>여행 회화</b> 정도는 거뜬히 해내요.<br>"
                     "농담이나 <b>말 속에 숨은 마음</b>까지 전하는 건 아직 사람이 잘해요."),
                ]),
            ]
            _JOBS = [j for _, group in _JOB_GROUPS for j in group]
            _OPTS = ["선택 안 함",
                     "🤖 AI가 거의 다 하게 될 것 같아",
                     "🤝 사람과 AI가 함께 할 것 같아",
                     "💪 그래도 사람이 더 중요할 것 같아"]

            # 한 분야(5개)만 끝내도 정리 카드가 열리게 한다.
            # (6개로 하면 첫 탭만 본 학생은 아무리 골라도 정리를 못 봐서 막힌다)
            _GOAL = 5
            _answered = 0
            _job_tabs = st.tabs([name for name, _ in _JOB_GROUPS])
            _ji = 0
            for _tab, (_gname, _group) in zip(_job_tabs, _JOB_GROUPS):
                with _tab:
                    for _emo, _job, _story in _group:
                        st.markdown(f"##### {_emo} {_job}")
                        _pick = st.radio(f"{_job}의 앞날은?", _OPTS, horizontal=True,
                                         key=f"ethics_job_{_ji}", label_visibility="collapsed")
                        if _pick != "선택 안 함":
                            _answered += 1
                            md_html(f"""
                            <div style="background:#E8F5E9; border-left:6px solid #43A047; border-radius:12px;
                                        padding:11px 15px; margin:8px 0 14px 0;
                                        font-family:'Gowun Dodum',sans-serif; font-size:15px;
                                        color:#1B3A24; line-height:1.75;">
                              🔎 <b>이렇게 바뀌는 중이에요</b><br>{_story}
                            </div>
                            """)
                        else:
                            st.caption("👆 내 생각을 골라보면 어떻게 바뀌는 중인지 알려줄게요!")
                        _ji += 1

            st.write("")
            st.progress(min(_answered, _GOAL) / _GOAL)
            st.caption(f"직업 {_answered}개를 살펴봤어요 (총 {len(_JOBS)}개 중) · "
                       f"한 분야({_GOAL}개)만 끝내도 아래 정리가 열려요 · 점수는 매기지 않아요")

            if _answered >= _GOAL:
                st.success(f"🌟 직업 {_answered}개를 살펴봤어요! 어떤 점을 느꼈나요?")
                md_html("""
                <div style="background:#FFF8E1; border:3px solid #FFB300; border-radius:16px;
                            padding:16px 20px; margin:6px 0;
                            font-family:'Gowun Dodum',sans-serif; font-size:16px;
                            color:#5D4037; line-height:1.85;">
                  💡 <b style="font-family:'Jua',sans-serif; font-size:18px; color:#E65100;">
                  오늘의 큰 깨달음</b><br>
                  AI가 온다고 직업이 <b>사라지기만 하는 게 아니에요.</b><br>
                  ① 힘들고 반복되는 일은 <b>AI가 도와주고</b>,
                  ② 사람은 <b>마음을 알아주고 새로 상상하는 일</b>에 더 집중하게 되고,
                  ③ 예전에 <b>없던 직업</b>이 새로 생겨나요.
                </div>
                """)

                st.write("")
                st.markdown("#### ✨ AI 때문에 **새로 생긴** 직업도 있어요!")
                _nc1, _nc2, _nc3 = st.columns(3)
                _newjobs = [
                    (_nc1, "🧑‍🏫", "AI 훈련사", "#1565C0", "#E3F2FD",
                     "AI에게 <b>좋은 예시</b>를 골라 가르치는 사람"),
                    (_nc2, "🕵️", "AI 검사관", "#6A1B9A", "#F3E5F5",
                     "AI가 <b>공평하게 판단하는지</b> 확인하는 사람"),
                    (_nc3, "💬", "AI 소통 전문가", "#2E7D32", "#E8F5E9",
                     "AI에게 <b>잘 물어보는 방법</b>을 연구하는 사람"),
                ]
                for _col, _e, _n, _c, _bg, _d in _newjobs:
                    with _col:
                        md_html(f"""
                        <div style="background:{_bg}; border:2px solid {_c}55; border-radius:16px;
                                    padding:14px 10px; text-align:center; min-height:150px;">
                          <div style="font-size:34px;">{_e}</div>
                          <div style="font-family:'Jua',sans-serif; font-size:16px; color:{_c};
                                      margin-top:5px;">{_n}</div>
                          <div style="font-family:'Gowun Dodum',sans-serif; font-size:13px;
                                      color:#3A4A6B; margin-top:5px; line-height:1.5;">{_d}</div>
                        </div>
                        """)
                st.write("")
                neo_says("어? 잠깐만요! <b>🧑‍🏫 AI 훈련사</b>가 하는 일… "
                         "여러분이 <b>0번 섬에서 나에게 이름표를 붙여 가르쳐 준 일</b>과 똑같아요!<br>"
                         "여러분은 이미 <b>미래 직업 하나를 체험해 본</b> 셈이에요. 멋지죠? 🚀", mood="happy")
                st.info("🗣️ **함께 이야기해봐요** — 내가 되고 싶은 직업은 AI 때문에 어떻게 바뀔까요? "
                        "그때 나는 무엇을 더 잘해야 할까요?")

            st.write("")
            st.caption("④ 탭에서 O/X 퀴즈로 배운 것을 확인해봐요.")

        # [④ 확인하기 · O/X 스피드 퀴즈] ---------------------------------------
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
                    celebrate_once("_celeb_ethics_ox")
                    md_html("""
                    <div style='background:#E0F2F1; border:3px solid #00897B; border-radius:16px;
                                padding:18px; text-align:center;'>
                        <div style='font-size:40px;'>🛡️</div>
                        <div style="font-family:'Jua',sans-serif; font-size:22px; color:#00695C;">
                            6문제 모두 정답! 당신은 AI 지킴이! 🎉</div>
                    </div>
                    """)
                else:
                    clear_celebration("_celeb_ethics_ox")
                    st.info(f"**{ox_correct} / {len(_OX_QUIZ)}문제** 맞혔어요! 틀린 문제의 설명을 읽고 다시 골라 만점에 도전해봐요.")
            else:
                clear_celebration("_celeb_ethics_ox")
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
                celebrate_once("_celeb_ethics_pledge")
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
                clear_celebration("_celeb_ethics_pledge")
                st.caption("네 가지를 모두 체크하면 '슬기로운 AI 사용자 인증'을 받을 수 있어요!")



    # --- [복습 게임: 로봇 네오 종합 점검 미션] ---
    elif menu == "🔧 로봇 네오 종합 점검":
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
            wrong = int(st.session_state.get("review_wrong", 0) or 0)
            correct_first_try = total - wrong if wrong <= total else 0

            celebrate_once("_celeb_review_done")
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
                        go_to("🏆 사후평가 (수료증)")
                else:
                    if st.button("🏠 탐험 본부로", key="review_to_home", **_STRETCH):
                        go_to("🏠 탐험 본부 (홈)")

        # 진행 중 화면 (문제 풀이)
        else:
            quiz = st.session_state["review_quiz"]
            total = len(quiz)
            # 인덱스가 범위를 벗어나면(새로고침·엉뚱한 클릭) 가장 가까운 문제로 안전하게 되돌린다
            idx = int(st.session_state.get("review_idx", 0) or 0)
            if not (0 <= idx < total):
                idx = max(0, min(total - 1, idx))
                st.session_state["review_idx"] = idx
            q = quiz[idx]
            if not isinstance(q, dict) or not (0 <= int(q.get("part_idx", -1)) < 6):
                start_review_game()      # 문제 데이터가 깨졌으면 새 판으로 시작
                st.rerun()
            part_idx = q["part_idx"]
            part_name = ALL_BADGES[part_idx][3]

            # 상단 진행바 + 이미 고친 부품 표시
            st.progress(idx / total)
            fixed_row = " ".join(
                badge_img_tag(i, size=30, grayscale=(i not in st.session_state["review_fixed"]))
                for i in range(6)
            )
            md_html(f"<div style='display:flex; gap:6px; flex-wrap:wrap; margin:4px 0 12px 0;'>{fixed_row}</div>")
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


    # --- [메뉴: 사후평가 · 수료증 발급소] ---
    elif menu == "🏆 사후평가 (수료증)":
        hero_card("🏆", "사후평가 · 수료증 발급소",
                   "사후 검사는 언제든지 할 수 있어요. 부품 6개를 모두 모으면 수료증도 짠! 하고 나타나요.",
                   "#FFC93C")

        render_bgm_player(BGM_CERT_FILE, "cert", label="🎵 축하 음악")

        # 한 화면에 사후 검사와 수료증이 모두 있어 스크롤이 매우 길었다.
        # 두 탭으로 나눠서 학생이 지금 할 일만 보이게 한다.
        tab_post, tab_cert = st.tabs(["📊 사후 검사", "🎓 수료증 만들기"])

        with tab_post:

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
                        st.session_state.pop("_diag_part1_done_post", None)
                        st.rerun()
            else:
                if not st.session_state.get("student_name", "").strip():
                    st.info("💡 '나의 이름 & 사전 평가' 방에서 먼저 이름을 적으면, 사전 결과와 비교할 수 있어요! (안 해도 사후 검사는 할 수 있어요.)")
                render_diagnostic_quiz("post", "post_score")


        with tab_cert:

            st.markdown("### 🎓 나만의 수료증 만들기")

            _cert_badge_count = len(st.session_state["badges"])
            # 심사위원·참관 선생님은 6개 섬을 다 하실 시간이 없다.
            # 그래서 부품을 다 모으지 않아도 수료증을 '미리 보기'로 확인할 수 있게 한다.
            # (미리 보기로 만든 수료증은 학급 기록에 저장하지 않는다.)
            _cert_preview = bool(st.session_state.get("cert_preview"))
            if _cert_badge_count < 6 and not _cert_preview:
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

                # 심사·참관용 미리 보기 (학생에게는 '어떤 모습인지 구경만' 하는 기능)
                st.write("")
                md_html("""
                <div style='background:#E8F0FE; border:2px solid #5C6BC0; border-radius:12px;
                            padding:10px 14px;'>
                  <b style="font-family:'Jua',sans-serif; color:#283593;">
                    👩‍🏫 선생님 · 참관자님께</b><br>
                  <span style='font-size:15px; color:#28406B;'>
                    부품을 다 모으지 않아도 <b>수료증이 어떻게 나오는지 미리 볼 수 있어요.</b>
                    미리 보기로 만든 수료증은 <b>학급 기록에 저장되지 않습니다.</b>
                  </span>
                </div>
                """)
                if st.button("👀 수료증 미리 보기 (심사·참관용)", key="cert_preview_on", **_STRETCH):
                    st.session_state["cert_preview"] = True
                    st.rerun()
                name = ""
            else:
                if _cert_badge_count < 6:
                    st.info(f"👀 **미리 보기 모드**예요. (지금 모은 부품 {_cert_badge_count}/6개) "
                            "이름을 적으면 수료증이 어떻게 나오는지 바로 볼 수 있어요. "
                            "**이 수료증은 학급 기록에 저장되지 않아요.**")
                    if st.button("↩️ 미리 보기 끄기", key="cert_preview_off"):
                        st.session_state["cert_preview"] = False
                        st.rerun()
                else:
                    st.write("아래에 탐험대장님의 이름을 적으면, 로봇 부품 6개가 모두 새겨진 멋진 수료증이 짠! 하고 나타납니다.")
                name = st.text_input("나의 이름은?", value=st.session_state.get("student_name", ""),
                                     placeholder="이름을 적어주세요 (예: 홍길동)", key="cert_name")

            if name and name.strip():
                name = name.strip()[:20]           # 수료증에 예쁘게 들어가도록 이름 길이는 20자까지
                _is_preview = _cert_badge_count < 6      # 미리 보기로 들어온 경우
                if not _is_preview:
                    st.session_state["student_name"] = name
                    if not save_student_record(name):
                        st.caption("⚠️ 학습 기록을 파일에 저장하지 못했어요. (수료증은 그대로 만들 수 있어요)")
                num_badges = len(st.session_state["badges"])
                num_bonus = len(st.session_state["bonus_badges"])
                now_kr = datetime.now(timezone(timedelta(hours=9)))
                today_kr = f"{now_kr.year} 년  {now_kr.month} 월  {now_kr.day} 일"

                cert_img, png_bytes = None, None
                try:
                    with st.spinner("수료증을 만들고 있어요..."):
                        cert_img = build_certificate_image(name, num_badges, num_bonus, today_kr)
                    buf = io.BytesIO()
                    cert_img.save(buf, format="PNG", optimize=True)
                    png_bytes = buf.getvalue()
                except Exception:
                    _LOG.exception("수료증 이미지 생성 실패")
                    cert_img = None

                if cert_img is None:
                    neo_says("앗, 수료증 그림을 만드는 데 문제가 생겼어요. 그래도 여러분이 6개 섬을 모두 완주한 건 "
                             "변하지 않아요! 선생님께 말씀드리면 <b>media/fonts</b> 폴더를 확인해 주실 거예요.", mood="oops")
                else:
                    st.image(cert_img, **_STRETCH)
                    celebrate_once(f"_celeb_cert_{name}")   # 같은 이름으로 다시 그려도 축하는 한 번만
                    if _is_preview:
                        st.caption(f"👀 미리 보기입니다. 지금 모은 부품 {_cert_badge_count}/6개가 그대로 새겨져요. "
                                   "6개를 다 모으면 부품이 모두 채워진 수료증이 나옵니다.")

                    st.write("")
                    _safe_name = sanitize_filename(name)
                    render_share_buttons(
                        png_bytes,
                        filename=f"AI탐험대_수료증_{_safe_name}.png",
                        share_title=f"{_safe_name} 탐험대장님의 인공지능 원리 탐험대 수료증",
                    )
                    st.caption("💡 다운로드한 이미지는 구글 클래스룸, 클래스팅, 카카오톡 등 어디에나 파일로 첨부해서 제출할 수 있어요!")

    # --- [교사용: 선생님 방 (학습 분석)] ---
    elif menu == "📜 자료 출처 · 저작권":
        hero_card("📜", "자료 출처 · 저작권",
                  "이 프로그램에 쓰인 모든 자료의 출처예요. 저작권을 지키는 것도 중요한 공부랍니다.",
                  "#607D8B")

        st.info("🔎 이 프로그램에 사용된 그림·소리·글꼴의 출처를 모두 밝혀 둔 곳이에요. "
                "아래 자료들은 모두 **이용 허락을 확인하고** 사용했어요.")

        t_img, t_snd, t_font = st.tabs(["🖼️ 그림 · 영상", "🎵 소리 · 음악", "🔤 글꼴 · 소프트웨어"])

        with t_img:
            st.markdown("### 🖼️ 그림 · 영상")
            md_text("""
        | 자료 | 출처 · 이용 조건 |
        |---|---|
        | 로봇 부품 배지 18종 (안테나·팔·뇌·심장·다리·센서) | **출품자 제작** (Canva AI) |
        | 소개 영상 및 영상 속 그림 (아이콘, 배경, 로봇 마스코트 등) | **출품자 제작** (Canva AI) |
        | 수료증 배경 이미지 | **출품자 제작** (Python·Pillow 코드 생성) |
        | 시작 화면 이미지(start) | 공유마당(공공누리) 내려받음 |
        """)


        with t_snd:
            st.markdown("### 🎵 소리 · 음악")
            md_text("""
        | 자료 | 출처 · 이용 조건 |
        |---|---|
        | 영상 내레이션 (sound_intro) | 네이버 **클로바더빙**으로 제작 |
        | 배경음악 `Cheek To Cheek` (music1) | **공유마당 CC BY** — 저작자 표시 |
        | 배경음악 `장난감마을` (music2) | **공유마당 CC BY** — 저작자 표시 |
        | 효과음 `코드7` (key, sound0) | **공유마당 CC BY** — 저작자 표시 |
        | 효과음 `K영화 열두번째 효과음` | **공유마당 CC BY** — 저작자 표시 |
        | 효과음 `파괴적인 분위기 음악` (sound1) | **공유마당 CC BY** — 저작자 표시 |
        """)


        with t_font:
            st.markdown("### 🔤 글꼴")
            md_text("""
        | 글꼴 | 출처 · 이용 조건 |
        |---|---|
        | Noto Sans KR / Noto Serif KR | **Google Fonts** — SIL Open Font License |
        | Jua (주아체) | **Google Fonts** — SIL Open Font License |
        | Gowun Dodum (고운돋움) | **Google Fonts** — SIL Open Font License |
        | 학교안심 보드마카R | **한국교육학술정보원(KERIS)** — 공공누리 제1유형 |
        """)

            st.markdown("### 🧰 사용한 공개 소프트웨어")
            st.caption("Streamlit · scikit-learn · Plotly · pandas · NumPy · Pillow "
                       "— 모두 누구나 무료로 쓸 수 있는 오픈소스예요.")


        st.divider()
        st.success("✅ 위 자료들은 모두 **원저작자의 이용 허락 범위 안에서** 사용했고, "
                   "이용 허락 증빙 자료는 연구보고서 참고자료에 함께 담았어요.")

        st.caption("🌱 여러분도 자료를 쓸 때는 **누가 만들었는지 꼭 밝혀주세요.** "
                   "그게 만든 사람에 대한 예의랍니다.")


    elif menu == "👩‍🏫 선생님 방 (학습 분석)":
        hero_card("👩‍🏫", "선생님 방 (학습 분석)",
                   "학생들의 학습 데이터를 확인하고, 사전·사후 향상도를 분석할 수 있어요.",
                   "#455A64")

        # 처음 방문한 사람(심사위원·다른 학교 선생님)이 절대 놓치지 않도록
        # 안내 문장 속이 아니라 '독립된 박스'로 크게 보여준다.
        md_html(f"""
        <div style="background:#FFF8E1; border:3px solid #FFB300; border-radius:16px;
                    padding:16px 20px; margin-bottom:14px;">
          <div style="font-family:'Jua',sans-serif; font-size:20px; color:#E65100; margin-bottom:6px;">
            🔑 처음 오셨나요? 체험용 비밀번호를 그대로 쓰세요
          </div>
          <div style="font-family:'Gowun Dodum',sans-serif; font-size:16px; color:#5D4037; line-height:1.7;">
            체험용 비밀번호 →
            <span style="display:inline-block; background:#FFFFFF; border:2px dashed #E65100;
                         border-radius:8px; padding:2px 16px; margin:0 4px;
                         font-family:'Jua',sans-serif; font-size:22px; color:#BF360C;">
              {TEACHER_PASSWORD}
            </span><br>
            아래 칸에 적거나, <b>‘체험용으로 바로 열기’ 버튼</b>을 누르면 곧바로 들어올 수 있어요.
          </div>
        </div>
        """)

        if st.button("👀 체험용으로 바로 열기 (비밀번호 없이)", key="teacher_demo_open", **_STRETCH):
            st.session_state["teacher_unlocked"] = True
            st.rerun()

        st.caption("🔒 이 방은 선생님용이에요. 학급 전체의 학습 현황과 사전·사후 향상도를 볼 수 있어요.")
        pw = st.text_input("비밀번호", type="password", key="teacher_pw")

        if not (st.session_state.get("teacher_unlocked") or pw == TEACHER_PASSWORD):
            if pw:
                st.error("비밀번호가 맞지 않아요. 위에 적힌 체험용 비밀번호를 그대로 적어주세요.")
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
            # 숫자형 변환 (분석용). 오래된 CSV에 열이 없어도 멈추지 않도록 빈 열을 채워 둔다.
            num_df = df.copy()
            for col in ["사전점수", "사후점수", "향상점수", "획득배지수", "보너스배지수"]:
                if col not in num_df.columns:
                    num_df[col] = np.nan
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce")

            # 한 화면에 요약·설문·학생별 기록이 모두 있어 스크롤이 길었다. 세 탭으로 나눈다.
            t_sum, t_survey, t_rec = st.tabs(
                ["📊 우리 반 요약", "💬 영역별 설문", "📋 학생별 기록 · 내려받기"])

            with t_sum:
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

            with t_survey:
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
                    st.info("아직 사후 설문 응답이 없어요. 학생들이 “🏆 사후평가 (수료증)”에서 사후 검사를 마치면 여기에 표시됩니다.")


            with t_rec:
                st.markdown("### 📋 학생별 상세 기록")
                st.dataframe(df, **_STRETCH)

                csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                safe_pick = "전체" if pick == "(전체 보기)" else str(pick).replace(" ", "")
                st.download_button(f"⬇️ '{safe_pick}' 기록 CSV 다운로드", data=csv_bytes,
                                   file_name=f"AI탐험대_학습기록_{safe_pick}.csv", mime="text/csv", **_STRETCH)
                st.caption("🔒 학생 기록은 이 프로그램이 실행 중인 곳의 data 폴더에만 저장돼요. "
                           "공용 PC라면 수업 뒤 아래에서 기록을 지워주세요.")
                # 웹(클라우드)으로 배포해 쓰는 경우, 서버가 다시 시작되면 저장된 파일이 사라진다.
                # 교사가 이 사실을 모르면 한 학기 기록을 통째로 잃을 수 있어 분명히 안내한다.
                st.warning("⚠️ **수업이 끝나면 꼭 위 버튼으로 CSV를 내려받아 보관해 주세요.** "
                           "인터넷 주소로 접속해 쓰는 경우, 서버가 다시 시작되면 "
                           "**여기 쌓인 기록이 사라질 수 있어요.**")

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


# 화면 그리기 (전역 오류 방어막 포함). 진행상황(배지)의 주소창 동기화도 이 안에서 항상 실행된다.
guarded_render(menu)
