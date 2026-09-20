# -*- coding: utf-8 -*-

import os
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "").strip()
if not APP_URL:
    print("❌ APP_URL 환경변수가 비어 있습니다. (워크플로 파일의 APP_URL 을 확인하세요)")
    sys.exit(1)
if not APP_URL.startswith("http"):
    APP_URL = "https://" + APP_URL

# 앱이 살아났다는 신호 — 본문 영역과 사이드바가 모두 그려져야 진짜 실행된 것이다
READY_ANY = [
    '[data-testid="stSidebar"]',
    '[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"]',
]
# 잠들었을 때 나오는 버튼 문구 (스트림릿이 문구를 조금 바꿔도 걸리도록 여러 개)
WAKE_TEXTS = [
    "Yes, get this app back up",
    "get this app back up",
    "Wake up",
    "이 앱을 다시",
]

DWELL_SECONDS = 45       # 접속을 유지하는 시간 (스쳐 가는 방문으로 처리되지 않도록)
WAIT_ROUNDS = 36         # 5초 × 36 = 최대 3분까지 기다린다


def try_wake(page):
    """잠들어 있으면 깨우기 버튼을 누른다. 눌렀으면 True."""
    # ① 버튼 모양으로 먼저 찾아본다
    for txt in WAKE_TEXTS:
        try:
            btn = page.get_by_role("button", name=txt, exact=False).first
            if btn.is_visible(timeout=2_000):
                print(f"💤 잠들어 있었음 → 버튼 '{txt}' 누름")
                btn.click(timeout=10_000)
                return True
        except Exception:
            pass
    # ② 버튼이 아니라 글자로만 되어 있을 수도 있다
    for txt in WAKE_TEXTS:
        try:
            el = page.get_by_text(txt, exact=False).first
            if el.is_visible(timeout=2_000):
                print(f"💤 잠들어 있었음 → 글자 '{txt}' 누름")
                el.click(timeout=10_000)
                return True
        except Exception:
            pass
    return False


def app_is_up(page):
    for sel in READY_ANY:
        try:
            if page.query_selector(sel):
                return True
        except Exception:
            pass
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_default_timeout(60_000)

        print(f"▶ 접속: {APP_URL}")
        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        except Exception as e:
            print(f"❌ 주소를 열지 못했습니다: {e}")
            browser.close()
            sys.exit(1)

        time.sleep(5)
        woke = try_wake(page)
        if not woke:
            print("🟢 잠자기 버튼 없음 (이미 깨어 있거나 바로 실행 중)")

        # ── 화면이 진짜 그려질 때까지 기다린다 ────────────────
        ok = False
        for i in range(WAIT_ROUNDS):
            if app_is_up(page):
                ok = True
                break
            # 늦게 뜬 깨우기 버튼도 한 번 더 눌러본다
            if i in (3, 8):
                try_wake(page)
            time.sleep(5)

        if not ok:
            print("❌ 앱 화면이 끝내 뜨지 않았습니다.")
            try:
                page.screenshot(path="wake_fail.png")
            except Exception:
                pass
            browser.close()
            sys.exit(1)

        print(f"⏳ {DWELL_SECONDS}초 동안 접속을 유지하는 중…")
        time.sleep(DWELL_SECONDS)

        print(f"✅ 앱이 살아 있습니다 — 제목: {page.title()}")
        try:
            page.screenshot(path="wake_ok.png")
        except Exception:
            pass
        browser.close()


if __name__ == "__main__":
    main()
