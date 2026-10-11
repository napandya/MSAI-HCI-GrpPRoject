"""Automated accessibility checks for the StreamHub chat page (target: WCAG 2.2 AA).

Setup (once):   pip install playwright axe-playwright-python
                python -m playwright install chromium
Run:            python -m scripts.accessibility_check

The page runs with a stand-in model that streams a fixed reply, so no model download is
needed and results are repeatable. The script runs axe-core (an automated WCAG rule
engine) at desktop and phone widths in light and dark mode, then checks things axe
cannot: keyboard order and focus visibility, speaker labels as screen readers get them,
what the live announcer says while a reply streams, reflow, text spacing, and 200% zoom.

Automated checks catch only part of WCAG. docs/ACCESSIBILITY.md lists the manual checks
(real screen readers, voice control, cognitive load) that still need a person.
"""
import sys
import time

import streamhub.service as support

PORT = 7890
URL = f"http://127.0.0.1:{PORT}"
REPLY = "StreamHub uses a monthly billing cycle. Plan changes apply at the next cycle."
AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]

results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


class StandInModel:
    """Streams a fixed reply word by word, like the real model but instant to load."""
    is_chat = True

    def off_topic_probability(self, message, history):
        return 0.0

    def stream(self, message, history, system_prompt=None):
        for word in REPLY.split(" "):
            time.sleep(0.08)
            yield word + " "


def start_app():
    support.CACHE = support.ResponseCache(None)
    support._get_generator = lambda: StandInModel()
    import app
    app.demo.launch(server_port=PORT, prevent_thread_lock=True, quiet=True, **app.LAUNCH_KWARGS)


def open_page(browser, width=1280, height=900, scheme="light"):
    page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=scheme)
    page.goto(URL, wait_until="load", timeout=60000)
    page.wait_for_selector("#message-box textarea", timeout=30000)
    page.wait_for_timeout(1000)
    return page


def send(page, text="What is my billing cycle?"):
    page.locator("#message-box textarea").fill(text)
    page.keyboard.press("Enter")
    page.wait_for_function(
        f"() => [...document.querySelectorAll('#support-chat .bot-row')].some(r => r.innerText.includes('next cycle'))",
        timeout=30000)
    page.wait_for_timeout(600)


def run_axe(page, label):
    from axe_playwright_python.sync_playwright import Axe
    response = Axe().run(page, options={"runOnly": {"type": "tag", "values": AXE_TAGS}}).response
    violations = response["violations"]
    detail = "; ".join(f"{v['id']} x{len(v['nodes'])}" for v in violations)
    check(f"axe-core finds no WCAG violations ({label})", not violations, detail)


def main():
    from playwright.sync_api import sync_playwright

    start_app()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        # 1. Automated rules at desktop and phone widths, light and dark OS settings.
        for width, scheme in ((1280, "light"), (320, "light"), (1280, "dark")):
            page = open_page(browser, width=width, scheme=scheme)
            send(page)
            run_axe(page, f"{width}px, {scheme} OS setting")
            if scheme == "dark":
                forced = page.evaluate("!document.body.classList.contains('dark')")
                check("Page stays in the contrast-checked light theme when the OS uses dark mode", forced)
            if width == 320:
                overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
                check("No horizontal scrolling at 320px wide (1.4.10 Reflow)", not overflow)
            page.close()

        page = open_page(browser)

        # 2. Page basics.
        check("Page language is set (3.1.1)", page.evaluate("document.documentElement.lang") != "")
        check("Page has a descriptive title (2.4.2)", page.title() == "StreamHub Support")
        headings = page.evaluate("[...document.querySelectorAll('h1,h2')].map(h => h.tagName + ':' + h.textContent.trim())")
        check("Headings: one h1, and h2s for the topics and the conversation (1.3.1, 2.4.6)",
              headings[:1] == ["H1:StreamHub Support"] and "H2:How can we help?" in headings
              and "H2:Conversation" in headings, ", ".join(headings))

        # 3. Keyboard: skip link first, every stop named and visibly focused.
        page.mouse.click(2, 2)
        page.keyboard.press("Tab")
        first = page.evaluate("document.activeElement.textContent.trim()")
        check("First Tab stop is the skip link (2.4.1 Bypass Blocks)", first == "Skip to message box", first)
        page.keyboard.press("Enter")
        page.wait_for_timeout(200)
        in_box = page.evaluate("!!document.activeElement.closest('#message-box')")
        check("Skip link moves focus into the message box", in_box)

        page.mouse.click(2, 2)
        stops = []
        for _ in range(12):
            page.keyboard.press("Tab")
            page.wait_for_timeout(350)  # Gradio animates button styles for 0.2s; read after it settles
            stops.append(page.evaluate("""() => {
                const e = document.activeElement;
                if (e === document.body) return null;  // focus wrapped past the last control
                const s = getComputedStyle(e), r = e.getBoundingClientRect();
                const name = (e.getAttribute('aria-label') || (e.labels && e.labels[0] && e.labels[0].textContent)
                              || e.textContent || '').trim();
                return {name: name.slice(0, 40), tag: e.tagName,
                        focusRing: s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) >= 2,
                        w: r.width, h: r.height}; }"""))
        stops = [s for s in stops if s]
        unnamed = [s for s in stops if not s["name"]]
        no_ring = [s["name"] or s["tag"] for s in stops if not s["focusRing"]]
        small = [s["name"] for s in stops if s["w"] < 24 or s["h"] < 24]
        check("Every keyboard stop has an accessible name (4.1.2)", not unnamed, str(unnamed))
        check("Every keyboard stop shows a visible focus ring (2.4.7)", not no_ring, ", ".join(no_ring))
        check("Every control is at least 24x24 px (2.5.8 Target Size)", not small, ", ".join(small))
        box_name = page.evaluate("document.querySelector('#message-box textarea').labels[0].textContent.trim()")
        check("Message box is labeled, and the label contains its visible text (3.3.2, 2.5.3, 4.1.2)",
              box_name == "Type a message to StreamHub support", box_name)

        # 4. Screen-reader behavior while a reply streams.
        page.evaluate("""() => {
            window.__announced = [];
            const region = document.getElementById('sr-announcer');
            new MutationObserver(() => { if (region.textContent) window.__announced.push(region.textContent); })
                .observe(region, {childList: true, characterData: true, subtree: true}); }""")
        send(page)
        announced = page.evaluate("window.__announced")
        check("Announces 'typing' when a reply starts (4.1.3 Status Messages)",
              bool(announced) and announced[0] == "StreamHub assistant is typing.", str(announced[:1]))
        check("Announces the finished reply once, in full, not word by word",
              len(announced) == 2 and announced[1] == "StreamHub assistant said: " + REPLY, str(announced))
        live = page.evaluate("document.querySelector('#support-chat [role=log]').getAttribute('aria-live')")
        check("Chat log itself is not a live region (prevents fragmented announcements)", live == "off", str(live))
        tree = page.locator("#support-chat").aria_snapshot()
        check("Screen readers hear who sent each message (1.3.1)",
              "You said:" in tree and "StreamHub assistant said:" in tree)

        # 5. Text spacing (1.4.12): user styles must not clip text.
        page.add_style_tag(content="* { line-height: 1.5 !important; letter-spacing: .12em !important;"
                                   " word-spacing: .16em !important; } p { margin-bottom: 2em !important; }")
        page.wait_for_timeout(300)
        clipped = page.evaluate("""[...document.querySelectorAll('button, h1, h2, p')]
            .filter(e => e.offsetParent && !e.closest('.sr-only') && getComputedStyle(e).overflow === 'hidden' && e.scrollHeight > e.clientHeight + 2)
            .map(e => e.textContent.trim().slice(0, 30))""")
        check("Text still fits with increased text spacing (1.4.12)", not clipped, str(clipped))
        page.close()

        # 6. 200% zoom (1.4.4): a 1280px screen at 200% behaves like a 640px viewport.
        page = open_page(browser, width=640, height=450)
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        check("Usable at 200% zoom without horizontal scrolling (1.4.4)", not overflow)
        page.close()
        browser.close()

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
