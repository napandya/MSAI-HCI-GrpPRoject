# Accessibility

**Target:** WCAG 2.2 Level AA.

In the United States, the Web Content Accessibility Guidelines (WCAG) are the standard
commonly used to judge whether a website meets the Americans with Disabilities Act (ADA).
The Department of Justice's 2024 rule for state and local government websites adopts
WCAG 2.1 AA. WCAG 2.2 AA includes everything in 2.1 AA except one retired criterion
(4.1.1 Parsing), plus newer criteria such as minimum target size. This document records
how the chatbot was tested and fixed; it is a course-project record, not a legal
conformance claim.

## Audit findings and fixes (September 27, 2026)

The audit used axe-core (an automated WCAG rule engine) in Chromium, plus scripted
checks for keyboard use, screen-reader output, zoom, and reflow.

| # | Finding | WCAG criterion | Fix |
|---|---|---|---|
| 1 | Send button: white text on `#6366f1` was 4.46:1, below the 4.5:1 minimum. | 1.4.3 Contrast (Minimum) | Button color changed to `#4338ca` (about 7.9:1). |
| 2 | With the operating system in dark mode, the "How can we help?" heading turned near-white on a white panel (1.09:1). | 1.4.3 Contrast (Minimum) | The page stays in its contrast-checked light theme regardless of the OS setting. |
| 3 | Gradio's footer links ("Use via API", "Built with Gradio", "Settings") were 1.75:1. | 1.4.3 Contrast (Minimum) | Footer links removed; they are not part of the support experience. |
| 4 | The message box showed no visible keyboard focus indicator. | 2.4.7 Focus Visible | A 3px dark-blue focus ring on every control, offset so it stays visible on colored buttons. |
| 5 | The chat toolbar's share button had no accessible name, so screen readers announced just "button". | 4.1.2 Name, Role, Value | Share button removed; the remaining toolbar buttons are labeled. |
| 6 | Toolbar and copy buttons were 18×18 px. | 2.5.8 Target Size (Minimum) | Minimum size set to 28×28 px. |
| 7 | The message box's accessible name was "Textbox"; the placeholder is not a label. | 3.3.2 Labels or Instructions; 4.1.2 | Labeled "Type a message to StreamHub support". The label includes the visible placeholder words, so voice-control users can say "click Type a message" (2.5.3 Label in Name). |
| 8 | Screen readers could not tell who sent each message; it was shown only by bubble position and color. | 1.3.1 Info and Relationships | Each message is announced as "You said: …" or "StreamHub assistant said: …". The labels are visually hidden because the layout already shows the speaker. |
| 9 | Streaming a reply updated the chat's live region 11 times, so screen readers could read replies in fragments. | 4.1.3 Status Messages (usability) | The chat log is no longer a live region. A separate status region announces "StreamHub assistant is typing." when a reply starts and the complete reply once when it finishes. |

Additional improvements (not failures, but recommended practice):

- **Skip link.** The first Tab stop is "Skip to message box", which moves focus straight
  to the message box (2.4.1 Bypass Blocks).
- **Reduced motion.** Animations and transitions turn off when the operating system's
  "reduce motion" setting is on (2.3.3 Animation from Interactions).
- **Conversation heading.** A heading, hidden visually, marks the chat area so
  screen-reader users can jump to it (2.4.6 Headings and Labels).
- **Readable placeholder.** Placeholder text is `#6b6b6b` (about 5.3:1), and the message
  box border is `#767676` (4.5:1) so the input is easy to find (1.4.11 Non-text Contrast).
- **Plain language.** The bot is instructed to answer in one or two short sentences and to
  give one troubleshooting step at a time, which reduces reading and memory load.

## What already met the guidelines

Page language is set (`en`), the page has a descriptive title, headings are in a logical
order, every control can be reached and used with the keyboard in a logical order, there
is no keyboard trap, there are no time limits, and the layout reflows to a 320px-wide
screen without horizontal scrolling.

## How it is tested

**Automated browser check** (about a minute, no model download):

```powershell
pip install -r requirements-dev.txt
python -m playwright install chromium
python -m scripts.accessibility_check
```

It runs the real page with a stand-in model and checks:

- axe-core WCAG 2.0/2.1/2.2 A and AA rules at 1280px and 320px, and with the OS in dark mode;
- the skip link is the first Tab stop and moves focus to the message box;
- every keyboard stop has an accessible name, a visible focus ring, and a size of at least 24×24 px;
- the message box label;
- screen-reader announcements while a reply streams: "typing" first, then the full reply exactly once;
- speaker labels in the accessibility tree;
- no horizontal scrolling at 320px width or at 200% zoom;
- no clipped text with increased text spacing (1.4.12).

Current result: **20/20 checks pass**.

**Unit tests** (`python -m unittest discover`) also guard the key settings (message-box
label, no share button, focus and reduced-motion CSS, the announcer region, the skip link)
without a browser.

## Manual checks still needed

Automated tools catch only part of WCAG. These need a person before the project is
submitted:

- [ ] **Screen reader on Windows:** with NVDA (free) or Narrator, ask a question, confirm you
  hear "StreamHub assistant is typing" and then the full reply once. Use heading navigation
  (the H key in NVDA) to reach the topics and the conversation, and confirm each message
  starts with who sent it.
- [ ] **Keyboard only:** unplug or ignore the mouse and complete each of the three scenarios
  using Tab, Shift+Tab, Enter, and Space.
- [ ] **Browser zoom at 200% and 400%:** everything readable, nothing overlapping or cut off.
- [ ] **Windows contrast themes** (Settings > Accessibility > Contrast themes): controls,
  focus rings, and text stay visible.
- [ ] **Voice Access** (Windows 11): "click Send", "click Troubleshoot playback", and
  "click Type a message" work.
- [ ] **Reply content:** replies stay short, plain, and free of jargon for each scenario.

## Known limitations

- **No dark mode.** The page always uses its light theme, because that palette is the one
  checked for contrast. Adding a checked dark palette would be a good future improvement.
- **Moving content during streaming.** The chat scrolls as a reply streams. This starts
  only after the user sends a message and stops when the reply finishes, but screen
  magnifier users may prefer to wait until the reply is complete.
- **Third-party component.** The chat window is a Gradio component. The fixes above
  override its styles and behavior; a future Gradio update could change its markup, so
  rerun `python -m scripts.accessibility_check` after upgrading Gradio.
