# app.py
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
from streamlit.components.v1 import html as st_html
from PIL import Image

# Azure
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

# AWS
import boto3


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Notes/Quiz → OCR → GPT → AMP",
    page_icon="🧠",
    layout="centered",
)
st.title("🧠 Notes/Quiz OCR → GPT Structuring → AMP Web Story")
st.caption("Upload notes/quiz image(s) (or JSON) + an AMP HTML template → choose count → build final HTML (dynamic repeater supported).")


# =========================
# READ SECRETS / CONFIG
# =========================
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        return default

# Azure
AZURE_DI_ENDPOINT       = get_secret("AZURE_DI_ENDPOINT")
AZURE_API_KEY           = get_secret("AZURE_API_KEY")
AZURE_OPENAI_ENDPOINT   = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY    = get_secret("AZURE_OPENAI_API_KEY", AZURE_API_KEY)
AZURE_OPENAI_API_VERSION= get_secret("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
GPT_DEPLOYMENT          = get_secret("GPT_DEPLOYMENT", "gpt-4o-mini")

# AWS / S3
AWS_ACCESS_KEY_ID       = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY   = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_REGION              = get_secret("AWS_REGION", "ap-south-1")
AWS_BUCKET              = get_secret("AWS_BUCKET", "suvichaarapp")
HTML_S3_PREFIX          = get_secret("HTML_S3_PREFIX", "webstory-html")
CDN_HTML_BASE           = get_secret("CDN_HTML_BASE", "https://stories.suvichaar.org/")

# Validate essentials
_missing = []
for k, v in [
    ("AZURE_DI_ENDPOINT", AZURE_DI_ENDPOINT),
    ("AZURE_API_KEY", AZURE_API_KEY),
    ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
    ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
    ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
    ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
]:
    if not v:
        _missing.append(k)
if _missing:
    st.error("Missing secrets: " + ", ".join(_missing))
    st.stop()


# =========================
# CLIENTS
# =========================
di_client = DocumentIntelligenceClient(
    endpoint=AZURE_DI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY),
)
gpt_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT_NOTES_TO_QA = """
You are given raw study notes text (could be Hindi or English).
Generate EXACTLY {N} high-quality multiple-choice questions (MCQs) strictly grounded in these notes.

For each question:
- Provide four options labeled A–D.
- Ensure exactly one correct option.
- Add a 1–2 sentence explanation that justifies the correct answer using the notes.

If the user provides a short "hint" (style/tone/focus), follow it sensibly without inventing facts.

Respond ONLY with valid JSON:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "..."
    }
  ]
}

Language: Use the same language as the notes (auto-detect). Keep questions concise and unambiguous.
""".strip()

SYSTEM_PROMPT_OCR_TO_QA = """
You receive OCR text that already contains multiple-choice questions in Hindi or English.
Each question has options (A)-(D), a single correct answer, and ideally an explanation.

Return ONLY valid JSON:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "..."
    }
  ]
}

- If explanations are missing, write a concise 1–2 sentence explanation grounded in the text.
- Preserve original language.
""".strip()

SYSTEM_PROMPT_NOTES_TO_SLIDES = """
You will split study notes into exactly {N} concise slide paragraphs.
Each paragraph should be 1–3 sentences, readable on a phone screen.
Language: keep the same as input.

Return ONLY valid JSON:
{"slides": ["p1", "p2", "..."]}
""".strip()


# =========================
# HELPERS
# =========================
def clean_model_json(txt: str) -> str:
    """Remove code fences if model returns ```json ... ``` or ``` ... ```."""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", txt, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return txt.strip()

def ocr_extract(image_bytes: bytes) -> str:
    """OCR via Azure Document Intelligence prebuilt-read for one image."""
    poller = di_client.begin_analyze_document(
        model_id="prebuilt-read",
        body=image_bytes
    )
    result = poller.result()
    if getattr(result, "paragraphs", None):
        return "\n".join([p.content for p in result.paragraphs]).strip()
    if getattr(result, "content", None):
        return result.content.strip()
    lines = []
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            if getattr(line, "content", None):
                lines.append(line.content)
    return "\n".join(lines).strip()

def ocr_extract_many(images_bytes_list: List[bytes]) -> str:
    chunks = []
    for idx, b in enumerate(images_bytes_list, start=1):
        text = ocr_extract(b)
        if text:
            chunks.append(f"[[PAGE {idx}]]\n{text}")
    return "\n\n".join(chunks).strip()

def gpt_notes_to_questions(notes_text: str, n: int, hint: str = "") -> dict:
    sys_prompt = SYSTEM_PROMPT_NOTES_TO_QA.replace("{N}", str(n))
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": notes_text}]
    if hint and hint.strip():
        msgs.append({"role": "user", "content": f"HINT: {hint.strip()}"} )

    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=0,
        messages=msgs,
    )
    content = clean_model_json(resp.choices[0].message.content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def gpt_ocr_text_to_questions(raw_text: str) -> dict:
    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_OCR_TO_QA},
            {"role": "user", "content": raw_text}
        ],
    )
    content = clean_model_json(resp.choices[0].message.content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def gpt_notes_to_slides(notes_text: str, n: int, hint: str = "") -> List[str]:
    sys_prompt = SYSTEM_PROMPT_NOTES_TO_SLIDES.replace("{N}", str(n))
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": notes_text}]
    if hint and hint.strip():
        msgs.append({"role": "user", "content": f"HINT: {hint.strip()}"} )

    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=0,
        messages=msgs,
    )
    content = clean_model_json(resp.choices[0].message.content)
    j = json.loads(content)
    slides_text = j.get("slides", [])
    if len(slides_text) < n:
        slides_text += [""] * (n - len(slides_text))
    else:
        slides_text = slides_text[:n]
    return slides_text

def upload_html_to_s3(html_text: str, filename: str) -> Tuple[str, str]:
    """Upload HTML to S3 (no ACL) and return (s3_key, cdn_url)."""
    if not filename.lower().endswith(".html"):
        filename = f"{filename}.html"
    s3_key = f"{HTML_S3_PREFIX.strip('/')}/{filename}" if HTML_S3_PREFIX else filename
    s3 = get_s3_client()
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=s3_key,
        Body=html_text.encode("utf-8"),
        ContentType="text/html; charset=utf-8",
        CacheControl="public, max-age=300",
        ContentDisposition=f'inline; filename="{filename}"',
    )
    cdn_url = f"{CDN_HTML_BASE.rstrip('/')}/{s3_key}"
    return s3_key, cdn_url

# ---------- Fixed-slot helpers (fallback) ----------
def build_attr_value(key: str, val: str) -> str:
    """sNoptionKattr + 'correct' → 'option-K-correct'."""
    if not key.endswith("attr") or not val:
        return ""
    m = re.match(r"s(\d+)option(\d)attr$", key)
    if m and val.strip().lower() == "correct":
        return f"option-{m.group(2)}-correct"
    return val

def fill_template(template: str, data: dict) -> str:
    """Replace {{key}} and {{key|safe}}, handling *attr keys."""
    rendered = {}
    for k, v in data.items():
        if k.endswith("attr"):
            rendered[k] = build_attr_value(k, str(v))
        else:
            rendered[k] = "" if v is None else str(v)
    html = template
    for k, v in rendered.items():
        html = html.replace(f"{{{{{k}}}}}", v)
        html = html.replace(f"{{{{{k}|safe}}}}", v)
    return html

def count_question_slots(template_html: str) -> int:
    found = set(int(m.group(1)) for m in re.finditer(r"\bs(\d+)question1\b", template_html))
    found = {n for n in found if n >= 2}
    return len(found)

# ---------- Repeater helpers ----------
RE_START = r"<!--SLIDE_TEMPLATE_START-->"
RE_END   = r"<!--SLIDE_TEMPLATE_END-->"

def extract_repeat_block(html: str) -> Tuple[str, str, str]:
    """Returns (prefix, block, suffix)."""
    m = re.search(f"{RE_START}(.*?){RE_END}", html, flags=re.DOTALL)
    if not m:
        return html, "", ""
    block = m.group(1)
    prefix = html[:m.start()]
    suffix = html[m.end():]
    return prefix, block, suffix

def render_block(block_html: str, slide: Dict[str, str], index: int) -> str:
    """Replace placeholders in repeater block."""
    rendered = block_html
    repl = {
        "index": str(index),
        "img": slide.get("img", ""),
        "audio": slide.get("audio", ""),
        "paragraph": slide.get("paragraph", ""),
        "page_title": slide.get("page_title", "Notes Chapter"),
    }
    for k, v in repl.items():
        rendered = rendered.replace(f"{{{{{k}}}}}", v)
    return rendered

def render_repeater(template_html: str, slides: List[Dict[str, str]]) -> str:
    """Expands the repeater block for len(slides) slides."""
    prefix, block, suffix = extract_repeat_block(template_html)
    if not block:
        return template_html  # no repeater present
    parts = [prefix]
    for i, s in enumerate(slides, start=1):
        parts.append(render_block(block, s, i))
    parts.append(suffix)
    return "".join(parts)


# =========================
# UI — MODES
# =========================
mode = st.radio(
    "Choose story type:",
    ["Notes Story (paragraph slides)", "Quiz Story (MCQs)"],
)

col_a, col_b = st.columns([1, 2])
with col_a:
    desired_count = st.number_input(
        "How many slides/questions?",
        min_value=1, max_value=12, value=5, step=1
    )
with col_b:
    user_hint = st.text_input(
        "Optional prompt hint (style/tone/focus)",
        placeholder="Keep it concise, emphasize definitions, Hindi/English tone..."
    )

up_tpl = st.file_uploader("📎 Upload AMP HTML template (.html)", type=["html", "htm"], key="tpl")
show_debug = st.toggle("Show OCR / JSON previews", value=False)


# =========================
# MODE 1: NOTES STORY
# =========================
notes_text: Optional[str] = None
questions_data: Optional[dict] = None
slides_list: Optional[List[Dict[str, str]]] = None

if mode == "Notes Story (paragraph slides)":
    up_imgs = st.file_uploader(
        "📎 Upload notes image(s) (JPG/PNG/WebP/TIFF) — multiple allowed",
        type=["jpg", "jpeg", "png", "webp", "tiff"],
        accept_multiple_files=True,
        key="notes_imgs",
    )
    if up_imgs:
        if show_debug:
            for i, f in enumerate(up_imgs, start=1):
                try:
                    st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                             caption=f"Notes page {i}", use_container_width=True)
                except Exception:
                    pass
        try:
            with st.spinner("🔍 OCR (Azure Document Intelligence) on all pages…"):
                all_bytes = [f.getvalue() for f in up_imgs]
                notes_text = ocr_extract_many(all_bytes)
            if not notes_text.strip():
                st.error("OCR returned empty text. Try clearer images.")
                st.stop()
            if show_debug:
                with st.expander("📄 OCR Notes Text"):
                    st.text(notes_text[:8000] if len(notes_text) > 8000 else notes_text)

            with st.spinner("📝 Splitting into slide paragraphs…"):
                paragraphs = gpt_notes_to_slides(notes_text, int(desired_count), user_hint or "")
            # Build slides for repeater (leave media empty unless you attach URLs)
            slides_list = [{
                "img": "",                # set a default background URL if desired
                "audio": "",              # attach your TTS URL per slide if available
                "paragraph": p,
                "page_title": "📘 Notes Chapter"
            } for p in paragraphs]
        except Exception as e:
            st.error(f"Failed to process notes → slides: {e}")
            st.stop()

# =========================
# MODE 2: QUIZ STORY
# =========================
else:  # Quiz Story (MCQs)
    submode = st.selectbox(
        "Input source for MCQs",
        ["Notes image(s) → generate MCQs", "Quiz image → parse MCQs", "Upload structured questions JSON"]
    )

    if submode == "Notes image(s) → generate MCQs":
        up_imgs = st.file_uploader(
            "📎 Upload notes image(s) (JPG/PNG/WebP/TIFF)",
            type=["jpg", "jpeg", "png", "webp", "tiff"],
            accept_multiple_files=True,
            key="quiz_from_notes_imgs",
        )
        if up_imgs:
            if show_debug:
                for i, f in enumerate(up_imgs, start=1):
                    try:
                        st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                                 caption=f"Notes page {i}", use_container_width=True)
                    except Exception:
                        pass
            try:
                with st.spinner("🔍 OCR…"):
                    all_bytes = [f.getvalue() for f in up_imgs]
                    notes_text = ocr_extract_many(all_bytes)
                if not notes_text.strip():
                    st.error("OCR returned empty text.")
                    st.stop()
                if show_debug:
                    with st.expander("📄 OCR Notes Text"):
                        st.text(notes_text[:8000] if len(notes_text) > 8000 else notes_text)
                with st.spinner("🧠 Generating MCQs…"):
                    questions_data = gpt_notes_to_questions(notes_text, int(desired_count), user_hint or "")
            except Exception as e:
                st.error(f"Failed to process notes → MCQs: {e}")
                st.stop()

    elif submode == "Quiz image → parse MCQs":
        up_img = st.file_uploader(
            "📎 Upload a quiz image (JPG/PNG)",
            type=["jpg", "jpeg", "png"], key="quiz_img"
        )
        if up_img:
            try:
                if show_debug:
                    st.image(Image.open(io.BytesIO(up_img.getvalue())).convert("RGB"),
                             caption="Uploaded quiz image", use_container_width=True)
                with st.spinner("🔍 OCR…"):
                    raw_text = ocr_extract(up_img.getvalue())
                if not raw_text.strip():
                    st.error("OCR returned empty text.")
                    st.stop()
                if show_debug:
                    with st.expander("📄 OCR Text"):
                        st.text(raw_text[:8000] if len(raw_text) > 8000 else raw_text)
                with st.spinner("🤖 Parsing OCR into MCQs…"):
                    questions_data = gpt_ocr_text_to_questions(raw_text)
                # Trim/pad to desired_count
                qs = questions_data.get("questions", [])
                if len(qs) > desired_count:
                    questions_data["questions"] = qs[:desired_count]
                elif len(qs) < desired_count:
                    pad = desired_count - len(qs)
                    for _ in range(pad):
                        questions_data.setdefault("questions", []).append({
                            "question": "",
                            "options": {"A":"", "B":"", "C":"", "D":""},
                            "correct_option": "A",
                            "explanation": ""
                        })
                if show_debug:
                    with st.expander("🧱 Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:8000], language="json")
            except Exception as e:
                st.error(f"Failed to parse image → MCQs: {e}")
                st.stop()

    else:  # Upload structured questions JSON
        up_json = st.file_uploader("📎 Upload structured questions JSON", type=["json"], key="quiz_json")
        if up_json:
            try:
                questions_data = json.loads(up_json.getvalue().decode("utf-8"))
                qs = questions_data.get("questions", [])
                if len(qs) > desired_count:
                    questions_data["questions"] = qs[:desired_count]
                elif len(qs) < desired_count:
                    pad = desired_count - len(qs)
                    for _ in range(pad):
                        questions_data.setdefault("questions", []).append({
                            "question": "",
                            "options": {"A":"", "B":"", "C":"", "D":""},
                            "correct_option": "A",
                            "explanation": ""
                        })
                if show_debug:
                    with st.expander("🧱 Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:8000], language="json")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()


# =========================
# QUIZ MAPPING (LOCAL)
# =========================
def questions_to_placeholders_locally(questions_data: dict) -> dict:
    """
    Map {"questions":[...]} → flat placeholders for N questions using s2..s{N+1}.
    """
    qs = questions_data.get("questions", []) or []
    n = len(qs)

    # Basic titles
    first_q_text = (qs[0].get("question", "") if qs else "").strip()
    default_title = "Quick Quiz" if not first_q_text else (first_q_text[:40] + ("…" if len(first_q_text) > 40 else ""))
    lang_is_hindi = any(re.search(r"[\u0900-\u097F]", (q.get("question","") + q.get("explanation",""))) for q in qs)

    def L(en, hi):
        return hi if lang_is_hindi else en

    data = {
        "pagetitle": default_title,
        "storytitle": default_title,
        "typeofquiz": L("Educational", "शैक्षिक"),
        "potraitcoverurl": "",
        "s1title1": L("Let’s Begin", "चलिए शुरू करें"),
        "s1text1": L("Answer the questions that follow.", "आगे दिए गए प्रश्नों के उत्तर दें।"),
        "results_bg_image": "",
        "results_prompt_text": L("Share your score!", "अपना स्कोर साझा करें!"),
        "results1_text": L("Nice attempt!", "अच्छा प्रयास!"),
        "results2_text": L("Great job!", "बहुत बढ़िया!"),
        "results3_text": L("Outstanding!", "शानदार!"),
    }

    # Fill s2..s{n+1}
    for i, q in enumerate(qs, start=2):
        idx = i - 1
        data[f"s{i}questionHeading"] = L(f"Question {idx}", f"प्रश्न {idx}")
        data[f"s{i}question1"] = q.get("question","")

        a1, a2, a3, a4 = (q.get("options", {}).get(k, "") for k in ("A","B","C","D"))
        correct = (q.get("correct_option","A") or "A").strip().upper()
        expl = q.get("explanation","")

        data[f"s{i}option1"] = a1
        data[f"s{i}option2"] = a2
        data[f"s{i}option3"] = a3
        data[f"s{i}option4"] = a4
        data[f"s{i}option1attr"] = "correct" if correct == "A" else ""
        data[f"s{i}option2attr"] = "correct" if correct == "B" else ""
        data[f"s{i}option3attr"] = "correct" if correct == "C" else ""
        data[f"s{i}option4attr"] = "correct" if correct == "D" else ""
        data[f"s{i}attachment1"] = expl

        # Optional media placeholders (if your template uses them)
        data[f"s{i}image1"] = data.get(f"s{i}image1", "")
        data[f"s{i}audio1"] = data.get(f"s{i}audio1", "")

    return data


# =========================
# BUILD
# =========================
can_build = bool(up_tpl and (slides_list or questions_data))
build = st.button("🛠️ Build final HTML", disabled=not can_build)

if build and up_tpl:
    try:
        template_html = up_tpl.getvalue().decode("utf-8")

        # 1) If template has a repeater block, we’ll use it for Notes Story (paragraph slides)
        has_repeater = bool(re.search(f"{RE_START}.*?{RE_END}", template_html, flags=re.DOTALL))

        if mode == "Notes Story (paragraph slides)":
            if not slides_list:
                st.error("No slides prepared. Please upload images first.")
                st.stop()

            if has_repeater:
                with st.spinner("🧩 Expanding dynamic repeater…"):
                    expanded_html = render_repeater(template_html, slides_list)
                # Optionally fill a few global keys for cover pages
                top_level = {
                    "storytitle": "Notes Chapter Story",
                    "s0image1": "",
                    "s0audio1": "",
                    "s1image1": "",
                    "s1audio1": "",
                    "s1paragraph1": slides_list[0]["paragraph"] if slides_list else "",
                }
                final_html = fill_template(expanded_html, top_level)
            else:
                # Fallback: try to stuff paragraphs into s2..s{N+1} if present
                available_slots = count_question_slots(template_html)
                if available_slots and available_slots < len(slides_list):
                    st.warning(f"Template has {available_slots} fixed slots but you requested {len(slides_list)}. Only first {available_slots} will render.")

                # Build a placeholder dict with s{i}paragraph1 (common in your template)
                data = {
                    "storytitle": "Notes Chapter Story",
                    "s0image1": "",
                    "s0audio1": "",
                    "s1image1": "",
                    "s1audio1": "",
                }
                for i, slide in enumerate(slides_list, start=2):
                    data[f"s{i}paragraph1"] = slide["paragraph"]
                    data[f"s{i}image1"] = slide["img"]
                    data[f"s{i}audio1"] = slide["audio"]
                final_html = fill_template(template_html, data)

        else:
            # QUIZ STORY
            if not questions_data:
                st.error("No questions JSON available.")
                st.stop()

            # Always map to placeholders locally
            with st.spinner("🧩 Generating placeholders for quiz…"):
                placeholders = questions_to_placeholders_locally(questions_data)
                if show_debug:
                    with st.expander("🧩 Placeholder JSON"):
                        st.code(json.dumps(placeholders, ensure_ascii=False, indent=2)[:12000], language="json")

            # If template has repeater, we can also render MCQs as slides (question+explanation) using repeater
            if has_repeater:
                qs = questions_data.get("questions", [])
                slides = []
                for q in qs:
                    # Combine Q + options into a readable paragraph
                    para = q.get("question","").strip()
                    opt = q.get("options", {})
                    a, b, c, d = (opt.get("A",""), opt.get("B",""), opt.get("C",""), opt.get("D",""))
                    para += f"\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
                    slides.append({
                        "img": "",
                        "audio": "",
                        "paragraph": para,
                        "page_title": "📘 Quick Quiz"
                    })
                expanded_html = render_repeater(template_html, slides)
                # still allow top-level fill (cover pages, etc.)
                final_html = fill_template(expanded_html, {
                    "storytitle": placeholders.get("storytitle","Quick Quiz"),
                    "s0image1": "",
                    "s0audio1": "",
                    "s1image1": "",
                    "s1audio1": "",
                    "s1paragraph1": "Answer the following questions."
                })
            else:
                # Fixed-slot fallback: warn if fewer slots
                available_slots = count_question_slots(template_html)
                requested = len(questions_data.get("questions", []))
                if available_slots and available_slots < requested:
                    st.warning(f"Template has {available_slots} question slots but you requested {requested}. Only first {available_slots} will render.")
                final_html = fill_template(template_html, placeholders)

        # 2) Save timestamped file locally
        ts_name = f"final_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        Path(ts_name).write_text(final_html, encoding="utf-8")

        # 3) Upload to S3
        with st.spinner("☁️ Uploading to S3…"):
            s3_key, cdn_url = upload_html_to_s3(final_html, ts_name)

        st.success(f"✅ Final HTML generated and uploaded to S3: s3://{AWS_BUCKET}/{s3_key}")
        st.markdown(f"**CDN URL:** {cdn_url}")

        # 4) Show preview + download
        with st.expander("🔍 HTML Preview (source)"):
            st.code(final_html[:120000], language="html")

        st.download_button(
            "⬇️ Download final HTML",
            data=final_html.encode("utf-8"),
            file_name=ts_name,
            mime="text/html"
        )

        st.markdown("### 👀 Live HTML Preview")
        h = st.slider("Preview height (px)", min_value=400, max_value=1600, value=900, step=50)
        full_width = st.checkbox("Force full viewport width (100vw)", value=True,
                                 help="Overrides container width so the preview stretches edge-to-edge.")
        style = f"width: {'100vw' if full_width else '100%'}; height: {h}px; border: 0; margin: 0; padding: 0;"
        st_html(final_html, height=h, scrolling=True) if not full_width else st_html(
            f'<div style="position:relative;left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;{style}">{final_html}</div>',
            height=h,
            scrolling=True
        )
        st.info("Note: AMP may not fully render inside Streamlit due to sandbox/CSP. For a faithful view, open the HTML in a browser or use the CDN URL.")

    except Exception as e:
        st.error(f"Build failed: {e}")

elif not can_build:
    st.info("Upload an AMP HTML template and provide inputs (images or JSON) to enable the Build button.")
