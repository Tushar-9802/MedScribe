"""
MedScribe — Clinical Documentation Workstation
================================================
Voice-to-SOAP + Clinical Intelligence powered by HAI-DEF models.

Tabs:
1. Voice to SOAP    — MedASR transcription + MedGemma SOAP generation
2. Text to SOAP     — Paste transcript, generate SOAP
3. Clinical Tools   — ICD-10, patient summary, completeness, DDx, med check
4. About            — Architecture, metrics, methodology

Launch: python app.py
"""
import os
import time
import gradio as gr
from src.pipeline import MedScribePipeline
from src.inference import format_soap_html

# ============================================================
# CONFIGURATION
# ============================================================
ADAPTER_PATH = "./models/checkpoints/medgemma_v2_soap/final_model"

EXAMPLE_TRANSCRIPTS = [
    [
        "45-year-old male presents with substernal chest pain for 2 hours, "
        "7/10 severity, radiating to left arm. Associated with diaphoresis "
        "and anxiety. No nausea or vomiting. BP 145/92, HR 98, RR 16. "
        "Anxious appearing. Regular rhythm, no murmurs."
    ],
    [
        "62-year-old female with type 2 diabetes returns for 3-month follow-up. "
        "Reports good compliance with metformin 1000mg twice daily. Occasional "
        "fasting glucose readings of 140-160. No hypoglycemic episodes. Denies "
        "polyuria, polydipsia, blurred vision. BP 132/78, HR 72, BMI 31.2. "
        "A1C today 7.4%, down from 8.1%. Creatinine 1.1, eGFR 68. Foot exam: "
        "intact sensation, no ulcers. Eyes: last retinal exam 6 months ago, "
        "no retinopathy."
    ],
    [
        "4-year-old male brought in by mother for 3 days of runny nose, cough, "
        "and low-grade fever. Max temp 100.4 at home. Eating and drinking well. "
        "No ear pulling. No history of asthma. Temp 99.8, HR 110, RR 22, "
        "SpO2 99%. Alert, playful. TMs clear bilaterally. Throat mildly "
        "erythematous, no exudate. Lungs clear. No lymphadenopathy."
    ],
    [
        "58-year-old male with hypertension and CKD stage 3b, GFR 38. On "
        "lisinopril 20mg daily, amlodipine 5mg daily. BP today 142/88. Labs: "
        "Cr 1.8, BUN 32, K 4.9, bicarb 20, phosphorus 4.8, PTH 98. Urine "
        "albumin-to-creatinine ratio 450. No edema. Denies fatigue, nausea, "
        "pruritus."
    ],
    [
        "34-year-old female presenting with worsening anxiety and depressed mood "
        "over 3 months since job loss. Reports difficulty sleeping, poor appetite, "
        "loss of interest in activities. Denies suicidal ideation, hallucinations, "
        "or substance use. Currently on sertraline 50mg daily started 6 weeks ago "
        "with minimal improvement. PHQ-9 score 14, GAD-7 score 12."
    ],
]


# ============================================================
# CUSTOM CSS — fixed-height status bar prevents layout reflow
# ============================================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif !important;
}
.app-header {
    text-align: center;
    padding: 20px 0 14px 0;
    border-bottom: 2px solid #1a5276;
    margin-bottom: 20px;
}
.app-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #1a5276;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 13px;
    color: #566573;
    margin: 3px 0 0 0;
}

/* FIXED-HEIGHT status bar — prevents layout stuttering */
.status-bar {
    min-height: 36px;
    max-height: 36px;
    overflow: hidden;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 13px;
    margin-bottom: 12px;
    box-sizing: border-box;
    transition: background 0.2s, border-color 0.2s, color 0.2s;
    background: #eaf2f8;
    border: 1px solid #aed6f1;
    color: #1a5276;
    line-height: 20px;
}
.status-bar.ready {
    background: #e8f8f5;
    border-color: #a3e4d7;
    color: #0e6655;
}
.status-bar.processing {
    background: #fef9e7;
    border-color: #f9e79f;
    color: #7d6608;
}
.status-bar.error {
    background: #fdedec;
    border-color: #f5b7b1;
    color: #922b21;
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #808b96;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 16px 0 6px 0;
    padding-bottom: 3px;
    border-bottom: 1px solid #eaecee;
}
.tab-nav button {
    font-weight: 600 !important;
    font-size: 13px !important;
}
.app-footer {
    text-align: center;
    padding: 14px 0;
    margin-top: 20px;
    border-top: 1px solid #eaecee;
    font-size: 11px;
    color: #aab7b8;
}
.primary-btn {
    background: #1a5276 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
.primary-btn:hover {
    background: #154360 !important;
}
.tool-output {
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    padding: 12px;
    background: #fafbfc;
    border: 1px solid #d5dbdb;
    border-radius: 4px;
    white-space: pre-wrap;
}

/* Prevent Gradio output containers from resizing during generation */
.prose, .markdown-text, .output-html {
    min-height: 0 !important;
}

/* ANTI-SHAKE: Disable Gradio's loading/generating animations that cause reflow */
.generating {
    min-height: inherit !important;
    border: none !important;
    animation: none !important;
}
.wrap.generating {
    border: none !important;
    animation: none !important;
}
/* Prevent the pulsing border effect on all components during generation */
.border-none {
    border: none !important;
}
div[data-testid] .wrap {
    transition: none !important;
}
/* Lock the main container width to prevent horizontal jitter */
.gradio-container > .main {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}
/* Prevent scroll jumps */
.contain {
    overflow: auto !important;
    scroll-behavior: auto !important;
}
"""


# ============================================================
# GLOBAL STATE
# ============================================================
pipeline = None
last_soap_note = ""


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = MedScribePipeline(adapter_path=ADAPTER_PATH)
    return pipeline


def _status(level, msg):
    return f'<div class="status-bar {level}">{msg}</div>'


def _tool_html(text):
    """Wrap clinical tool output in styled div."""
    if not text:
        return ""
    import html as html_mod
    safe = html_mod.escape(text)
    safe = safe.replace('\n', '<br>')
    return (
        f'<div class="tool-output" style="color:#1a1a1a !important; '
        f'background:#fafbfc !important;">{safe}</div>'
    )


# ============================================================
# HANDLER: Load Models
# ============================================================
def load_models():
    pipe = get_pipeline()
    try:
        if not pipe.asr_loaded:
            pipe.load_asr()
        if not pipe.soap_loaded:
            pipe.load_soap()
        return _status("ready", "✓ Models loaded. Ready for input.")
    except Exception as e:
        return _status("error", f"Load failed: {str(e)}")


# ============================================================
# HANDLER: Voice to SOAP
# ============================================================
def process_audio(audio_path):
    global last_soap_note
    pipe = get_pipeline()

    if not pipe.fully_loaded:
        return ("", "", _status("error", "Models not loaded. Click 'Load Models' first."), "", "", "")
    if audio_path is None:
        return ("", "", _status("error", "No audio provided."), "", "", "")

    try:
        asr_result = pipe.transcribe(audio_path)
        transcript = asr_result["transcript"]

        soap_result = pipe.generate_soap(transcript)
        last_soap_note = soap_result["soap_note"]
        total = round(asr_result["time_s"] + soap_result["time_s"], 1)

        return (
            transcript,
            format_soap_html(soap_result["soap_note"]),
            _status("ready", f"✓ Complete in {total}s"),
            f'{asr_result["time_s"]}s',
            f'{soap_result["time_s"]}s',
            f'{total}s',
        )
    except Exception as e:
        return ("", "", _status("error", f"Error: {str(e)}"), "", "", "")


# ============================================================
# HANDLER: Text to SOAP
# ============================================================
def process_text(transcript):
    global last_soap_note
    pipe = get_pipeline()

    if not pipe.soap_loaded:
        return ("", _status("error", "MedGemma not loaded. Click 'Load Models' first."), "", "")
    if not transcript or not transcript.strip():
        return ("", _status("error", "No transcript provided."), "", "")

    try:
        result = pipe.generate_soap(transcript.strip())
        last_soap_note = result["soap_note"]

        return (
            format_soap_html(result["soap_note"]),
            _status("ready", f'✓ Complete in {result["time_s"]}s  |  {result["word_count"]} words'),
            f'{result["time_s"]}s',
            f'{result["word_count"]} words',
        )
    except Exception as e:
        return ("", _status("error", f"Error: {str(e)}"), "", "")


# ============================================================
# HANDLER: Clinical Tools (generic wrapper)
# ============================================================
def _run_tool(tool_fn, tool_name, soap_note_input):
    """Generic wrapper for all clinical tools."""
    pipe = get_pipeline()
    note = soap_note_input.strip() if soap_note_input else last_soap_note
    if not note:
        return _status("error", "No SOAP note available. Generate one first."), ""
    if not pipe.soap_loaded:
        return _status("error", "Model not loaded."), ""
    try:
        result = tool_fn(note)
        return (
            _status("ready", f'✓ {tool_name} complete in {result["time_s"]}s'),
            _tool_html(result["text"]),
        )
    except Exception as e:
        return _status("error", str(e)), ""


def run_icd10(soap_input):
    return _run_tool(get_pipeline().suggest_icd10, "ICD-10 coding", soap_input)


def run_patient_summary(soap_input):
    return _run_tool(get_pipeline().patient_summary, "Patient summary", soap_input)


def run_completeness(soap_input):
    return _run_tool(get_pipeline().completeness_check, "Documentation review", soap_input)


def run_differential(soap_input):
    return _run_tool(get_pipeline().differential_diagnosis, "Differential diagnosis", soap_input)


def run_med_check(soap_input):
    return _run_tool(get_pipeline().medication_check, "Medication review", soap_input)


# ============================================================
# BUILD UI
# ============================================================
def build_app():
    with gr.Blocks(
        title="MedScribe",
    ) as app:

        # Header
        gr.HTML(
            '<div class="app-header">'
            "<h1>MedScribe</h1>"
            "<p>Clinical Documentation Workstation</p>"
            "<p style='font-size:11px; color:#aab7b8; margin-top:2px;'>"
            "MedASR + MedGemma  |  HAI-DEF Pipeline  |  Voice-to-SOAP + Clinical Intelligence</p>"
            "</div>"
        )

        # Status bar (shared, FIXED HEIGHT via CSS)
        status_html = gr.HTML(
            _status("", "Models not loaded. Click 'Load Models' to begin.")
        )

        # Load button
        load_btn = gr.Button("Load Models", variant="secondary", size="sm")
        load_btn.click(fn=load_models, outputs=[status_html], show_progress="hidden")

        with gr.Tabs():

            # ============================================
            # TAB 1: Voice to SOAP
            # ============================================
            with gr.TabItem("Voice to SOAP"):
                gr.HTML('<div class="section-label">Audio Input</div>')
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload clinical dictation",
                )
                voice_btn = gr.Button(
                    "Transcribe and Generate", variant="primary",
                    elem_classes=["primary-btn"],
                )

                gr.HTML('<div class="section-label">Transcript (MedASR)</div>')
                voice_transcript = gr.Textbox(
                    lines=4, interactive=True, show_label=False,
                    placeholder="Transcript appears here. You can edit before regenerating.",
                )

                voice_regen_btn = gr.Button(
                    "Regenerate SOAP from Edited Transcript",
                    variant="secondary", size="sm",
                )

                gr.HTML('<div class="section-label">SOAP Note (MedGemma, fine-tuned)</div>')
                voice_soap = gr.HTML()

                with gr.Row():
                    voice_asr_time = gr.Textbox(label="ASR Time", interactive=False, scale=1)
                    voice_soap_time = gr.Textbox(label="SOAP Time", interactive=False, scale=1)
                    voice_total_time = gr.Textbox(label="Total Time", interactive=False, scale=1)

                voice_btn.click(
                    fn=process_audio,
                    inputs=[audio_input],
                    outputs=[voice_transcript, voice_soap, status_html,
                             voice_asr_time, voice_soap_time, voice_total_time],
                    show_progress="hidden",
                )

                def regen_from_transcript(transcript):
                    result = process_text(transcript)
                    return (result[0], result[1], "", result[2], "")

                voice_regen_btn.click(
                    fn=regen_from_transcript,
                    inputs=[voice_transcript],
                    outputs=[voice_soap, status_html,
                             voice_asr_time, voice_soap_time, voice_total_time],
                    show_progress="hidden",
                )

            # ============================================
            # TAB 2: Text to SOAP
            # ============================================
            with gr.TabItem("Text to SOAP"):
                gr.HTML('<div class="section-label">Medical Transcript</div>')
                text_input = gr.Textbox(
                    lines=6, show_label=False,
                    placeholder="Paste or type a medical encounter transcript...",
                )
                text_btn = gr.Button(
                    "Generate SOAP Note", variant="primary",
                    elem_classes=["primary-btn"],
                )

                gr.HTML('<div class="section-label">SOAP Note (MedGemma, fine-tuned)</div>')
                text_soap = gr.HTML()

                with gr.Row():
                    text_gen_time = gr.Textbox(label="Generation Time", interactive=False, scale=1)
                    text_word_count = gr.Textbox(label="Output Size", interactive=False, scale=1)

                text_btn.click(
                    fn=process_text,
                    inputs=[text_input],
                    outputs=[text_soap, status_html, text_gen_time, text_word_count],
                    show_progress="hidden",
                )

                gr.HTML('<div class="section-label">Example Transcripts</div>')
                gr.Examples(examples=EXAMPLE_TRANSCRIPTS, inputs=[text_input], label="")

            # ============================================
            # TAB 3: Clinical Tools (5 tools)
            # ============================================
            with gr.TabItem("Clinical Tools"):
                gr.HTML(
                    '<div class="section-label">Clinical Intelligence (Base MedGemma)</div>'
                    '<p style="font-size:12px; color:#808b96; margin-bottom:12px;">'
                    'Analyze SOAP notes using MedGemma\'s base instruction-following capability. '
                    'Generate a SOAP note first, or paste one below.</p>'
                )

                tools_soap_input = gr.Textbox(
                    lines=8, show_label=False,
                    placeholder="Paste a SOAP note here, or generate one in Voice/Text tabs first. "
                    "The most recently generated note is used automatically if this is empty.",
                )

                # Row 1: Documentation tools
                with gr.Row():
                    icd_btn = gr.Button("ICD-10 Codes", variant="secondary", scale=1)
                    summary_btn = gr.Button("Patient Summary", variant="secondary", scale=1)
                    complete_btn = gr.Button("Completeness Check", variant="secondary", scale=1)

                # Row 2: Clinical reasoning tools
                with gr.Row():
                    ddx_btn = gr.Button("Differential Diagnosis", variant="secondary", scale=1)
                    med_btn = gr.Button("Medication Check", variant="secondary", scale=1)

                tools_status = gr.HTML(
                    _status("", "Select a tool above to analyze the SOAP note.")
                )

                gr.HTML('<div class="section-label">Result</div>')
                tools_output = gr.HTML()

                icd_btn.click(
                    fn=run_icd10, inputs=[tools_soap_input],
                    outputs=[tools_status, tools_output],
                    show_progress="hidden",
                )
                summary_btn.click(
                    fn=run_patient_summary, inputs=[tools_soap_input],
                    outputs=[tools_status, tools_output],
                    show_progress="hidden",
                )
                complete_btn.click(
                    fn=run_completeness, inputs=[tools_soap_input],
                    outputs=[tools_status, tools_output],
                    show_progress="hidden",
                )
                ddx_btn.click(
                    fn=run_differential, inputs=[tools_soap_input],
                    outputs=[tools_status, tools_output],
                    show_progress="hidden",
                )
                med_btn.click(
                    fn=run_med_check, inputs=[tools_soap_input],
                    outputs=[tools_status, tools_output],
                    show_progress="hidden",
                )

            # ============================================
            # TAB 4: About
            # ============================================
            with gr.TabItem("About"):
                gr.Markdown("""
### MedScribe — Clinical Documentation Workstation

**Problem**: AI documentation tools like Dragon Copilot generate verbose, textbook-style
notes. A nephrologist reports: *"More often than not I have to go and edit the notes and
shorten them, because they read like textbook lexicon rather than shorthand designed to
deliver efficient summaries with alacrity."* Editing AI notes takes longer than writing
from scratch.

**Solution**: MedScribe generates concise clinical shorthand using a fine-tuned MedGemma
model, then provides clinical intelligence tools for the complete documentation workflow.

---

### Architecture — Three HAI-DEF Models, One Pipeline

| Component | Model | Role |
|-----------|-------|------|
| Speech Recognition | MedASR (105M, Conformer) | Medical dictation → text, 5.2% WER |
| SOAP Generation | MedGemma 1.5 4B (LoRA fine-tuned) | Concise structured notes (~100 words) |
| Clinical Intelligence | MedGemma 1.5 4B (base, instruction-tuned) | ICD-10, DDx, med check, summaries, QA |

### Clinical Intelligence Tools

| Tool | Function |
|------|----------|
| ICD-10 Coding | Suggests billing codes from SOAP documentation |
| Patient Summary | Plain-language visit summary for patients |
| Completeness Check | Identifies documentation gaps |
| Differential Diagnosis | Ranked DDx with supporting/refuting evidence |
| Medication Check | Drug interactions, contraindications, dosage review |

### Training Methodology

- 712 curated samples generated by GPT-4o with anti-hallucination constraints
- LoRA fine-tuning (rank 16, alpha 32) on MedGemma 1.5 4B
- Anti-hallucination: "Not documented in source" for missing data
- Zero WNL shortcuts enforced in training data
- Validation loss 0.782 < Train loss 0.828 (no overfitting)

### Key Metrics

| Metric | Value |
|--------|-------|
| Quality score | 90/100 |
| Section completeness | 100% (S/O/A/P) |
| WNL present | 0% |
| Hallucinated findings | 0% |
| Avg word count | 104 words (vs ~200+ for verbose tools) |
| Avg inference time | ~25s |
| PLAN items | 2-4 per note |

### Deployment

- Runs on consumer GPU (RTX 5070 Ti, 16GB VRAM)
- 4-bit quantization via bitsandbytes (NF4)
- Fully offline capable — no cloud dependency
- MedASR + MedGemma coexist in 16GB VRAM

### Limitations

- English only (MedASR training constraint)
- Research prototype — not validated for clinical use
- Training data derived from synthetic encounters
- Inference speed hardware-dependent
- Clinical tools use base model instruction-following, not fine-tuned for coding accuracy

---

MedGemma Impact Challenge 2026 | Main Track + Novel Task Prize
""")

        # Footer
        gr.HTML(
            '<div class="app-footer">'
            "MedScribe | MedGemma Impact Challenge 2026 | "
            "HAI-DEF: MedASR + MedGemma 1.5 4B (fine-tuned + base) | "
            "Research prototype — not for clinical use"
            "</div>"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
            font=["IBM Plex Sans", "system-ui", "sans-serif"],
            font_mono=["IBM Plex Mono", "Consolas", "monospace"],
        ),
    )