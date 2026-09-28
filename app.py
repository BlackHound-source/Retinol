"""
Gradio user interface components and event binding handlers.
"""

import socket
import gradio as gr
from inference import frontend_predict
from storage import execute_database_save

RETINOL_CSS = """
:root {
    --bg-base: #0f1115;
    --bg-surface: #16191f;
    --bg-surface-elevated: #1e222b;
    --border-subtle: #272c38;
    --text-primary: #e6edf3;
    --text-muted: #8b949e;
    --accent-red: #e50914;
}
body, .gradio-container {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}
.brand-header {
    background: linear-gradient(180deg, rgba(22,25,31,0.95) 0%, rgba(15,17,21,0.4) 100%);
    border-bottom: 1px solid var(--border-subtle);
    padding: 22px 0;
    margin-bottom: 24px;
}
.brand-title {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
}
.brand-title span { color: var(--accent-red); }
.brand-tagline { font-size: 13px; color: var(--text-muted); }
.retinol-card {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 18px !important;
}
.retinol-metric-card {
    background: var(--bg-surface-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 14px !important;
}
button.primary-btn {
    background-color: var(--accent-red) !important;
    color: #ffffff !important;
}
"""

def find_free_port(start_port=7860, max_port=7999):
    for port in range(start_port, max_port + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            sock.close()
            return port
        except OSError:
            sock.close()
    raise RuntimeError("No open TCP port found.")

def build_app(model, ood_mean, ood_std):
    """Instantiates and configures the Gradio Blocks UI."""
    port = find_free_port()
    with gr.Blocks(title="Retinol Diagnostic Engine", css=RETINOL_CSS) as demo:
        gr.HTML("""
        <div class="brand-header">
            <div class="brand-title">RETIN<span>O</span>L</div>
            <div class="brand-tagline">Clinical Grade Diabetic Retinopathy Engine & Perforation Guardian</div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group(elem_classes="retinol-card"):
                    input_fundus = gr.Image(type="numpy", label="Retinal Fundus Sensor Input", height=390)
                    btn_run_screening = gr.Button("Execute Clinical Evaluation", variant="primary", elem_classes="primary-btn")

            with gr.Column(scale=5):
                with gr.Group(elem_classes="retinol-card"):
                    gr.Markdown("### Primary Diagnostic Verdict")
                    with gr.Row():
                        out_grade = gr.Textbox(label="Identified Stage", interactive=False)
                        out_confidence = gr.Number(label="Confidence Rating (%)", interactive=False)
                    out_triage = gr.Textbox(label="Triage & Integrity Status", interactive=False)
                    out_advisory = gr.Textbox(label="Morphology & Perforation Advisory", lines=4, interactive=False)

        gr.Markdown("#### Computational Vision Pipelines")
        with gr.Row():
            with gr.Column(elem_classes="retinol-metric-card"):
                out_img_original = gr.Image(label="Field Sensor Capture", height=240)
            with gr.Column(elem_classes="retinol-metric-card"):
                out_img_normalized = gr.Image(label="CLAHE + ROI Illumination Normalization", height=240)
            with gr.Column(elem_classes="retinol-metric-card"):
                out_img_edges = gr.Image(label="Structural Edge & Microvascular Extraction", height=240)

        gr.Markdown("#### Severity Probability Vector")
        with gr.Group(elem_classes="retinol-card"):
            out_table_probs = gr.Dataframe(headers=["Grade", "Severity", "Probability (%)"], datatype=["str", "str", "number"], interactive=False)

        gr.Markdown("#### Patient Triage Record Sync")
        with gr.Group(elem_classes="retinol-card"):
            with gr.Row():
                in_pid = gr.Textbox(label="Patient ID", placeholder="PAT-88390")
                in_name = gr.Textbox(label="Full Name", placeholder="Optional")
                in_age = gr.Number(label="Age", precision=0)
            with gr.Row():
                in_gender = gr.Dropdown(choices=["Unspecified", "Male", "Female", "Other"], label="Gender", value="Unspecified")
                in_contact = gr.Textbox(label="Contact Info", placeholder="+91-...")
                btn_save_registry = gr.Button("Commit Record to Database", variant="secondary")

            out_db_status = gr.Textbox(label="Registry Synchronization Status", lines=1, interactive=False)

        session_image = gr.State(None)
        session_state = gr.State(None)

        btn_run_screening.click(
            fn=lambda img: frontend_predict(img, model, ood_mean, ood_std),
            inputs=[input_fundus],
            outputs=[
                out_img_original, out_img_normalized, out_img_edges,
                out_grade, out_confidence, out_triage, out_advisory,
                out_table_probs, session_image, session_state
            ],
            queue=False
        )

        btn_save_registry.click(
            fn=execute_database_save,
            inputs=[in_pid, in_name, in_age, in_gender, in_contact, session_image, session_state],
            outputs=[out_db_status],
            queue=False
        )

    return demo, port
