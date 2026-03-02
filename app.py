from urllib.parse import parse_qs, urlparse

import gradio as gr

from pipeline import get_index_id, ingest, query


def _extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)
    params = parse_qs(parsed.query)
    if "v" in params:
        return params["v"][0]
    return parsed.path.rstrip("/").split("/")[-1]


def _build_response(answer: str, timestamps: list[dict]) -> str:
    text = answer
    if timestamps:
        text += "\n\n---\nTimestamps in video:"
        for ts in timestamps:
            text += f"\n[{ts['timestamp_label']}] {ts['title']} — [Watch here]({ts['youtube_url']})"
    return text


def _build_embeds(timestamps: list[dict]) -> str:
    if not timestamps:
        return ""
    items = ""
    for ts in timestamps:
        vid = _extract_video_id(ts["youtube_url"])
        start = int(ts["start_time"])
        items += (
            f'<iframe width="100%" height="215" '
            f'src="https://www.youtube.com/embed/{vid}?start={start}" '
            f'frameborder="0" allowfullscreen></iframe>'
            f'<p style="font-size:13px; color:gray;">{ts["title"]} — {ts["timestamp_label"]}</p>'
        )
    return (
        f'<div style="display:flex; flex-direction:column; gap:12px;">'
        f'<p><b>Jump to relevant moments:</b></p>{items}</div>'
    )


def load_video(url: str, index_id: str) -> tuple[str, str]:
    if not url.strip():
        return "Please enter a YouTube URL.", index_id
    status = "Processing... this may take a few minutes."
    yield status, index_id
    result = ingest(url)
    new_index_id = get_index_id(url)
    yield result, new_index_id


def send_message(
    user_message: str,
    history: list[dict],
    index_id: str,
) -> tuple[list[dict], str, str]:
    if not index_id:
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "Please load a video first."},
        ]
        return history, "", ""

    result = query(user_message, index_id)
    answer = result["answer"]
    timestamps = result["timestamps"]

    response_text = _build_response(answer, timestamps)
    embed_html = _build_embeds(timestamps)

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response_text},
    ]
    return history, "", embed_html


with gr.Blocks() as demo:
    gr.Markdown("## YouTube Knowledge Base")

    current_index_id = gr.State("")

    url_input = gr.Textbox(
        placeholder="Paste a YouTube video or playlist URL",
        label="YouTube URL",
    )
    load_btn = gr.Button("Load Video")
    status_box = gr.Textbox(
        value="Paste a URL and click Load Video to begin.",
        label="Status",
        interactive=False,
    )

    chatbot = gr.Chatbot(label="Ask anything about the video", height=400)
    video_embeds = gr.HTML(value="")
    message_input = gr.Textbox(
        placeholder="Ask a question about the video...",
        label="Your Question",
    )
    send_btn = gr.Button("Send")

    load_btn.click(
        fn=load_video,
        inputs=[url_input, current_index_id],
        outputs=[status_box, current_index_id],
    )

    send_btn.click(
        fn=send_message,
        inputs=[message_input, chatbot, current_index_id],
        outputs=[chatbot, message_input, video_embeds],
    )

demo.launch()
