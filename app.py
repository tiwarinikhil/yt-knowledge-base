import gradio as gr

from pipeline import get_index_id, ingest, query


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
) -> tuple[list[dict], str]:
    if not index_id:
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "Please load a video first."},
        ]
        return history, ""
    response = query(user_message, index_id)
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]
    return history, ""


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
        outputs=[chatbot, message_input],
    )

demo.launch()
