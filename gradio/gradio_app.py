import gradio as gr

from swsearch.logutil import get_logger
from swsearch.search.engine import SearchEngine

logger = get_logger(__name__)

try:
    engine = SearchEngine()
except (FileNotFoundError, RuntimeError) as e:
    logger.warning("Could not load SearchEngine backend: %s", e)
    engine = None


def search_wiki(query, k=5):
    if engine is None:
        return "⚠️ Backend data not available (no FAISS index / metadata found). You're in frontend-only mode."
    if not query or not query.strip():
        return ""

    results = engine.search(query, k=int(k))
    if not results:
        return "No results found."

    return "\n".join(f"\U0001f539 [{r['title']}]({r['url']})" for r in results)


css = """
#custom-query-box, #custom-output-box {
    max-width: 600px;
    width: 70%;
    margin-left: auto;
    margin-right: auto;
}
"""

with gr.Blocks(css=css) as iface:
    with gr.Column():
        gr.Markdown("## \U0001f4da Wikipedia Semantic Search")

        query_input = gr.Textbox(
            lines=2,
            placeholder="Enter your semantic query...",
            label="Search Query",
            elem_id="custom-query-box",
        )

        k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=5,
            label="Top K Results",
        )

        output_box = gr.Markdown(elem_id="custom-output-box")

        query_input.change(fn=search_wiki, inputs=[query_input, k_slider], outputs=output_box)
        k_slider.change(fn=search_wiki, inputs=[query_input, k_slider], outputs=output_box)

if __name__ == "__main__":
    iface.launch()
