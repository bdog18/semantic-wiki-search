import os
import gradio as gr


# Baseline (off-the-shelf all-MiniLM-L6-v2) for now: on the clean corpus it
# still edges out the fine-tuned lr5e-6_steps8000 run on the metrics that
# decide what a user sees first -- reranked MRR 0.7947 vs 0.7444 and Top-1
# 0.7032 vs 0.6065 -- even though the fine-tune is slightly ahead on Top-5/
# Top-10 and MAP. These are also the CLI's default paths
# (settings.paths.faiss_index_path / faiss_meta_db_path), so the app and
# `swsearch search` now answer the same query the same way. Swap back to the
# transfer run's paths + its model/ directory to compare.
INDEX_PATH = "data/processed/models/baseline/runs/all-MiniLM-L6/paragraphs.index"
META_DB_PATH = "data/processed/models/baseline/runs/all-MiniLM-L6/paragraphs.index.meta.db"
MODEL_NAME = "all-MiniLM-L6-v2"

SNIPPET_MIN_CHARS = 50
SNIPPET_MAX_CHARS = 220

EXAMPLE_QUERIES = [
    "Who wrote The Odyssey?",
    "What causes the seasons to change?",
    "History of the United States",
    "How do neural networks learn?",
]

engine = None

def _truncate_snippet(text: str,limit: int = SNIPPET_MAX_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def search_wiki(query, k=5, rerank=True):
    if engine is None:
        return "⚠️ Backend data not available (no FAISS index / metadata found). You're in frontend-only mode."
    if not query or not query.strip():
        return ""

    results = engine.search(query, k=int(k), rerank_enabled=rerank)
    if not results:
        return "No results found."

    blocks = [
        f"**{i}. [{r['title']}]({r['url']})**\n\n> {_truncate_snippet(r['snippet'])}"
        for i, r in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


css = """
.gradio-container {
    max-width: 700px !important;
    margin: 0 auto !important;
}
#header {
    text-align: center;
    margin-bottom: 0;
}
#subheader {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin-bottom: 12px;
}
.search-box {
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 8px 12px;
    --block-padding: 6px 10px;
    --layout-gap: 8px;
}
.results-box {
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 12px 16px;
    min-height: 96px;
}
"""

with gr.Blocks(title="Wikipedia Semantic Search") as iface:
    gr.Markdown("# \U0001f4da Wikipedia Semantic Search", elem_id="header")
    gr.Markdown(
        "Ask a question and search the Wikipedia corpus by meaning, not just keywords.",
        elem_id="subheader",
    )

    with gr.Group(elem_classes=["search-box"]):
        query_input = gr.Textbox(
            placeholder="Enter your semantic query...",
            label="Search Query",
        )
        with gr.Row():
            k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=5,
                label="Top K Results",
                scale=3,
            )
            rerank_checkbox = gr.Checkbox(
                value=True, label="Enable reranking", scale=1
            )
    with gr.Row():
        search_button = gr.Button("Search", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLE_QUERIES, inputs=[query_input])

    with gr.Group(elem_classes=["results-box"]):
        gr.Markdown("### Results")
        output_box = gr.Markdown()

    search_inputs = [query_input, k_slider, rerank_checkbox]
    search_button.click(fn=search_wiki, inputs=search_inputs, outputs=output_box)
    query_input.submit(fn=search_wiki, inputs=search_inputs, outputs=output_box)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),
        css=css,
    )   
