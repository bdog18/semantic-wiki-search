import json
import logging
import os

import gradio as gr
import requests

# Deliberately imports nothing from swsearch: this app is a client of the
# search API, not part of the package. That keeps its container down to
# gradio + requests instead of dragging in torch, faiss and
# sentence-transformers for a process that only formats markdown.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Which index answers a query is the API's business now, set by its
# deployment rather than here. Default suits `uvicorn swsearch.api.app:app`
# on the same machine; set SWSEARCH_API_URL in the Railway service to point
# at the deployed backend.
API_URL = os.environ.get("SWSEARCH_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("SWSEARCH_API_KEY")

# Long enough to survive a Lambda cold start reading a ~1GB index through
# lazily-loaded image layers, which has measured 112-173s. The EventBridge
# warmup schedule should make that rare; when it isn't rare, showing the
# error beats a browser spinning for three minutes, so this is a compromise
# rather than a ceiling that fits the worst case.
API_TIMEOUT_SECONDS = int(os.environ.get("SWSEARCH_API_TIMEOUT", "45"))

# Requests are SigV4-signed rather than the endpoint being public: AWS blocks
# unauthenticated Lambda Function URLs on this account, and signing is the
# better answer anyway -- the endpoint stops being reachable by anyone
# without credentials, which makes the shared API key a second layer instead
# of the only one. Unset credentials means unsigned requests, so a local
# uvicorn backend still works with no AWS involvement.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

_auth = None
if _ACCESS_KEY and _SECRET_KEY:
    from requests_aws4auth import AWS4Auth

    _auth = AWS4Auth(_ACCESS_KEY, _SECRET_KEY, AWS_REGION, "lambda")
    logger.info("Signing requests to %s with SigV4 (region %s)", API_URL, AWS_REGION)
else:
    logger.info("No AWS credentials set; calling %s unsigned", API_URL)

SNIPPET_MIN_CHARS = 50
SNIPPET_MAX_CHARS = 220

EXAMPLE_QUERIES = [
    "Who wrote The Odyssey?",
    "What causes the seasons to change?",
    "History of the United States",
    "How do neural networks learn?",
]

def _truncate_snippet(text: str,limit: int = SNIPPET_MAX_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def search_wiki(query, k=5, rerank=True):
    if not query or not query.strip():
        return ""

    headers = {"content-type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    # Serialised here rather than passed as json=: SigV4 hashes the exact
    # request body, so the bytes that get signed have to be the bytes that
    # get sent.
    payload = json.dumps({"query": query.strip(), "k": int(k), "rerank": bool(rerank)})
    try:
        response = requests.post(
            f"{API_URL}/search",
            data=payload,
            headers=headers,
            auth=_auth,
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json()["results"]
    except (requests.RequestException, ValueError, KeyError) as e:
        # Covers the backend still loading its index (503 for the ~30s a
        # ~1GB index takes to read) as well as genuine outages. Both look
        # the same to a user and both come right on a retry, so they get one
        # message rather than an exception class.
        logger.warning("Search request to %s failed: %s", API_URL, e)
        return "⚠️ Search backend unavailable. Give it a moment and try again."

    if not results:
        return "No results found."

    # Already ordered by the API; the engine guarantees descending score.

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
