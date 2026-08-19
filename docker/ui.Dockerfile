# Gradio front end: a client of the search API, not the search engine itself.
# It imports nothing from swsearch, so this image needs neither the package
# nor torch/faiss/sentence-transformers -- only gradio and requests.
#
# Build from the repo root so both this file and gradio/ are in context:
#   docker build -f docker/ui.Dockerfile -t swsearch-ui .
# On Railway: leave Root Directory empty and set Dockerfile Path to
# docker/ui.Dockerfile. The repo-root .dockerignore keeps data/ (399GB) out
# of the build context.
FROM python:3.12-slim

WORKDIR /app

# Deps in their own layer, so editing the UI rebuilds in seconds rather than
# reinstalling gradio every time.
COPY docker/ui-requirements.txt .
RUN pip install --no-cache-dir -r ui-requirements.txt

COPY gradio/gradio_app.py .

# Railway overrides PORT at runtime; this is the local-run default.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "gradio_app.py"]
