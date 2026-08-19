FROM python:3.12-slim

WORKDIR /app

# Deps first, in their own layer: this only re-runs when requirements.txt
# changes, so editing the UI rebuilds in seconds instead of reinstalling gradio.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gradio_app.py .

# Railway overrides PORT at runtime; this is just the local-run default.
ENV PORT=7860

CMD ["python", "gradio_app.py"]