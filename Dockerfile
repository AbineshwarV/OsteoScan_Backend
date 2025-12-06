# Read: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.10

# Create non-root user (HF recommendation)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all app files
COPY --chown=user . /app

# Hugging Face Spaces expect port 7860
EXPOSE 7860

# Run Flask via gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app_api:app"]
