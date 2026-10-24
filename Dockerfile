# Packages the eval runner, golden dataset, prompts and reporting layer.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVAL_PROVIDER=mock

WORKDIR /app

# Install dependencies first so Docker can cache this layer between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY regdetect/ regdetect/
COPY prompts/ prompts/
COPY data/ data/
COPY baselines/ baselines/
COPY config.yaml .

# Run as a non-root user.
RUN useradd --create-home runner && mkdir -p results reports && chown -R runner /app
USER runner

# Mount ./reports and ./results to keep output on your machine.
VOLUME ["/app/reports", "/app/results"]

# Configure with env vars: OPENAI_API_KEY, SLACK_WEBHOOK_URL, EVAL_PROVIDER,
# ACTIVE_PROMPT, WARN_DELTA_PCT, CRITICAL_DELTA_PCT, DRIFT_WINDOW, ...
ENTRYPOINT ["python", "-m", "regdetect"]
CMD ["run"]
