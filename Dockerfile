FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
