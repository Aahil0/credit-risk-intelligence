FROM python:3.12-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m demo && chown -R demo:demo /workspace
USER demo
EXPOSE 8501 8000
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
