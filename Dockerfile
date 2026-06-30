FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["sh", "-c", "python src/slope_displacement.py && python src/response_spectrum.py"]