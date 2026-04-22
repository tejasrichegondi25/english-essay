FROM python:3.9-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

COPY . /app/

RUN python manage.py collectstatic --noinput

EXPOSE 8080

CMD python manage.py migrate --noinput && gunicorn --bind 0.0.0.0:${PORT:-8080} Automatic_English_Essay_Scoring_Algorithm_Based_On_Ml.wsgi:application
