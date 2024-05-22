FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y build-essential

WORKDIR /usr/src/app

EXPOSE 8050

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src .

CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]