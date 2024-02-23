FROM python:3.11-slim

WORKDIR /App


COPY . /App


RUN pip install --no-cache-dir --upgrade pip flask torch==2.0.1 torchvision matplotlib numpy

# Puerto donde se genera la app
EXPOSE 5000

CMD ["python", "app.py"]


