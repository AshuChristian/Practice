FROM python:3.8

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "automl_script.py"]

RUN python train.py
EXPOSE 5000

RUN pip install scikit-learn pandas numpy joblib

