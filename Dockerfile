ARG LTS=24.04

FROM ubuntu:${LTS}

USER root

FROM python:3.11

COPY . /tmp/app

WORKDIR /tmp/app

RUN apt-get update

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "run.py"]
