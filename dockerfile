FROM python:3.6.12-slim as build

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

FROM build as social_distancing
COPY social.distancing.py /root/
WORKDIR /root

CMD ["python3", "social_distancing.py"]