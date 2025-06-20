FROM python:3.10
ARG COORDINATES
ENV COORDINATES=$COORDINATES
WORKDIR /work
ADD . .
RUN pip install -r requirements.txt
CMD python testing_robots.py -i ${COORDINATES} --scenery scenario1/scenery.txt -d /output -e --train-rounds 1