FROM hieupth/blueforest:env

COPY resources resources
COPY warmup.py warmup.py

SHELL ["/bin/bash", "-c"]
RUN apt-get update -y && \
    apt-get install -y git && \
    ls -la resources
RUN source /venv/bin/activate && \
    pip install git+https://github.com/hieupth/blueforest && \
    python warmup.py