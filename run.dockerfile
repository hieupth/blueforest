from hieupth/demoblueforest:base 

ADD ./resources ./resources
ADD ./*.py ./
SHELL ["/bin/bash", "-c"]
RUN ls
RUN source /venv/bin/activate && python warmup.py
ENTRYPOINT source /venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000