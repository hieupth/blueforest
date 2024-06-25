from hieupth/blueforest

ENTRYPOINT source /venv/bin/activate && uvicorn blueforest.api:app --host 0.0.0.0 --port 8080