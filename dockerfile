# Build stage:
  FROM hieupth/mamba AS build

  ADD . .
  RUN mamba install -c conda-forge conda-pack && \
      mamba env create -f environment.yml
  # Make RUN commands use the new environment:
  SHELL ["conda", "run", "-n", "demoblueforest", "/bin/bash", "-c"]
  RUN mamba install -y pytorch torchvision torchaudio cpuonly -c pytorch
  # Pack environment.
  RUN conda-pack -n demoblueforest -o /tmp/env.tar && \
      mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
      rm /tmp/env.tar
  # Unpack environment.
  RUN /venv/bin/conda-unpack
  
  # Runtime stage:
  FROM ubuntu:22.04 AS runtime
  # Copy /venv from the previous stage:
  COPY --from=build /venv /venv
  COPY resources resources
  COPY app.py app.py
  COPY warmup.py warmup.py
  COPY faceparser.py faceparser.py
  COPY imgutils.py imgutils.py
  #
  RUN apt-get update -y && \
      apt-get install -y libgl1-mesa-glx libegl1-mesa libopengl0 && \
      # Clean cache.
      apt-get clean && rm -rf /var/lib/apt/lists/*
  #
  SHELL ["/bin/bash", "-c"]
  RUN source /venv/bin/activate && python warmup.py
  EXPOSE 7860
  ENTRYPOINT source /venv/bin/activate && python app.py