# Build stage:
  FROM hieupth/mamba AS build

  ADD . .
  RUN apt-get update && \
      apt-get install -y build-essential pkg-config libssl-dev && \
      mamba install -c conda-forge conda-pack && \
      mamba env create -f environment.yml
  # Make RUN commands use the new environment:
  SHELL ["conda", "run", "-n", "blueforest", "/bin/bash", "-c"]
  RUN mamba install -y pytorch torchvision torchaudio cpuonly -c pytorch
  # Pack environment.
  RUN conda-pack -n gfn -o /tmp/env.tar && \
      mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
      rm /tmp/env.tar
  # Unpack environment.
  RUN /venv/bin/conda-unpack
  
  # Runtime stage:
  FROM debian:buster AS runtime
  # Copy /venv from the previous stage:
  COPY --from=build /venv /venv
  ADD . .
  #
  RUN apt-get update -y && \
      apt-get upgrade -y
  #
  SHELL ["/bin/bash", "-c"]
  EXPOSE 7860
  ENTRYPOINT source /venv/bin/activate && python app.py