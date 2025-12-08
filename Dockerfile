FROM nvcr.io/nvidia/isaac-lab:2.2.0

COPY requirements.txt /tmp/requirements.txt
RUN /workspace/isaaclab/_isaac_sim/python.sh -m \
    pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt
RUN echo "export ISAACSIM_PATH=/workspace/isaaclab/_isaac_sim" >> ~/.bashrc
WORKDIR /app
