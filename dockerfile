FROM nvcr.io/nvidia/pytorch:22.12-py3
RUN pip uninstall -y torch && pip uninstall -y transformer-engine
RUN python -m pip install --upgrade pip
RUN pip install \
    ninja \
    psutil \
    ray \
    sentencepiece \
    numpy \
    torch>=2.0.0 \
    transformers@git+https://github.com/ertkonuk/transformers.git \
    xformers>=0.0.19 \
    fastapi \
    uvicorn \
    pydantic==1.10.11 \
    fschat \
    einops
    
RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.10.0
