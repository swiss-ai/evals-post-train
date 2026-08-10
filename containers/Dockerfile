FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /workspace

RUN apt-get update -y
RUN apt-get install -y \
    git \
    curl \
    wget \
    sudo \
    libibverbs-dev

# CUDA & build environment
ENV TORCH_CUDA_ARCH_LIST='9.0+PTX'
ENV VLLM_FA_CMAKE_GPU_ARCHES='90-real'
ENV FORCE_CUDA=1
ENV MAX_JOBS=16
ENV NVCC_THREADS=8
ENV USE_CUDNN=1
ENV VLLM_FLASH_ATTN_VERSION=3
ENV SGL_KERNEL_ENABLE_BF16=1
ENV SGL_KERNEL_ENABLE_SM90A=1
ENV USE_CUDNN=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=16
ENV FLASHINFER_ENABLE_AOT=1
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# Freeze packages so pip won't touch it
COPY constraints-packages.txt /workspace/constraints-packages.txt
RUN REGEX=$(paste -sd'|' /workspace/constraints-packages.txt | sed 's/^/^(/;s/$/)==?/') && \
    pip list --format=freeze | grep -E "$REGEX" > /workspace/frozen.txt
RUN echo "=== frozen.txt ===" && cat /workspace/frozen.txt && echo "=== end frozen.txt ==="

COPY requirements-setup.txt /workspace/requirements-setup.txt
COPY requirements-packages.txt /workspace/requirements-packages.txt

RUN pip install --no-cache-dir --no-build-isolation -U \
    -r /workspace/requirements-setup.txt

RUN pip install --no-cache-dir --no-build-isolation -U \
    git+https://github.com/nickjbrowning/XIELU.git@main \
    --constraint /workspace/frozen.txt

RUN pip install -v --no-cache-dir --no-build-isolation -U \
    git+https://github.com/swiss-ai/transformers.git@aimv2-fix \
    --constraint /workspace/frozen.txt

RUN pip list --format=freeze | grep -E 'transformers' >> /workspace/frozen.txt && \
    echo "=== updated frozen.txt ===" && cat /workspace/frozen.txt && echo "=== end updated frozen.txt ==="
    
RUN pip install --no-cache-dir --no-build-isolation -U \
    -r /workspace/requirements-packages.txt \
    --constraint /workspace/frozen.txt

RUN pip install -v --no-cache-dir --no-build-isolation -U \
    git+https://github.com/Dao-AILab/flash-attention.git@v2.8.1#subdirectory=hopper \
    --constraint /workspace/frozen.txt

RUN pip install -v --no-cache-dir --no-build-isolation -U \
    git+https://github.com/facebookresearch/xformers.git@v0.0.29.post3#egg=xformers \
    --constraint /workspace/frozen.txt

RUN pip install -v --no-cache-dir --no-build-isolation -U \
    git+https://github.com/flashinfer-ai/flashinfer.git@v0.2.7.post1 \
    --constraint /workspace/frozen.txt

RUN git clone https://github.com/swiss-ai/vllm.git /workspace/vllm && \
    cd /workspace/vllm && \
    git checkout v0.9.0.1+swissai && \
    python use_existing_torch.py && \
    pip install -v --no-cache-dir --no-build-isolation --constraint /workspace/frozen.txt -r requirements/build.txt && \
    pip install -v --no-cache-dir --no-build-isolation --constraint /workspace/frozen.txt -e .

RUN pip install -v --no-cache-dir --no-build-isolation  -U \
    "lm-eval[vllm] @ git+https://github.com/swiss-ai/lm-evaluation-harness.git@main" \
    --constraint /workspace/frozen.txt

######## ADDED
RUN pip install "huggingface_hub<1.0" --system-site-packages
RUN pip freeze | grep -iE "huggingface_hub|^datasets|^transformers"

