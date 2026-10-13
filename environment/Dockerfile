# hash:sha256:4ad05058c9876e31fee487016cbdccd7a78e6c0f2d5e52f0d43168884a4c7ae0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gh \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    aind-analysis-arch-result-access==0.8.0 \
    aind-behavior-gym==0.4.5 \
    aind-dynamic-foraging-basic-analysis==0.3.34 \
    aind-dynamic-foraging-data-utils==0.1.41 \
    aind-dynamic-foraging-models==0.13.0 \
    aind-dynamic-foraging-multisession-analysis==0.0.18 \
    beaker-py==1.38.1 \
    hydra-core==1.3.0 \
    jax==0.6.1 \
    omegaconf==2.3.0 \
    pandas==2.2.3 \
    pyarrow==21.0.0 \
    requests==2.34.2 \
    scipy==1.14.1 \
    wandb==0.23.0 \
    -e git+https://github.com/AllenNeuralDynamics/aind-disentangled-rnns.git@a9b9978831cb22d37e2a75c15805c621dfe00b1f#egg=disentangled-rnns \
    -e git+https://github.com/AllenNeuralDynamics/aind_disrnn_utils.git@74de874d93b951d9ef3ae6bff6453e6ae805b649#egg=aind-disrnn-utils

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
