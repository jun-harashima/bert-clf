FROM ubuntu:22.04 AS base

RUN apt update -y \
    && apt install -y curl cmake python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists

RUN pip3 install -U pip \
    && pip3 install poetry

WORKDIR /work
COPY . ./

FROM base AS bert

RUN poetry install --no-root --with bert

CMD ["/bin/bash"]

FROM base AS jumanpp

RUN curl -LO https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz \
    && tar -xf jumanpp-2.0.0-rc3.tar.xz \
    && cd jumanpp-2.0.0-rc3 \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make install \
    && rm -rf /work/jumanpp-2.0.0-rc3 \
    && rm /work/jumanpp-2.0.0-rc3.tar.xz

FROM jumanpp AS roberta

RUN poetry install --no-root --with roberta

CMD ["/bin/bash"]

FROM jumanpp AS deberta

RUN poetry install --no-root

CMD ["/bin/bash"]
