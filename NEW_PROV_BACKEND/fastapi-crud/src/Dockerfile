# pull official base image
FROM python:3.11.0

# set work directory
WORKDIR /usr/src/app

# copy requirements file
COPY ./requirements.txt /usr/src/app/requirements.txt

# install dependencies
RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        gcc \
        libc6-dev \
        python3-dev \
        cmake \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /usr/src/app/requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# copy project
COPY . /usr/src/app/
