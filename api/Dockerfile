# pull official base image
FROM python:3.9

# set work directory
WORKDIR /usr/app

# RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
#     echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
#     echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

# copy requirements file
COPY ./requirements.txt /usr/app/requirements.txt

# install dependencies
RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libspatialindex-dev \
        libgdal-dev \
        build-essential \
        libssl-dev \
        libffi-dev \
        gcc \
        libc6-dev \
        python3-dev \
        cmake \
    # && pip cache purge \
    && pip install --no-cache-dir Cython \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /usr/app/requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# copy project
COPY . /usr/app/

VOLUME /usr/app


CMD  ["uvicorn", "app.main:app", "--reload", "--workers", "-1", "--host", "0.0.0.0", "--port", "8000"]