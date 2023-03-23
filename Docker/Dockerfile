FROM tensorflow/tensorflow:1.14.0-gpu

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils # To remove a warning

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        htop \
        libsm6 \
        libxext6 \
        libxrender-dev \
        sshfs \
        sudo \
        vim \
        nano

# The installation of "python-tk" is a little bit special:
RUN ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime
RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python-tk
RUN dpkg-reconfigure --frontend noninteractive tzdata
RUN DEBIAN_FRONTEND=


# To share display
RUN apt-get install -qqy x11-apps
ENV DISPLAY :0


RUN pip install --upgrade pip
RUN pip install absl-py==0.8.1 \
    astor==0.8.0 \
    atomicwrites==1.3.0 \
    attrs==19.3.0 \
    backports.functools-lru-cache==1.5 \
    backports.shutil-get-terminal-size==1.0.0 \
    backports.weakref==1.0.post1 \
    cachetools==3.1.1 \
    certifi==2019.9.11 \
    chardet==3.0.4 \
    configparser==4.0.2 \
    contextlib2==0.6.0.post1 \
    cycler==0.10.0 \
    decorator==4.4.2 \
    docutils==0.16 \
    entrypoints==0.3 \
    enum34==1.1.6 \
    fire==0.2.1 \
    flake8==3.7.9 \
    flake8-import-order==0.18.1 \
    funcsigs==1.0.2 \
    functools32==3.2.3.post2 \
    future==0.18.2 \
    futures==3.3.0 \
    gast==0.2.2 \
    google-auth==1.7.1 \
    google-auth-oauthlib==0.4.1 \
    google-pasta==0.1.7 \
    got10k==0.1.3 \
    graphviz==0.16 \
    grpcio==1.24.1 \
    h5py==2.10.0 \
    idna==2.8 \
    importlib-metadata==1.5.0 \
    ipython==5.10.0 \
    ipython-genutils==0.2.0 \
    Keras-Applications==1.0.8 \
    Keras-Preprocessing==1.1.0 \
    kiwisolver==1.1.0 \
    llvmlite==0.31.0 \
    Markdown==3.1.1 \
    matplotlib==2.2.4 \
    mccabe==0.6.1 \
    mock==3.0.5 \
    more-itertools==5.0.0 \
    motmetrics==1.2.0 \
    numba==0.47.0 \
    numpy==1.16.5 \
    oauthlib==3.1.0 \
    opencv-python==4.2.0.32 \
    opt-einsum==2.3.2 \
    packaging==20.1 \
    pandas==0.24.2 \
    pathlib==1.0.1 \
    pathlib2==2.3.5 \
    pexpect==4.8.0 \
    pickleshare==0.7.5 \
    Pillow==6.2.1 \
    pluggy==0.13.1 \
    prompt-toolkit==1.0.18 \
    protobuf==3.10.0 \
    ptyprocess==0.7.0 \
    py==1.8.1 \
    py-cpuinfo==5.0.0 \
    pyasn1==0.4.8 \
    pyasn1-modules==0.2.7 \
    pycodestyle==2.5.0 \
    pydot==1.2.3 \
    pyflakes==2.1.1 \
    Pygments==2.5.2 \
    pyparsing==2.4.2 \
    pytest==4.6.9 \
    pytest-benchmark==3.2.3 \
    python-dateutil==2.8.0 \
    pytz==2019.3 \
    requests==2.22.0 \
    requests-oauthlib==1.3.0 \
    rsa==4.0 \
    scandir==1.10.0 \
    scikit-learn==0.20.4 \
    scipy==1.2.3 \
    Shapely==1.6.4.post2 \
    simplegeneric==0.8.1 \
    singledispatch==3.4.0.3 \
    six==1.12.0 \
    statistics==1.0.3.5 \
    subprocess32==3.5.4 \
    termcolor==1.1.0 \
    tflearn==0.3.2 \
    torch==1.4.0 \
    torchvision==0.5.0 \
    traitlets==4.3.3 \
    typing==3.7.4.1 \
    urllib3==1.25.7 \
    wcwidth==0.1.8 \
    Werkzeug==0.16.0 \
    wget==3.2 \
    wrapt==1.11.2 \
    xmltodict==0.12.0 \
    zipp==1.1.0

# To use local graphical display
RUN echo "QT_X11_NO_MITSHM=1" >> /etc/environment

RUN apt install -y cifs-utils


# To seamlessly share files
ARG HOST_USER_NAME=lorenzo.vaquero
ARG HOST_UID=1004  # 3316
ARG HOST_GID=1004  # 2000


RUN mkdir -p /mnt/shared_ctdesk143
RUN echo -e '#!/usr/bin/env bash\nsshfs -o allow_other -o idmap=user,uid=$(id -u '${HOST_USER_NAME}'),gid=$(id -g '${HOST_USER_NAME}') lorenzo.vaquero@ctdesk143.inv.usc.es:/home/lorenzo.vaquero/PHD/SiamMT/Aplicacion/python /mnt/shared_ctdesk143' > /root/share_ctdesk143.sh
RUN chmod +x /root/share_ctdesk143.sh

# To mimic my workspace
RUN mkdir -p /home/lorenzo.vaquero/PHD/SiamMT/Aplicacion/
RUN mkdir -p /home/lorenzo.vaquero/Datasets

RUN ln -s /mnt/shared_ctdesk143/ /home/lorenzo.vaquero/PHD/SiamMT/Aplicacion/python
RUN cp /root/share_ctdesk143.sh /home/lorenzo.vaquero/share_ctdesk143.sh
RUN chmod 777 /home/lorenzo.vaquero/share_ctdesk143.sh

RUN echo 'ALL ALL=NOPASSWD: /home/lorenzo.vaquero/share_ctdesk143.sh' >> /etc/sudoers

RUN groupadd -g $HOST_GID $HOST_USER_NAME
RUN useradd --uid $HOST_UID --gid $HOST_GID --shell /bin/bash $HOST_USER_NAME
RUN usermod -aG sudo $HOST_USER_NAME
USER $HOST_USER_NAME
WORKDIR "/home/lorenzo.vaquero"


CMD bash
