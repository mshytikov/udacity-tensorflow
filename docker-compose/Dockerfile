#
# Docker file to build  local image to fix the users permission
# and install extra packages
#

FROM jupyter/tensorflow-notebook

ARG host_uid
ARG host_gid

ENV NB_UID ${host_uid}
ENV NB_GID ${host_gid}

USER root

RUN usermod -u $NB_UID $NB_USER && chown -R $NB_UID $CONDA_DIR
RUN groupmod -g $NB_GID -o $(id -g -n $NB_USER)

RUN echo "$NB_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/notebook

USER jovyan

# Install jupyterlab

RUN conda install --yes -c conda-forge jupyterlab

RUN jupyter serverextension enable --py jupyterlab --sys-prefix
