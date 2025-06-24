FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
RUN apt -o Acquire::AllowInsecureRepositories=true \
-o Acquire::AllowDowngradeToInsecureRepositories=true \
update
RUN apt install -y screen
RUN apt install -y build-essential
# RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install wget
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install libopenmpi-dev
RUN apt-get -y install unzip
RUN apt-get -y install htop
RUN pip install --upgrade pip
RUN apt-get -y update
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt