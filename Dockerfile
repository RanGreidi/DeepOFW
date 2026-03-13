# Minimal Dockerfile for Sionna inference, no requiemrnts file needed
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory and copy necessary files
WORKDIR /tmp
COPY requirment.txt /tmp/requirment.txt
# COPY inf.py /tmp/inf.py

# Set timezone
ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install necessary rand recommanded packages
RUN apt-get update && \             
    apt-get install -y --no-install-recommends \  
    # Compiler tools (gcc, make, etc.)
    build-essential \ 
    # Git version control
    git \     
    # Command-line tool for data transfers (HTTP, FTP, etc.)
    curl  \                                  
    # Python3 pip package manager
    python3-pip \                         
    # Python3 headers for building C extensions - CAREFUL: if using pip, it is unlikely that this is needed
    #python3-dev                                                                                       
    # Clean up apt cache to reduce image size and Upgrade pip without cache to keep image small
    && apt-get clean && rm -rf /var/lib/apt/lists/* \  
    && pip3 install --no-cache-dir --upgrade pip 

# Install requirment packages
RUN pip3 install -r /tmp/requirment.txt
RUN pip3 install jupyter notebook

# keep the container running (this is an infinit loop command, no inf.py script needed)
CMD ["tail", "-f", "/dev/null"]