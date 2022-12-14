
FROM python:3.7.8-windowsservercore-1809
# FROM python:3.7

LABEL maintainer="NP3DQN"

# Working directory is / by default. We explictly state it here for posterity
WORKDIR /

# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#        apt-utils \
#        build-essential \
#        curl \
#        cmake 

# Upgrade pip
RUN pip install --upgrade pip --user 

# - copy src files
COPY . /user/src/ml 

# # Move the requirements file into the image
# COPY requirements.txt /test/

# # Move the requirements file into the image
COPY requirements.txt ./user/src/ml  
RUN pip install --no-cache-dir -r ./user/src/ml/requirements.txt --user 
COPY . . 

# # Install the python requirements on the image
# RUN pip install --trusted-host pypi.python.org --no-cache-dir -r /test/requirements.txt

# Remove the requirements file - this is no longer needed
RUN rm /user/src/ml/requirements.txt

# Set it as the working directory
WORKDIR /user/src/ml 

### probably need Cmake to dynamically compile CTM or init Vissim as env here; then do it in bash script

### --- --- --- use following when sh bash is used --- --- ---
# # Copy over the start-up script
# ADD scripts/startup_script.sh /usr/local/bin/startup_script.sh

# # Give permissions to execute
# RUN chmod 777 /usr/local/bin/startup_script.sh

# # Set the display when we run the container. This allows us to record without the user needing to type anything explicitly
# # This code snippet was taken from https://github.com/duckietown/gym-duckietown/issues/123
# ENTRYPOINT ["/usr/local/bin/startup_script.sh"]

### start run python
CMD [ "python", "main.py"]
