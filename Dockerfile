
FROM python:3.7

LABEL maintainer="Test"

# Working directory is / by default. We explictly state it here for posterity
WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       cmake 

# Upgrade pip3
RUN pip3 install --upgrade pip

# - copy src files
COPY . /test

# Move the requirements file into the image
COPY requirements.txt /test/

# Install the python requirements on the image
RUN pip3 install --trusted-host pypi.python.org --no-cache-dir -r /test/requirements.txt

# Remove the requirements file - this is no longer needed
RUN rm /test/requirements.txt

# Set it as the working directory
WORKDIR /test/

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
CMD [ "python3", "main.py"]
