# Docker image for xx using xx and xx
It is based on a Ubuntu 18.04 distribution and cuda 10.0.

## Getting up and running
### Step 1: Place the contents in an appropriate location
```
git clone
```

### Step 2: Build the docker image
It can take a while, a lot of things are being included. Only has to be done when the dockerfile has been changed.
```
cd docker && sh build.sh
```

### Step 3: run docker
Look at run.sh and make sure that the folders are mapped as you want. If the container is already running, use the clean.sh script.
```
sh run.sh # gives you a cmdline to the container
```

