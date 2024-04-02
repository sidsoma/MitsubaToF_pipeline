# MitsubaToF_pipeline

## (1) ssh into matlaber using the following command:

ssh -K username@matlaber12.media.mit.edu

## (2) Create a docker container with the image containing MitsubaToF

docker run --gpus all -dit --shm-size 50G -v /mas/u/$(whoami):/mas/u/$(whoami) -v /u/$(whoami)/:/u/$(whoami) -v /dtmp:/dtmp -v /tmp:/tmp -p $(id -u):8888 -u $(id -u):2000 -e YOUR_HOST=$(hostname) -e YOUR_USERNAME=$(whoami) -e YOUR_UID=$(id -u) --name $(whoami)-mitsubatof zigzagzackey/mitsubatof_pip

## (3) Enter the docker container as root username

docker exec -it -u root $(whoami)-mitsubatof bash

## (4) Navigate to local directory "run_single_frame" on matlaber (replace dir_name with path)

cd dir_name/run_single_frame

## (5) Run setup.py to install dependencies from within the docker container.

python setup.py

## (6) You can run run_scan.py to render a scene. Be sure to adjust the parameters in the file beforehand.

python run_scan.py

## (7) At some point during the code execution, you will access the docker container as a user. Once this happens, 

just type ctrl+D to exit. This will complete execution and the files will be saved to whatever global directory
you specificed in the file.

## (8) Run reconstruction.ipynb to see your output. 