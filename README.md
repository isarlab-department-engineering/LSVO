# LSVO
LS-VO code repository

# INTRODUCTION

This repository contains the source code of the LS-VO approach described in the article "LS-VO: Learning Dense Optical Subspace for Robust Visual Odometry Estimation" by Gabriele Costante and Thomas A. Ciarfuglia

# ENVIRONMENT SETUP

The LS-VO code has thee following dependencies: 
* Keras framework https://keras.io/ 
* Tensorflow as the backend framework for Keras https://www.tensorflow.org/. 
* Python 3.5. 

We suggest to create a virtual environment to install the project dependencies:
    
    #Install python 3 and virtualenv
    sudo apt-get install python3-pip python3-dev python-virtualenv
    
    #Create a virtualenv directory
    virtualenv --system-site-packages -p python3 lsvo-environment
    
    #Activate the ennvironment on the current shell
    source ~/lsvo-environment/bin/activate
    
    #Install tensorflow with GPU support
    pip3 install --upgrade tensorflow-gpu
    
    
    
    

    
