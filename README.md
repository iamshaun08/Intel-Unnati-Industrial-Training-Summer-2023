# Intel-Unnati-Industrial-Training-Summer-2023
This repository is meant for the sole purpose of participating and submitting the submissions for the Intel Unnati Industrial Training 2023 for the team Go SOLO

It contains 5 folders inside the main folder

1. code - contains the python notebook file and the lda model simulatioin
2. demo videos - contains the demo video link
3. data - contains the train and test dataset
4. docs - contains the project report
5. model - contains all the files for creating docker image
   (docker image was too large to be uploaded)

Steps to create docker image and run it:
1. download all the files in model to a folder
2. in the same folder, run the command
     sudo docker build -t {image-name} .
     sudo docker run {image-name}
