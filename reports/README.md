---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [X] Create a git repository
* [X] Make sure that all team members have write access to the github repository
* [X] Create a dedicated environment for you project to keep track of your packages
* [X] Create the initial file structure using cookiecutter
* [X] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [X] Add a model file and a training script and get that running
* [X] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [X] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [X] Setup version control for your data or part of your data
* [X] Construct one or multiple docker files for your code
* [X] Build the docker files locally and make sure they work as intended
* [X] Write one or multiple configurations files for your experiments
* [X] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [X] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [X] Write unit tests related to the data part of your code
* [X] Write unit tests related to model construction and or model training
* [X] Calculate the coverage.
* [X] Get some continuous integration running on the github repository
* [X] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [X] Create a trigger workflow for automatically building your docker images
* [X] Get your model training in GCP using either the Engine or Vertex AI
* [X] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [X] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [X] Revisit your initial project description. Did the project turn out as you wanted?
* [X] Make sure all group members have a understanding about all parts of the project
* [X] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

47

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s222612, s222486, s212554, s222381

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

In our project we used Segmentation models pytorch which is built over timm. It provides standard encoders trained on imagenet data. It also provides aditional functionalities to choose loss function, decoder model and APIs for train and val loop.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

We used conda for managing dependencies in our project. List of dependencies required in our project was autogenerated by pipreqs to get a requirements.txt file. This was not necessary for local work, but turned really helpful when working with docker. Then we exported our environment to yml file. If new member wanted to fully copy environment he should use command "conda env create -f conda.yml"

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

From the cookiecutter template we have filled out every folder except for visualization in src directory. In folders in src directory we keep all functions necessary for preparing data, extracting features, training and testing our models. All those scripts are kept in folders with corresponding names. Folders in main directory serve more as an input and output. User should insert images to data folder and after training and validation, models and outputs will be created in directory with according name. We removed visualization folder because results of our training are kept on wandb and we did not prepare another script for presenting those. Local wandb version of runs are also stored locally in additionally created output folder. 

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

Although we did not create any new rule, we used both isort and black to adapt to pep8 format and improve quality of our code. If every member of big project writes code in his/her own style, it becomes very complicated to read and edit. Especially, after person responsible for some code part leaves project.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 3 tests to test data, model and training. Data test verifies our make_dataset function by checking number of elements in our dataset. Model test compare shape of model output's shape given random tensor of correct input shape. Training test mainly verifies if types of data in training functions are adequate.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our code had coverage of 69%. 
Even though Code coverage of 100% sounds nice, because every part of the code was tested, we should not trust the code to be completely bug free. To achieve 100% code coverage in bigger project we would probably have to change code parts to invoke certain assertion and artificially improve code coverage without improving code quality. In fact these tests could slow down runtime of our program. Also if too many changes like this were to happen bugs can appear undetected sooner or later.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In our project we made use of branches to increase productivity. Every member had his own branch to test new functionalities and fix bugs without interfering in work of others. Also the main branch was mostly free of bugs and unfinished "TODO" parts. Although even with that we had to browse through commit history to pull previous version replaced by omitment. We abstained from using pull requests though. Mainly due to the rush and inconveniences with pushing data to google cloud platform. Despite that, we know value of pull requests is even more evident in bigger projects. Code verification before publishing prevents plenty of bugs from happening.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer: 

Yes, we implemented DVC to store our data on the google drive initially and then shifted to google cloud at a later phase in the project. It was useful in multiple instances in the project. For eg, we were able to carryout DVC pull during the cloud build without copying the data. DVC also enables us to have a continuously improving dataset in the case where our project is deployed in real life where the car would record more data when it is driving and this data can also be incorporated to our dataset.



### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer: 

The CI that we are running are unittesting, github actions and precommit. Unittesting ran tests on the training data and the model scripts. Github actions was used to test for building the docker and carrying out the unittesting. Precommit was used to check the coding standards before committing to main.
The link to our github actions workflow: https://github.com/harshnehal1996/ML_OPS/blob/main/.github/workflows/run_pytests.yml


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer: 

We used config file to run our experiments. The config files for different experiments have been created in the config folder and they were incorporated into the code using hydra. Different experiments are run by changing the experiment name in the config.yaml file and just running the train_model.py script. We plan to improve the code by passing the experiment config file for different experiments through the run command. 

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer: 

We ensured reproducibility by using config files and hydra to store parameters for each experiment so that by running the code with different config files anyone could reproduce the experiments. Whenever an experiment is run the train_model.py takes in the config.yaml file which points to the particular experiment that we wish to run. The parameters are loaded from the config file for the particular experiment and these are used to run the experiment. We also use docker to enhance the reproducibility.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer: 

![wandb1](figures/wandb1.png)
![wandb2](figures/wandb2.png)
      
As seen in the figures, we tracked the Train IOU score, Validation IOU score, Train Accuracy and Validation Accuracy. The IOU scores tell us the extent of overlap between the predicted segments and the actual segmentation. The accuracy score gives us the pixelwise prediction accuracy. Both these metrics tell us the extent to which our predictions concur with the ground truth.


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer: 
      
We used seperate docker images for training and fast API predictions. The training docker image carried out dvc pull inside the docker image and ran the make_data.py and the train_model.py scripts to generate the trained model which is stored in a google cloud bucket. The second docker image carries out the inference and deployment by running the predict_model.py on Fast API to provide us predictions for images passed to it as input. 
To run the training docker: docker run -it gcr.io/snappy-byte-374310/segmentation_project /bin/bash
To run the Fast API docker: docker run --name gcr.io/snappy-byte-374310/fastapi -p 80:80 myimage

The link to the docker files are: 
      https://github.com/harshnehal1996/ML_OPS/blob/main/Dockerfile
      https://github.com/harshnehal1996/ML_OPS/blob/main/fastapi.dockerfile

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer: 

Debugging methods depended on the group members and the specific bug that was encountered. When faced with bugs related to data path, connectivity etc we tried debugging using different approaches from print statements to the inbuilt python debugging. We also generated the coverage reports along with logical analysis along with stackoverflow and chatgpt for debugging and optimizing our code.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer: 

We used the following services in GCP:
1. Compute Engine: We created a pytorch instance with GPU, pulled the docker container from the container registry and did the training of the model. The model after training will store the trained model in the bucket.
2. Bucket: Two buckets were created one to store our dataset after DVC push to the remote storage and one to store our trained model.
3. Container Registry: This was used to store our docker images for training and FastAPI prediction. 
4. Cloud Build: We used cloud build to create the infrastructure to build and push the docker images to the container registry and for creating the instances.
5. Vertex AI: Used this service to run a custom training job on a CPU. This was configured inside the cloudbuild.yaml file.
6. Cloud Run: This is a serverless service which ran the FastAPI container to respond to curl API requests from the end user.
7. Secrets Manager: Stores the kaggle credentials as a secret to not expose it to the public incase if the dataset needs to be downloaded from the internet. Uploaded the kaggle.json file and it parsed the information from here.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine to train our model. It has the following hardware: 

machine type: n1-standard-8,
 accelerator: 1 x NVIDIA V100,
 size: 128 GB,
 RAM: 32 gb

We pulled the docker image from the container registry and did the training of the model inside this. It was found that the docker container needed more memory to run, so we configured a memory swap to use the full memory of the instance for training. 

We also tried creating the instance through cloud build but the docker image was too big(around 24 GB) to be pushed to the container registry which resulted in a time out. This is the reason that we manually had to do the training on the compute instance. 

We followed a hybrid model where the build of docker image was done locally and pushed to the container registry, the training and the prediction using FastAPI was done on the cloud.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![bucket](figures/bucket1.png)


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![registry](figures/container_registry.png)
      
### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![build_history](figures/build1.png)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes, we did manage to deploy our application in the cloud. We imported and used the FastAPI decorator in our prediction script and packaged the application into a docker container. It was built and pushed to the container registry from the local machine and the cloud run used the container to respond to curl requests from the end user. The following is the curl command:

curl -X 'POST' 'https://modified-docker-dazyopfg7a-ew.a.run.app/prediction/b1_cycle' ?? -H 'accept: application/json' ?? -H 'Content-Type: multipart/form-data' ?? -F 'data=@image.png;type=image/png'

This curl command calls our application in the cloud by giving parameters in the URL such as the raw image of a city scene. The API is hit and it returns a segmented image downloaded into the user's local machine.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We could not implement monitoring mainly due to time constraints. We would have liked to do so to check for data drifting using tools like the Evidently AI framework and also create an alert system to send out alerts to people when some metric which needs to be tracked is not behaving as expected.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

In total we used around 45 dollars. A lot of credits (around 38.68 dollars) were used by the compute engine instance which used a high end machine with a GPU and the training was done using this. The other services which used up credits were the cloud build and storage which took about 2 dollars each.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![architecture](figures/Architecture.png)

Our starting point of the diagram is the local setup, where we integrated Hydra and W&B to create our model and training environment including logging and configuration. Then we pushed data to github firstly to store changes, but also to make use of github actions and testing implemented within these which is a part of continuous integration. We also integrated DVC to pull the latest version of the dataset from the remote storage such as Google Drive or the GCloud Bucket. Next the Cloud Build Trigger is setup in the GCloud console and will start the build process when all the tests have passed in GitHub Actions. When the docker image is built, it is then pushed to the container registry. 

In our case, the docker file built for training was too large (around 24 GB), so the build failed due to a timeout issue. Then we decided to locally build the docker image and push to the GCloud container registry. Next the Compute Engine instance pulls the image from the container registry and starts the training process. The trained models are then stored in the GCloud Bucket. 

Next for Cloud Deployment we use a service called Cloud Run which will pull the container which has the FastAPI application. The FastAPI application is used for prediction and fetches the trained model weights from the GCloud Bucket. Cloud Run then responds to POST API requests from the end user where the user uploads a raw image of the cityscapes scene. The user then receives a response with the segmented image of the scene and sends feedback to the developer to improvize on the model if needed. This completes the MLOps pipeline.
### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest issue faced during the project was implementing the cloud build since our image has a size of 24gb since it carries out a dvc pull to incorporate data into it. Due to this large size, the cloud build timed out and hence we had to build the files in the local machine and then push it onto the cloud registry. We also faced certain authentication issues in cloud build due to the data transfer protocols for cloud. 
Also when training the docker image in the compute engine instance, there were issues of limited RAM for the docker image. We had to perform a memory swap so that the docker image could utilize the full memory which was allocated to the instance. We had another issue where, by mistake we added the google project credentials to the github repository and it was exposed to the public which resulted in high end instances being used for crypto mining created by unknown users. Later it was realized that secrets need to be stored in the secrets manager. While building the docker image and pushing it for use by the Vertex AI service, the cloudbuid service did not have permission to create the instance. So, we had to give permissions via the IAM console. The buckets created were not able to receive the data because it was private at first and then made it public so that all services and group members could read and write data into the bucket. 

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:
* Navaneeth Kizhakkumbadan Pacha (s222486) was in charge of implementing the raw data processing and pipeline development to generate the datasets that would be used for training. He was also responsible for implementing unittesting on testdata and setting up the entire logging protocols using wandb.
* Joshua Sebastian (s212554): Involved in creating the initial cookie cutter structure, creating docker files and the cloudbuild.yaml file, DVC integration pointing to remote bucket in GCloud. Involved in continuous integration activites like github actions, cloud build, coverage and precommit configurations. Also setup the instances, managing secrets, vertex AI, bucket and pushing docker images to the container registry. Helped in preparing the script for the prediction using FastAPI.
* Micha?? Reczulski (s222612) was inlolved in raw data extraction, feature extraction and pixel classification with use of preprocessing. Also involved in creating docker file, setup DVC and training model.
* Harsh Vardhan Rai(s222381) was in charge of designing training experiments using hydra, developing the training model and script and training the model inside the compute engine. He was also incharge of model prediction and deployment using fast api and cloud run. He created unit test for checking correct behaviour of training scripts against different model inputs and added branch protection in repo.