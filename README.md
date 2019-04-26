# STarFish

STarFish is a Stacked Ensemble Target Fishing tool for small molecule target
identification. STarFish takes a chemical structure as an input and provides
a ranked list of potential protein targets along with a probability score 
for each.

![](https://i.imgur.com/B8xJKGC.gif)

### Prerequisites

STarFish was built using python and anaconda was used to manage the required 
packages and dependencies. To get started, download and install the python3 
version of miniconda:

https://docs.conda.io/en/latest/miniconda.html

### Installing and Running STarFish

Download the STarFish repository and extract files to desired location.
Open Anaconda Prompt (This was installed with anaconda).

Note: After extraction, make sure that opening the "STarFish-master" folder 
that you see folders and files named "build_model", "web_app",
"environment.yml", "LICENSE", and "README.md". If you see another folder named
"STarFish-master" then move that folder to your desired location and confirm
that you see the aforementioned folders and files inside of that one. Otherwise
the following commands will not work.

In the prompt type:
```
cd /your/path/to/STarFish-master
```
For windows use "\\" instead of "/", for example:
```
cd \your\path\to\STarFish-master
```

Replace "/your/path/to/" with the pathname of the directory where you 
extracted the repository files to.

So if you extracted to your Downloads folder you would type in the prompt:
```
cd Desktop/STarFish-master
```
or for windows:
```
cd Desktop\STarFish-master
```

If you extracted to a different location and are unfamiliar with naviagation
via pathnames, then open the file explore GUI, navigate to the directory,
and copy the path name from the address bar into the anaconda prompt.
For example:

![](https://i.imgur.com/H5vy9Ht.png)
```
cd C:\Users\cockr\Documents\STarFish-master
```

Once you have navigated to the repository in the Anaconda Prompt install
the anaconda environment from the environment.yml file.

In the prompt type:
```
conda env create -f environment.yml
```
The creation of the conda environment may take several minutes.

Following successulf creation of the conda environment, activate the 
environment:
```
conda activate STarFish
```

Change to the web_app directory:
```
cd web_app
```

Set the FLASK_APP environmental variable:
```
export FLASK_APP=app.py
```
or for windows:
```
set FLASK_APP=app.py
```

Run the STarFish web application:
```
flask run
```

In your web browser, navigate to the "Running on" url that was listed in the
prompt:
```
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

### Building the models used in STarFish

The build_model folder contains the python scripts that were used to generate 
the models in STarFish. The models were trained using HPC cluster resources and
it is not realistic to recreate them on a single machine. Bash scripts and pbs 
files submission files are provided which can be used to re-create the models.
However, these require the use of a bash shell and access to a HPC cluster that
uses a PBS batch scheduling system. It is extremely likely the .sh and .pbs 
files would need to be modified for your environment prior to running them.

