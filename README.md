# Plugin package: Shot Type Classification

This package includes all methods to classify a given shot/or image sequence in one of the categories Extreme Long Shot 
(ELS), Long Shot (LS), Medium Shot (MS) or Close-Up Shot (CU).

## Package Description

PDF format: [vhh_stc_pdf](https://github.com/dahe-cvl/vhh_stc/blob/master/ApiSphinxDocumentation/build/latex/vhhpluginpackageshottypeclassificationvhh_stc.pdf)
    
HTML format (only usable if repository is available in local storage): [vhh_stc_html](https://github.com/dahe-cvl/vhh_stc/blob/master/ApiSphinxDocumentation/build/html/index.html)
    
    
## Quick Setup

This package includes a setup.py script and a requirements.txt file which are needed to install this package for custom applications.
The following instructions have to be done to used this library in your own application:

**Requirements:**

   * Ubuntu 18.04 LTS
   * CUDA 10.1 + cuDNN
   * python version 3.6.x

**Create a virtual environment:**

   * create a folder to a specified path (e.g. /xxx/vhh_stc/)
   * python3 -m venv /xxx/vhh_stc/

**Activate the environment:**

   * source /xxx/vhh_stc/bin/activate

**Checkout vhh_stc repository to a specified folder:**

   * git clone https://github.com/dahe-cvl/vhh_stc

**Install the stc package and all dependencies:**

   * Update pip and setuptools (tested using pip\==20.2.3 and setuptools==50.3.0)
   * Install the Wheel package: ```pip install wheel```
   * change to the root directory of the repository (includes setup.py)
   * ```python setup.py bdist_wheel```
   * The aforementioned command should create a /dist directory containing a wheel. Install the package using ```python -m pip install dist/xxx.whl```
   
> **_NOTE:_**
You can check the success of the installation by using the commend *pip list*. This command should give you a list
with all installed python packages and it should include *vhh-stc*.

**Install PyTorch :**

Install a Version of PyTorch depending on your setup. Consult the [PyTorch website](https://pytorch.org/get-started/locally/) for detailed instructions.

**Setup environment variables:**

   * source /data/dhelm/python_virtenv/vhh_sbd_env/bin/activate
   * export CUDA_VISIBLE_DEVICES=1
   * export PYTHONPATH=$PYTHONPATH:/XXX/vhh_stc/:/XXX/vhh_stc/Develop/:/XXX/vhh_stc/Demo/

**Run demo script**

   * change to root directory of the repository
   * python Demo/vhh_stc_run_on_single_video.py
