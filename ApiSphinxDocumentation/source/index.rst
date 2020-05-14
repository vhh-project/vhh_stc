.. vhh_stc documentation master file, created by
   sphinx-quickstart on Wed May  6 18:41:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#############################################
Welcome to the vhh_stc package documentation!
#############################################

This python package is developed within the project `Visual History of the Holocaust`_ (VHH) started in Januray 2019.
The major objective of this package is to provide interfaces and functions to classify image sequences
(shots) in on out the classes: Extreme Long Shot (ELS), Long Shot (LS), Medium Shot (MS) or Close-Up shot (CU).

This package is installable and designed to reuse it in customized applications such as the `vhh_core`_ package. This
module represents the main controller in the context of the VHH project.

This documentation provide an API description of all classes, modules and member functions as well as
the required setup descriptions.

Package Overview
================
name of repository: vhh_stc
   * ApiSphinxDocumentation: includes all files to generate the documentation as well as the created documentations (html, pdf)
   * config: this folder includes the required configuration file
   * stc: this folder represents the shot-type-classification module and builds the main part of this repository
   * Demo: this folder includes a demo script to demonstrate how the package have to be used in customized applications
   * Develop: includes scripts to train and evaluate the pytorch models
   * README.md: this file gives a brief description of this repository (e.g. link to this documentation)
   * requirements.txt: this file holds all python lib dependencies and is needed to install the package in your own virtual environment
   * setup.py: this script is needed to install the stc package in your own virtual environment

Setup  instructions
===================


API Description
===============

.. toctree::
   :maxdepth: 4

   Configuration.rst
   STC.rst
   Video.rst
   Models.rst
   Datasets.rst
   CustomTransforms.rst
   Shot.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

**********
References
**********
.. target-notes::

.. _`Visual History of the Holocaust`: https://www.vhh-project.eu/
.. _`vhh_core`: https://github.com/dahe-cvl/vhh_core

