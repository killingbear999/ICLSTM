# ICLSTM for fast MPC
**Library for code execution: Tensorflow Keras, Pyipopt, Numpy** </br> </br>
File description:
* CSTR_data_generation.py is used to generate training data </br>
* icrnn.py, lstm.py, iclstm.py, cnn1d.py, rescnn1d.py, icnn.py are deep learning models built using Tensorflow Keras </br>
* results_summary.py is used to analyse model performance </br>
* mpc.py is used to incorporate deep learning model into mpc and compute runtime </br>
* CSTR_data_testing.ipynb was used in model design and model evaluation during the experimental process

**If .py files do not work, you may run the CSTR_data_testing.ipynb. It is a bit messy. I will clean up the code asap.** </br>
**Pyipopt can be installed and run on Docker (i.e., mpc.py).**
