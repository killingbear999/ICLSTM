# ICLSTM for Fast MPC
**Library for code execution: Tensorflow Keras, Pyipopt, Numpy, Sklearn** </br>

File description:
* docker.pptx includes the instruction on how to install Pyipopt into Docker on your laptop. </br>
* CSTR_ICLSTM.ipynb and CSTR_NNs.ipynb are used to train neural networks to learn the system dynamics. </br>
* cstr_iclstm.py and cstr_nns.py are .py version to train neural networks to learn the system dynamics. They are identical to CSTR_ICLSTM.ipynb and CSTR_NNs.ipynb. </br>
* model26.h5, model27.h5, model28.h5, model29.h5 are trained RNN, LSTM, ICRNN, and ICLSTM respectively. You may regenerate the models using CSTR_ICLSTM.ipynb and CSTR_NNs.ipynb. <br>
* mpc_rnn.ipynb, mpc_lstm.ipynb, mpc_icrnn.ipynb, mpc_iclstm.ipynb are used to integrate NNs into LMPC and solve the MPC optimization problem.

FYI:
* .ipynb files can be run on Jupyter Notebook or Google Colab.
* Pyipopt can be installed and run on Docker. mpc_rnn.ipynb, mpc_lstm.ipynb, mpc_icrnn.ipynb, mpc_iclstm.ipynb use Pyipopt.
