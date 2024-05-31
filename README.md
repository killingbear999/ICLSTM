# Input Convex LSTM: A Convex Approach for Fast Model Predictive Control

Zihao Wang, Donghan Yu, Zhe Wu </br>
Paper: https://arxiv.org/abs/2311.07202 </br>

**Requires: Python 3.11.3, Tensorflow Keras 2.13.0, Pyipopt, Numpy, Sklearn** </br>
File description:
* docker.pptx includes the instruction on how to install Pyipopt into Docker on your laptop. </br>
* ICLSTM_toy_examples.ipynb demonstrates the input convexity of ICLSTM in some 3D toy examples. <\br>
* Under CSTR subfolder:
  1. CSTR_ICLSTM.ipynb and CSTR_NNs.ipynb are used to train neural networks to learn the system dynamics. </br>
* Under MPC subfolder:
  1. rnn.h5, lstm.h5, icrnn.h5, iclstm.h5 are trained RNN, LSTM, ICRNN, and ICLSTM respectively. You may regenerate the models using CSTR_ICLSTM.ipynb and CSTR_NNs.ipynb. <br>
  2. mpc_rnn.ipynb, mpc_lstm.ipynb, mpc_icrnn.ipynb, mpc_iclstm.ipynb are used to integrate NNs into LMPC and solve the MPC optimization problem.

FYI:
* .ipynb files can be run on Jupyter Notebook or Google Colab.
* Pyipopt can be installed and run on Docker. mpc_rnn.ipynb, mpc_lstm.ipynb, mpc_icrnn.ipynb, mpc_iclstm.ipynb use Pyipopt.

The ICLSTM cell follows the structure as follows:
![alt text](https://github.com/killingbear999/ICLSTM/blob/main/ICLSTM_cell.pdf)

## Citation </br>
If you find our work relevant to your research, please cite:
```
@article{wang2023input,
  title={Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control},
  author={Wang, Zihao and Wu, Zhe},
  journal={arXiv preprint arXiv:2311.07202},
  year={2023}
}
```
