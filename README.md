# Input Convex LSTM: A Convex Approach for Fast Model Predictive Control

Zihao Wang, Donghan Yu, Zhe Wu </br>
Paper: https://arxiv.org/abs/2311.07202 (This version is outdated, we are working on the updated version) </br>

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

The ICLSTM cell follows the structure as follows: </br>
</br>
![alt text](https://github.com/killingbear999/ICLSTM/blob/main/ICLSTM_cell.png)

Specifically, </br>
$f_t = ReLU[D_f(W_hh_{t-1}) + U_f(W_x[x_t,-x_t]) + b_f]$ </br>
$i_t = ReLU[D_i(W_hh_{t-1}) + U_i(W_x[x_t,-x_t]) + b_i]$ </br>
$c_{temp} = ReLU[D_c(W_hh_{t-1}) + U_c(W_x[x_t,-x_t]) + b_c]$ </br>
$o_t = ReLU[D_o(W_hh_{t-1}) + U_o(W_x[x_t,-x_t]) + b_o]$ </br>
$c_t = f_tc_{t-1} + i_tc_{temp}$ </br>
$h_t = o_tReLU(c_t)$ </br>
where 
* $D_f$, $D_i$, $D_c$, $D_o$, $U_f$, $U_i$, $U_c$, $U_o$ are non-negative trainable scaling vectors
* $W_h$, $W_x$ are non-nagative trainable weights (i.e., sharing weights across all gates)
* $b_f$, $b_i$, $b_c$, $b_o$ are trainable bias

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
