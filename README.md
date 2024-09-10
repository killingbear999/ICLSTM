# Real-Time Machine-Learning-Based Optimization Using Input Convex Long Short-Term Memory Network

Zihao Wang, Donghan Yu, Zhe Wu </br>
Applied Energy (in press) </br>
Paper: https://arxiv.org/abs/2311.07202 </br>

**Requires: Python 3.11.3, Tensorflow Keras 2.13.0, Pyipopt, Numpy, Sklearn** </br>

**File description** </br>
* ICLSTM_poster.pdf is the poster version of the paper </br>
* docker.pptx includes the instruction on how to install Pyipopt into Docker on your laptop </br>
* ICLSTM_toy_examples.ipynb demonstrates the input convexity of ICLSTM in some 3D toy examples on surface fitting of non-convex bivariate scalar functions (we have constructed three examples for you to play around) </br>
* Under CSTR subfolder:
  1. CSTR_ICLSTM.ipynb and CSTR_NNs.ipynb are used to train neural networks to learn the system dynamics </br>
* Under MPC subfolder:
  1. rnn.h5, lstm.h5, icrnn.h5, iclstm.h5 are trained RNN, LSTM, ICRNN, and ICLSTM respectively. You may regenerate the models using CSTR_ICLSTM.ipynb and CSTR_NNs.ipynb <br>
  2. mpc_rnn.ipynb, mpc_lstm.ipynb, mpc_icrnn.ipynb, mpc_iclstm.ipynb are used to integrate NNs into LMPC and solve the MPC optimization problem <br>
  3. mpc_first_principles.ipynb uses the first principle model to solve the MPC optimization problem
* FYI:
  1. .ipynb files can be run on Jupyter Notebook or Google Colab </br>
  2. Pyipopt can be installed and run on Docker. mpc_rnn.ipynb, mpc_lstm.ipynb, mpc_icrnn.ipynb, mpc_iclstm.ipynb use Pyipopt </br>

**Motivation** </br>
* Traditional model-based optimization and control rely on the development of first-principles models, a process that is resource-intensive </br>
* Neural network-based optimization suffers from slow computation times, limiting its applicability in real-time tasks </br>
* Computational efficiency is a critical parameter for real-world and real-time implementation of neural network-based optimization </br>
* The optima of convex optimization problems are easier and faster to obtain than those of non-convex optimization problems </br>
* Long Short-Term Memory (LSTM) network's advanced gating architecture, which has been well documented in the literature </br>

**Objective** </br>
* Proposes an Input Convex Long Short-Term Memory (ICLSTM) neural network to increase computational efficiency (by preserving the convexity in neural network-based optimization) for real-time neural network-based optimization (e.g., model predictive control (MPC))

**Architecture** </br>

The ICLSTM cell follows the structure: </br>
</br>
![alt text](https://github.com/killingbear999/ICLSTM/blob/main/ICLSTM_cell.png)

Specifically, </br>

$f_t = g[D_f(W_hh_{t-1} + W_x[x_t,-x_t]) + b_f]$ </br>
$i_t = g[D_i(W_hh_{t-1} + W_x[x_t,-x_t]) + b_i]$ </br>
$c_{temp} = g[D_c(W_hh_{t-1} + W_x[x_t,-x_t]) + b_c]$ </br>
$o_t = g[D_o(W_hh_{t-1} + W_x[x_t,-x_t]) + b_o]$ </br>
$c_t = f_t * c_{t-1} + i_t * c_{temp}$ </br>
$h_t = o_t * g(c_t)$ </br>

where 
* $D_f$, $D_i$, $D_c$, $D_o$ are non-negative trainable scaling vectors to differentiate different gates
* $W_h$, $W_x$ are non-negative trainable weights (i.e., sharing weights across all gates)
* $b_f$, $b_i$, $b_c$, $b_o$ are trainable bias
* $g$ are convex, non-negative, and non-decreasing activation function (e.g., ReLU)
* $*$ denotes element-wise multiplication (i.e., Hadamard product)

The output of $L$-layer ICLSTM follows the structure: </br>
</br>
![alt text](https://github.com/killingbear999/ICLSTM/blob/main/ICLSTM_nlayer.png)

Specifically, </br>

$z = g^d(W_dh_t + b_d) + [x_t,-x_t]$ </br>
$y = g^y(W_yz + b_y)$ </br>

where
* $W_d$, $W_y$ are non-negative trainable weights
* $b_d$, $b_y$ are trainable bias
* $g^d$ is convex, non-negative, and non-decreasing activation function (e.g., ReLU)
* $g^y$ is convex, non-decreasing activation function

**Results** </br>
* ICLSTM-based MPC achieved convergence in 15 initial conditions (i.e., it achieved the fastest convergence in 13 out of 15 different initial conditions) on a continuous stirred tank reactor (CSTR) example, with an average percentage decrease in computational time of **54.4%**, **40.0%**, and **41.3%** compared to plain RNN, plain LSTM, and ICRNN, respectively </br>
* ICLSTM-based MPC enjoys a faster (at least **4 $\times$**) solving time compared to LSTM on a solar PV energy system example (i.e., for a scaled-up solar PV energy system or a longer prediction horizon, the time discrepancy will be even greater)

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
