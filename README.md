# Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control
Zihao Wang, Zhe Wu </br>
Paper: https://arxiv.org/abs/2311.07202 </br>

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
