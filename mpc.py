from __future__ import print_function
import tensorflow as tf
from keras import backend as K
import numpy
from keras.models import load_model
from pathlib import Path
import os.path
import time
import os
import pyipopt
from numpy import *

class MyRNNCell(tf.keras.layers.Layer):

    def __init__(self, units, input_shape_custom, **kwargs):
        self.units = units
        self.input_shape_custom = input_shape_custom
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([input_shape_custom])]
        super(MyRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer='uniform',
                                                name='recurrent_kernel',
                                                constraint=tf.keras.constraints.NonNeg(),
                                                trainable=True)
        self.D1 = self.add_weight(shape=(self.units, self.units),
                                 initializer='uniform',
                                 name='D1',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.D2 = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='uniform',
                                 name='D2',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.D3 = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='uniform',
                                 name='D3',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.V = self.add_weight(shape=(self.units, self.units),
                                 initializer='uniform',
                                 name='V',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.built = True

    def call(self, inputs, states):
        # ICRNN
        prev_h, prev_input = states
        h = K.dot(inputs, self.kernel) + K.dot(prev_h, self.recurrent_kernel) + K.dot(prev_input, self.D2)
        h = tf.nn.elu(h)
        y = K.dot(h, self.V) + K.dot(prev_h, self.D1) + K.dot(inputs, self.D3)
        y = tf.nn.elu(y)
        return y, [h, inputs]

    def get_config(self):
        config = super(MyRNNCell, self).get_config()
        config.update({"units": self.units, "input_shape_custom": self.input_shape_custom})
        return config

#####Simulation time step
delta=0.005
hc=1e-4 #delta/100
oper_time=0.01
short_factor=int(0.01/delta)

####Initial states
CAi=1.5
Ti=-70

x1_nn=CAi
x2_nn=Ti
x1_record=[CAi]
x2_record=[Ti]
u1_record=[]
u2_record=[]
time_record=[]

a=1060
b=22
d=0.52

F=5
V=1
k0=8460000
E=50000
R=8.314
T0=300
Dh=-11500
rho=1000
sigma=1000
Cp=0.231
cp=0.231
Qs=0
CA0s=4
x_record=[0,0]

#steady-state
CAs= 1.9537
Ts=  401.8727

w1_std=2.5
w2_std=70
state_ss=numpy.array([Ts, CAs])
input_ss=numpy.array([Qs, CA0s])
ROOT_FOLDER=os.getcwd()

#### CONSTANTS ####
NUM_MPC_ITERATION=20*short_factor   #10000000000
OUTPUT_NO=0
TOTAL_MODELS=12
NUM_SUBMODELS=1
NUM_OUTPUTS=2
NUM_INPUTS=8 
HORIZON=2
NUM_IN_SEQUENCE=10
PREDICTION_STORE=0
deviation=0
NUM_MPC_INPUTS=2*HORIZON
NUM_MPC_CONSTRAINTS=HORIZON
realtime_data=None
setpoint=[0, 0]

def my_ens_prediction(num_horizon,my_rawdata,my_inputs):
    xx = []
    nn_inputs = []
    ensemble_output = numpy.zeros((num_horizon,NUM_OUTPUTS,NUM_IN_SEQUENCE))
    ensemble_output = ensemble_output.reshape(num_horizon,NUM_IN_SEQUENCE,NUM_OUTPUTS)
    predict_output = []
    x_test2 = my_rawdata[0:NUM_OUTPUTS].astype(float)
    x_test2= (x_test2-state_mean)/state_std

    predict_output_normal=[[0 for i in range(NUM_OUTPUTS)] for j in range(NUM_IN_SEQUENCE)]
    for i_model in range(num_horizon):      
        COUNT_CORRECT_MODEL=0
        my_inputs_normalized = (my_inputs[2*i_model:2*(i_model+1)] - input_mean) / input_std
        sum=[[0 for i in range(NUM_OUTPUTS)] for j in range(NUM_IN_SEQUENCE)]
        xx = numpy.concatenate((x_test2,  my_inputs_normalized, -x_test2, -my_inputs_normalized), axis=None).reshape((1, NUM_INPUTS))
        xx = numpy.tile(xx, (NUM_IN_SEQUENCE, 1))

        nn_inputs = xx.reshape(1, NUM_IN_SEQUENCE, NUM_INPUTS)
        for j_submodel in range (NUM_SUBMODELS):
            # j_submodel=0
            predict_output = numpy.array(model[j_submodel].predict(nn_inputs,verbose=0))
            predict_output = predict_output.reshape(NUM_IN_SEQUENCE, NUM_OUTPUTS)
            sum=sum+predict_output

        # MODEL AVERAGING (ENSEMBLE LEARNING)
        predict_output=sum/NUM_SUBMODELS
        x_test2=predict_output[-1,0:2]

        #########  if delta=0.005##########
        x_test2 = predict_output[int(NUM_IN_SEQUENCE/short_factor-1), 0:2]
        x_test2=x_test2 * output_std + output_mean
        x_test2 = (x_test2 - state_mean) / state_std

        # RESCALING BY THE CORRESPONDING STANDARD DEVIATION & THE MEAN OF THE OUTPUT STATISTICS OF THE EXITING SURFACE
        predict_output_normal = predict_output * output_std + output_mean
        ensemble_output[i_model,:,:]=predict_output_normal

    return ensemble_output    

#################################################
################## MPC PROGRAM ##################
#################################################
### DEFINE THE UPPER BOUND AND LOWER BOUND OF THE MANIPULATED INPUTS ###

def eval_f(x):
    assert len(x) == int(NUM_MPC_INPUTS)
    offset=0
    global PREDICTION_STORE
    #### CALCULATE OUTLET CONC ###########
    df_ensemble_output = my_ens_prediction(num_horizon=int(NUM_MPC_INPUTS/2),my_rawdata=realtime_data,my_inputs=x)

    #### account for all intermediate steps ####
    for j in range (int(NUM_MPC_INPUTS/2)):
        est_outlet_product = df_ensemble_output[j, :, 0:2]
        for i in range (int(NUM_IN_SEQUENCE/short_factor)):  #NUM_IN_SEQUENCE/2
             offset = offset + (setpoint[0] - (est_outlet_product[i, 0]))  + (setpoint[1] - (est_outlet_product[i, 1])) * 1000
        offset=offset+x[2*j] *3e-10 + 1* x[2*j+1]

    return offset/100

def eval_grad_f(x):
    assert len(x) == int(NUM_MPC_INPUTS)
    step = 1e-1 # we just have a small step
    objp=objm=0
    grad_f = [0]*NUM_MPC_INPUTS
    xpstep = [0]*NUM_MPC_INPUTS
    xmstep = [0]*NUM_MPC_INPUTS
    for i_mpc_input in range(NUM_MPC_INPUTS):
        xpstep=x.copy()
        xmstep=x.copy()
        # for each variables, we need to evaluate the derivative of the function with respect to that variable, This is why we have the for loop
        xpstep[i_mpc_input]  = xpstep[i_mpc_input]+step 
        xmstep[i_mpc_input] = xmstep[i_mpc_input]-step

        # Evaluate the objective function at xpstep and xmstep
        objp=eval_f(xpstep) # This function returns the value of the objective function evaluated with the variable x[i] is perturebed +step
        objm=eval_f(xmstep) # This function returns the value of the objective function evaluated with the variable x[i] is perturebed -step
        grad_f[i_mpc_input] = (objp - objm) / (2 * step) # This evaluates the gradient of the objetive function with repect to the optimization variable x[i]
    return array(grad_f)

def eval_g(x):
    assert len(x) == int(NUM_MPC_INPUTS)
    #### CALCULATE FLUID TEMPERATURE ALONG THE FIRST THREE SURFACES ###########
    CAd2=realtime_data[1]
    Td2=realtime_data[0]
    g=array([-5.0]*NUM_MPC_CONSTRAINTS)

    df_ensemble_output2 = my_ens_prediction(num_horizon=int(NUM_MPC_INPUTS / 2), my_rawdata=realtime_data, my_inputs=x)
    for j in range(int(NUM_MPC_INPUTS / 2)):
        est_outlet_product2 = df_ensemble_output2[j, int(NUM_IN_SEQUENCE/short_factor-1), 0:2]
        g[j]= d * (est_outlet_product2[0]) ** 2+ 2 * b * (est_outlet_product2[0])*(est_outlet_product2[1]) + \
                a*(est_outlet_product2[1]) ** 2 - a*CAd2**2 - 2*b*CAd2*Td2 - d*Td2**2

    return  g

nnzj = NUM_MPC_CONSTRAINTS*NUM_MPC_INPUTS

def eval_jac_g(x, flag):
    if flag:
        list_x = []
        list_y=[]
        for i in range(int(NUM_MPC_INPUTS / 2)):
            list_x = list_x + [i] * NUM_MPC_INPUTS
            list_y = list_y +list(range(0, int(NUM_MPC_INPUTS)))
        return (array(list_x),
                array(list_y))
    else:
        assert len(x) == int(NUM_MPC_INPUTS)
        step = 1e-1 # we just have a small step
        gp=gm=numpy.zeros(NUM_MPC_CONSTRAINTS)
        xpstep=xmstep=numpy.zeros(NUM_MPC_INPUTS)
        jac_g = [[0]*int(NUM_MPC_INPUTS) for _ in range(NUM_MPC_CONSTRAINTS)]
        for i_mpc_input in range(NUM_MPC_INPUTS):
            xpstep=x.copy()
            xmstep=x.copy()
            # for each variables, we need to evaluate the derivative of the function with respect to that variable, This is why we have the for loop
            xpstep[i_mpc_input] += step 
            xmstep[i_mpc_input] -= step
            gp=eval_g(xpstep)
            gm=eval_g(xmstep)
            for num_constraint in range(NUM_MPC_CONSTRAINTS):
                jac_g[num_constraint][i_mpc_input] = (gp[num_constraint] - gm[num_constraint]) / (2 * step)
        return array(jac_g)

def apply_new(x):
    return True
def print_variable(variable_name, value):
    for i in range(len(value)):
        print("{} {}".format(variable_name + "["+str(i)+"] =", value[i]))

nnzh = NUM_MPC_INPUTS**2

#####################################################################
##### PRE-PROCESSING (THE FOLLOWING COMMANDS ARE EXECUTED ONCE) #####
#####################################################################
#### LOAD MEAN AND STD FILES###########
#### READ MEANS & STD FROM THE FILE #####
x1_mean=1.6712e-02 # CA
x1_std=8.4936e-01 
x2_mean=-6.1691e-01# T
x2_std=3.8528e+01 
u1_mean=1.1605e-02   # CA0
u1_std=2.6116e+00 
u2_mean=1.9277e+02    # Q
u2_std=3.7388e+05
y1_mean=0.02164    # CA
y1_std=0.8431
y2_mean=-0.8497  # T
y2_std=39.6907
state_mean=numpy.array([x2_mean, x1_mean])
state_std=numpy.array([x2_std, x1_std])
input_mean=numpy.array([u2_mean, u1_mean])
input_std=numpy.array([u2_std, u1_std])
output_mean=numpy.array([y2_mean, y1_mean])
output_std=numpy.array([y2_std, y1_std])


model=[1]*(NUM_SUBMODELS) 
model[0]=load_model("model14.h5", custom_objects={"MyRNNCell": MyRNNCell})
    
####################################################################
##### SOLVING THE MPC PROGRAM TO FIND THE OPTIMIZED MPC INPUTS #####
####################################################################
##########  KEEP RUNNING MPC ###############

dir_name = os.getcwd()
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".txt"):
        os.remove(os.path.join(dir_name, item))

nvar = NUM_MPC_INPUTS
x_lower=[0]* nvar
x_upper=[0]* nvar
for i in range(int(NUM_MPC_INPUTS/2)):
    x_lower[2*i]=-5e5
    x_lower[2 * i+1] = -3.5
    x_upper[2 * i] = 5e5
    x_upper[2 * i + 1] = 3.5
x_L = array(x_lower) #array([-5e5, -3.5])
x_U = array(x_upper) #array([5e5, 3.5])

### DEFINE THE UPPER BOUND AND LOWER BOUND OF THE CONSTRAINT ###
ncon = NUM_MPC_CONSTRAINTS
g_L = array([-2e19]*HORIZON)
g_U = array([0]*HORIZON)

print ("g_L", g_L, g_U)

for main_iteration in range(NUM_MPC_ITERATION):
    print ("Num Iteratin: ", main_iteration)

    rawdata=numpy.array([Ti, CAi])
    realtime_data=rawdata

    start = time.time()
    nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
    if main_iteration ==0 :
        x0 = array([0.0]*int(NUM_MPC_INPUTS))
    else:
        x0=x
        x0[0:-2]=x[2:]
        x0[-2:]=x[-2:]
        x_record=x

    """
    print x0
    print nvar, ncon, nnzj
    print x_L,  x_U
    print g_L, g_U
    print eval_f(x0)
    print eval_grad_f(x0)
    print eval_g(x0)
    a =  eval_jac_g(x0, True)
    print "a = ", a[1], a[0]
    print eval_jac_g(x0, False)
    print eval_h(x0, pi0, 1.0, False)
    print eval_h(x0, pi0, 1.0, True)
    """

    """ You CAd2 set Ipopt options by calling nlp.num_option, nlp.str_option
    or nlp.int_option. For instance, to set the tolarance by calling

        nlp.num_option('tol', 1e-8)

    For a complete list of Ipopt options, refer to

        http://www.coin-or.org/Ipopt/documentation/node59.html

    Note that Ipopt distinguishs between Int, Num, and Str options, yet sometimes
    does not explicitly tell you which option is which.  If you are not sure about
    the option's type, just try it in PyIpopt.  If you try to set one type of
    option using the wrong function, Pyipopt will remind you of it. """
    nlp.int_option('max_iter', 200)
    nlp.num_option('tol', 1e-5)
    nlp.int_option('print_level', 2)
    print("Going to call solve")
    print("x0 = {}".format(x0))
    x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)

    nlp.close()
    end = time.time()
    
    print("The elapsed time is", end - start, "s")
    time_record.append(end-start)

    print("Solution of the primal variables, x")
    print_variable("x", x)
    print ("status=", status)
    print("Objective value")
    print("f(x*) = {}".format(obj))
    print ("Control action=:  ", x[1], x[0])

    x1=CAi
    x2=Ti

    w1 =numpy.random.normal(0, w1_std, 1)
    w2 =numpy.random.normal(0, w2_std, 1)
    if w1>w1_std:
        w1=w1_std
    if w1<-w1_std:
        w1=-w1_std
    if w2>w2_std:
        w2=w2_std
    if w2>w2_std:
        w2=w2_std

    for kk in range (int(delta/hc)):
        x1_new = x1 + hc * ((F / V) * (x[1] - x1) -
                            k0 * ((numpy.exp(-E / (R * (x2 + Ts)))*(x1 + CAs) * (x1 + CAs))
                                  - numpy.exp(-E / (R * Ts)) * CAs * CAs)+5*float(w1))

        x2_new = x2 + hc * (((F / V) * (-x2) + (-Dh / (sigma * cp)) *
                             (k0 * ((numpy.exp(-E / (R * (x2 + Ts))) * (x1 + CAs) * (x1 + CAs)) -
                                      numpy.exp(-E / (R * Ts)) * CAs * CAs)) + (x[0] / (sigma * cp * V)))+5*float(w2))

        x1 = x1_new
        x2 = x2_new

        if (kk%5==1):
            x1_record.append(x1)
            x2_record.append(x2)
            u1_record.append(x[1])
            u2_record.append(x[0])

    CAi=x1
    Ti=x2

    print('Real model output x1 x2 in deviation form:   ', x1, x2)

print ("x1_record: ",x1_record)
print ("x2_record: ",x2_record)

print ("u1_record: ",u1_record)
print ("u2_record: ",u2_record)

print("time_record: ", time_record)
print("total time elapsed: ", sum(time_record), 's')
print("average time for each iteration:", sum(time_record)/len(time_record), 's')
