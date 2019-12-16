from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import datetime
from neural_networks_digits_cpu import *


digits = datasets.load_digits()
data_train, data_test, declabels_train, declabels_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=0)
labels_train = np.zeros((declabels_train.shape[0], 10), dtype=np.int)
for i in range(declabels_train.shape[0]):
    labels_train[i][declabels_train[i]] = 1
labels_test = np.zeros((declabels_test.shape[0], 10), dtype=np.int)
for i in range(declabels_test.shape[0]):
    labels_test[i][declabels_test[i]] = 1
data_train /= 16
data_test /= 16


max_iter = 10000
INPUT_LAYER_SIZE = 64
HIDDEN_LAYER_SIZE = 100
OUTPUT_LAYER_SIZE = 10
LEARNING_RATE = 0.01
macros = ["INPUT_LAYER_SIZE", "HIDDEN_LAYER_SIZE", "OUTPUT_LAYER_SIZE", "LEARNING_RATE"]
values = [str(INPUT_LAYER_SIZE), str(HIDDEN_LAYER_SIZE), str(OUTPUT_LAYER_SIZE), str(LEARNING_RATE)]

mynn = neural_network_cpu(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)


def define(func, macro, value):
    for i in range(len(macro)):
        func = func.replace(macro[i], value[i])
    return func

# kernal function generation
sample_select = """
__global__ void sample_select(float *input_out, float *label, float *data_train, float *labels_train, float *rands, float *iter)
{
    const int tid = threadIdx.x;
    __shared__ int sample_selector;
    
    if(tid == 0){
        sample_selector = (int)(rands[(int)(*iter)]);
    }
    __syncthreads();

    if(tid < INPUT_LAYER_SIZE){
        input_out[tid] = data_train[sample_selector*INPUT_LAYER_SIZE + tid];
    }
    if(tid < OUTPUT_LAYER_SIZE){
        label[tid] = labels_train[sample_selector*OUTPUT_LAYER_SIZE + tid];
    }
    __syncthreads();
    if(tid == 0 ){
        *iter += 1;
    }
}
"""
sample_select = define(sample_select, macros, values)
sample_select_gpu = SourceModule(sample_select).get_function("sample_select")


initialize_nodes = """
__global__ void initialize_nodes(float *delta_h, float* hidden_in, float *hidden_out, float *output_in, float *output_out)
{
    const int tid = threadIdx.x;
    if(tid < HIDDEN_LAYER_SIZE){
        hidden_in[tid] = 0;
        hidden_out[tid] = 0;
        delta_h[tid] = 0;
    }
    if(tid < OUTPUT_LAYER_SIZE){
        output_in[tid] = 0;
        output_out[tid] = 0;
    }
}
"""
initialize_nodes = define(initialize_nodes, macros, values)
initialize_nodes_gpu = SourceModule(initialize_nodes).get_function("initialize_nodes")

forward_input2hidden = """
__global__ void forward_input2hidden(float *v, float *input_out, float *hidden_in, float *hidden_out)
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bid = blockIdx.x;
    int i,j;
    i = tidx;
    j = bid*BLOCK_SIZE + tidy;
    //hidden_in[j] += v[i*HIDDEN_LAYER_SIZE + j] * input_out[i];  
    atomicAdd(&hidden_in[j],v[i*HIDDEN_LAYER_SIZE + j] * input_out[i]);
    /*__syncthreads();
    __threadfence();*/

}
"""
BLOCK_SIZE = int(HIDDEN_LAYER_SIZE / 10)
forward_input2hidden = define(forward_input2hidden, macros + ["BLOCK_SIZE"], values + [str(BLOCK_SIZE)])
forward_input2hidden_gpu = SourceModule(forward_input2hidden).get_function("forward_input2hidden")

relu = """
__global__ void relu(float* hidden_in, float *hidden_out)
{
    const int j = threadIdx.x;
    float temp;

    //relu
    temp = hidden_in[j];
    if(temp > 0){
        hidden_out[j] = temp;
    }
}
"""
relu = define(relu, macros, values)
relu_gpu = SourceModule(relu).get_function("relu")

forward_hidden2output = """
__global__ void forward_hidden2output(float *w, float *hidden_out, float *output_in, float *output_out)
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    int j,k;
    __shared__ float temp_sum;
    if(tidx == 0 && tidy == 0){
        temp_sum = 0;
    }
    __syncthreads();
    j = tidx;
    k = tidy;
    //output_in[k] += w[j*OUTPUT_LAYER_SIZE + k] * hidden_out[j];
    atomicAdd(&output_in[k], w[j*OUTPUT_LAYER_SIZE + k] * hidden_out[j]);
    __syncthreads();

    if(j == 0){
        output_in[k] = expf(output_in[k]);

        //temp_sum += output_in[k];
        atomicAdd(&temp_sum,output_in[k]);
    }
    __syncthreads();

    if(j == 0){
        output_out[k] = output_in[k] / temp_sum;
    }

}
"""
forward_hidden2output = define(forward_hidden2output, macros, values)
forward_hidden2output_gpu = SourceModule(forward_hidden2output).get_function("forward_hidden2output")


backward_output2hidden = """
__global__ void backward_output2hidden(float *label, float *output_out, float *delta_o)
{
    const int tidx = threadIdx.x;
    delta_o[tidx] = label[tidx] - output_out[tidx];

}
"""
backward_output2hidden = define(backward_output2hidden, macros, values)
backward_output2hidden_gpu = SourceModule(backward_output2hidden).get_function("backward_output2hidden")

backward_hidden2input = """
__global__ void backward_hidden2input(float *w, float *hidden_in, float *delta_o, float *delta_h)
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    
    int j,k;
    j = tidx;
    k = tidy;

    __syncthreads();

    //diff_relu
    if(hidden_in[j] > 0){
        //delta_h[j] += delta_o[k] * w[j*HIDDEN_LAYER_SIZE + k];
        atomicAdd(&delta_h[j], delta_o[k] * w[j*OUTPUT_LAYER_SIZE + k]);
    }
}
"""
backward_hidden2input = define(backward_hidden2input, macros, values)
backward_hidden2input_gpu = SourceModule(backward_hidden2input).get_function("backward_hidden2input")

update_w = """
__global__ void update_w(float *learning_rate, float *w, float *delta_o, float *hidden_out)
{
    //extern __shared__ float lr;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    int j,k;
    j = tidx;
    k = tidy;

    //w[j*HIDDEN_LAYER_SIZE + k] += (*learning_rate) * (hidden_out[j] * delta_o[k]);
    atomicAdd(&w[j*OUTPUT_LAYER_SIZE + k], (*learning_rate) * (hidden_out[j] * delta_o[k]));
}
"""
update_w = define(update_w, macros, values)
update_w_gpu = SourceModule(update_w).get_function("update_w")

update_v = """
__global__ void update_v(float *learning_rate, float *v, float *delta_h, float *input_out)
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bid = blockIdx.x;
    int i,j;
    i = tidx;
    j = bid*BLOCK_SIZE + tidy;

    //v[i*INPUT_LAYER_SIZE + j] += (*learning_rate) * (input_out[i] * delta_h[j]);
    atomicAdd(&v[i*HIDDEN_LAYER_SIZE + j], (*learning_rate) * (input_out[i] * delta_h[j]));
}
"""
update_v = define(update_v, macros + ["BLOCK_SIZE"], values + [str(BLOCK_SIZE)])
update_v_gpu = SourceModule(update_v).get_function("update_v")

# malloc
v = mynn.v.astype(np.float32) #flatten to send to GPU
w = mynn.w.astype(np.float32)
data_train = data_train.astype(np.float32)
labels_train = labels_train.astype(np.float32)
input_out = data_train[0]
label = labels_train[0]
hidden_inout = np.zeros(HIDDEN_LAYER_SIZE, dtype=np.float32)
output_inout = np.zeros(OUTPUT_LAYER_SIZE, dtype=np.float32)
learning_rate = np.float32(LEARNING_RATE)

rands = np.random.randint(data_train.shape[0], size=max_iter).astype(np.float32)
zero_iter = 0
zero_iter = np.float32(zero_iter)

w_fetch = np.empty_like(w)
v_fetch = np.empty_like(v)

time_start = datetime.datetime.now()

v_gpu = cuda.mem_alloc(v.nbytes)
w_gpu = cuda.mem_alloc(w.nbytes)
data_train_gpu = cuda.mem_alloc(data_train.nbytes)
labels_train_gpu = cuda.mem_alloc(labels_train.nbytes)
input_out_gpu = cuda.mem_alloc(input_out.nbytes)
label_gpu = cuda.mem_alloc(label.nbytes)
hidden_in_gpu = cuda.mem_alloc(hidden_inout.nbytes)
hidden_out_gpu = cuda.mem_alloc(hidden_inout.nbytes)
delta_h_gpu = cuda.mem_alloc(hidden_inout.nbytes)
output_in_gpu = cuda.mem_alloc(output_inout.nbytes)
output_out_gpu = cuda.mem_alloc(output_inout.nbytes)
delta_o_gpu = cuda.mem_alloc(output_inout.nbytes)
learning_rate_gpu = cuda.mem_alloc(learning_rate.nbytes)

# data transfer: host -> device
cuda.memcpy_htod(v_gpu, v)
cuda.memcpy_htod(w_gpu, w)
cuda.memcpy_htod(data_train_gpu, data_train)
cuda.memcpy_htod(labels_train_gpu, labels_train)
cuda.memcpy_htod(learning_rate_gpu, learning_rate)

rands_gpu = cuda.mem_alloc(rands.nbytes)
cuda.memcpy_htod(rands_gpu, rands)

iter_gpu = cuda.mem_alloc(zero_iter.nbytes)
cuda.memcpy_htod(iter_gpu, zero_iter)

# execute
debug = 0
for i in range(max_iter):
    sample_select_gpu(input_out_gpu, label_gpu, data_train_gpu, labels_train_gpu, rands_gpu, iter_gpu, block=(INPUT_LAYER_SIZE, 1, 1)) # input_layer_size should be bigger than output
    initialize_nodes_gpu(delta_h_gpu, hidden_in_gpu, hidden_out_gpu, output_in_gpu, output_out_gpu, block=(HIDDEN_LAYER_SIZE, 1, 1))

    forward_input2hidden_gpu(v_gpu, input_out_gpu, hidden_in_gpu, hidden_out_gpu, block=(INPUT_LAYER_SIZE, 10, 1), grid=(BLOCK_SIZE, 1))
    relu_gpu(hidden_in_gpu, hidden_out_gpu, block=(HIDDEN_LAYER_SIZE, 1, 1))

    forward_hidden2output_gpu(w_gpu, hidden_out_gpu, output_in_gpu, output_out_gpu, block=(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 1))


    backward_output2hidden_gpu(label_gpu, output_out_gpu, delta_o_gpu, block=(OUTPUT_LAYER_SIZE, 1, 1))
    
    backward_hidden2input_gpu(w_gpu, hidden_in_gpu, delta_o_gpu, delta_h_gpu, block=(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 1))
    
    update_w_gpu(learning_rate_gpu, w_gpu, delta_o_gpu, hidden_out_gpu, block=(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 1))
    update_v_gpu(learning_rate_gpu, v_gpu, delta_h_gpu, input_out_gpu, block=(INPUT_LAYER_SIZE, 10, 1), grid=(BLOCK_SIZE, 1))

    if(debug == 1 or i==max_iter-1):
        tmp2 = np.empty_like(input_out)
        cuda.memcpy_dtoh(tmp2, input_out_gpu)
        print("io", tmp2)
        tmp0 = np.empty_like(hidden_inout)
        cuda.memcpy_dtoh(tmp0, hidden_in_gpu)
        print("hi", tmp0)
        tmp1 = np.empty_like(hidden_inout)
        cuda.memcpy_dtoh(tmp1, hidden_out_gpu)
        print("ho", tmp1)
        '''
        tmp1 = np.empty_like(v)
        cuda.memcpy_dtoh(tmp1, v_gpu)
        print("v", tmp1)
        tmp1 = np.empty_like(w)
        cuda.memcpy_dtoh(tmp1, w_gpu)
        print("w", tmp1)
        '''
        tmp0 = np.empty_like(output_inout)
        cuda.memcpy_dtoh(tmp0, output_in_gpu)
        print("oi", tmp0)
        tmp1 = np.empty_like(output_inout)
        cuda.memcpy_dtoh(tmp1, output_out_gpu)
        print("oo", tmp1)
        tmp2 = np.empty_like(output_inout)
        cuda.memcpy_dtoh(tmp2, delta_o_gpu)
        print("delta_o", tmp2)
        tmp2 = np.empty_like(hidden_inout)
        cuda.memcpy_dtoh(tmp2, delta_h_gpu)
        print("delta_h", tmp2)
        print("---------------------")
# variables: device -> host
cuda.memcpy_dtoh(w_fetch, w_gpu)
cuda.memcpy_dtoh(v_fetch, v_gpu)
time_total = datetime.datetime.now() - time_start
print("w",w_fetch)
print("v",v_fetch)

mynn.w = w_fetch.copy()
mynn.v = v_fetch.copy()

mynn.test(data_train, labels_train)

print("time", time_total)








