from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

def relu(x):
    if x <= 0:
        return 0
    else:
        return x

def diff_relu(x):
    if x <= 0:
        return 0
    else:
        return 1

def classify(output_out):
    return np.argmax(output_out)

class neural_network_cpu:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.v, self.w = self.intialize_weight()
        self.activation = relu
        self.diff_activation = diff_relu

    def intialize_weight(self):
        # input -> hidden
        v = np.random.uniform(0, 1, [self.input_layer_size, self.hidden_layer_size])
        v = v / np.sum(v, axis=0)  #normalization!
    
        # hidden -> output
        w = np.random.uniform(0, 1, [self.hidden_layer_size, self.output_layer_size])
        w = w / np.sum(w, axis=0)
        
        
        return v, w
        
    def forward(self, input_out):
        v = self.v
        w = self.w
        # calculate hidden layer
        hidden_in = v.T @ input_out.reshape(-1,1)
        hidden_out = np.maximum(hidden_in, 0)
        
        # calculate output layer
        output_in = w.T @ hidden_out
        e_output_in = np.exp(output_in)
        output_out = e_output_in / np.sum(e_output_in)
              
        return hidden_out, output_out

    def backward(self, input_out, label):
        hidden_out, output_out = self.forward(input_out)

        # update wjk: output -> hidden
        delta_o = label.reshape(-1, 1) - output_out
        delta_w = hidden_out @ delta_o.T

        
        # update vij: hidden -> input
        delta_h = np.zeros(self.hidden_layer_size)
        for j in range(self.hidden_layer_size):
            for k in range(self.output_layer_size):
                temp_hidden_in_j = np.sum(self.v[:, j] * input_out)
                if (temp_hidden_in_j > 0):
                    delta_h[j] += delta_o[k] * self.w[j, k]

        delta_v = input_out.reshape(-1, 1) @ delta_h.reshape(1,-1)      

        return delta_v, delta_w
      
    def train(self, data, labels, learning_rate, threshold, max_iter):  #BP,SGD
        iter = 0
        change = threshold + 1
        while (change >= threshold and iter < max_iter):
            # SGD
            # print(iter)
            sample_select = np.random.randint(data.shape[0])
            input_out = data[sample_select]
            label = labels[sample_select]
            delta_v, delta_w = self.backward(input_out, label)
            self.w += learning_rate * delta_w
            self.v += learning_rate * delta_v
            iter += 1
            
    def test(self, data, labels):
        num_correct = 0
        num_total = data.shape[0]
        
        for i in range(data.shape[0]):
            hidden_out, output_out = self.forward(data[i])
            a = classify(output_out)
            b = classify(labels[i])
            '''print(a, "output", output_out)
            print(b,"label",labels[i])'''
            
            if a == b:
                num_correct += 1
        
        correct_rate = num_correct / num_total
        
        print("正确率：",correct_rate)
        return correct_rate


if __name__ == '__main__':
            
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

    mynn = neural_network_cpu(64, 100, 10)

    time_start = datetime.datetime.now()
    mynn.train(data_train, labels_train, 0.01, 1, 10000)
    time_total = datetime.datetime.now() - time_start

    print(mynn.w)
    print(mynn.v)

    mynn.test(data_train, labels_train)


    print("time", time_total)
