import numpy as np
import random
import sys
from tensorflow.examples.tutorials.mnist import input_data

#Activation Funcs    
class logistic(object):
    @classmethod
    def f(cls,x):
        return 1/(1+np.exp(-x))

    @classmethod
    def f_inv(cls,x):
        f = cls.f(x)
        if f >  0.99:
            return 9999
        return f/(1-f)

class relu(object):
    @classmethod
    def f(cls,x):
        def fij(xij):
            return max(0,xij)
        return np.vectorize(fij)(x)

    @classmethod
    def f_dash(cls,x):
        def fdij(xij):
            if xij > 0:
                return 1
            else:
                return 0
        return np.vectorize(fdij)(x)

class softmax(object):
    @classmethod
    def fu(cls,u):
        result = np.array(u)
        s = 0
        for i in range(len(u)):
            ui = np.exp(u[i])
            result[i] = ui
            s += ui
        result = result/s
        return result

    @classmethod
    def f(cls,batch):
        return matrix_f(cls.fu)(batch)

    @classmethod
    def fu_dash(cls,u):
        fu = cls.fu(u)
        y = fu*(1-fu)
        return y

    @classmethod
    def f_dash(cls,batch):
        return matrix_f(cls.fu)(batch)

def status_indicator(n,total):
    finish = 30 
    percent = n/total
    sharps = int(percent*finish)
    print("\033[2K\033[0G",end="")
    print("["+"#"*sharps+" "*(finish-sharps)+"] "+str(percent*100)+"%",end="")
    if percent >= 1:
        print("")
    sys.stdout.flush()
    return

def matrix_f(f):
    def fm(batch):
        batch = np.array(batch)
        result = np.zeros_like(batch)
        for i in range(len(batch[0])):
            result[:,i] = f(batch[:,i])
        return result
    return fm

class DeepLearning(object):
    def __init__(self,network,train_steps = 10,test_output = False,lerning_rate = 0.01):
        self.network = network
        self.train_steps = train_steps
        self.lerning_rate = lerning_rate
        self.minbatch_size = 20
        self.test_output = test_output

    def summary(self):
        print("-------------------------------")
        print("#Deep Lerning Summary#")

        print("Training Steps: "+str(self.train_steps))
        print("Lerning Rate: "+str(self.lerning_rate))
        print("Minibatch Size: "+str(self.minbatch_size))
        print("-------------------------------")

    def train(self,batch_x,batch_y):
        if self.test_output:
            print("")
            print("Training Progress")
            status_indicator(0,self.train_steps)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        for step in range(self.train_steps):
            mb_x,mb_y = self.generate_minibatch(batch_x,batch_y)
            mb_xt = mb_x.T
            mb_yt = mb_y.T
            y = self._fpropagation(mb_xt)
            w_new = self._bpropagation(y,mb_yt)
            self.network.weights = w_new
            if self.test_output:
                status_indicator(step+1,self.train_steps)

    def generate_minibatch(self,batch_x,batch_y):
        input_nodes_n = len(batch_x[0])
        output_nodes_n = len(batch_y[0])
        batch_size = len(batch_x)
        mb_x = np.zeros((self.minbatch_size,input_nodes_n))
        mb_y = np.zeros((self.minbatch_size,output_nodes_n))
        for i in range(self.minbatch_size):
            ri = np.random.randint(0,batch_size-1)
            mb_x[i] = batch_x[ri]
            mb_y[i] = batch_y[ri]
        return (mb_x,mb_y)
 
    def predict(self,x):
        z = np.array(x).T
        counter = 0
        for w in self.layers.weights_matrixes():
            z = w.dot(z)
            s = 0
            for i in range(len(z)):
                s += np.exp(z[i])
            tes = 0
            for i in range(len(z)):
                z[i] = np.exp(z[i])/s
                tes += z[i]
            counter += 1
        return z 
    
    def predict_batch(self,batch_x):
        u = np.array(batch_x).T
        for i in range(len(self.network.layers)-1):
            f = self.network.layers[i].afunc.f
            w = self.network.weights[i]
            u =f(w.dot(u))

        return u.T

    def _fpropagation(self,batch_x):
        u = [np.array(batch_x)]
        counter = 0
        pairs = list(self.network.wb_pairs())
        for i in range(self.network.length()-1):
            w,b = pairs[i]
            ones = np.ones((1,len(batch_x[0])))
            bias = b.dot(ones)
            af = self.network.layers[i].afunc.f
            u_next = af(w.dot(u[counter])+bias)
            u.append(u_next)
            counter += 1
        return u

    def _bpropagation(self,u,batch_y):
        deltas = []
        for ui in u:
            deltas.append(np.zeros_like(ui))
        oi = len(deltas)-1
        n = len(batch_y[0])
        deltas[oi] = (u[oi] - batch_y)/n
        weights = self.network.weights
        bias = self.network.biases
        batch_n = len(batch_y)
        for i in range(1,oi+1):
            l = oi-i
            w_inv = weights[l].T
            wd = w_inv.dot(deltas[l+1])
            afi = self.network.layers[l].afunc.f_dash
            fu = afi(u[l])
            deltas[l] = np.asarray(wd) * np.asarray(fu)
        for i in range(oi):
            ones = np.zeros((len(deltas[i+1][0]),1))
            dw = np.array(deltas[i+1].dot(u[i].T))
            db = np.array(deltas[i+1].dot(ones))
            weights[i] = weights[i] - (self.lerning_rate*dw + 0.01*weights[i])
            bias[i] = bias[i]-self.lerning_rate*db
        return weights
            


class Network(object):
    def __init__(self,layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(self.layers)-1):
            layer = self.layers[i]
            wi = np.zeros((self.layers[i+1].size,self.layers[i].size))
            biasi  = np.zeros((self.layers[i+1].size,1))
            self.init_weight(wi)
            self.weights.append(wi)
            self.biases.append(biasi)

    def init_weight(self,w):
        for i in range(len(w)):
            for j in range(len(w[i])):
                w[i][j] = np.random.normal(0,0.2)
        return w

    def wb_pairs(self):
        return zip(self.weights,self.biases)

    def length(self):
        return len(self.layers)

    def print(self):
        for l in self.layers:
            print(l.size)
            print("->")
        print("output")


class Layer(object):
    def __init__(self,size,afunc):
        self.size = size
        self.afunc = afunc

    def print(self):
        print(self.size)

if __name__=="__main__":
    image_size = 28*28
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    batch_x,batch_y = mnist.train.next_batch(100)
    layers = Network([Layer(784,softmax),Layer(10,None)])
    dl = DeepLearning(layers,train_steps = 200,lerning_rate = 0.01,test_output = True)
    dl.summary()
    dl.train(batch_x,batch_y)
    test_x = mnist.test.images
    test_y = mnist.test.labels
    output = dl.predict_batch(test_x)
    total = 0
    correct = 0
    for i in range(len(output)):
        prd = np.argmax(output[i])
        ans = np.argmax(test_y[i])
        total += 1
        if prd == ans:
            correct += 1
    print("Correct Ratio")
    print(correct/total*100,end="%\n")
