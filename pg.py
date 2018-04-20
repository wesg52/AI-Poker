import numpy as np

def reward(inpt, out):
    inp = np.squeeze(inpt)
    if inp[1] == 1.:
        if out == 1:
            return 1.
        else:
            return -1.
    else:
        if out == 1:
            return -1.
        else:
            return 1.


class Network(object):
    def __init__(self):
        # initialize parameters randomly
        self.h1 = 8 # size of hidden layer1
        self.h2 = 4 #size of hidden layer2
        self.D = 3 #data dimension
        self.K = 2 #num classes
        self.W = np.random.randn(self.D,self.h1)
        self.b = np.zeros((1,self.h1))
        self.W2 =np.random.randn(self.h1,self.h2)
        self.b2 = np.zeros((1,self.h2))
        self.W3= np.random.randn(self.h2, self.K)
        self.b3= np.zeros((1,self.K))
        # some hyperparameters
        self.step_size = .01
        self.reg = 1e-3 # regularization strength
        self.alpha = .1 #Leaky RELU  parameter

# gradient descent loop

    def update_weights(self, X, y, reward, i):
        num_examples = X.shape[0]
        alpha = self.alpha
        # evaluate class scores, [N x K]
        hidden1_out = np.dot(X, self.W) + self.b
        hidden_layer1 = np.maximum(alpha*hidden1_out, hidden1_out) # Leaky ReLU activation
        hidden2_out = np.dot(hidden_layer1, self.W2) + self.b2
        hidden_layer2 = np.maximum(alpha*hidden2_out, hidden2_out) #Leaky ReLU activation
        scores = np.dot(hidden_layer2, self.W3) + self.b3
        print(scores)
        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        correct_logprobs = -np.log(probs[range(num_examples),y])
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5*self.reg*np.sum(self.W*self.W) + 0.5*self.reg*np.sum(self.W2*self.W2)
        loss = data_loss + reg_loss
        if i % 100 == 0:
            print ("iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        print('Prob: ', probs)
        print('Dscore: ', dscores)
        dW3 = np.dot(hidden_layer2.T, dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)

        # next backprop into hidden layer
        dhidden2 = np.dot(dscores, self.W3.T)
        # backprop the ReLU non-linearity
        dhidden2[hidden_layer2 <= 0] = dhidden2[hidden_layer2 <= 0] * alpha

        # backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer1.T, dhidden2)
        db2 = np.sum(dhidden2, axis=0, keepdims=True)

        dhidden1 = np.dot(dhidden2, self.W2.T)
        # backprop the ReLU non-linearity
        dhidden1[hidden_layer1 <= 0] = dhidden1[hidden_layer1 <= 0] * alpha

        # finally into W,b
        dW = np.dot(X.T, dhidden1)
        db = np.sum(dhidden1, axis=0, keepdims=True)

        # add regularization gradient contribution
        reg = self.reg
        dW3 += reg * self.W3
        dW2 += reg * self.W2
        dW += reg * self.W

        step_size = self.step_size
        # perform a parameter update
        old_W = self.W
        self.W += -step_size * dW * reward
        self.b += -step_size * db * reward
        self.W2 += -step_size * dW2 * reward
        self.b2 += -step_size * db2 * reward
        self.W3 += -step_size * dW3 * reward
        self.b3 += -step_size * db3 * reward
        #print('Deriv:', dW)
        #print('OLD:  ',  old_W)
        #print('NEW:  ', self.W)


def gen_data(Network,i):
    X = np.random.choice([0., 1.], size=(1,3))
    hidden1_out = np.dot(X, Network.W) + Network.b
    hidden_layer1 = np.maximum(Network.alpha*hidden1_out, hidden1_out) # Leaky ReLU activation
    hidden2_out = np.dot(hidden_layer1, Network.W2) + Network.b2
    hidden_layer2 = np.maximum(Network.alpha*hidden2_out, hidden2_out) #Leaky ReLU activation
    scores = np.dot(hidden_layer2, Network.W3) + Network.b3

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    y = np.argmax(np.random.multinomial(1, np.squeeze(probs)))
    r = reward(X, y)
    #print(X, y, r)
    Network.update_weights(X, y, r, i)
    return r

def train(Network):
    reward = 0
    for i in range(10000):
        reward = gen_data(Network,i )
    print(reward)

def test(Network):
    reward = 0
    for i in range(1000):
        reward += gen_data(Network,i)
    print(reward)

if __name__ == '__main__':
    NeuralNet = Network()
    b_bef = NeuralNet.b
    train(NeuralNet)
    test(NeuralNet)
