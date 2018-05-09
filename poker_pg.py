import pickle
import numpy as np


def learning_rate_scheduler(num_games):
    if num_games < 5000:
        return .001
    elif num_games < 10000:
        return .0005
    elif num_games < 20000:
        return .00025
    elif num_games < 40000:
        return .000125
    elif num_games < 100000:
        return .00005
    else:
        return .00001

class Network(object):
    def __init__(self):
        """NN initialization, set architecture hyperparameters here."""
        self.h1 = 128 # size of hidden layer1
        self.h2 = 64 #size of hidden layer2
        self.h3 = 32 #size of hidden layer 3
        self.D = 112 #data dimension
        self.K = 6 #num classes
        self.W = .1*np.random.randn(self.D,self.h1)
        self.b = np.zeros((1,self.h1))
        self.W2 = .1*np.random.randn(self.h1,self.h2)
        self.b2 = np.zeros((1,self.h2))
        self.W3= .1*np.random.randn(self.h2, self.h3)
        self.b3= np.zeros((1,self.h3))
        self.W4 = .1*np.random.randn(self.h3, self.K)
        self.b4= np.zeros((1,self.K))

        # some hyperparameters
        self.step_size = .00001
        self.reg = 1e-3 # regularization strength
        self.alpha = .1 #Leaky RELU  parameter

    def save_network(self, save_name):
        pickle.dump(self, open(save_name, 'wb'))

    def choose_action(self, input_vec):
        """Use network weights to compute a probabililty distribution over actions
         based on the game state given by the input_vec.
         See bot.py for more details on the input_vec."""
        alpha = self.alpha
        hidden1_out = np.dot(input_vec, self.W) + self.b
        hidden_layer1 = np.maximum(alpha*hidden1_out, hidden1_out) # Leaky ReLU activation
        hidden2_out = np.dot(hidden_layer1, self.W2) + self.b2
        hidden_layer2 = np.maximum(alpha*hidden2_out, hidden2_out) #Leaky ReLU activation
        hidden3_out = np.dot(hidden_layer2, self.W3) + self.b3
        hidden_layer3 = np.maximum(alpha*hidden3_out, hidden3_out) #Leaky ReLU activation
        scores = np.dot(hidden_layer3, self.W4) + self.b4

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        y = np.argmax(np.random.multinomial(1, np.squeeze(probs)))
        return y

    def update_weights(self, X, y, reward):
        """Based on the reward given from the game output, adjust the likelihood
        of selecting action y based on game state X using policy gradients."""
        num_examples = X.shape[0]
        alpha = self.alpha
        # evaluate class scores, [N x K]
        hidden1_out = np.dot(X, self.W) + self.b
        hidden_layer1 = np.maximum(alpha*hidden1_out, hidden1_out) # Leaky ReLU activation
        hidden2_out = np.dot(hidden_layer1, self.W2) + self.b2
        hidden_layer2 = np.maximum(alpha*hidden2_out, hidden2_out) #Leaky ReLU activation
        hidden3_out = np.dot(hidden_layer2, self.W3) + self.b3
        hidden_layer3 = np.maximum(alpha*hidden3_out, hidden3_out) #Leaky ReLU activation
        scores = np.dot(hidden_layer3, self.W4) + self.b4
        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        #print(probs)

        # compute the loss: average cross-entropy loss and regularization
        #correct_logprobs = -np.log(probs[range(num_examples),y])
        #data_loss = np.sum(correct_logprobs)/num_examples
        #reg_loss = 0.5*self.reg*np.sum(self.W*self.W) + 0.5*self.reg*np.sum(self.W2*self.W2)
        #loss = data_loss + reg_loss


        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        dW4 = np.dot(hidden_layer3.T, dscores)
        db4 = np.sum(dscores, axis=0, keepdims=True)

        # next backprop into hidden layer
        dhidden3 = np.dot(dscores, self.W4.T)
        # backprop the ReLU non-linearity
        dhidden3[hidden_layer3 <= 0] = dhidden3[hidden_layer3 <= 0] * alpha

        # backprop into parameters W2 and b2
        dW3 = np.dot(hidden_layer2.T, dhidden3)
        db3 = np.sum(dhidden3, axis=0, keepdims=True)

        dhidden2 = np.dot(dhidden3, self.W3.T)
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
        dW4 += reg * self.W4
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
        self.W4 += -step_size * dW4 * reward
        self.b4 += -step_size * db4 * reward
