import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, x, y, learning_rate=0.1, bias=0):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.bias = bias

        self.weights = np.zeros((x.shape[1], 1))
        self.r = np.max(np.linalg.norm(x, axis=1))
        
        self.error_rates = []
        self.weights_history = []
        self.bias_history = []
        
        
    def fit(self, verbose=False):                
        iteration = 1
        while True:
            if verbose:
                print(f'[EPOCH]:\t{iteration}')
            self.error_rates.append(self.calculate_err_rate())
            
            for idx, sample in enumerate(x):
                self.weights_history.append(np.copy(self.weights))
                self.bias_history.append(np.copy(self.bias))
                    
                if np.sign(np.dot(sample, self.weights) - self.bias) != y[idx]:
                    self.weights += y[idx] * self.learning_rate * sample.reshape(len(sample), 1)
                    self.bias -= self.learning_rate * y[idx] * self.r ** 2
            
            
            if verbose:
                print(f'Weights: {self.weights.T}')
                print(f'Bias: {self.bias}')
                self.plot_decision_boundary()
            iteration += 1
            if np.all(np.sign(np.dot(x, self.weights) - self.bias) == y):
                self.error_rates.append(self.calculate_err_rate())
                break
    
    def predict(self, x):
        return np.sign(np.dot(x, self.weights) - self.bias)
    
    def calculate_err_rate(self):
        classification_results = np.sign(np.dot(self.x, self.weights) - self.bias) == self.y
        return np.sum(np.where(classification_results == False, 1, 0)) / self.x.shape[0] * 100
        
    def plot_decision_boundary(self):     
        # Plot the samples
        h = plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
        
        if np.all(self.weights == 0):
            print("[INFO]:\tWeight vector is a zero vector, hence no decision boundary.\n")
        else:
            if self.weights[1] == 0:
                plt.plot([0, 0], [plt.ylim()[0], plt.ylim()[1]])
            else:
                f = lambda x : float(-(self.weights[0]/ self.weights[1]) * x + (self.bias / self.weights[1]))
            
                x1, x2 = plt.xlim()
                plt.plot([x1, x2], [f(x1), f(x2)])        

        plt.quiver(0, 0, self.weights[0], self.weights[1], angles='xy', scale_units='xy', scale=1)
        plt.grid()

        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        plt.show()
