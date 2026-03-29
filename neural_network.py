import numpy as np 

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)*np.sqrt(2/input_size)
        self.bias = np.zeros((1, output_size))
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad):
        self.d_weights = np.dot(self.x.T, grad)
        self.d_bias = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)

class Relu:
    def __init__(self):
        self.x = None

    def forward (self, x):
        self.x = x
        return np.maximum(0,x)

    def backward (self, grad):
        return grad * (self.x > 0)    

class Softmax:
    def __init__(self):
        self.out = None

    def forward(self, x):
        exp_x = np.exp(x-np.max(x, axis=1, keepdims=True))
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        return self.out * (grad - np.sum(grad * self.out, axis=1, keepdims=True))

class CrossEntropyLoss:
    def __init__(self):
        self.pred = None
        self.y = None

    def forward(self, pred, y):
        self.pred = pred
        self.y= y
        pred_clipped = np.clip(pred, 1e-12, 1-1e-12)
        loss = -np.mean(np.sum(y * np.log(pred_clipped), axis=1))
        return loss

    def backward(self):
        pred_clipped = np.clip(self.pred, 1e-12, 1-1e-12)
        return -self.y / pred_clipped / self.pred.shape[0]

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.loss = CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self):
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename):
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)

def SGD(model, lr):
    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            layer.weights -= lr*layer.d_weights
            layer.bias -= lr*layer.d_bias