import numpy as np
from neural_network import LinearLayer, Relu, Softmax, CrossEntropyLoss, NeuralNetwork, SGD

def training_loop(model, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
    for epoch in range(epochs):

        #shuffling data between each epoch
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        num_batches = len(X_train)// batch_size

        for i in range(num_batches):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train[i*batch_size:(i+1)*batch_size]

            pred = model.forward(batch_x)
            loss = model.loss.forward(pred, batch_y)
            epoch_loss += loss
            model.backward()
            SGD(model, lr)

        # validation
        val_pred = model.forward(X_val)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}, Validation Accuracy: {val_acc:.4f}")
            



if __name__ ==  "__main__":
    X_train = np.load('X_train.npy',allow_pickle=True)
    y_train = np.load('y_train.npy',allow_pickle=True).astype(int)
    X_val = np.load('X_val.npy',allow_pickle=True)
    y_val = np.load('y_val.npy',allow_pickle=True).astype(int)
    X_test = np.load('X_test.npy',allow_pickle=True)
    y_test = np.load('y_test.npy',allow_pickle=True).astype(int)

    num_classes = 10
    y_onehot = np.zeros((len(y_train), num_classes))
    y_onehot[np.arange(len(y_train)), y_train] = 1

    y_val_onehot = np.zeros((len(y_val), num_classes))
    y_val_onehot[np.arange(len(y_val)), y_val] = 1

    model = NeuralNetwork([
        LinearLayer(784, 128),
        Relu(),
        LinearLayer(128, 64),
        Relu(),
        LinearLayer(64, 10),
        Softmax()
    ])

    training_loop(model, X_train, y_onehot, X_val, y_val_onehot, 30, 0.01, 32)

    model.save('model.pkl')
    