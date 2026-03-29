import numpy as np
from scipy.ndimage import shift as ndimage_shift
from neural_network import LinearLayer, Relu, Softmax, CrossEntropyLoss, NeuralNetwork, Adam

# ── Improvement 3: Data Augmentation ──────────────────────────────────────────
def augment(X, y, shift_range=2):
    """Randomly shift each image by up to ±shift_range pixels in x and y.
    This forces the model to be position-invariant, greatly reducing real-world error.
    """
    X_aug = np.empty_like(X)
    for i in range(len(X)):
        img = X[i].reshape(28, 28)
        dx, dy = np.random.randint(-shift_range, shift_range + 1, size=2)
        shifted = ndimage_shift(img, [dx, dy], cval=0.0)
        X_aug[i] = shifted.flatten()
    return X_aug, y


def training_loop(model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    for epoch in range(epochs):

        # Augment THEN shuffle every epoch so the model sees different shifts each time
        X_aug, y_aug = augment(X_train, y_train)            # Improvement 3
        indices = np.random.permutation(len(X_aug))
        X_aug  = X_aug[indices]
        y_aug  = y_aug[indices]

        epoch_loss  = 0
        num_batches = len(X_aug) // batch_size

        for i in range(num_batches):
            batch_x = X_aug[i*batch_size:(i+1)*batch_size]
            batch_y = y_aug[i*batch_size:(i+1)*batch_size]

            pred = model.forward(batch_x)
            loss = model.loss.forward(pred, batch_y)
            epoch_loss += loss
            model.backward()
            optimizer.step()                                 # Improvement 1 (Adam)

        # Validate on original (unaugmented) data
        val_pred = model.forward(X_val)
        val_acc  = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}, Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True).astype(int)
    X_val   = np.load('X_val.npy',   allow_pickle=True)
    y_val   = np.load('y_val.npy',   allow_pickle=True).astype(int)

    num_classes = 10
    y_onehot     = np.zeros((len(y_train), num_classes))
    y_onehot[np.arange(len(y_train)), y_train] = 1

    y_val_onehot = np.zeros((len(y_val), num_classes))
    y_val_onehot[np.arange(len(y_val)), y_val] = 1

    # Improvement 2: Wider hidden layer (784 → 256 → 128 → 10)
    model = NeuralNetwork([
        LinearLayer(784, 256),
        Relu(),
        LinearLayer(256, 128),
        Relu(),
        LinearLayer(128, 10),
        Softmax()
    ])

    # Improvement 1: Adam optimizer (lr=0.001 works well with Adam)
    optimizer = Adam(model.layers, lr=0.001)

    training_loop(model, optimizer, X_train, y_onehot, X_val, y_val_onehot, epochs=30, batch_size=64)

    model.save('model.pkl')