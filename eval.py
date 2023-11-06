# evaluate models 

from simple_model import SimpleModel
from loader import Loader
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    loader = Loader()
    test_loader = loader.get_test_loader(batch_size=32)
    model = SimpleModel()
    error = 0
    for batch in test_loader:
        x, y = batch
        y_hat = model(x)
        # reduce error over the batch dimension
        for i in range(len(y_hat)):
            error += mean_squared_error(y_hat[i], y[i])
    # save error to file
    with open("eval.txt", "w") as f:
        f.write("cumulative MSE of SimpleModel on test set: " + str(error))