import numpy as np
from minisom import MiniSom
from sklearn.metrics.pairwise import rbf_kernel
import pickle

class RBF(object):

    def __init__(
            self,
            input_dim=33,
            num_classes=10,
            inputdatasize = 330,
            num_iterations = 100,
            data = {},
            datafull=[],
            **kwargs

    ):

        self.params = {}
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.data_full = datafull
        self.lr = kwargs.pop("lr",0.0001)
        self.num_iterations=num_iterations
        self.reg_rbf = kwargs.pop("reg_rbf",0.5)
        self.input_dim = input_dim
        self.weight_scale = kwargs.pop("weight_scale",1e-3)
        self.no_of_hidden = kwargs.pop("no_of_hidden",50)
        self.reg_rbf = kwargs.pop("reg_rbf",1e-3)
        self.centroid = np.random.randn(self.no_of_hidden, input_dim)
        self.center_sigma = kwargs.pop("center_sigma",0.4)
        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.verbose = kwargs.pop("verbose", True)
        self.hidden_output = np.random.randn(inputdatasize, self.no_of_hidden)
        self.score = 0
        self.params['W1'] = self.weight_scale * np.random.randn(self.no_of_hidden, 1) #1N
        self.params['b1'] = np.zeros(num_classes)


        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.best_val_acc = 0
        self.best_params = {}

        self.loss_history.clear()
        self.train_acc_history.clear()
        self.val_acc_history.clear()

    def setup_center(self):
        # print('set up center')
        # SOM select neuron
        som = MiniSom(1, self.no_of_hidden, self.input_dim, sigma=0.1, learning_rate=.8,
                      neighborhood_function='gaussian')  # initialization of 6x6 SOM
        # som.pca_weights_init(self.data_full)
        som.train_batch(self.data_full, 10000, verbose=True)
        for i in som.get_weights():
            self.centroid = i
        return self.centroid


    def hidden_wetight(self, data): # 1*N and m*N = 1*N n*hidden hidden*1= n*1
        self.hidden_output = rbf_kernel(data,self.centroid,gamma=self.center_sigma)
        return self.hidden_output


    def train(self, X, y):
        for i in range(self.num_iterations):
            W1, b1 = self.params['W1'], self.params['b1']
            x = self.hidden_wetight(X)
            N, D = X.shape
            scores = x.dot(W1)
            loss = 0.5 * np.sum(np.square(y-scores), axis=0, keepdims=True)
            loss /= N
            loss += self.reg_rbf * (np.sum(W1 * W1))   # regularization
            dW1 = (((scores-y).T).dot(x)).T
            db1=0
            # store the parameters
            train_acc = self.check_accuracy(
                self.X_train, self.y_train
            )
            val_acc = self.check_accuracy(
                self.X_val, self.y_val
            )
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            self.loss_history.append(loss)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {}
                for k, v in self.params.items():
                    self.best_params[k] = v.copy()
            W1 -= self.lr * dW1
        self._save_checkpoint()

        self.params = self.best_params

        return scores

    def check_accuracy(self,X,y):
        acc = (self.predict(X) == y).mean()
        return acc

    def predict(self,X):
        scores = self.hidden_wetight(X).dot(self.params['W1'])
        predict = np.where(scores > 0, 1, -1)
        return predict

    def check_final_accuracy(self,X,y,params):
        final_acc = (self.final_predict(X,params)== y).mean()
        return final_acc

    def final_predict(self,X,params):
        scores = self.hidden_wetight(X).dot(params)
        final_predict = np.where(scores > 0, 1, -1)
        return final_predict

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            'lr': self.lr,
            'weight_scale':self.weight_scale,
            'no_of_hidden':self.no_of_hidden,
            'center_sigma': self.center_sigma,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.num_iterations)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)



