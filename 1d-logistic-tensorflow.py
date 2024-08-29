# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For NN
import tensorflow as tf
from scipy import optimize
from tensorflow.keras.optimizers import Adam  # , SGD, Adadelta, Adagrad, Nadam
from tensorflow import keras
from tensorflow.python.keras import backend
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set seed for reproducability
tf.random.set_seed(1234)
np.random.seed(1234)

tf.config.run_functions_eagerly(True)

class PhysicsInformedNN(object):
    def __init__(self, X, Y, X_domain, layers, acti="tanh", tf_epochs=10000, lr=0.04,lbfgs_=True):
        """
        The PINN.
        Takes in input (and respective output) sets as X and Y respectively.
        Takes in hyper-parameters including layer structure, activation functions, number of epochs and
        learning rate for Adam optimizer.
        """
        # Set dtype to float32
        tf.keras.backend.set_floatx("float32")

        self.X = tf.convert_to_tensor(X)
        self.Y = tf.convert_to_tensor(Y)

        self.X_domain = tf.convert_to_tensor(X_domain)

        self.X_boundary = np.array([[0]],dtype=np.float32)
        self.Y_boundary = np.array([[1]],dtype=np.float32)
        self.X_boundary = tf.convert_to_tensor(self.X_boundary)
        self.Y_boundary = tf.convert_to_tensor(self.Y_boundary)

        # # Get the input bounds:
        # self.ub = self.X_domain.max(0)
        # self.lb = self.X_domain.min(0)

        self.R = 1.0

        ## Model
        # Using the keras library to build the underlying model:
        self.model = tf.keras.Sequential(name='PINN')
        self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        # self.model.add(tf.keras.layers.Lambda(
        #     lambda X: 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0))
        self.initializer = "he_normal"
        if acti == "tanh":
            self.initializer = "glorot_normal"
        for width in layers[1:-1]:
            self.model.add(tf.keras.layers.Dense(
                width, activation=acti,
                kernel_initializer=self.initializer))
        self.model.add(tf.keras.layers.Dense(
            layers[-1], activation=None,
            kernel_initializer=self.initializer))

        # Set keras optimizer
        self.opt = Adam(learning_rate=lr, decay=0)

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        # Hyper-parameter saving
        self.tf_epochs = tf_epochs

        # set l-bfg-s optimization
        self.lbfgs_switch = lbfgs_

    # The actual PINN
    def ns_net(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we'll need later, x and t
            tape.watch(self.X_domain)
            # Getting the prediction
            Y = self.model(self.X_domain)

            df_dt = tape.gradient(Y,self.X_domain)

        # Letting the tape go
        del tape

        return df_dt

    def boundary(self):
        Y_pred_boundary = self.model(self.X_boundary)
        return self.Y_boundary, Y_pred_boundary

    @tf.autograph.experimental.do_not_convert
    # Custom loss function
    def loss(self, Y_actual, Y_pred):
        """
        Neural Network custom loss function.
        Takes in two inputs, Y_actual and Y_pred
        Outputs a scalar loss term, which is based on the Navier-Stokes equations.
        MSE0 returns the error between the predicted Y and actual Y.
        MSE1 returns the error term of the conservation of volume equation (flow is assumed incompressible).
        MSE2 and MSE3 represent the horizontal and vertical conservation of momentum equations.
        """

        df_dt = self.ns_net()

        Y_bound_actual, Y_bound_pred = self.boundary()

        t = self.X_domain

        mse_0 = tf.reduce_mean(tf.square(Y_actual - Y_pred))

        mse_1 = tf.reduce_mean(tf.square(df_dt - self.R * t * (1-t)))

        mse_2 = tf.reduce_mean(tf.square(Y_bound_actual - Y_bound_pred))

        mse = mse_0 + mse_1 + mse_2

        return mse

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    def fit_pinn(self):
        self.tf_optimization(self.X, self.Y)
        print("TF done")
        if self.lbfgs_switch:
            self.lbfgs_optimization(self.X, self.Y)
            print("L-BFG-S done")

    def tf_optimization(self, X, Y):
        for epoch in range(self.tf_epochs):
            loss_value = self.tf_optimization_step(X, Y)
            if epoch % 10 == 0:
                tf.print(f"Epoch {epoch}: Loss = {self.loss(Y, self.model(X))}")

    @tf.function
    def tf_optimization_step(self, X, Y):
        loss_value, grads = self.grad(X, Y)
        self.opt.apply_gradients(
            zip(grads, self.wrap_training_variables()))
        return loss_value

    @tf.function
    def grad(self, X, Y):
        with tf.GradientTape() as tape:
            loss_value = self.loss(Y, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def lbfgs_optimization(self, X, Y):
        """
        The lbfgs optimization function makes use of scipy's in-built optimizer to reduce the loss function.
        Scipy's optimizer takes in arguments:
        1. The function to be minimized
            a. For LBFGS, with Jac = True, this is assumed to output (f, g) representing the functional output
               and gradient respectively.
        2. The initial "guess"--this for us is the array of weights.p has type float32 that does not match type in
        Essentially, we wish to minimize the loss with respect to the weights.
        Notably, f, g and x must all be arrays with shape (n,)
        """
        # optimizer options
        options = {'disp': True, 'maxfun': 50000, 'maxiter': 50000, 'maxcor': 50, 'maxls': 50,
                   'ftol': 1.0 * np.finfo(float).eps, 'gtol': 1.0 * np.finfo(float).eps}

        # define x0 as an array with shape (n,)
        x0 = self.flatten_trainable_variables()
        self.results = optimize.minimize(fun=self.val_and_grad, x0=x0, args=(X, Y),
                                               jac=True, method='L-BFGS-B', options=options)

    def flatten_trainable_variables(self):
        wtv = pinn.wrap_training_variables()
        w = np.array([])
        for i in wtv:
            w = np.append(w, i.numpy())
        return w

    def apply_trainable_variables(self, w):
        for i, layer in enumerate(self.model.layers[1:]):
            if not layer.trainable_variables:
                continue
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i + 1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    def val_and_grad(self, w, X, Y):
        self.apply_trainable_variables(w)
        with tf.GradientTape() as tape:
            loss_value = self.loss(Y, self.model(X))
            grads = tape.gradient(loss_value, self.wrap_training_variables())
        k = np.array([])
        for i in range(len(grads)):
            k = np.append(k, tf.reshape(grads[i], [-1]).numpy())
        return loss_value.numpy(), k

    def predict(self,X_test):
        return self.model.predict(X_test)

    def save_weights(self,file_path):
        self.model.save_weights(file_path)
        print(f'Weights and Biases are saved in location:\n{file_path}')

    def load_weights(self,filename):
        self.model.load_weights(filename)

if __name__ == "__main__":
    # prepare training data
    inputNo = 1
    outputNo = 1
    iterations = 2000
    noLayers = 3
    noNeurons = 50
    acti = 'tanh'
    lr = 1e-2

    X = np.array([[1.1437e-04],
        [1.4676e-01],
        [3.0233e-01],
        [4.1702e-01],
        [7.2032e-01]],dtype=np.float32)

    Y = np.array([[1.0000],
        [1.0141],
        [1.0456],
        [1.0753],
        [1.1565]],dtype=np.float32)

    # layers = [inputNo] + [noNeurons] * noLayers + [outputNo]
    layers = [inputNo] + [16] + [32] + [16] + [outputNo]
    # for _ in range(noLayers):
    #     layers.append(noNeurons)
    # layers.append(outputNo)

    domain = [0,1.5]

    t = np.linspace(domain[0],domain[1],101,dtype=np.float32).reshape(-1,1)

    # PINN initiate, train and save
    pinn = PhysicsInformedNN(X, Y, t, layers, lr=lr, acti=acti, tf_epochs=iterations, lbfgs_=False)
    pinn.fit_pinn()

    # plot graph
    R = 1
    def logistic_eq_fn(t,f):
        return R * t * (1 - t)

    domain = [0,1.5]
    ft0 = 1.0

    x_eval = np.linspace(domain[0],domain[1],100)
    t = np.linspace(domain[0],domain[1],10)
    numeric_solution = solve_ivp(
        logistic_eq_fn, domain, [ft0],t_eval=x_eval
    ).y.T
    numeric_solution_points = solve_ivp(
        logistic_eq_fn, domain, [ft0],t_eval= t
    ).y.T
    f_eval = pinn.predict(x_eval.reshape(-1,1))

    fig, ax = plt.subplots(figsize=(12,5))
    ax.scatter(X, Y, label='Observation data',color='blue')
    ax.scatter(t,numeric_solution_points, label='Collocation points',color='magenta',alpha=0.75)
    ax.plot(x_eval,numeric_solution,label='Analytic solution',color='magenta',alpha=0.75)
    ax.plot(x_eval,f_eval,label='NN solution', color='black')
    ax.legend()
    ax.set(title='Logisitic equation solved with NNs',xlabel='t',ylabel='f(t)')
    plt.show()
