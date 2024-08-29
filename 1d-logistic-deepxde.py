import deepxde as dde
import numpy as np

R = 1

def pde(t,f):
    df_dt = dde.grad.jacobian(f,t)

    pde_loss = df_dt - R * t * (1-t)

    return pde_loss

# def soln(t,f):

geom = dde.geometry.Interval(0,1.5)

def IC_boundary(x,on_boundary):
    return on_boundary and np.isclose(x[0],0)

IC_t_0 = dde.icbc.DirichletBC(geom, lambda t: 1, IC_boundary)

x_train = np.array([[1.1437e-04],
        [1.4676e-01],
        [3.0233e-01],
        [4.1702e-01],
        [7.2032e-01]])
y_train = np.array([[1.0000],
        [1.0141],
        [1.0456],
        [1.0753],
        [1.1565]])

data = dde.data.TimePDE(
    geom,
    pde,
    [IC_t_0],
    num_domain=200,
    num_boundary=2,
    num_test=100
)

data.add_anchors(np.vstack((x_train, y_train)))

net = dde.nn.FNN([1] + [16] + [32] + [16] + [1], "tanh",'Glorot uniform')

model = dde.Model(data,net)

model.compile("adam",lr=1e-2)
losshistory, train_state = model.train(iterations=2000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
f_eval = model.predict(x_eval.reshape(-1,1),operator=None)

fig, ax = plt.subplots(figsize=(12,5))
ax.scatter(x_train, y_train, label='Observation data',color='blue')
ax.scatter(t,numeric_solution_points, label='Collocation points',color='magenta',alpha=0.75)
ax.plot(x_eval,numeric_solution,label='Analytic solution',color='magenta',alpha=0.75)
ax.plot(x_eval,f_eval,label='NN solution', color='black')
ax.legend()
ax.set(title='Logisitic equation solved with NNs',xlabel='t',ylabel='f(t)')
plt.show()
