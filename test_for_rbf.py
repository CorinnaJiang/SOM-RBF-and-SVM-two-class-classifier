import optuna
from rbf_o import *
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_slice
import pickle

X_test = loadmat('dataset/data_test.mat')['data_test']
X_train_full = loadmat('dataset/data_train.mat')['data_train']
y_train_full = loadmat('dataset/label_train.mat')['label_train']

kf = StratifiedKFold(n_splits=6,shuffle=True,random_state=1)
best_net = 0
for fold,(train_index,val_index) in enumerate(kf.split(X_train_full,y_train_full)):
    X_train,X_val = X_train_full[train_index],X_train_full[val_index]
    y_train,y_val = y_train_full[train_index],y_train_full[val_index]


    data = {'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test}


def objective(trial: optuna.trial.Trial):
    params ={
            'lr': trial.suggest_loguniform('lr',1e-8, 1e-3),
            'weight_scale': trial.suggest_uniform('weight_scale', 0, 1e-5),
            'no_of_hidden': trial.suggest_int('no_of_hidden', 100,330),
            'center_sigma': trial.suggest_uniform('center_sigma', 0.3, 0.8),
            "reg_rbf":trial.suggest_uniform('reg_rbf', 0,0.8)
             }

    net = RBF(
        input_dim=33,
        num_classes=1,
        inputdatasize=X_train.shape[0],
        num_iterations=5000,
        data=data,
        datafull=X_train_full,
        kwargs=params,
    )

    for step in range(10):
        loss = net.train(X_train, y_train)

        # Report intermediate objective value.
        trial.report(loss, step)
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.TrialPruned()
    with open("{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(net, fout)

    return loss

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=300)


trial_ = study.best_trial
print(f'min loss: {trial_.value}')
print(f'best params: {trial_.params}')


with open("{}.pickle".format(study.best_trial.number), "rb") as fin:
    best_net = pickle.load(fin)
print(best_net.train(X_val,y_val))

# plot
plot_intermediate_values(study).show()
plot_optimization_history(study)
plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_contour(study, params=['lr',
            'weight_scale',
            'no_of_hidden',
            'center_sigma',
            'reg_rbf']).show()
plot_slice(study, params=['lr'])
plot_slice(study, params=['weight_scale']).show()
plot_slice(study, params=['no_of_hidden']).show()
plot_slice(study, params=['center_sigma']).show()
plot_slice(study, params=['reg_rbf']).show()
