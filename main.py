from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from rbf_o import *
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import time

# load data
X_test = loadmat('dataset/data_test.mat')['data_test']
X_train_full = loadmat('dataset/data_train.mat')['data_train']
y_train_full = loadmat('dataset/label_train.mat')['label_train']

# Normalize
def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std

def Test_for_RBFNN():
    kf = StratifiedKFold(n_splits=6,shuffle=True,random_state=1)
    best_acc , k = 0, 0
    for fold,(train_index,val_index) in enumerate(kf.split(X_train_full,y_train_full)):
        X_train,X_val = X_train_full[train_index],X_train_full[val_index]
        y_train,y_val = y_train_full[train_index],y_train_full[val_index]

        data = {'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test}

        net= RBF(
            input_dim=33,
            num_classes=1,
            inputdatasize=X_train.shape[0],
            # lr= 4.289457971611662e-06,
            # lr=2.289457971611662e-04,
            # lr=2.5e-03,
            lr=2.5e-03,
            num_iterations= 700,
            data = data,
            datafull = X_train_full,
            reg_rbf=0,  # 0.5-0.9
            weight_scale=9.415128343773481e-09,  # <1e-10
            # weight_scale=0.0000,
            # no_of_hidden=117,
            no_of_hidden=200,
            verbose=True,
            checkpoint_name= 'Test',
            center_sigma=3# no a big deal
        )

        net.setup_center()
        net.train(X_train,y_train)
        acc_val = net.check_final_accuracy(X_val,y_val,net.params['W1'])
        print('val accuracy:',acc_val)

        k +=1
        if acc_val > best_acc:
            best_param = net.params['W1']
            if k >=6:
                df =np.array(net.final_predict(X_test,best_param))
                df.reshape(1,-1)
                # print("RBF Result",df)
                rbf_val_result = net.final_predict(X_val, best_param)
                rbf_test_result = net.final_predict(X_test,best_param)

                df = pd.DataFrame(rbf_test_result, columns=['RBF'])
                df.to_csv('result1.csv', sep=' ',index=False)

                # Output the classification report RBFNN
                print('-----------------Test_for_RBFNN-------------------')
                print(classification_report(y_val, rbf_val_result))
                print (net.train(X_val,y_val))

# plot the loss function and training and validation accuracy curve
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(np.array(net.loss_history).reshape(-1,1), 'o', color='green', label='RBF_Network')
    # plt.plot(solver_svm.loss_history, 'o',color='blue',label='SVM')
    plt.xlabel('Iteration')
    plt.legend(loc='lower right')
    plt.subplot(2, 1, 2)
    plt.plot(np.array(net.train_acc_history).reshape(-1,1), '-o', color='green', label='RBFNet_train')
    plt.plot(np.array(net.val_acc_history).reshape(-1,), '-o', color='blue', label='RBFNet_val')
    plt.xlabel('Iteration')
    plt.legend(loc='lower right')
    # plt.plot(net.loss_history, 'o', clor='green', label='RBF_Network')
    plt.gcf().set_size_inches(20, 10)
    plt.show()
    return df


def Test_for_SVM_Gaussian():
    kf = StratifiedKFold(n_splits=6,shuffle=True,random_state=1)
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]
        model = svm.SVC(kernel='rbf',C=3,gamma=0.4)
        model.fit(X_train,y_train.ravel())
        prediction = model.predict(X_val)
        print('acc_gaussian:',accuracy_score(prediction,y_val))

    SVM_result = model.predict(X_test)
    df = SVM_result
    df = pd.DataFrame(df, columns=['SVM'])
    df.to_csv('result1.txt', sep=' ',index=False)
    # Output the classification report SVM Gaussian
    print('-----------------Test_for_SVM_Gaussian-----------------')
    print(classification_report(y_val, prediction))
    return df

def Test_for_SVM_linear():
    kf = StratifiedKFold(n_splits=6,shuffle=True,random_state=1)
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]
        model = svm.SVC(kernel='linear',C=10)
        model.fit(X_train,y_train.ravel())
        prediction = model.predict(X_val)
        print(accuracy_score(prediction,y_val))
        # print('acc_linear:',accuracy_score(prediction,y_val))
    # Output the classification report SVM linear
    print('------------------Test_for_SVM_linear-------------------')
    print(classification_report(y_val, prediction))

if __name__ == "__main__":
    feature_normalize(X_train_full)
    start_time = time.time()
    # Test for RBF
    rbf=Test_for_RBFNN()

    # Test for SVM_Gaussian
    svm= Test_for_SVM_Gaussian()

    # Test for SVM_linear
    Test_for_SVM_linear()

    # coverge two dataframe
    df = pd.concat((rbf,svm),axis=1)
    df.to_csv('result.csv', header=True, index=True)
    data ={"rbf":rbf,"svm":svm}
    f1 = pd.DataFrame(data, columns=['rbf', 'svm'])
    print(f1)
    df = pd.DataFrame(data)
    df.to_csv('result.csv',header=True,index=True)
    print(df.to_latex(index=True))
    # print the running time
    print("--- %s seconds ---" % (time.time() - start_time))
