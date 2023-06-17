import random
import numpy as np




def generate_f1(D,train_samples,feature_m,d_data1,feature_MFm, feature_MFd):
    vect_len1 = feature_m.shape[1]
    vect_len2 = d_data1.shape[1]
    train_n = train_samples.shape[0]
    train_feature = np.zeros([train_n, 2 * vect_len1 + 2 * D])
    # train_feature = np.zeros([train_n, vect_len1+vect_len2])
    train_label = np.zeros([train_n])
    for i in range(train_n):
        train_feature[i, 0:vect_len1] = feature_m[train_samples[i, 0], :]
        train_feature[i, vect_len1:(vect_len1 + vect_len2)] = d_data1[216*train_samples[i, 0]+train_samples[i, 1], :]
        train_feature[i, (vect_len1 + vect_len2):(vect_len1 + vect_len2 + D)] = feature_MFm[train_samples[i, 0], :]
        train_feature[i, (vect_len1 + vect_len2 + D):(vect_len1 + vect_len2 + 2 * D)] = feature_MFd[train_samples[i, 1],:]
        train_label[i] = train_samples[i, 2]
    return train_feature, train_label







def get_low_feature(k,lam, th, A):
    m, n = A.shape
    arr1=np.random.randint(0,100,size=(m,k))
    U = arr1/100
    arr2=np.random.randint(0,100,size=(k,n))
    V = arr2/100
    obj_value = objective_function(A, A, U, V, lam)
    obj_value1 = obj_value + 1
    i = 0
    diff = abs(obj_value1 - obj_value)
    while i < 1000:
        i =i + 1
        U = updating_U(A, A, U, V, lam)
        V = updating_V(A, A, U, V, lam)

    return U, V.transpose()

def objective_function(W, A, U, V, lam):
    m, n = A.shape
    sum_obj = 0
    for i in range(m):
        for j in range(n):
            sum_obj = sum_obj + W[i,j]*(A[i,j] - U[i,:].dot(V[:,j]))+ lam*(np.linalg.norm(U[i, :], ord=1,keepdims= False) + np.linalg.norm(V[:, j], ord = 1, keepdims = False))
    return  sum_obj

def updating_U (W, A, U, V, lam):
    m, n = U.shape
    upper = (W*A).dot(V.T)
    down = (W*(U.dot(V))).dot((V.T)) + (lam/2) *(np.ones([m, n]))
    U_new = U
    for i in range(m):
        for j in range(n):
            U_new[i,j] = U[i, j]*(upper[i,j]/down[i, j])
    return U_new


def updating_V (W, A, U, V, lam):
        m,n = V.shape
        upper = (U.T).dot(W*A)
        down = (U.T).dot(W*(U.dot(V)))+(lam/2)*(np.ones([m,n]))
        V_new = V
        for i in range(m):
            for j in range(n):
                V_new[i,j] = V[i, j]*(upper[i,j]/down[i,j])
        return V_new
