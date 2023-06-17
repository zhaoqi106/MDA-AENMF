import numpy as np
import pandas as pd

def get_fusion_sim (k1, k2):


    sim_m1 = pd.read_csv("mydata/data/metabolite_GIP_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    sim_m2 = pd.read_csv("mydata/data/metabolites_information_entropy_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    sim_m3 = pd.read_csv("mydata/data/metabolites_structure_similarity.csv", index_col=0, dtype=np.float32).to_numpy()


    sim_d1 = pd.read_csv("mydata/data/disease_semantic_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    sim_d2 = pd.read_csv("mydata/data/disease_GIP_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    sim_d3 = pd.read_csv("mydata/data/disease _information_entropy_similarity .csv", index_col=0, dtype=np.float32).to_numpy()


    m1 = new_normalization1(sim_m1)
    m2 = new_normalization1(sim_m2)
    m3 = new_normalization1(sim_m3)

    Sm_1 = KNN_kernel1(sim_m1, k1)
    Sm_2 = KNN_kernel1(sim_m2, k1)
    Sm_3 = KNN_kernel1(sim_m3, k1)

    Pm = Updating1(Sm_1,Sm_2,Sm_3, m1, m2,m3)
    Pm_final = (Pm + Pm.T)/2


    d1 = new_normalization1(sim_d1)
    d2 = new_normalization1(sim_d2)
    d3 = new_normalization1(sim_d3)


    Sd_1 = KNN_kernel1(sim_d1, k2)
    Sd_2 = KNN_kernel1(sim_d2, k2)
    Sd_3 = KNN_kernel1(sim_d3, k2)

    Pd = Updating1(Sd_1,Sd_2,Sd_3, d1, d2,d3)
    Pd_final = (Pd+Pd.T)/2



    return Pm_final, Pd_final


def new_normalization1 (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p


def KNN_kernel1 (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn



def Updating1 (S1,S2,S3, P1,P2,P3):
    it = 0
    P = (P1+P2+P3)/3
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,(P2+P3)/2),S1.T)
        P111 = new_normalization1(P111)
        P222 =np.dot (np.dot(S2,(P1+P3)/2),S2.T)
        P222 = new_normalization1(P222)
        P333 = np.dot (np.dot(S3,(P1+P2)/2),S3.T)
        P333 = new_normalization1(P333)

        P1 = P111
        P2 = P222
        P3 = P333

        P_New = (P1+P2+P3)/3
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P






























def GIP_kernel(association):

    nc = association.shape[0]
    matrix = np.zeros((nc, nc))
    r = getGosiR(association)
    for i in range(nc):
        for j in range(nc):
            temp_up = np.square(np.linalg.norm(association[i, :] - association[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix


def getGosiR(association):

    nc = association.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(association[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy








