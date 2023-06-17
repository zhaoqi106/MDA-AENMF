import itertools
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
from classifiers import *
from GAE_trainer import *
from GAE import *
from NMF import *
from metric import *
from similarity_fusion import *
from five_AE import *
import warnings
warnings.filterwarnings("ignore")


# parameter
n_splits = 5
classifier_epochs = 50
m_threshold = [0.7]
epochs=[200]
fold = 0
result = np.zeros((1, 7), float)
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)

for s in itertools.product(m_threshold,epochs):

        association = pd.read_csv("mydata/data/M_D.csv", index_col=0).to_numpy()
        samples = get_all_samples(association)


        k1 = 226
        k2 = 21
        m_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)  # Integration of similarity networks for metabolites or diseases
        # m_fusion_sim = pd.read_csv("mydata/data/m_fusion_sim.csv", index_col=0, dtype=np.float32).to_numpy()
        # d_fusion_sim = pd.read_csv("mydata/data/d_fusion_sim.csv", index_col=0, dtype=np.float32).to_numpy()

        kf = KFold(n_splits=n_splits, shuffle=True)

        # Metabolite and disease features extraction from NMF
        D = 90
        NMF_mfeature, NMF_dfeature = get_low_feature(D, 0.01, pow(10, -4), association)

        for train_index, val_index in kf.split(samples):
            fold += 1
            train_samples = samples[train_index, :]
            val_samples = samples[val_index, :]
            new_association = association.copy()
            for i in val_samples:
                new_association[i[0], i[1]] = 0

            # Metabolite features extraction from GAE
            m_network = sim_thresholding(m_fusion_sim, s[0])
            m_adj, meta_features = generate_adj_and_feature(m_network, new_association)
            m_features = get_gae_feature(m_adj, meta_features, s[1], 1)

            # Disease features extraction from five-layer auto-encoder
            d_features = five_AE(d_fusion_sim)

            # get feature and label
            train_feature, train_label = generate_f1(D, train_samples, m_features, d_features, NMF_mfeature, NMF_dfeature)
            val_feature, val_label = generate_f1(D, val_samples, m_features, d_features, NMF_mfeature, NMF_dfeature)

            # MLP classfier
            model = BuildModel(train_feature, train_label)
            test_N = val_samples.shape[0]
            y_score = np.zeros(test_N)
            y_score = model.predict(val_feature)[:, 0]

            # calculate metrics
            fpr, tpr, thresholds = roc_curve(val_label, y_score)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            result += get_metrics(val_label, y_score)
            print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]',
                  get_metrics(val_label, y_score))


        print("==================================================")
        print(result / n_splits)

        # plot ROC curve
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g', label='diagonal', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='r', label=r'MDA_AENMF ROC (area=%0.3f)' % mean_auc, lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc='lower right')
        plt.show()








