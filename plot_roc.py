
import itertools

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from classifiers import *
from GAE_trainer import *
from GAE import *
from NMF import *
from metric import *
from similarity_fusion import get_syn_sim, sim_thresholding,get_syn_sim1
from five_AE import *
from sklearn.metrics import roc_curve,auc
from scipy import interp


# tprs=[]
# aucs=[]
# mean_fpr=np.linspace(0,1,100)

# draw a diagonal line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g', alpha=.8)



# plot DEEP-RAM ROC curve
# DEEP_DRM_mean_fpr = pd.read_csv("mydata/ROC/DEEP_DRM_mean_fpr.csv", index_col=0, dtype=np.float32).to_numpy()
# DEEP_DRM_mean_tpr = pd.read_csv("mydata/ROC/DEEP_DRM_mean_tpr.csv", index_col=0, dtype=np.float32).to_numpy()
# DEEP_DRM_mean_auc = auc(DEEP_DRM_mean_fpr, DEEP_DRM_mean_tpr)
# plt.plot(DEEP_DRM_mean_fpr, DEEP_DRM_mean_tpr, color='b', label='Deep_DRM ROC (AUC=%0.3f)' % DEEP_DRM_mean_auc, lw=2,alpha=.8)
#





#  plot AENMF ROC curve
AENMF_mean_fpr = pd.read_csv("mydata/ROC/AENMF_mean_fpr.csv", index_col=0, dtype=np.float32).to_numpy()
AENMF_mean_tpr = pd.read_csv("mydata/ROC/AENMF_mean_tpr.csv", index_col=0, dtype=np.float32).to_numpy()
AENMF_mean_auc = auc(AENMF_mean_fpr, AENMF_mean_tpr)
plt.plot(AENMF_mean_fpr, AENMF_mean_tpr, color='r', label=r'MDA_AENMF ROC (AUC=%0.3f)' % AENMF_mean_auc, lw=2, alpha=.8)



#  parameter and label
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')


plt.show()