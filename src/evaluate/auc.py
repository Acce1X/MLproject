import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np

def aucPlot(y_one_hot,y_score):
    # 手动计算micro类型的AUC
    # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.ravel())
    auc_val = auc(fpr, tpr)
    print ('手动计算auc：', auc_val)
    # 绘图
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    #'plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc_val)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'ROC和AUC', fontsize=17)
    plt.show()
