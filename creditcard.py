import readline
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, recall_score, precision_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import seaborn as sns

class CreditCard(object):
    """
    Process Credit card data from AER (r package) dataset, build and
    persist a model
    """
    def __init__(self):
        """
        Read in data when class is initialized and binarize "yes"/"no"
        """
        self._read_data()
        self._binarize_cat_strings(['card', 'owner', 'selfemp'])
        self.y = self.df['card']
        self.X = self.df.drop('card', axis=1)
    def _read_data(self):
        """
        Activate R to Pandas, import package from R and extract dataset.
        convert to pandas dataframe

        Input: None
        Output: CreditCard data read to dataframe in Object
        """
        pandas2ri.activate()
        aer = importr('AER')
        credcard = data(aer).fetch('CreditCard')
        self.df = pandas2ri.ri2py(credcard['CreditCard'])
    def _binarize_cat_strings(self, cols):
        """
        Input: list of (yes/no) columns
        Output:
        Convert columns that are setup as 'yes'/'no' into 1 for yes, 2 for no
        """
        for col in cols:
            self.df[col] = self.df[col].apply(lambda x: 1 if x == 'yes' else 0)

    def gen_eda_plots(self, violin=True, hist=['share',
                                               'expenditure',
                                               'reports']):
        """
        Generate violin and histogram plots used in EDA
        violin plots are normalized and converted from wide to long form to
        show distr diffs between cards.

        Input:  Desired plots
        Output: generated graphs to graph folder
        """
        if violin == True:
            cols = ['reports', 'share', 'expenditure',
                    'majorcards', 'owner', 'card']

            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(self.df)
            df_norm = pd.DataFrame(x_scaled)
            df_norm.columns = self.df.columns

            df_melt = pd.melt(df_norm[cols], id_vars='card')
            sns.violinplot(x='variable', y='value', hue='card',
                            data=df_melt, split=True, width=0.8)
            plt.title('Distribution of Variables to Application Approval')
            plt.savefig('graphs/violin.png')
        if hist != False:
            for column in hist:
                plt.title('Distribution for %s by Approval Status' % (column))
                self.df.hist(column=column, by='card')
                plt.savefig('graphs/%s_hist.png' % (column))
                plt.clf()

    def test_models(self, model_dict):
        """
        Run models established, and return score to STOUT, and generate ROC
        curve for the model

        Input: Dictionary of model names and untrained models
        Output: ROC graph across models and score/Precision/Recall scores to
        STOUT
        """
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.min_max_scaler.fit(X_train)
        X_train = pd.DataFrame(self.min_max_scaler.transform(X_train))
        X_test = pd.DataFrame(self.min_max_scaler.transform(X_test))
        X_train.columns = self.X.columns
        X_test.columns = self.X.columns

        for mod_name, model in model_dict.iteritems():
            model.fit(X_train, y_train)
            mod_score = model.score(X_test, y_test)
            print "average score for %s is %s" % (mod_name,
                                                    round(mod_score,2)
                                                  )
            y_pred = model.predict(X_test)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            print "%s has a recall of %s and precision of %s" \
                    % (mod_name, round(precision, 4), round(recall, 4))
            fpr, tpr, _ = roc_curve(y_test.values, model.predict(X_test))
            plt.plot(fpr, tpr, label=mod_name)
        legend = plt.legend(frameon = 1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        plt.title('ROC Curve For Across Models')
        plt.xlabel('False Positive Rate (1-Recall)')
        plt.ylabel('True Positive Rate')
        plt.plot([0,1],[0,1], ls="--", color='black')
        plt.savefig('graphs/roc_curves.png')

    def final_train_and_persist(self, model, plot=True):
        """
        Select final model and train for entire dataset and persist

        Input: Untrained Model object
        Output: save trained model to model.pkl.  If plot is selected,
                it will generate a plot of the Coefficients (logistic by
                default)
        """
        self.model = model
        scaled_X = self.min_max_scaler.transform(self.X)
        self.model.fit(scaled_X,self.y)
        joblib.dump(self.model, 'model.pkl', compress=1)
        if plot == True:
            plt.clf()
            ypos = np.arange(len(self.X.columns))
            plt.barh(ypos, self.model.feature_importances_)
            plt.yticks(ypos, self.X.columns)
            plt.grid(b=None)
            plt.title('Feature Importance Random Forest')
            plt.ylabel('Features')
            plt.xlabel('Importance')
            plt.savefig('graphs/Coefficients_model.png')

if __name__ == '__main__':
    cc = CreditCard()
    cc.gen_eda_plots()
    model_tests = {
                'rfc': RandomForestClassifier(),
                'knn': KNeighborsClassifier(),
                'logit': LogisticRegression()
                }
    cc.test_models(model_tests)
    cc.final_train_and_persist(RandomForestClassifier())
