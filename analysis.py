import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

def prec_recall(y_actual,y_predict,threshold):
    ''' returns precision and recall given actual known data y_actual,
    the predicted probability y_predicted, and the threshold (0-1.0)
    above which the predicted probability is 1, below 0.
    '''
    n, fp, fn, tp = confusion_matrix(y_actual,
                                     (y_predict>threshold).astype(int)).ravel()
    labelpos = tp+fp
    actpos = tp+fn
    if (labelpos>0):
        precision = tp/float(labelpos)
    else: precision = np.nan
    recall =  tp/float(actpos)
    return precision, recall


def fitPlotMult(data, pred,response ='biss_severity_9',N=10,test_size=.20):
    ''' data = dataframe to use with all the data
    pred = list of predictors, column names from data
    response = output variable, biss 9+ is default
    N= number of times to test out of sample
    test_size = training testing split, default at 80/20
    
    plots prec/recall curve for each trial
    returns dataframe showing optimal thresholds
    optimal thresholds are those that maximize recall without sacraficing precision.
    '''
    
    # figure out the baseline DMV prec/recall
    y_predDMV = (data.SEVERITY.isin(['K','A'])).astype(int)
    y_actualAll = (data[response]=='severe').astype(int)
    dmv_prec, dmv_recall =prec_recall(y_actual=y_actualAll,
                                      y_predict=y_predDMV, 
                                      threshold=.5)
    
    # isolate the prediction columns and outcome to work with
    y,X = dmatrices(response + '~' + '+'.join(pred),
                    data, return_type='dataframe')
    y = np.ravel(y[response+'[severe]'])
    
    print pred
    print 'number of variables',X.shape[1]
    print 'number of data points', y.shape[0]
    print 'number of severe instances', y.sum()
    print 'test fraction', test_size
    
    thresholds = np.arange(0,1,.01)
    prec_avg = np.zeros(shape=thresholds.shape)
    recall_avg = np.zeros(shape=thresholds.shape)
    col=['index','precision','recall']
    optimal = pd.DataFrame(columns=col)
    
    for _ in range(N):
        try:
            # train the model on 90% of the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            model = sm.Logit(y_train,X_train).fit()
            y_predict = model.predict(X_test)

            # store the precision recall points, for each threshold
            prec_list = []
            recall_list = []
            # run through all thresholds and calculate precision recall
            for threshold in thresholds:
                precision, recall = prec_recall(y_test,y_predict,threshold)
                # store:
                prec_list.append(precision)
                recall_list.append(recall)

            df = pd.DataFrame({'precision':prec_list,'recall':recall_list},index=thresholds)

            plt.plot(recall_list,prec_list,'o')

            prec_avg += np.array(prec_list)/float(N)
            recall_avg += np.array(recall_list)/float(N)

            # choose the threshold that maximizes recall, 
            # holding to the dmv precision
            subset = df[(df.precision >= dmv_prec)]
            threshold = subset.recall.idxmax()
            optimal = optimal.append(df[df.index == threshold].reset_index(),
                                     ignore_index=True)
        except:
            print 'error - singular matrix?'
    
    #plotting graphs
    plt.plot(dmv_recall,dmv_prec,marker='*',markersize=30,color='b',label='DMV')
    plt.plot(recall_avg,prec_avg,'-',label='avg')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.ylabel('precision',fontsize=18)
    plt.xlabel('recall',fontsize=18)
    plt.legend()

    print 'baseline dmv \n precision   recall\n',\
    '{:.2f}'.format(dmv_prec),'        ',  '{:.2f}'.format(dmv_recall)
    print 'optimal recall', float(median_recall(optimal)[0])
    return optimal


def median_recall(df):
    #values = {'precision':prec_dmv_9,'recall':rec_dmv_9}
    #des = df.fillna(value=values).describe([.1,.5,.9])
    des = df.describe([.1,.5,.9])
    med = des.loc[['50%']].recall[0]
    low = des.loc[['10%']].recall[0]
    high = des.loc[['90%']].recall[0]
    return format(med,'.2f'),format(low,'.2f'),format(high,'.2f')
    
def forwardSelect(data,startVar,unusedVar, epsilon=0):
    '''forwards selection'''
    # starting recall
    opRec = fitPlotMult(data=data,pred=startVar)
    curRec = float(median_recall(opRec)[0])
    print 'start curRec',curRec
    # placeholder for the winning variable
    curVar='notnone'
    while(curVar!='none'):
        curVar='none'
        # lower the recall a bit so it's easier to surpass
        curRec = curRec - epsilon
        # run through all the unused variables
        for v in unusedVar:
            # add in one aditional variable to inputs and run it
            # keep the variable with the largest median optimal recall
            try:
                opRec = fitPlotMult(data=data, pred=startVar+[v])
                medRec = float(median_recall(opRec)[0])
                print 'Recall',medRec
            except:
                print 'error'
            if (medRec>curRec):
                curRec = medRec
                curVar = v
                print curRec
        #the variable with the highest recall gets added to starting vars
        if (curVar!='none'):
            startVar.append(curVar)
            unusedVar.remove(curVar)
    
    print 'curRec',curRec
    return startVar

def backSelect(data,startVar,epsilon=0):
    '''backwards selection'''
    opRecdf = fitPlotMult(data=data,pred=startVar)
    curRec = float(median_recall(opRecdf)[0])

    # placeholder for the winning variable
    curVar = 'notnone'

    while (curVar!='none'):
        curVar='none'
        # lower the recall a bit so it's easier to surpass
        curRec = curRec - epsilon
        # go through all variables and test recall value
        for v in startVar:
            inputs = startVar[:]
            inputs.remove(v)
            opRecdf = fitPlotMult(data=data,pred=inputs)
            opRec = float(median_recall(opRecdf)[0])
            # winning recall gets saved
            if (opRec>curRec):
                curRec = opRec
                curVar = v
                print curRec   
        if (curVar!='none'):
            startVar.remove(curVar)
    print 'curRec', curRec
    return startVar
