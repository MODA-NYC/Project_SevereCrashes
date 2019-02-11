import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import itertools



def sratio(data):
    ''' 
    returns a dataframe where: 
        rows are columns from input data, prefixed by f_  
        columns are (1) SevereCount - number of severe cases for that attribute(factor)
                    (2) AllCount - total number of cases, severe or not, for that attribute
                    (3) SeverePct - percent of of cases which are severe for that attribute
                    (4) SeverityRatio - SeverePct divided by total percent severe 
                                    irrespective of attribute.
    input data: each row is a person injured in a crash
    '''
    factors = data.filter(like='f_').columns
    factors = factors.drop(['f_AgeYear','f_DriverAgeYear']) #redundant
    
    ctlist = []
    keys = []
    
    # probability of severe conditional on a factor being present 
    # P(severe|factor)
    for factor in factors:
        # severe counts by factor
        df = data[data.biss_severity_9=='severe'].groupby(factor).count()[['CI_ID']]\
        .rename(columns={'CI_ID':'SevereCount'}) 
        
        # counts of all people by factor
        df['AllCount']= data.groupby(factor).count()[['CI_ID']]
        
        # percent
        df['SeverePct'] = df.SevereCount/df.AllCount
        
        ctlist.append(df)
        keys.append(factor)    
    
    factordf = pd.concat(ctlist,keys=keys).sort_values('SeverePct')
    
    # baseline risk is not conditional on any factors P(severe)
    baseRisk = data[data.biss_severity_9=='severe'].shape[0]/float(data.shape[0])
    
    #normalizing all by baseline risk
    factordf['SeverityRatio'] = factordf['SeverePct']/baseRisk

    # combine index into one column
    factordf.index = factordf.index.map('{0[0]}:{0[1]}'.format)
    
    # renaming 
    factordf.index = factordf.index.str.replace('f_','')
    factordf.index = factordf.index.str.replace(':',' : ')
    
    return factordf

def bootstrapSR(data,N=10):
    ''' Sampling with replacement.
    Returns a dataframe where each column is a random sample from 
    the input data. Total of N samples (columns)
    Used to create confidence intervals around the severity ratio metric
    '''
    sample=pd.DataFrame()
    for i in range(N):
        sample[i] = sratio(data.sample(frac=1,replace=True))['SeverityRatio']
    return sample

def plotSR(value, sample, minSR = 1.3,colors='lightblue'):
    ''' returns a plot of the values, 
    with error bars set by the sample dataframe (90% confidence intervals)
    only showing attributes where the severity ratio is greater than maxSR
    '''
    errmax = sample.quantile(.95, axis =  1) - value['SeverityRatio']
    errmin = value['SeverityRatio'] - sample.quantile(.05, axis =  1)
    df = value[value.SeverityRatio > minSR] #only plot attributes with large ratios
    error = [[errmin[df.index],errmax[df.index]]]

    df.SeverityRatio.T.plot(kind='barh',xerr=error,color=colors,figsize=(8,8))
    plt.axvline(x=1,color='grey')
    plt.xlabel('severity ratio',fontsize=20)
    plt.title('Top Predictors of Severe Injury \n for pedestrians or bicyclists',fontsize=22)
    
def renameLabelSR(df):
    '''rename labels in the output of sratio() and bootstrapSR()
    this is to make the graph have more readable labels.
    it's ridiculously difficult to change the labels in matplotlib, 
    so I'm changing them in the source dataframes instead. 
    ''' 
    index = df.index
    shortIndex = list(set([i.split(' :')[0] for i in index]))

    changeIndex = {
        'Sex':'Sex',
        'RoadSurface':'Road Surface',
        'OtherVehTypeVIN':'Vehicle Type',
        'Eject':'Eject',
        'Age70':'Age70',
        'TrafficControl':'Traffic Control',
        'InjuryStatus': 'Injury Status',
        'DriverSex':'Driver Sex',
        'PedLoc': 'Pedestrian Location',
        'InjuryType':'Injury Type',
        'InjuryLoc': 'Injury Location',
        'PedAction': 'Pedestrian Action',
        'Weather':'Weather',
        'Role':'Role',
        'OtherVehAction':'Vehicle Action',
        'DriverAge70': 'Driver Age70',
        'TimeOfDay':'Time of Day',
        'Lighting':'Lighting'}
 
    t = [i.replace(k,changeIndex[k]) for i in index for k in changeIndex.keys() \
                                 if i.startswith(k)]
    
    df.index = pd.unique(t).tolist() #remove duplicates but keep order
    return df


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
    ''' 
    fitting a logistic regression model to the data and plotting the
    precission/recall curves
    
    data = dataframe, each row is an injured person
    pred = list of predictors (column names from data)
    response = output variable, biss 9+ is default
    N = number of times to test out of sample
    test_size = training testing split, default at 80/20
    
    plots prec/recall curve for each trial
    returns dataframe showing optimal thresholds
    optimal thresholds are those that maximize recall without sacrificing precision.
    '''
    
    # baseline DMV prec/recall
    y_predDMV = (data.SEVERITY.isin(['K','A'])).astype(int)
    y_actualAll = (data[response]=='severe').astype(int)
    dmv_prec, dmv_recall = prec_recall(y_actual=y_actualAll,
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
            # train the model on part of the data
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
    '''
    returns the 5%, 50%, 95% recall values
    df = output from fitPlotMult()
    '''
    des = df.describe([.05,.50,.95])
    med = des.loc[['50%']].recall[0]
    low = des.loc[['5%']].recall[0]
    high = des.loc[['95%']].recall[0]
    return format(med,'.2f'),format(low,'.2f'),format(high,'.2f')
    

def varSelect(data,varOfInt,N=10):
    '''
    run through all variables and collect optimal recall
    returns model dataframe with optimal recall for each model
    '''
    # set up dataframe to collect recall info for all the models
    models = pd.DataFrame(columns=['model','recall','recall low','recall high'])

    # include injury variables in all models
    inj=['f_InjuryType','f_InjuryLoc','f_InjuryStatus']

    # adding additional variable pairs and calculating recall
    for x in list(itertools.combinations(varOfInt,2)):
        pred = inj + [x[0],x[1]]
        op = fitPlotMult(data=data, pred=pred, N=N)
        models = models.append([{'model':'+'.join([x[0],x[1]]),
                                'recall':float(median_recall(op)[0]),
                                'recall low':float(median_recall(op)[1]),
                                'recall high': float(median_recall(op)[2]),
                                }],
                                ignore_index=True)
    models.index = models.model
    models.drop('model',axis=1,inplace=True)

    #clean up model names
    models.index = models.index.str.replace('f_','')#.str.replace('dec','')\
    #.str.replace('period','time').str.replace('road_','')

    # error bars for graphing will need the change in high and low values
    # not the values themselves
    models['delta low'] = models['recall low'] - models['recall']
    models['delta high'] = models['recall high'] - models['recall']
    
    return models

    
#note: forward and backward select are not used
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
