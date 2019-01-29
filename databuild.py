import pandas as pd
import numpy as np
import statsmodels.api as sm

import sys
sys.path.insert(0,'/home/deena/Documents/data_munge/ModaCode/')
import moda

def readDMV():
    '''read in DMV data tables'''
    crash = moda.databridge("select * from anaphi.crash", encrypted='yes')
    ind = moda.databridge("select * from anaphi.individual", encrypted='yes')
    veh = moda.databridge("select * from anaphi.vehicle", encrypted='yes')
    
    # header got repreated in a couple of the files, removing it
    crash.drop(133,inplace=True)
    ind.drop(484,inplace=True)
    
    # reformating date 
    crash['year']=crash.CSACCDTE.str[:4]
    crash['day']=crash.CSACCDTE.str[8:10]
    crash['month']=crash.CSACCDTE.str[5:7]
    crash['date'] = pd.to_datetime(crash[['year','month','day']])

    # reformating age and vehicle type to number
    veh.VEHBDYT_ID = pd.to_numeric(veh.VEHBDYT_ID,errors='coearse')
    ind.INDIV_AGE = pd.to_numeric(ind.INDIV_AGE,errors='coearse')
    
    print 'full crash table',crash.shape
    print 'full person table', ind.shape
    print 'full vehicle table', veh.shape
    
    return crash, ind, veh

def buildTablesDMV(crash,ind,veh):
    '''
    returns pedestrian table (only one vehicle crashes) and
    twoVeh table with people involved in two-vehicle crashes 
    (a single crash may be represented multiple times since multiple 
    people are involved
    inputs are the three DMV tables    
    '''
    # only police reported crashes
    # I'm assuming the POL_REPT variables says whether or not a police report was completed
    # According to the codebook the INJT_ID variable 16,17,18 corresponds 
    # to the civilian crashform
    # so filtering out those as well
    crashPR = crash[crash.POL_REPT=='Y']
    indPR = ind[ind.CS_ID.isin(crashPR.CS_ID)&(~ind.INJT_ID.isin(['16','17','18']))]

    # combining data using person info at the row level.
    indPR = indPR.drop(['ORG_HOSP_CDE','LAGSTP','LAGINJT','LAGHOSP'],axis=1)
    
    #merging in crash info
    base = indPR.merge(crash.drop(['RDSYST_ID'],axis=1),
                      how='left',on='CS_ID')

    ####
    
    # other vehicle info
    # motor vehicles (not pedestrians etc) 
    vehMotor = veh[(veh.VEHBDYT_ID<100)&(veh.VEHBDYT_ID>0)]
    vehMotor = vehMotor.drop(['CV_WEIGHT_LBS','SHZMTT_ID','DMV_VIN_NUM'],axis=1)
    
    ####
    # 2 veh crashes only
    '''
    twoVeh = base[base.VEH_CNT=='2']

    # merging in own vehicle info
    twoVeh = twoVeh.merge(veh.drop(['CV_WEIGHT_LBS','SHZMTT_ID','DMV_VIN_NUM','CS_ID'],
                                 axis=1),
                      how='left',on='CV_ID')
    # own vehicle is requrired to be a motor vehicle (not pedestrian or bicycle)
    twoVeh = twoVeh[(twoVeh.VEHBDYT_ID<100)&(twoVeh.VEHBDYT_ID>0)]

    #dataframe to join crash and vehicle id's
    cv = crash[crash.VEH_CNT=='2'][['CS_ID']].merge(vehMotor[['CS_ID','CV_ID']],
                                               how='left',on='CS_ID')
    cv.columns=['CS_ID','CV2_ID']

    twoVeh = twoVeh.merge(cv,how='left',on='CS_ID')
    twoVeh = twoVeh[~(twoVeh.CV_ID==twoVeh.CV2_ID)]
    twoVeh = twoVeh.merge(vehMotor.add_suffix('_other'),
                      how='left',left_on='CV2_ID',right_on='CV_ID_other')
    print 'two veh', twoVeh.shape
    '''
    ####
    # only keeping 1 veh crashes
    ped = base[(base.CIROLET_ID.isin(['6','7','14']))&(base.VEH_CNT=='1')]

    # bring in other vehicle info

    #dataframe to join crash and vehicle id's
    cv1 = crash[crash.VEH_CNT=='1'][['CS_ID']].merge(vehMotor[['CS_ID','CV_ID']],
                                               how='left',on='CS_ID')
    cv1.columns=['CS_ID','CV2_ID']

    ped = ped.merge(cv1,how='left',on='CS_ID')
    ped = ped[~(ped.CV_ID==ped.CV2_ID)]
    ped = ped.merge(vehMotor.add_suffix('_other'),
                    how='left',left_on='CV2_ID', right_on='CV_ID_other')

    # merge in driver info
    driver = indPR[indPR.CIROLET_ID=='1']

    ped = ped.merge(driver.add_suffix('_driver'), how='left',left_on='CV_ID_other',
             right_on='CV_ID_driver')
    ped = ped[~(ped.duplicated('CI_ID'))] #one of the drivers is duplicated, so dropping.
    
    # read in vin decoded vehicle type info
    vin_api_data = pd.read_csv('vin_api_data.csv')
    # merge onto matched data
    ped = ped.merge(vin_api_data[['VIN','BodyClass']],
                    how='left',left_on='VIN_other',right_on='VIN')
    
    print 'pedestrians/bicyclists (police reported) (single vehicle)',ped.shape
    
    return ped

def readLinked():
    ''' read in linked(matched) DMV/SPARKS data'''

    linked = moda.databridge('select * from anaphi.dohmh_traffic_dot_moda2', encrypted='yes')

    # some col names end with underscore, removing them.
    linked.columns = [x[:-1] if x[-1]=='_' else x for x in linked.columns ]

    print 'linked',linked.shape

    # dropping anything without a police report
    linked.drop(linked[linked.POL_REPT=='N'].index,inplace=True)
    # dropping anything with injury type = 15-18 since that corresponds to the civilian form
    linked.drop(index = linked[linked.INJT_ID.isin(['15','16','17','18'])].index,
            inplace=True)

    linked.reset_index(drop=True,inplace=True)

    print 'linked after dropping no police reports', linked.shape
    return linked

def mergeBiss(df,bissdf):
    ''' 
    returns original dataframe df with biss info merged in
    '''
    df = df.merge(bissdf[['CI_ID','BISS']],how='left',on='CI_ID')

    # everyone who is killed, set their BISS to 75
    #ped.loc[ped.SEVERITY=='K','BISS'] = 75
    #twoVeh.loc[twoVeh.SEVERITY=='K','BISS'] = 75

    # keep only rows with a BISS score
    df = df[df.BISS>0]
    df['hosp_biss'] = pd.to_numeric(df.BISS)

    # b-iss severity, setting 9+ as severe
    df.loc[df.hosp_biss>8,'biss_severity_9'] = 'severe'
    df.loc[df.hosp_biss<9,'biss_severity_9'] = 'not severe'

    return df

def formatVars(df):
    '''catagorizing and formating variables'''
    #injury variables f_inj
    # injury type
    inj_type = {1: 'Amputation',
                2 : 'Concusion',
                3 : 'Internal',
                4 : 'Minor Bleeding',
                5 : 'Severe Bleeding',
                6 : 'Minor Burn',
                7 : 'Moderate/Severe Burn',#'Moderate Burn',
                8 : 'Moderate/Severe Burn',#'Severe Burn',
                9 : 'Fracture-Dislocation',
                10 : 'Contusion-Bruise',
                11 : 'Abrasion',
                12 : 'Complaint of Pain',
                13 : 'None Visible',
                14 : 'Whiplash'}

    df['f_InjuryType'] = df.INJT_ID.astype('float').map(inj_type)

    ###
    # injury status
    status = {1:'not conscious states',#'death',
              2:'not conscious states',3:'not conscious states',4:'not conscious states',
              5:'conscious states',6:'conscious states'}

    df['f_InjuryStatus'] = df.EMTNSTATT_CDE.astype(float).map(status)
    ###
    # injury location
    inj_loc = {1 : 'Head',
               2 : 'Face',
               3 : 'Eye',
               4 : 'Neck',
               5 : 'Chest',
               6 : 'Back',
               7 : 'Shoulder-Upper Arm',
               8 : 'Elbow-Lower Arm-Hand',
               9 : 'Abdomen-Pelvis',
               10 : 'Hip-Upper Leg',
               11 : 'Knee-Lower Leg-Foot',
               12 : 'Entire Body'}

    df['f_InjuryLoc'] = df.INJLOCT_CDE.astype(float).map(inj_loc)
    ###
    # person variables f_per
    # sex
    sex = {'M':'male','m':'male',
           'F':'female','f':'female'}

    df['f_Sex'] = df.CI_SEX_CDE.map(sex)
    # incase df does not have other driver variables use try/except
    try:
        df['f_DriverSex'] = df.CI_SEX_CDE_driver.map(sex)
    except:
        0
    ###
    # age

    df['f_AgeYear'] = df.INDIV_AGE
    df['f_AgeDecade'] = (np.floor(df.f_AgeYear/10)*10)
    df['f_AgeDecade'] = df.f_AgeDecade.fillna('unknown').astype(str)
    # age over and under 70
    df.loc[df.INDIV_AGE<70,'f_Age70'] = 'age < 70'
    df.loc[df.INDIV_AGE>=70,'f_Age70'] = 'age >= 70'

    try:
        df['f_DriverAgeYear'] = df.INDIV_AGE_driver
        df['f_DriverAgeDecade'] = (np.floor(df.f_DriverAgeYear/10)*10)
        df['f_DriverAgeDecade'] = df.f_DriverAgeDecade.fillna('unknown').astype(str)
         # age over and under 70
        df.loc[df.INDIV_AGE_driver<70,'f_DriverAge70'] = 'age < 70'
        df.loc[df.INDIV_AGE_driver>=70,'f_DriverAge70'] = 'age >= 70'
        
    except:
        0
    ###
    # role

    role = {1:'driver',2:'passenger',
            6:'pedestrian',9:'pedestrian',
            7:'bicyclist',14:'bicyclist'}
    # not sure what 11=Registrant means, keeping it as unknown.

    df['f_Role'] = df.CIROLET_ID.astype('int').map(role)
    ###
    # ejected from vehicle

    ejected = {1 : 'not ejected',2:'ejected',3:'ejected'}

    try:
        df['f_Eject'] = df.EJCTT_ID.astype(float).map(ejected)
    except:
        0
    ###
    # pedestrian/bicyclist at interesection
    loc = {'1':'at intersection','2':'not at intersection'}

    try:
        df['f_PedLoc'] = df.PBLOCT_ID.map(loc)
    except:
        0    
    ###
    # road conditions f_road
    # day light condition

    light = {1 : 'Daylight',
             2 : 'Dawn/Dusk',
             3 : 'Dawn/Dusk',
             4 : 'Dark-Road', #lit
             5 : 'Dark-Road'} #unlit

    df['f_Lighting'] = df.LGHTCNDT_ID.astype(int).map(light)

    ###
    # time of day    
    timeperiod = {0:'midnight-3am',
             1:'3am-6am',
             2:'6am-9am',
             3:'9am-noon',
             4:'noon-3pm',
             5:'3pm-6pm',
             6:'6pm-9pm',
             7:'9pm-midnight'}

    df['f_TimeOfDay']= np.floor(df.HR1.astype(float)/3).map(timeperiod)
    
    ###
    # road surface
    surf = {1 : 'Dry',
            2 : 'Not Dry',
            3 : 'Not Dry',
            4 : 'Not Dry',
            5 : 'Not Dry',
            6 : 'Not Dry'}       
    df['f_RoadSurface'] = df.RDSRFT_ID.astype(int).map(surf)

    ###
    # weather
    weather = {
        1 : 'Clear',
        2 : 'Cloudy',
        3 : 'Percipitation',#'Rain',
        4 : 'Percipitation',#'Snow',
        5 : 'Percipitation',#'Sleet/Hail/Freezing Rain',
        6 : 'Cloudy',#'Fog/Smog/Smoke',
        #9 : 'Other' 
    }
    df['f_Weather'] = df.WTHRT_ID.astype(int).map(weather)
    
    ###
    # traffic control
    control ={
        '1' : 'None',
        '2' : 'Traffic signal',
        '3' : 'Stop sign',
        '4' : 'Other',#'Flashing light',
        '5' : 'Other',#'Yield sign',
        '6' : 'Other',#'Officer/Flagman/Guard',
        '7' : 'Other',#'No passing zone',
        '8' : 'Other',#'RR crossing sign',
        '9' : 'Other',#'RR crossing flash light',
        '10' : 'Other',#'RR crossing gates',
        '11' : 'Other',#'Stopped school bus w/ red light flash',
        '12' : 'Other',#'Highway work area (construction)',
        '13' : 'Other',#'Maintenance work area',
        '14' : 'Other',#'Utility work area',
        '15' : 'Other',#'Police/Fire emergency',
        '16' : 'Other',#'School zone',
        '20' : 'Other'}

    df['f_TrafficControl'] = df.TFCCTRLT_ID.map(control)
    ###
    # pre-collision action f_act

    # pedestrian bicyclist action
    # even if the injured is not a pedestrian, this can be filled out
    # it comes from the DMV crash table (not the person table)
    pedaction = {
        1 : 'Crossing, With Signal',
        2 : 'Crossing, Against Signal',
        3 : 'Crossing, No Signal, Marked Crosswalk',
        4 : 'Crossing, No Signal or Crosswalk',
        5 : 'Along Highway',
        6 : 'Along Highway',
        7 : 'Other',#'Emerging from in Front of/Behind Parked Vehicle',
        8 : 'Other',#'Going to/From Stopped School Bus',
        9 : 'Other',#'Getting On/Off Vehicle Other than School Bus',
        10 : 'Other',#'Pushing/Working on Car',
        11 : 'Other',#'Working in Roadway',
        12 : 'Other',#'Playing in Roadway',
        13 : 'Other',#'Other Acrions in Roadway',
        14 : 'Other',#'Not in Roadway (Indicate)'
    }
    df['f_PedAction'] = pd.to_numeric(df.PBACTT_DMV_CDE).map(pedaction)

    ###
    # vehicle action
    action = {
        1 : 'Going Straight Ahead',
        2 : 'Making Right Turn',
        3 : 'Making Left Turn',
        4 : 'Making Left Turn',#'Making U Turn',
        5 : 'Stopping Starting',#'Starting from Parking',
        6 : 'Stopping Starting',#'Starting in Traffic',
        7 : 'Stopping Starting',#'Slowing or Stopping',
        8 : 'Stopping Starting',#'Stopped in Traffic',
        9 : 'Stopping Starting',#'Entering Parked Position',
        10 : 'Stopping Starting',#'Parked',
        11 : 'Other',#'Avoiding Object in Roadway',
        12 : 'Other',#'Changing Lanes',
        13 : 'Other',#'Overtaking/Passing',
        14 : 'Other',#'Merging',
        15 : 'Backing',
        16 : 'Making Right Turn',
        17 : 'Making Left Turn',
        18 : 'Other',#'Police Pursuit',
        20 : 'Other'}
    try:
        df['f_OtherVehAction'] = df.PACCACTT_ID_other.astype(float).map(action)
        df['f_VehAction'] = df.PACCACTT_ID.astype(float).map(action)
    except:
        0

   
    ###
    # vehicle body type from DMV encoding
    vehicle = {
        1 : 'Car',
        2 : 'Truck',
        3 : 'Car',
        4 : 'Car',
        5 : 'Suburban',
        6 : 'Car',
        7 : 'Car',    
        10 : 'Motorcycle',
        11 : 'Car',
        12 : 'Car',    
        15 : 'Truck',
        16 : 'Car',
        22 : 'Truck',
        27 : 'Truck',   
        28 : 'Truck',
        33 : 'Truck',
        34 : 'Truck',
        40 : 'Truck',
        41 : 'Truck',
        42 : 'Truck',
        43 : 'Truck',
        44 : 'Pickup',
        45 : 'Truck',
        46 : 'Truck',
        47 : 'Truck',
        48 : 'Truck',
        49 : 'Van',
        50 : 'Truck',
        51 : 'Truck',
        52 : 'Truck',
        53 : 'Truck',
        55 : 'Truck',
        65 : 'Truck',
        57 : 'Truck',
        59 : 'Truck',
        60 : 'Bus',
        63 : 'Car',
        101 : 'Pedestrian',
        102 : 'Pedestrian',
        103 : 'Bicyclist',
    }

    df['f_OtherVehType'] = df.VEHBDYT_ID_other.map(vehicle)
    try:
        df['f_VehType'] = df.VEHBDYT_ID.map(vehicle)
    except:
        0
    
    ###
    # veh body type from decoded VIN
    # pedestrian only has other vehicle.
    try: 
        df['f_OtherVehTypeVIN']='unknown'
        df.loc[df.BodyClass.fillna('-').str.contains('Motorcycle'),
                'f_OtherVehTypeVIN'] = 'Motorcycle'

        df.loc[df.BodyClass.fillna('-').str.contains('SUV'),
                'f_OtherVehTypeVIN'] = 'SUV'

        df.loc[df.BodyClass.fillna('-').str.contains('SUV'),
                'f_OtherVehTypeVIN'] = 'SUV'

        df.loc[df.BodyClass.fillna('-').str.contains('Bus'),
                'f_OtherVehTypeVIN'] = 'Bus'

        df.loc[df.BodyClass.fillna('-').str.contains('Truck'),
                'f_OtherVehTypeVIN'] = 'Truck'

        df.loc[df.BodyClass.fillna('-').str.contains('Sedan'),
                'f_OtherVehTypeVIN'] = 'Car'

        df.loc[df.BodyClass.fillna('-').str.contains('CUV'),
                'f_OtherVehTypeVIN'] = 'Car'

        df.loc[df.BodyClass.fillna('-').str.contains('Hatchback'),
                'f_OtherVehTypeVIN'] = 'Car'

        df.loc[df.BodyClass.fillna('-').str.contains('Convertible'),
                'f_OtherVehTypeVIN'] = 'Car'

        df.loc[df.BodyClass.fillna('-').str.contains('Coupe'),
                'f_OtherVehTypeVIN'] = 'Car'

        df.loc[df.BodyClass.fillna('-').str.contains('Van'),
                'f_OtherVehTypeVIN'] = 'Van'

        df.loc[df.BodyClass=='Minivan',
                'f_OtherVehTypeVIN'] = 'Minivan'

        df.loc[df.BodyClass=='Pickup',
                'f_OtherVehTypeVIN'] = 'Pickup'

        df.loc[df.BodyClass=='Wagon',
                'f_OtherVehTypeVIN'] = 'Car'
    except:
        0
        
    # final cleaning - fill all blanks
    df.fillna('unknown',inplace=True)

    return df
