Jupyter Workbooks:

PedAnalysis.ipynb - main notebook for analysis

LinkedDataExplore.ipynb - exploring the matched data

VariableMap.ipynb - explicitly showing the mapping from coded variables to descriptive variables using the data dictionary (actual mapping takes place in analysis.py)


# DOT Classifying Severe Crashes
Predicting which traffic crashes result in severe injuries helps to prioritize locations for street safety redesign.

# Overview
Using linked crash and hospital data we were able to evaluate DOT’s formula for assigning severity to pedestrian crashes. 

Our findings confirm that the current formula, which relies on police officer’s reporting of injury, largely tracks with hospital designated injury outcomes. However some improvement is possible. We developed two methods that improve upon the current formula.

In addition we found that the older adults are over twice as likely to be severely injured in a crash as a random pedestrian. 

This project was a partnership between MODA, DOT, and DOHMH. 


# Scoping
Every year in NYC there are over 40 thousand traffic crashes where someone is injured. The vast majority of these injuries are minor. Around 250 crashes are fatal, but many more result in serious injuries, close to death, with potentially long term effects on the injured and their loved ones. Being able to separate out the most severe crashes from the rest allows for the City to have a more focused response on safety interventions, including where to prioritize street safety redesign.

NYC Department of Transportation (DOT) identifies areas with high concentrations of killed or severely injured (KSI) pedestrians as priority geographies for street safety redesign. For instance, “priority corridors” are defined as corridors with the highest KSI per mile over a 5 year time period, where at least 50% of KSI are covered in those areas. Whether or not a new street redesign project is in a priority area is one of the criteria that goes into determining when it gets completed.

Crashes resulting in deaths are relatively straightforward to identify but whether someone has sustained serious injuries is not well defined. Currently DOT uses a formula called KABCO created by NY State Department of Motor Vehicle (DMV). It categorizes people involved in crashes into five categories: K (killed), A (severe injury), B (moderate injury), C (minor injury), and O (no injury). Hence the KABCO name. These KABCO scores are based on three questions from the police crash report on the injured person(s): injury type, injury location, and injury status.

DOT’s current KSI metric refers to crashes with a K or A score from KABCO. These are crashes where the reported injury status is one of the following: Death, Unconscious, Semiconscious, Incoherent; the reported injury type is one of the following: Amputation, Concussion, Internal, Severe Bleeding, Severe/Moderate Burn, Fracture-Dislocation; or the injury location is the eye. 

While this formula has been used for some time, there hasn’t been a way to evaluate it or improve on it.

# Data 
In 2017 DOHMH linked police crash report data with hospital data. Reference: Conderino, S, Fung, L  et al. “Linkage of traffic crash and hospitalization records with limited identifiers for enhanced public health surveillance” AA&P 2017.   
We used this data for our analysis.

LinkedDataExplore.ipynb


# Analysis
We found that most attributes listed as K or A from KABCO ranked high on the Severity Ratio scale (i.e. they also tracked as severe as measured by the hospital outcomes). The exceptions are eye injuries and burns, both of which do not have enough data points to assign a meaningful Severity Ratio. 

The other finding is that age is extremely important in classifying severity. Pedestrians over age 70 were 2.5 times more likely to sustain severe injuries than a random pedestrian who is hit by a motor vehicle. 

The Severity Ratio is allows us to see which crash attributes are more indicative of severe crashes than other, but it doesn’t give a quantitatively understanding of  improving on the original DMV KABCO formula.

We propose two new methods for developing with a new KSI formula. We evaluate these formulas using precision and recall metrics where the ground truth is the hospital outcomes (b-ISS). 

**New KSI formula 1: KABCO plus**
This is the most straightforward approach to changing the KABCO formula. In addition to crashes that are ranked K or A in KABCO, we also include other crash attributes that had high SR. These include: head injuries, age 70+, and the other vehicle being a motorcycle or truck. We found that adding in any or all of these categories captures more severe cases, but decreases the percentage of severe cases in the target group. 

**New KSI formula 2: Scoring system using logistic regression models**
This is a more flexible method allowing us to keep precision constant while improving recall by around 20%

VariableMap.ipynb

# Pilot
DOT decided, based on this analysis, the current KABCO system for identifying severe crashes is sound and they will not change it going forward. 

Independent of this analysis, DOT has started researching safety engineering standards focused on older adults. 

# Handoff
Not Applicable.
