MatchedAnalysis-Ped.ipynb - main notebook for analysis

MatchedExplore.ipynb - exploring the matched data

MV104-data_explore.ipynb - exploring the full MV104 (crash) data

MV104-variable_map.ipynb - explicitly showing the mapping from coded variables to descriptive variables using the data dictionary (actual mapping takes place in analysis.py)


Edits in google doc

# DOT Classifying Severe Crashes
Predicting which traffic crashes result in severe injuries helps to prioritize locations for street safety redesign.

# Overview
Every year in NYC there are over 40 thousand traffic crashes where someone is injured. The vast majority of these injuries are minor. Around 250 crashes are fatal but many more result in serious injuries, close to death, with potentially long term effects on the injured and their loved ones. Being able to separate out the most severe crashes from the rest allows for the City to have a more focused response on safety interventions, including where to prioritize street redesign. This project proposes a few different formulas for classifying severe crashes.

RESULTS


# Scoping
NYC Department of Transportation (DOT) identifies areas with high concentrations of killed or severely injured (KSI) pedestrians to determine priority geographies for street safety redesign. For instance, “priority corridors” are defined as those with the highest KSI per mile over a 5 year time period, such that at least 50% of KSI are covered in those areas. Whether or not a new street redesign project is in a priority area is one of the criteria that goes into determining its precedence.

Crashes resulting in deaths are relatively straightforward to identify but whether anyone has sustained serious injuries is not well defined. Currently DOT uses a formula called KABCO. This formula was created by NY State Department of Motor Vehicle (DMV). It categorizes people involved in crashes into five categories: K (killed), A (severe injury), B (moderate injury), C (minor injury), and O (no injury). Hence the KABCO name. These KABCO scores are based on three questions from the police crash report that ask the police to evaluate the injured person(s). DOT’s KSI metric only refers to crashes with a K or A score from KABCO.

While this formula has been used for some time, there hasn’t been a way to evaluate it or improve on it.

# Data 

MV104 police reported crash database 
Based on MV104 crash report filled out by a police officer at the time of a crash
Reports sent to NY State DMV to compile into a database
Three tables: Crash table, Person table, and Vehicle table, linked by common identifiers.

Hospital records and crash report linked dataset
Hospital records were obtained from NY SPARCS (Statewide Planning and Research Cooperative System).
About half the hospitalizations and ER visits related to traffic crashes were able to be matched to an MV104 crash report.
Probabilistic matching to link to crash reports. Based on person level attributes (age, sex, crash role, collision type, date and time of crash, injury type, injury body location, and county). 
Time period: 2011 - 2013 (Billing code changes prior to 2011 made using earlier data unfeasible. Post 2013 hospital data was not available at the time.)
Patient bill records used to derive Injury Severity Score using the Barell matrix formula. (b-ISS). 

In 2017 DOHMH linked police crash report data with hospital data. [Reference: Conderino, S, Fung, L  et al. “Linkage of traffic crash and hospitalization records with limited identifiers for enhanced public health surveillance” AA&P 2017.]   In addition, they used the hospital administrative data on each patient to determine an injury severity score using the Barell Matrix method https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2564435/ .  The Barell Matrix is a classification system to translate billing codes into a matrix of injury types. These groupings are given an Abbreviated Injury Scale (AIS) code ranging from 1 to 6 ( from minor injury to death) (https://en.wikipedia.org/wiki/Abbreviated_Injury_Scale ).  Patients have an AIS for each injury, these are combined into one overall Injury Severity Score (ISS) by taking the top three AIS from different body regions, squaring them and summing. https://en.wikipedia.org/wiki/Injury_Severity_Score

This gave us a valuable dataset to start our analysis.


# Analysis
Defining severe: 
Patients with a derived Injury Severity Score (b-ISS) of 9 or greater we defined as a severe case. While generally the bar is set at ISS 16 or greater (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3217501/ ), we chose a lower threshold to take into account injuries such as a broken leg (ISS = 9) which may be low on a threat to life scale but can have significant impact on a person’s ability to work and get around in the near term. 
While we could have chosen to do an analysis without assigning a threshold (for instance using an AUC measure), most implementations would necessitate defining a threshold at some stage.

As a first attempt to understand which crash attributes are associated with severe outcomes we defined a severity ratio. The severity ratio is the probability of a severe outcome given a crash attribute is present divided by the probability of a severe outcome (irrespective of whether the attribute is present or not). (Note: this is slightly different from the usual definition of Risk Ratio where the denominator is probability of a severe outcome given the attribute is NOT present) 

The Severity Ratio is a useful measure because Severity RatioSR near 1 indicates that it is just as likely to have a severe crash with the attribute present as any random crash.  While an attribute that has a Severity Ratio greater than 1 indicates a higher likelihood of a severe crash compared to a random crash.

We found that most attributes listed as K or A from KABCO ranked high on the Severity Ratio. The exceptions are eye injuries and burns, both of which do not have enough data points to assign a meaningful Severity Ratio. 

The other finding was that age is extremely important in classifying severity. Pedestrians over age 70 were 2.5 times more likely to sustain severe injuries than a random pedestrian who is hit by a motor vehicle. 

New KSI formulas

Method 1 KABCO plus
Increase recall, decrease precision

Method 2 Scoring system.
Including age increases recall by about 20%, keeping precision same.




# Pilot
DOT decided, based on this analysis, the current KABCO system for identifying severe crashes is sound and they will not change it going forward. 

Independent of this analysis, DOT has started researching safety engineering standards focused on older adults. 

# Handoff
Not Applicable.
