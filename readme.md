# Impact at Home Plate
### By: Kevin Smith 08/18/2022
---

Whether you are a Major League Baseball coach making lineup decisions, or a general manager making free agency decisions, there is a lot to consider when putting your team together. Analytics have been great in part because they aschewed tradtional baseball wisdom and allowed teams to find new paths to success. The impact metric just might be something that can bridge the gap between analytics and "old school baseball."

## Project Goal
---
The goal of this project is to build a model to predict a players offensive wins above replacement (owar) utilizing machine learning and a metric that i invented called "impact."

being able to utilize impact in both free agent and lineup decisions simplifies the process by utilizing both analytics and tradition baseball wisdom.

We will be walking through a basic data science pipeline to reach our goal today. The steps in the pipeline are:

* Planning
* Acqusition
* Prep
* Exploration
* Clustering
* Feature Engineering
* Modeling
* Delivery

### Steps For Reproduction
---

1. Clone my repo (including the , <mark>acquire.py</mark>. <mark>explore.py</mark> , <mark>stats.csv</mark>, <mark>value.csv</mark>, <mark>model.py</mark> & <mark>wrangle.py</mark>files.
2. The libraries used are pandas, numpy, scipy, matplotlib, seaborn, and sklearn.
3. You should now be able to run the <mark>zillow_final_cluster.ipynb</mark> file.

## Planning
---
Their are two essential parts to any good plan. Identify your **Goals**, and the necessary **Steps** to get there. 

### Goals:
1. Identify variables driving owar.
2. Develop a model to make value predicitons based on those variable. 
3. Deliver actionable takeaways

### Steps:
1. Initial hypothesis
2. Acquire and cache the dataset
3. Clean, prep, and split the data to prevent data leakage
4. Do some preliminary exploration of the data (including visualiztions and statistical analyses)
5. Create meaningful clusters
6. Trim dataset of variables that are not statistically significant
7. Determine which machine learning model perfoms the best
8. Utilize the best model on the test dataset
9. Create a final report notebook with streamlined code optimized for a technical audience



## Data Library
---
| **Variable Name** | **Explanation** | **Values** |
| :---: | :---: | :---: |
| last | the players last name | object |
| first | the players first name | object |
| id | unique identification code | Numeric value |
| year | the year of the season of the observation | Numeric value |
| name | The players full name | object |
| team | Code representing the team played for during observation | object|
| owar | offensive wins above replacement | Numeric value |
| age | age of player during season in years | Numeric value |
| ops| on-base + slugging-pct | Numeric value |
| tb | total bases | Numeric vaue |
| pitches_faced | number of puitches faced by batter during season | Numeric value |
| pull | the percentage of balls hit to the 'pull' side | Numeric value |
| center| the percetage of balls hit 'up the middle' | Numeric value |
| oppo | percentage of balls hit to the 'opposite' side | Numeric value |
| batted | number of balls the hitter made contact with during the season | Numeric value |
| raa | runs above average | Numeric value|
| waa | wins above average | Numeric value |
| impact | measure of players average impact per plate appearance| Numeric value |
| ppa | number of pitches seen in average plate appearance | Numeric value |




## Initial Hypothesis
--- 
The initial hypothesis can be based on a gut instinct or the first question that comes to mind when encountering a dataset.

|**Initial hypothesis number** |**hypothesis** |
| :---: | :---: |
|Initial hypothesis 1 | pull, center, and oppo have a non-linear relationship with each other|
|Initial hypothesis 2 | There is a linear relationship between raa and waa, but a non-linear relationship between each of them and ops |

## Acquire and Prep
---
Utitlize the functions imported from the <mark>wrangle.py</mark> to create a DataFrame with pandas.

These functions will also cache the data to reduce execution time in the future should we need to create the DataFrame again.

In this step we will utilize the functions in the <mark>wrangle.py</mark> file to get our data ready for exploration. 

This means that we will be looking for columns that may be dropped because they are duplicates, and either dropping or filling any rows that contain blanks depending on the number of blank rows there are.

This also means that we will be splitting the data into 3 separate DataFrames in order to prevent data leakage corrupting our exploration and modeling phases.


## Exploration
---
This is the fun part! this is where we get to ask questions, form hypotheses based on the answers to those questions and use our skills as data scientist to evaluate those hypotheses!

For example, in the zillow dataset I asked "Does square footage drive up value?" and unsurprisingly the answer was generally yes. This lead me to the hypothesis that square footage would have a dependent relationship with tax value, which hypothesis testing confirmed. However I was able to find another variable that did a better job of predicting value.

## Clustering
---
I created 2 clusters and utilized them as features for the models.

|**Cluster** |**Elements** |
| :---: | :---: |
| tendency | pull, center, oppo |
| production | raa, waa, ops |

## Feature Engineering
---
I didnt try to reinvent the wheel here. I used Sklearn's Kbest function to find the most important variables for our model.

The tendency cluster was used in the models, but the production cluster proved less useful.


## Modeling
---
Here we determine the best model to use for predicting value. I ran the data through 3 different regression algorithms to determine which would perform the best

The 2 degree polynomial model performed the best against baseline when predicting owar. Predicting owar on average 4 times better than baseline.

## Delivery
---
So, you may be wondering how you will be able to use this model to make decisions.

1. By keeping track of a players impact at the plate over the course of the season, managers can make lineup decisions based on a players recent impact level knowing that it is a key driver in how many wins above or below that player is likely to be on that given day.

2. When making free agency decisions the general manager can take into account the average impact that player has had at the plate over the course of previous seasons, and decide if they would rather have a player with a lower average impact who is more consistent, or a player who has short bursts of dominance at the plate.

## Conclusion
---

In conclusion, with more time, and more robust data sources I would be able to track a players impact at the plate and identify potential free agents that would fit the identity of the team because of the metrics reliance on both analytics and tradition baseball wisdom. 