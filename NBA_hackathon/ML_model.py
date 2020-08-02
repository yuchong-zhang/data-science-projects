#!/usr/bin/env python3
# -*- coding: utf-8 -*-S
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

train = pd.read_csv('~/Documents/NBA hackathon/Business Analytics/training_set.csv')
test = pd.read_csv('~/Documents/NBA hackathon/Business Analytics/test_set.csv')
game = pd.read_csv('~/Documents/NBA hackathon/Business Analytics/game_data.csv')
player = pd.read_csv('~/Documents/NBA hackathon/Business Analytics/player_data.csv')

#Twitter followers of NBA teams in March 2017 (in millions)
twitter_follower_2017={"LAL":6.17,"GSW":3.5,"MIA":4.09,"CHI":3.6,"SAS":2.3,"BOS":2.3,"CLE":2.1,"OKC":1.8,"HOU":1.71,"NYK":1.78,"ORL":1.5,
"TOR":1.4,"DAL":1.2,"PHI":0.93,"LAC":1.1,"ATL":0.99,"POR":0.82,"IND":0.93,"PHX":0.75,"SAC":0.71,"MEM":0.77,
                                        "BKN":0.76,"CHA":0.73,"MIL":0.7,"MIN":0.65,"WAS":0.66,"DET":0.71,"NOP":0.66,
                                                                    "UTA":0.63,"DEN":0.63}
#Twitter followers of NBA teams in March 2018 (in millions)
twitter_follower_2018={"LAL":7.39,"GSW":5.63,"MIA":4.68,"CHI":4.09,"SAS":3.32,"BOS":3.21,"CLE":3.16,"OKC":2.55,"HOU":2.54,"NYK":2.07,"ORL":1.59,
"TOR":1.5,"DAL":1.48,"PHI":1.46,"LAC":1.37,"ATL":1.19,"POR":1.13,"IND":1.11,"PHX":0.99,"SAC":0.96,"MEM":0.92,
                                        "BKN":0.88,"CHA":0.88,"MIL":0.87,"MIN":0.85,"WAS":0.85,"DET":0.85,"NOP":0.8,
                                                                    "UTA":0.79,"DEN":0.79}
#Facebook fans of NBA teams in March 2018 (in millions)
facebook_fan_2018={"LAL":21.53,"GSW":11.08,"MIA":15.74,"CHI":18.37,"SAS":7.2,"BOS":9,
                       "CLE":8.68,"OKC":7.24,"HOU":3.82,"NYK":6.12,"ORL":2.76,
"TOR":2.18,"DAL":4.52,"PHI":1.53,"LAC":3.85,"ATL":1.6,"POR":2.4,"IND":3.32,
                       "PHX":1.89,"SAC":1.76,"MEM":1.81,
            "BKN":2.8,"CHA":1.74,"MIL":1.42,"MIN":1.84,"WAS":1.53,"DET":1.86,"NOP":1.64,
                                        "UTA":1.17,"DEN":1.96}
twitter_2018=pd.DataFrame(list(twitter_follower_2018.items()), columns=['team', 'twitter_follower_2018 (in million)'])
twitter_2017=pd.DataFrame(list(twitter_follower_2017.items()), columns=['team', 'twitter_follower_2017 (in million)'])
facebook_2018=pd.DataFrame(list(facebook_fan_2018.items()), columns=['team', 'facebook_fan_2018 (in million)'])

train['Game_ID']=train['Game_ID'].apply(str)
test['Game_ID']=test['Game_ID'].apply(str)
game['Game_ID']=game['Game_ID'].apply(str)
player['Game_ID']=player['Game_ID'].apply(str)

game['Win_Percentage']=game['Wins_Entering_Gm']/(game['Wins_Entering_Gm']+game['Losses_Entering_Gm'])
game['Win_Percentage'].fillna(value=0.5,inplace=True)
game_H=game[game['Location']=='H'].reset_index(drop=True)
game_A=game[game['Location']=='A'].reset_index(drop=True)
game_H.rename(index=str, columns={'Win_Percentage':'Win_Percentage_H'},inplace=True)
game_A.rename(index=str, columns={'Win_Percentage':'Win_Percentage_A'},inplace=True)

player_ASG=player.loc[player['ASG_Team']!='None',['Game_ID','Team','ASG_Team','Active_Status']]
player_ASG_total=player_ASG.groupby('Game_ID')['ASG_Team'].count().reset_index()
player_ASG_active=player_ASG[player_ASG['Active_Status']=='Active'].groupby('Game_ID')['Active_Status'].count().reset_index()
player_ASG_dual=player_ASG[player_ASG['Active_Status']=='Active'].groupby('Game_ID')['Team'].nunique().reset_index()
player_ASG_dual.rename(index=str, columns={'Team':'ASG_Dual'},inplace=True)

train2=train.groupby(['Season','Game_ID','Game_Date','Away_Team','Home_Team'])['Rounded Viewers'].sum().reset_index()
train2['Day_of_Week']=pd.to_datetime(train2['Game_Date'], format='%m/%d/%Y').dt.dayofweek
train2['Game_Date']=pd.to_datetime(train2['Game_Date'], format='%m/%d/%Y')
train2['is_weekend']=(train2['Day_of_Week']==4)|(train2['Day_of_Week']==5)|(train2['Day_of_Week']==6)
train2=train2.merge(game_H[['Game_ID','Win_Percentage_H']], how='left', on=('Game_ID'))
train2=train2.merge(game_A[['Game_ID','Win_Percentage_A']], how='left', on=('Game_ID'))
train2=train2.merge(player_ASG_total,how='left', on=('Game_ID'))
train2=train2.merge(player_ASG_active,how='left', on=('Game_ID'))
train2=train2.merge(player_ASG_dual,how='left', on=('Game_ID'))
train2.loc[:,['ASG_Team','Active_Status','ASG_Dual']]=train2.loc[:,['ASG_Team','Active_Status','ASG_Dual']].fillna(value=0)
train2[['ASG_Team','Active_Status','ASG_Dual']]=train2[['ASG_Team','Active_Status','ASG_Dual']].astype('int64')

def day_diff(x,y):
     return (x-pd.Timestamp(y)).days

train2['is_Christmas']=(train2['Game_Date'].apply(day_diff, args=('2016-12-25',))==0)|(train2['Game_Date'].apply(day_diff, args=('2017-12-25',))==0)
train2['is_opening_week']=((train2['Game_Date'].apply(day_diff, args=('2016-10-25',))>=0)&(train2['Game_Date'].apply(day_diff, args=('2016-10-25',))<=6))|((train2['Game_Date'].apply(day_diff, args=('2017-10-17',))>=0)&(train2['Game_Date'].apply(day_diff, args=('2017-10-17',))<=6))

train3=train2.copy()
train3=train3.merge(facebook_2018, how='left', left_on=('Away_Team'),right_on=('team'))
train3=train3.drop('team',axis=1)
train3.rename(index=str, columns={'facebook_fan_2018 (in million)':'facebook_fan_A'},inplace=True)
train3=train3.merge(facebook_2018, how='left', left_on=('Home_Team'),right_on=('team'))
train3=train3.drop('team',axis=1)
train3.rename(index=str, columns={'facebook_fan_2018 (in million)':'facebook_fan_H'},inplace=True)
train3_1617=train3[train3['Season']=='2016-17']
train3_1718=train3[train3['Season']=='2017-18']
train3_1617=train3_1617.merge(twitter_2017, how='left', left_on=('Away_Team'),right_on=('team'))
train3_1617=train3_1617.drop('team',axis=1)
train3_1617.rename(index=str, columns={'twitter_follower_2017 (in million)':'twitter_follower_A'},inplace=True)
train3_1617=train3_1617.merge(twitter_2017, how='left', left_on=('Home_Team'),right_on=('team'))
train3_1617=train3_1617.drop('team',axis=1)
train3_1617.rename(index=str, columns={'twitter_follower_2017 (in million)':'twitter_follower_H'},inplace=True)
train3_1718=train3_1718.merge(twitter_2018, how='left', left_on=('Away_Team'),right_on=('team'))
train3_1718=train3_1718.drop('team',axis=1)
train3_1718.rename(index=str, columns={'twitter_follower_2018 (in million)':'twitter_follower_A'},inplace=True)
train3_1718=train3_1718.merge(twitter_2018, how='left', left_on=('Home_Team'),right_on=('team'))
train3_1718=train3_1718.drop('team',axis=1)
train3_1718.rename(index=str, columns={'twitter_follower_2018 (in million)':'twitter_follower_H'},inplace=True)

test['Day_of_Week']=pd.to_datetime(test['Game_Date'], format='%m/%d/%Y').dt.dayofweek
test['Game_Date']=pd.to_datetime(test['Game_Date'], format='%m/%d/%Y')
test['is_weekend']=(test['Day_of_Week']==4)|(test['Day_of_Week']==5)|(test['Day_of_Week']==6)
test=test.merge(game_H[['Game_ID','Win_Percentage_H']], how='left', on=('Game_ID'))
test=test.merge(game_A[['Game_ID','Win_Percentage_A']], how='left', on=('Game_ID'))
test=test.merge(player_ASG_total,how='left', on=('Game_ID'))
test=test.merge(player_ASG_active,how='left', on=('Game_ID'))
test=test.merge(player_ASG_dual,how='left', on=('Game_ID'))
test.loc[:,['ASG_Team','Active_Status','ASG_Dual']]=test.loc[:,['ASG_Team','Active_Status','ASG_Dual']].fillna(value=0)
test[['ASG_Team','Active_Status','ASG_Dual']]=test[['ASG_Team','Active_Status','ASG_Dual']].astype('int64')

test['is_Christmas']=(test['Game_Date'].apply(day_diff, args=('2016-12-25',))==0)|(test['Game_Date'].apply(day_diff, args=('2017-12-25',))==0)
test['is_opening_week']=((test['Game_Date'].apply(day_diff, args=('2016-10-25',))>=0)&(test['Game_Date'].apply(day_diff, args=('2016-10-25',))<=6))|((test['Game_Date'].apply(day_diff, args=('2017-10-17',))>=0)&(test['Game_Date'].apply(day_diff, args=('2017-10-17',))<=6))

test=test.merge(facebook_2018, how='left', left_on=('Away_Team'),right_on=('team'))
test=test.drop('team',axis=1)
test.rename(index=str, columns={'facebook_fan_2018 (in million)':'facebook_fan_A'},inplace=True)
test=test.merge(facebook_2018, how='left', left_on=('Home_Team'),right_on=('team'))
test=test.drop('team',axis=1)
test.rename(index=str, columns={'facebook_fan_2018 (in million)':'facebook_fan_H'},inplace=True)

test_1617=test[test['Season']=='2016-17']
test_1718=test[test['Season']=='2017-18']
test_1617=test_1617.merge(twitter_2017, how='left', left_on=('Away_Team'),right_on=('team'))
test_1617=test_1617.drop('team',axis=1)
test_1617.rename(index=str, columns={'twitter_follower_2017 (in million)':'twitter_follower_A'},inplace=True)
test_1617=test_1617.merge(twitter_2017, how='left', left_on=('Home_Team'),right_on=('team'))
test_1617=test_1617.drop('team',axis=1)
test_1617.rename(index=str, columns={'twitter_follower_2017 (in million)':'twitter_follower_H'},inplace=True)
test_1718=test_1718.merge(twitter_2018, how='left', left_on=('Away_Team'),right_on=('team'))
test_1718=test_1718.drop('team',axis=1)
test_1718.rename(index=str, columns={'twitter_follower_2018 (in million)':'twitter_follower_A'},inplace=True)
test_1718=test_1718.merge(twitter_2018, how='left', left_on=('Home_Team'),right_on=('team'))
test_1718=test_1718.drop('team',axis=1)
test_1718.rename(index=str, columns={'twitter_follower_2018 (in million)':'twitter_follower_H'},inplace=True)

full_data=[train3_1617,train3_1718,test_1617,test_1718]

for dataset in full_data:
    dataset['ASG_Dual'] = dataset['ASG_Dual'].apply(lambda x: 1 if x == 2 else 0)
    dataset['is_Christmas'] = dataset['is_Christmas'].apply(lambda x: 1 if x == True else 0)
    dataset['is_opening_week'] = dataset['is_opening_week'].apply(lambda x: 1 if x == True else 0)
    dataset['is_weekend'] = dataset['is_weekend'].apply(lambda x: 1 if x == True else 0)


'''
for dataset in full_data:
    dataset['Win_Percentage_Total'] = dataset['Win_Percentage_H']+dataset['Win_Percentage_A']
    dataset['Win_Percentage_diff'] = np.absolute(dataset['Win_Percentage_H']-dataset['Win_Percentage_A'])
    dataset['facebook_fan_Total'] = dataset['facebook_fan_H']+dataset['facebook_fan_A']
    dataset['twitter_follower_Total'] = dataset['twitter_follower_H']+dataset['twitter_follower_A']
    
    
    
y_train_1617=train3_1617['Rounded Viewers'].ravel()
x_train_1617=train3_1617.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Rounded Viewers','Day_of_Week','Win_Percentage_H',
       'Win_Percentage_A','facebook_fan_A', 'facebook_fan_H',
       'twitter_follower_A', 'twitter_follower_H'],axis=1).values
y_train_1718=train3_1718['Rounded Viewers'].ravel()
x_train_1718=train3_1718.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Rounded Viewers','Day_of_Week','Win_Percentage_H',
       'Win_Percentage_A','facebook_fan_A', 'facebook_fan_H',
       'twitter_follower_A', 'twitter_follower_H'],axis=1).values
x_test_1617=test_1617.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Total_Viewers','Day_of_Week','Win_Percentage_H',
       'Win_Percentage_A','facebook_fan_A', 'facebook_fan_H',
       'twitter_follower_A', 'twitter_follower_H'],axis=1).values
x_test_1718=test_1718.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Total_Viewers','Day_of_Week','Win_Percentage_H',
       'Win_Percentage_A','facebook_fan_A', 'facebook_fan_H',
       'twitter_follower_A', 'twitter_follower_H'],axis=1).values
'''
y_train_1617=train3_1617['Rounded Viewers'].ravel()
x_train_1617=train3_1617.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Rounded Viewers'],axis=1).values
y_train_1718=train3_1718['Rounded Viewers'].ravel()
x_train_1718=train3_1718.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Rounded Viewers'],axis=1).values
x_test_1617=test_1617.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Total_Viewers'],axis=1).values
x_test_1718=test_1718.drop(['Season','Game_ID','Game_Date', 'Away_Team', 'Home_Team','Total_Viewers'],axis=1).values

    
class SklearnHelper():
    def __init__(self,clf):
        self.learner=clf
        self.clf=clf()
    def update(self,params):
        self.clf=self.learner(**params)
    def train(self,X,y,n_iter,k):
        print(self.clf.get_params())
        MAPE_list=[]
        r2_list=[]
        for i in range(n_iter):
            kf=KFold(n_splits=k,shuffle=True)
            kf.get_n_splits(X)
            MAPE=[]
            r2=[]
            for train_index,test_index in kf.split(X):
                self.clf.fit(X[train_index],y[train_index])                
                temp_MAPE=np.mean(np.absolute(y[test_index]-self.clf.predict(X[test_index]))/y[test_index])
                MAPE.append(temp_MAPE)
                temp_r2=r2_score(y[test_index], self.clf.predict(X[test_index]))
                r2.append(temp_r2)
            MAPE_list.append(np.mean(MAPE))
            r2_list.append(np.mean(r2))
        print("MAPE_mean",np.mean(MAPE_list))
        print("MAPE_std",np.std(MAPE_list))
        print("r2_mean",np.mean(r2_list))
        print("r2_std",np.std(r2_list))
    def randomsearch(self,X,y,n_iter,cv,verbose,n_jobs,params):
        clf_random=RandomizedSearchCV(estimator=self.clf, param_distributions=params, n_iter=n_iter,cv=cv, verbose=verbose, n_jobs=n_jobs)
        clf_random.fit(X,y)
        self.best_search_random=clf_random.best_params_
    def gridsearch(self,X,y,cv,verbose,n_jobs,params):
        clf_grid=GridSearchCV(estimator=self.clf, param_grid=params,cv=cv, verbose=verbose, n_jobs=n_jobs)
        clf_grid.fit(X,y)
        self.best_search_grid=clf_grid.best_params_
        
RF_learner=SklearnHelper(RandomForestRegressor)

n_estimators=[1000]
max_features=['sqrt']
max_depth=[int(x) for x in np.linspace(5,110,num=3)]
max_depth.append(None)
min_samples_split=[2,5,10,50]
min_samples_leaf=[1,2,4,10]
grid_grid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf}
#grid_grid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth}
#RF_learner.gridsearch(x_train_1617,y_train_1617,5,1,-1,grid_grid)
#RF_learner.randomsearch(x_train_1617,y_train_1617,10,5,2,-1,random_grid)
RF_learner.update({'n_estimators':1000,'min_samples_split': 10})
#RF_learner.train(x_train_1617,y_train_1617,5,5)
#RF_learner.train(x_train_1718,y_train_1718,5,5)
RF_learner_final=RF_learner.clf
RF_learner_final.fit(x_train_1617,y_train_1617)
total_viewer_1617=RF_learner_final.predict(x_test_1617).astype(int)
RF_learner_final.fit(x_train_1718,y_train_1718)
total_viewer_1718=RF_learner_final.predict(x_test_1718).astype(int)


                
                
