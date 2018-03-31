
import numpy as np
import pandas as pd
import os

def read_data():
    raw_data_path = os.path.join(os.pardir,'data','raw')
    train_file_path = os.path.join(raw_data_path,'train.csv')
    test_file_path = os.path.join(raw_data_path,'test.csv')
    # read the data with all default parameters
    train_df = pd.read_csv(train_file_path,index_col="PassengerId")
    test_df = pd.read_csv(test_file_path,index_col="PassengerId")
    test_df['Survived'] = -888
    df = pd.concat((train_df,test_df),axis=0)
    return df

def process_data(df):
    return(df
           ## create title attribute
           .assign(Title = lambda x: x.Name.map(get_title))
           #working with missing values
           .pipe(fill_missing_values)
           ## create fare bin attribute
           .assign(Fare_Bin = lambda x:
                   pd.qcut(x.Fare,4,labels=['very_low','low','high','very_high']))
           ##create age state
           .assign(AgeState = lambda x: np.where(x.Age >18,'Adult','Child'))
            ## Assign family size
           .assign(FamilySize = lambda x:x.Parch + x.SibSp +1) 
           .assign(IsMother= lambda x: np.where(((x.Sex == 'female')
                                                 & (x.Age > 18) &
                                                 (x.Parch > 0) &
                                                 (x.Title != 'Miss')),1,0))
           .assign(Cabin = lambda x:np.where(x.Cabin =='T',np.NaN,x.Cabin))
           .assign(Deck= lambda x: x.Cabin.map(get_deck))             
           .assign(IsMale = lambda x: np.where(x.Sex == 'Male',1,0))
           .pipe(pd.get_dummies,
                 columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])
            .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1)
            .pipe(reorder_columns)
    )


def reorder_columns(df):
    coloums = [column for column in df.columns if column != "Survived"  ]
    coloums = ['Survived'] + coloums
    return df[coloums]
                   
def fill_missing_values(df):
    #Embarked
    df.Embarked.fillna('C',inplace=True)
    #Fare
    median_fare = df[(df.Pclass==3) & (df.Embarked=='S')]['Fare'].median()
    df.Fare.fillna(median_fare,inplace=True)
    #age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median,inplace=True)
    return df

def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0],'Z')
                   

def write_data(df):
    processed_data_path = os.path.join(os.pardir,'data','processed')
    write_train_path = os.path.join(processed_data_path,'train.csv')
    write_test_path = os.path.join(processed_data_path,'test.csv')
    #train data
    df[df.Survived != -888].to_csv(write_train_path)
    #test data
    columns = [column for column in df.columns if column != "Survived"]
    df[df.Survived == -888][columns].to_csv(write_test_path)

                   
def get_title(name):
    title_groups = {
        "mr":"Mr",
        "mrs":"Mrs",
        "miss":"Miss",
        "master":"Master",
        "don":"Sir",
        "rev":"Sir",
        "dr":"Officer",
        "mme":"Mrs",
        "ms":"Mrs",
        "major":"Officer",
        "lady":"Lady", 
         "sir":"Sir",
        "mlle":"Miss",
        "col":"Officer",
        "capt":"Officer",
        "the countess":"Lady",
        "jonkheer":"Sir",
        "dona":"Lady" 
    }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_groups[title]

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)