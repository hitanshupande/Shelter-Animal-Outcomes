# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:56:43 2016

@author: hitanshu.pande
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.stats import hmean

def munge(data, train):
    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0,"HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)
    data['AnimalType'] = data['AnimalType'].map({'Cat':0,'Dog':1})
    data = data.apply(breedTransform, axis=1)   

    if(train):
        data.drop(['AnimalID','OutcomeSubtype'],axis=1, inplace=True)
        data['OutcomeType'] = data['OutcomeType'].map({'Return_to_owner':4, 'Euthanasia':3, 'Adoption':0, 'Transfer':5, 'Died':2})
            
    data['Gender'] = data['SexuponOutcome'].str.contains('Male').fillna(-1).astype(int)
    data.loc[data['SexuponOutcome']=='Unknown', 'Gender'] * -1
  
    data['Reproduction'] = data['SexuponOutcome'].str.contains('Intact').fillna(-1).astype(int)
    data.loc[data['SexuponOutcome']=='Unknown', 'Reproduction'] * -1
    #data['Reproduction'].fillna(0, inplace=True)     
    
    encoder = LabelEncoder()
    data['Color'] = encoder.fit_transform(data['Color'])
    
    def agetodays(x):
        try:
            y = x.split()
        except:
            return None 
        if 'year' in y[1]:
            return float(y[0]) * 365
        elif 'month' in y[1]:
            return float(y[0]) * (365/12)
        elif 'week' in y[1]:
            return float(y[0]) * 7
        elif 'day' in y[1]:
            return float(y[0])
        
    data['AgeInDays'] = data['AgeuponOutcome'].map(agetodays)
    data.loc[(data['AgeInDays'].isnull()),'AgeInDays'] = data['AgeInDays'].median()

    data['Year'] = data['DateTime'].str[:4].astype(int)
    data['Month'] = data['DateTime'].str[5:7].astype(int)
    data['Day'] = data['DateTime'].str[8:10].astype(int)
    data['Hour'] = data['DateTime'].str[11:13].astype(int)
    data['Minute'] = data['DateTime'].str[14:16].astype(int)

    data['IsMix'] = data['Breed'].str.contains('mix',case=False).astype(int)
    data['CrossBreed'] = data['Breed'].str.contains('/',case=False).astype(int)

           
    return data.drop(['AgeuponOutcome','Name','Breed','DateTime', 'SexuponOutcome'],axis=1)
    # return data.drop(['AgeuponOutcome','Name','Breed','Color','DateTime'],axis=1)

def breedTransform(i):
    for k,v in breed_dict.items():
        i[k] = v
    if(i['AnimalType'] == 'Cat'):
        i['Cat'] = 1
        return i
    else:
     
        breeds = ['Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Fris_','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
        groups = ['Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']

        breeds_group = np.array([breeds,groups]).T
        
        dog_groups = np.unique(breeds_group[:,1])
        
        group_values_dog = []
        
        
        not_found = []
        
       # for i in df['Breed']:
        i['Breed'] = i['Breed'].replace(' Shorthair','')
        i['Breed'] = i['Breed'].replace(' Longhair','')
        i['Breed'] = i['Breed'].replace(' Wirehair','')
        i['Breed'] = i['Breed'].replace(' Rough','')
        i['Breed'] = i['Breed'].replace(' Smooth Coat','')
        i['Breed'] = i['Breed'].replace(' Smooth','')
        i['Breed'] = i['Breed'].replace(' Black/Tan','')
        i['Breed'] = i['Breed'].replace('Black/Tan ','')
        i['Breed'] = i['Breed'].replace(' Flat Coat','')
        i['Breed'] = i['Breed'].replace('Flat Coat ','')
        i['Breed'] = i['Breed'].replace(' Coat','')
            
        groups = []
        if '/' in i['Breed']:
            split_i = i['Breed'].split('/')
            for j in split_i:
                if j[-3:] == 'Mix':
                    breed = j[:-4]               
                    if breed in breeds_group[:,0]:
                        indx = np.where(breeds_group[:,0] == breed)[0]
                        groups.append(breeds_group[indx,1][0])
                        groups.append('Mix')
                    elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                        find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                        groups.append(find_group[find_group != 'Unknown'])
                        groups.append('Mix')  
                    elif breed == 'Pit Bull':
                        groups.append('Pit Bull')
                        groups.append('Mix')  
                    elif 'Shepherd' in breed:
                        groups.append('Herding')
                        groups.append('Mix')  
                    else:
                        not_found.append(breed)
                        groups.append('Unknown')
                        groups.append('Mix')
                else:
                    if j in breeds_group[:,0]:
                        indx = np.where(breeds_group[:,0] == j)[0]
                        groups.append(breeds_group[indx,1][0])
                    elif np.any([s.lower() in j.lower() for s in dog_groups]):
                        find_group = [s if s.lower() in j.lower() else 'Unknown' for s in dog_groups]                    
                        groups.append(find_group[find_group != 'Unknown'])
                    elif j == 'Pit Bull':
                        groups.append('Pit Bull')
                    elif 'Shepherd' in j:
                        groups.append('Herding')
                        groups.append('Mix')  
                    else:
                        not_found.append(j)
                        groups.append('Unknown')
            else:
        
                if i['Breed'][-3:] == 'Mix':
                    breed = i['Breed'][:-4]
                    if breed in breeds_group[:,0]:
                        indx = np.where(breeds_group[:,0] == breed)[0]
                        groups.append(breeds_group[indx,1][0])
                        groups.append('Mix')
                    elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                        find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                        groups.append(find_group[find_group != 'Unknown'])
                        groups.append('Mix') 
                    elif breed == 'Pit Bull':
                        groups.append('Pit Bull')
                        groups.append('Mix') 
                    elif 'Shepherd' in breed:
                        groups.append('Herding')
                        groups.append('Mix')  
                    else:
                        groups.append('Unknown')
                        groups.append('Mix') 
                        not_found.append(breed)
        
                else:
                    if i['Breed'] in breeds_group[:,0]:
                        indx = np.where(breeds_group[:,0] == i)[0]
                        groups.append(breeds_group[indx,1][0])
                    elif np.any([s.lower() in i['Breed'].lower() for s in dog_groups]):
                        find_group = [s if s.lower() in i['Breed'].lower() else 'Unknown' for s in dog_groups]                    
                        groups.append(find_group[find_group != 'Unknown'])
                    elif i['Breed'] == 'Pit Bull':
                        groups.append('Pit Bull')
                    elif 'Shepherd' in i['Breed']:
                        groups.append('Herding')
                        groups.append('Mix') 
                    else:
                        groups.append('Unknown') 
                        not_found.append(i['Breed'])
            #group_values_dog.append(list(set(groups)))
            dog_group_list = list(set(groups))
            for dog_group in dog_group_list:
                i[dog_group] = 1
        
        
        
        #not_f_unique,counts = np.unique(not_found,return_counts=True)
        
        #unique_groups, counts = np.unique(group_values_dog,return_counts=True)
        
        # add mix, pit bull, and unknown to the groups
        #groups = np.unique(np.append(dog_groups,['Mix','Pit Bull','Unknown']))
        return i
    
# pd_train.head()
# Import data
breeds = ['Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Fris_','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
groups = ['Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']

breeds_group = np.array([breeds,groups]).T

dog_groups = np.unique(breeds_group[:,1])

pd_train = pd.read_csv('C:/Python workspace/Kaggle/Animal Shelter/train.csv')
pd_test = pd.read_csv('C:/Python workspace/Kaggle/Animal Shelter/test.csv')

#pd_train.isnull().any()

breed_dict = dict()
for x in dog_groups:
    breed_dict[x] = 0
breed_dict['Cat'] = 0



# Clean and transform data
pd_train = munge(pd_train,True)
pd_test = munge(pd_test,False)
pd_train.fillna(0)
pd_test.fillna(0)
pd_train.isnull().any()
pd_test.isnull().any()
print(pd_train.columns)

pd_test.drop('ID',inplace=True,axis=1)

train = pd_train.values
test = pd_test.values

#print("Calculating best case params...\n")
#print(best_params(train))


y = train[:,0]
from sklearn.cross_validation import StratifiedKFold
skf = StratifiedKFold(y, n_folds=5, shuffle=True)
len(skf)
skf_list = list(skf)

score_list = []
for skf_train, skf_val in skf:
    X_train, X_val = train[skf_train], train[skf_val]
    #print("Train:", skf_train, "Test:", skf_val)
    print("Predicting... \n")
    forest = RandomForestClassifier(n_estimators = 400, max_features='auto')
    forest = forest.fit(X_train[0::,1::],X_train[0::,0])
    predictions = forest.predict_proba(X_val[0::,1::])
    score = log_loss(X_val[0::, 0], predictions)
    print('Score:', score)
    score_list.append(score)

print('Average score:', hmean(score_list))