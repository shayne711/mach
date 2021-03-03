# Shayne Erickson
# ECON 9880-001
# Machine Learning Midterm
# 2 March 2021


import pandas
import numpy
from math import radians
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import haversine_distances


#######################################################################################################################
#######################################################################################################################


# We will sample 10% of the dataset and drop problematic rows
dataset = pandas.read_csv("Crimes_-_2001_to_Present.csv")
dataset = dataset.sample(frac=0.1).reset_index(drop=True)

# Drop any issues in the IUCR values.
for i in range(len(dataset.index)):
    word = dataset.loc[dataset.index[i], 'IUCR']
    if ord(word[-1]) >= 65:
        dataset.drop([dataset.index[i]])
dataset = dataset.dropna()

# Reassign the datetime and its components to the dataset variable for use
dataset['Date'] = pandas.to_datetime(dataset['Date'])
dataset['year'] = dataset['Date'].dt.year
dataset['month'] = dataset['Date'].dt.month
dataset['day_num'] = dataset['Date'].dt.day
dataset['hour'] = dataset['Date'].dt.hour
dataset['minute'] = dataset['Date'].dt.minute

# Categorize seasons (0=winter -> 3=fall)
dataset['season_slot'] = numpy.select([
    (dataset['month'] < 3),
    (dataset['month'] < 6),
    (dataset['month'] < 9),
    (dataset['month'] < 11),
    (dataset['month'] <= 12)],
    [0,1,2,3,0])

# Categorize the day of the week (0=monday -> 6=sunday)
dataset['day_slot'] = dataset['Date'].dt.dayofweek

# Categorize the 96 15-minute slots in the day
dataset['hour_slot'] = numpy.select([
    (dataset['hour'] < 4),
    (dataset['hour'] < 8),
    (dataset['hour'] < 12),
    (dataset['hour'] < 16),
    (dataset['hour'] < 20),
    (dataset['hour'] < 24)],
    [0,1,2,3,4,5])

dataset['minute_slot'] = numpy.select([
    (dataset['minute'] < 15),
    (dataset['minute'] < 30),
    (dataset['minute'] < 45),
    (dataset['minute'] < 60)],
    [0,1,2,3])

dataset['time_slot'] = dataset['hour']*4 + dataset['minute_slot']

# Find the distance to the nearest police station and store as our "position-related" input
def police_dist(x):
    lat1 = x['Latitude']
    long1 = x['Longitude']
    loc1 = [radians(lat1), radians(long1)]
    stations = [[41.862920,-87.658080], [41.858089,-87.627502], [41.801490,-87.630200],
                [41.7668848,-87.6062725], [41.7075005,-87.5686422], [41.6922434,-87.6033162],
                [41.7519916,-87.6444358], [41.779753,-87.660049], [41.7791805,-87.7087368],
                [41.8369402,-87.6466056], [41.8564248,-87.7084573], [41.8732445,-87.7055489],
                [41.9212062,-87.6978558], [41.8808898,-87.7134466], [41.9743749,-87.7661011],
                [41.9661645,-87.7282855], [41.903015,-87.643269], [41.9475958,-87.6511856],
                [41.9799283,-87.6930573], [41.6918747,-87.6690059], [41.9995655,-87.6718539],
                [41.9181983,-87.7648901]]

    dist = [None]*22
    for i in range(22):
        loc2 = [radians(stations[i][0]), radians(stations[i][1])]
        # Distance is in meters!
        distance = haversine_distances([loc1, loc2])*6357000
        dist[i] = distance[0,1]
    return min(dist)

dataset['distance_from_station'] = dataset.apply(police_dist, axis=1)

# Now let's change 'Arrest' and 'Domestic' columns to 1's and 0's
dataset['Arrest'] = dataset['Arrest']*1
dataset['Domestic'] = dataset['Domestic']*1

# Get dummies for location description
loc_description = pandas.get_dummies(dataset['Location Description'])


#######################################################################################################################
#######################################################################################################################


# Assign Data and Target
data = dataset.loc[:,['season_slot', 'day_slot', 'time_slot', 'distance_from_station', 'Arrest', 'Domestic']].values
data = numpy.append(data, loc_description.iloc[:,:].values, axis=1)
target = dataset['IUCR'].values
target = target.astype(numpy.int)


# We will randomly split our initial data set in 8 parts
kfold_object = KFold(n_splits=8)
kfold_object.get_n_splits(data)

i = 0
for training_index, test_index in kfold_object.split(data):
    i = i + 1

    # Set the different data sets (training sets or test sets)
    data_training = data[training_index]
    data_test = data[test_index]
    target_training = target[training_index]
    target_test = target[test_index]

# Run linear or logistic regression on training data
machine = linear_model.LinearRegression()
print(type(target_training[0]))
machine.fit(data_training, target_training)

# Predict new set of y-values to compare with "test" y-values
new_target = machine.predict(data_test)

# Print out comparison of predicted outputs to the "test" outputs
# print(metrics.r2_score(target_test, new_target))
print("Accuracy Score: ", metrics.accuracy_score(target_test, new_target))
