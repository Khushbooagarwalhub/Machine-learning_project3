from shutil import copyfile
import pandas as pd
import numpy as np
import os

# The following function collects features extracted 
# from every batch in a given data generator.
def feature_extraction(feature_model, generator, batch_size):

    total_samples = len(generator)*batch_size
    total_batches = len(generator)
    
    # Initialize features and labels arrays
    features = np.zeros(shape=(total_samples, 5, 5, 1536))
    labels = np.zeros(shape=(total_samples, 10))

    batch_num = 0
    for inputs_batch, labels_batch in generator:
        predicted_features = feature_model.predict(inputs_batch)

        size = len(predicted_features)
        lower_bound = batch_num*size
        upper_bound = (batch_num+1)*size

        # Insert features and labels into array
        features[lower_bound:upper_bound] = predicted_features
        labels[lower_bound:upper_bound] = labels_batch

        if (batch_num % 50) == 0: 
            print(f"Processed batch ---- {batch_num}/{total_batches}")

        batch_num += 1
        if batch_num >= total_batches:
            break

    # Reshape features into more convenient array
    features = np.reshape(features, (total_samples, 5*5*1536))
    
    # Build a dictionary in order to return both features and labels
    features_labels = {
        "Features": features,
        "Labels": labels
    }
    print("Completed Processing")
    
    return features_labels

# The following functions save extracted features 
# and labels into csv files
def features_to_csv(features, file):
    features_df = pd.DataFrame()
    
    print("Processing Features")
    print("-------------------")
    for idx in np.arange(features.shape[0]):
        col = f"s{idx}"
        features_df[col] = features[idx]

        if (idx % 100) == 0:
            print(f"Processed Row ---- {idx} of {features.shape[0]}")

    features_df = features_df.T
    features_df.to_csv(file)
    print("Complete\n")

def labels_to_csv(labels, file):
    labels_dict = {
        "Class": []
    }

    print("Processing Labels")
    print("-----------------")
    for idx in np.arange(labels.shape[0]):
        labels_dict['Class'].append(labels[idx].tolist().index(1))
        
    if (idx % 100) == 0:
            print(f"Processed Row ---- {idx} of {features.shape[0]}")

    labels_df = pd.DataFrame(labels_dict)
    labels_df.to_csv(file)
    print("Complete")

def features_labels_to_csv(features, labels, feature_file, label_file):
    features_to_csv(features, feature_file)
    labels_to_csv(labels, label_file)