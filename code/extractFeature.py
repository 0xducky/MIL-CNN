import os
import numpy as np
import cv2
import pickle
from skimage.feature import hog
import gist
from tqdm import tqdm
from skimage import data, feature, transform
'''
Used to produce 960d GIST feature and 1764d HOG feature.

Install lear-gist-python(https://github.com/whitphx/lear-gist-python) on a linux machine, then you are good to go.

'''
def extract_features(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return None
    resized_image = transform.resize(img, (256, 256))
    hog_features = hog(resized_image, orientations=9,pixels_per_cell=(32, 32), cells_per_block=(2, 2), block_norm='L2-Hys',feature_vector=True,channel_axis=-1)
    print(f"HOG feature vector length: {len(hog_features)}")
    gist_features = gist.extract(img)
    print(f"GIST feature vector length: {len(gist_features)}")
    features = np.concatenate((hog_features, gist_features))
    return features

def process_dataset(dataset_path):
    feature_list = []
    labels = []
    class_names = os.listdir(dataset_path)

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        image_files = os.listdir(class_path)
        for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
            image_path = os.path.join(class_path, image_file)
            features = extract_features(image_path)
            if features is not None:
                feature_list.append(features)
                labels.append(class_name)

    return np.array(feature_list), np.array(labels)

def save_features(features, labels, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump((features, labels), f)

if __name__ == "__main__":
    dataset_path = "" # your dataset directory path
    output_file = "Features.pkl"

    features, labels = process_dataset(dataset_path)
    save_features(features, labels, output_file)

    print(f"Features and labels saved to {output_file}")