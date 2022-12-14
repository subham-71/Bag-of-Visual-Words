# Name : Subham Subhasis Sahoo
# Entry No: 2020CSB1317

""" A classification of the Fashion - MNIST dataset using Bag of Visual words technique with K Means clustering. """

""" Importing Libraries """

from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq
from numpy.linalg import norm
from keras.datasets import fashion_mnist
from scipy.spatial.distance import cdist
import cv2


def img_dict(img,labels):
  image_dict = {}

  for i in range(0,labels.shape[0]) :
    image_dict[i] = (labels[i],img[i])

  return image_dict


"""Feature Extraction"""

def feature_extract(train_data) :
 
  # SIFT Feature

  extractor = cv2.SIFT_create(sigma=2)
  
  keypoints = {}
  descriptors = {}

  for index,img in train_data.items():

      img_keypoints, img_descriptors = extractor.detectAndCompute(img[1], None)
      keypoints[index] = (img[0],img_keypoints)
      descriptors[index]= (img[0],img_descriptors)
  
  return descriptors

""" K Means """

def util_k_means(distances):
  points=[]
  for i in distances:
    points.append(np.argmin(i))
  
  points = np.array(points)
  return points

def k_means(data,k,iters):

  indices = np.random.choice(len(data), k, replace=False)
  codebook = data[indices, :]
    
  distances = cdist(data, codebook ,'euclidean')
  points = util_k_means(distances)
    
  for iter in range(iters): 
      print(f"Iteration : {iter+1}")

      codebook = []
      for ind in range(k):
        center = data[points==ind].mean(axis=0) 
        codebook.append(center)

      codebook = np.vstack(codebook)
      distance_new = cdist(data, codebook ,'euclidean')
      points = util_k_means(distance_new)

  return codebook


"""Visual Dictionary"""

def CreateVisualDictionary(descriptors, k) :

  descriptor_list = []

  for index,des in descriptors.items():
    if des[1] is None :
      continue
    else :
      for v in des[1]:
        descriptor_list.append(v)
  
  descriptor_list = np.stack(descriptor_list)

  # KMeans Clustering

  codebook = k_means(descriptor_list,k,100)
  np.savetxt(f'codebook',codebook)

  return codebook

"""Frequency Vectors"""

# Computes Histograms (Frequency Vectors) for each image in train set
def ComputeHistogram(descriptors,codebook) :

  desc = {}
  for index,des in descriptors.items():
    if des[1] is None :
      continue
    else :
      desc[index] = (des[0],des[1])

  # Computing visual words in each image
  visual_words= {}
  for index, img_descriptors in desc.items():
      img_visual_words, distance = vq(img_descriptors[1], codebook)
      visual_words[index] = (img_descriptors[0], img_visual_words)

  # Generating frequency vector of visual words for each image 
  frequency_vectors = {}
  counter = 0

  for index, img_visual_words in visual_words.items():
      img_frequency_vector = np.zeros(len(codebook))
      for word in img_visual_words[1]:
          img_frequency_vector[word] += 1
      frequency_vectors[counter] = (img_visual_words[0] , img_frequency_vector)
      counter+=1

  return frequency_vectors

"""TF-IDF Vectors"""

def generate_tfidf_vectors(frequency_vectors):

  N = len(frequency_vectors)
  n = len(frequency_vectors[0][1])

  doc_frequency = np.zeros(n).tolist()

  for index,freq_vec in frequency_vectors.items():
    doc_frequency += freq_vec[1]

  inverse_doc_frequency = np.log(N/ doc_frequency)

  tfidf = {}

  counter =0
  for index, freq_vec in frequency_vectors.items() :
    tfidf[counter] = (freq_vec[0], freq_vec[1]*inverse_doc_frequency)
    counter+=1

  return tfidf 

"""MatchHistogram Function"""

# Returns the predicted labels of each image in the Test Set
def MatchHistogram(freq_vectors_test , train_vecs):
  actual = []
  pred = []

  for index, img in freq_vectors_test.items() :
    label_code = predict(img[1] , train_vecs)
    pred.append(label_code)
    actual.append(img[0])

  return (actual, pred)

# Predicts for a single test image
def predict(test_freq_vec , train_vecs):
  
  a = test_freq_vec
  b = train_vecs

  cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
  idx = np.argsort(-cosine_similarity)[:1]

  return  freq_vectors_train[idx[0]][0]

# Pre-processing frequency vectors to list 
def util_freq_vectors(freq_vectors_train) : 
  freq_vectors_train_match = []

  for index, vec in freq_vectors_train.items():
    freq_vectors_train_match.append(vec[1])

  freq_vectors_train_match = np.array(freq_vectors_train_match)
  
  return freq_vectors_train_match


""" Classification Results """

def get_metrics(actual,pred):

  report = classification_report(actual, pred, output_dict=True)
  df = pd.DataFrame(report).transpose()
  print(classification_report(actual,pred))

  df.to_csv(f'classification_report.csv')

""" Main Function """

if __name__ == "__main__":

  # Dataset
  (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
  labels = ['T-Shirt/Top' , 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandals' , 'Shirt' , 'Sneaker', 'Bag' , 'Ankle Boots']
  train_data = img_dict(trainX,trainy)
  test_data = img_dict(testX,testy)


  # Extracting Features
  print("Extracting Features ... ")
  descriptors_train = feature_extract(train_data)
  descriptors_test = feature_extract(test_data)

  # Creating Visual Dictionary
  print("Creating Visual Dictionary ...")
  n_clusters = 68
  codebook_train = CreateVisualDictionary(descriptors_train,n_clusters)

  # Compute Histogram
  print("Computing Histograms ...")
  freq_vectors_train = ComputeHistogram(descriptors_train,codebook_train)
  freq_vectors_test = ComputeHistogram(descriptors_test,codebook_train)

  # Using TFIDF to balance the frequency train vectors
  freq_vectors_train = generate_tfidf_vectors(freq_vectors_train)
  freq_vectors_train_match = util_freq_vectors(freq_vectors_train)

  # Match Histogram
  print("Predicting Labels for Test set ... ")
  actual_test_labels , predicted_test_labels= MatchHistogram(freq_vectors_test, freq_vectors_train_match)
  
  # Classification Results
  print("Classification Report")
  get_metrics(actual_test_labels,predicted_test_labels)
 
