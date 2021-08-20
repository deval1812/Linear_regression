import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        raise NotImplementedError
    def __call__(self,features, is_train=False):
        raise NotImplementedError

def get_features(csv_path,is_train=False,scaler=None):
  df = pd.read_csv(csv_path)
  df_X = df.iloc[:,0:59]
  df_X = df_X.to_numpy()
  m,n = df_X.shape
  mean_X = np.mean(df_X,axis=0)
  stddev_X = np.std(df_X,axis=0)
  for j in range(n):
    for i in range(m):
      df_X[i,j] = (df_X[i,j] - mean_X[j])/stddev_X[j]       #Z-score normalization
  return df_X

def get_targets(csv_path):
  df = pd.read_csv(csv_path)
  df_y = df.iloc[:,59:60]
  df_y = df_y.to_numpy()
  m, n = df_y.shape
  mean_y = np.mean(df_y, axis=0)
  stddev_y = np.std(df_y, axis=0)
  for i in range(m):
    df_y[i] = (df_y[i] - mean_y) / stddev_y
  return df_y

def analytical_solution(feature_matrix, targets, C=0.0):
  m,n = feature_matrix.shape
  w = np.array(np.zeros([m,1]))
  f_t = np.transpose(feature_matrix)
  temp1 = np.linalg.inv(np.matmul(f_t,feature_matrix) + C*np.eye(n))
  temp2 = np.matmul(f_t,targets)
  w = np.matmul(temp1,temp2)
  return w

def get_predictions(feature_matrix, weights):
  y = np.matmul(feature_matrix,weights)
  return y

def mse_loss(feature_matrix, weights, targets):
  err = 0
  m,n = feature_matrix.shape
  for i in range(m):
    err = err + ((np.matmul(feature_matrix[i,:],weights))-targets[i])**2
  err = err/m
  return err

def l2_regularizer(weights):
  m = weights.shape
  penalty = 0
  for i in range(m[0]):
    penalty = penalty + (weights[i]**2)
  return penalty

def loss_fn(feature_matrix, weights, targets, C=0.0):
  m,n = feature_matrix.shape
  err = 0
  penalty = 0
  for i in range(n+1):
    penalty = penalty + (weights[i]**2)
  for i in range(m):
    err = err + ((((weights[0] + np.matmul(feature_matrix[i,:],weights[1:60]))-targets[i])**2) + C*penalty)
  err = err/m
  loss = err + C*penalty
  return loss

def compute_gradients(feature_matrix, weights, targets, C=0.0):
  m,n = feature_matrix.shape
  grad = np.array(np.zeros([n,1]))
  sum = 0
  for i in range(n):
    sum = 0
    for j in range(m):
      sum = sum + ((np.matmul(feature_matrix[j,:],weights))-targets[j])*feature_matrix[j,i]
    grad[i] = (2/m)*(sum + 2*C*weights[i])
  return grad

def sample_random_batch(feature_matrix, targets, batch_size):
  array = np.column_stack((feature_matrix, targets))
  np.random.shuffle(array)
  arr = array[0:batch_size]
  df = pd.DataFrame(arr)
  df_feature = df.iloc[:,:-1]
  df_target = df.iloc[:,-1]
  feature_batch = df_feature.to_numpy()
  target_batch = df_target.to_numpy()
  return [feature_batch,target_batch]

def initialize_weights(n):
  w = np.array(np.zeros([n,1]))
  return w

def update_weights(weights, gradients, lr):
    weights = weights - (lr*gradients)
    return weights

def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None):
  raise NotImplementedError

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
  weights = initialize_weights(59)
  dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
  train_loss = mse_loss(train_feature_matrix, weights, train_targets)

  print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))

  for step in range(1,max_steps+1):

    features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
    gradients = compute_gradients(features, weights, targets, C)
    weights = update_weights(weights, gradients, lr)

    if step%eval_steps == 0:
      dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
      train_loss = mse_loss(train_feature_matrix, weights, train_targets)
      print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))
  return weights

def do_evaluation(feature_matrix, targets, weights):
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    train_features, train_targets = get_features('E:/iitb/sem3/fml_pj/assignment_1/dataset/train.csv',True,None), get_targets('E:/iitb/sem3/fml_pj/assignment_1/dataset/train.csv')
    dev_features, dev_targets = get_features('E:/iitb/sem3/fml_pj/assignment_1/dataset/dev.csv',False,None), get_targets('E:/iitb/sem3/fml_pj/assignment_1/dataset/dev.csv')

    test_features = get_features('E:/iitb/sem3/fml_pj/assignment_1/dataset/test.csv',False,None)
    print(test_features.shape)

    a_solution = analytical_solution(train_features, train_targets, C=1e-8)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    stddev_y = 1353.43644244
    mean_y = 5590.36601362
    test_y = get_predictions(test_features,a_solution)
    test_y = (test_y*stddev_y) + mean_y
    print(test_y.shape)
    df_test_y = pd.DataFrame(test_y)
    #temp = np.arange(0,11894)
    #df_temp = pd.DataFrame(temp)
    df_test_y.columns=['shares']
    #df_test_y.insert(0,"instance_id",df_temp)
    #print(df_test_y)
    df_test_y.to_csv('prediction.csv')

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.001,
                        C=0.1,
                        batch_size=320,
                        max_steps=2000000,
                        eval_steps=5)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

