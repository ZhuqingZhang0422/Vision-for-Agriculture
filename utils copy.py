import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import _pickle as pickle
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from numpy import asarray
from matplotlib import image as mpimg
import matplotlib


def load_agri_version(train_dir,labels_dir,name_save):
    '''
    Calculate classification rate according to the required 6 classes
    Input: Train_directory ---- str
           Labels_directory ---- str
           Name_save ---- str
    Output: Label data ---- dict[list]   
    '''
    labels_init = defaultdict(list)
    # initialize label dictionary according to the training data
    for f in [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]:
        labels_init[f[0:-4]] = [0,0,0,0,0,0]
    for ind, l in enumerate([f for f in os.listdir(labels_dir)][1:]):
        direc = labels_dir + l
        for file in tqdm([f for f in os.listdir(direc) if os.path.isfile(os.path.join(direc, f))]):
            if file != ".DS_Store":
                # Calculate coverage according to the corresponding label
                rate = (np.array(Image.open(os.path.join(direc, file)).convert('L'))/255).mean()
                labels_init[file[0:-4]][ind] = rate
    # save dictionary for later use
    filename = name_save + ".pkl"
    f = open(filename,"wb")
    pickle.dump(labels_init,f)
    f.close()
    return labels_init


def image_argment_train(fig_id):
    '''
    Process single image in training set
    Input: Image Id ---- str
    Output: Processed image (Apply mask + boundry, stack NIR) ---- np.array
    '''
    rgb_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/train/images/rgb'
    nir_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/train/images/nir'
    bdry_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/train/boundaries'
    mask_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/train/masks'
    
    rgb_img = mpimg.imread(os.path.join(rgb_path, fig_id + '.jpg'))/255
    nir_img = mpimg.imread(os.path.join(nir_path, fig_id + '.jpg')).reshape((512,512,1))/255
    bdry_img = mpimg.imread(os.path.join(bdry_path, fig_id + '.png'))
    mask_img = mpimg.imread(os.path.join(mask_path, fig_id + '.png'))
    #input_img = np.concatenate([rgb_img, nir_img], axis=2) / 255. # Concatenate the RGB and NIR
    
    # stack mask and boundry
    final_mask = np.multiply(mask_img,bdry_img).reshape(512,512,1)
    
    # add nir channel to original data
    input_img = np.concatenate([rgb_img, nir_img], axis=2) / 255. # Concatenate the RGB and NIR
    
    # apply mask and boundry to image
    final_img_rgb = np.multiply(final_mask,rgb_img)
    final_img_nir = np.multiply(final_mask,nir_img)

    return final_img_rgb, final_img_nir

def image_argment_val(fig_id):
    '''
    Process single image in validation set
    Input: Image Id ---- str
    Output: Processed image (Apply mask + boundry, stack NIR) ---- np.array
    '''
    rgb_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/val/images/rgb'
    nir_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/val/images/nir'
    bdry_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/val/boundaries'
    mask_path = '/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/val/masks'
    
    rgb_img = mpimg.imread(os.path.join(rgb_path, fig_id + '.jpg'))/255
    nir_img = mpimg.imread(os.path.join(nir_path, fig_id + '.jpg')).reshape((512,512,1))/255
    bdry_img = mpimg.imread(os.path.join(bdry_path, fig_id + '.png'))
    mask_img = mpimg.imread(os.path.join(mask_path, fig_id + '.png'))
    #input_img = np.concatenate([rgb_img, nir_img], axis=2) / 255. # Concatenate the RGB and NIR
    
    # stack mask and boundry
    final_mask = np.multiply(mask_img,bdry_img).reshape(512,512,1)
    
    # add nir channel to original data
    input_img = np.concatenate([rgb_img, nir_img], axis=2) / 255. # Concatenate the RGB and NIR
    
    # apply mask and boundry to image
    final_img_rgb = np.multiply(final_mask,rgb_img)
    final_img_nir = np.multiply(final_mask,nir_img)

    return final_img_rgb, final_img_nir


def save_process_img_train (fig_id, path_tar_rgb, path_tar_nir, resize_x = 512, resize_y = 512):
    
    '''
    Rescale the processed image and save to the required directory
    Input: fig_id ---- str
           path_tar_rgb ---- str
           path_tar_nir ---- nir
           resize_x ---- int
           resize_y ---- int
    '''
    try:
        path_tar_rgb = os.path.join(path_tar_rgb,str(resize_x) + "*" + str(resize_y))
        os.makedirs(path_tar_rgb)
    except:
        pass
            
    try:
        path_tar_nir = os.path.join(path_tar_nir,str(resize_x) + "*" + str(resize_y))
        os.makedirs(path_tar_nir)
    except:
        pass
    
    # process rgb and nir data with mask and boundry
    final_img_rgb, final_img_nir = image_argment_train(fig_id)
    # resize and save rgb data
    matplotlib.image.imsave(os.path.join(path_tar_rgb, fig_id + '.jpg'), final_img_rgb)
    rgb_resize = Image.open(os.path.join(path_tar_rgb, fig_id + '.jpg')).resize((resize_x,resize_y))
    rgb_resize.save(os.path.join(path_tar_rgb, fig_id + '.jpg'),quality = 90)
    
    # resize and save nir data
    final_img_nir = np.concatenate([final_img_nir, final_img_nir, final_img_nir], axis=2)
    matplotlib.image.imsave(os.path.join(path_tar_nir, fig_id + '.jpg'), final_img_nir)
    nir_resize = Image.open(os.path.join(path_tar_nir, fig_id + '.jpg')).resize((resize_x,resize_y))
    nir_resize.save(os.path.join(path_tar_nir, fig_id + '.jpg'),quality = 90)
    
    return rgb_resize, nir_resize



def load_agri_train(size_res,number):
    '''
    Load agricultural version data
    '''
    train_dir = "/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/train/images/rgb"
    labels_dir = "/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/train/labels/"
    y_train = utils.load_agri_version(train_dir,labels_dir,"y_train")
    
    train_val = "/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/val/images/rgb"
    labels_val = "/Users/zhuqing/Documents/Github/Vision-for-Agriculture/Agriculture-Vision/val/labels/"
    y_val = utils.load_agri_version(train_val,labels_val,"y_val")
    
    
def process_tarin(train_data, path_tar_rgb, path_tar_nir, resize_x = 30, resize_y = 30):
    '''
    Process training data, resize and save images of RGB and NIR channel
    Input: train_data ---- database.pkl
           path_tar_rgb ---- str
           path_tar_nir ---- str
           resize_x ---- int
           resize_y ---- int
    Output: None 
    '''
    dbfile = open(train_data, 'rb')     
    db = pickle.load(dbfile)
    for fig_id in tqdm(db):
        save_process_img_train (fig_id, path_tar_rgb, path_tar_nir, resize_x, resize_y)
    return None 






def get_CIFAR10_data(num_training = 49000, num_validation=1000, num_test=10000):
  #Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
  #it for the softmax classifier. 
  # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # subsample the data

    X_train, y_train, X_val, y_val, X_test, y_test = subsample(num_training,num_validation,num_test,X_train,y_train,X_test,y_test)

  # visualize a subset of the training data
  
    visualize_cifar10(X_train,y_train)

  # preprocess data
  
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
  
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(cifar10_root):
  # load all of cifar 
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(cifar10_root, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(cifar10_root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.

def visualize_cifar10(X_train,y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
    plt.close()

# subsampling  the data

def subsample(num_training,num_validation,num_test,X_train,y_train,X_test,y_test):

  # Our validation set will be num_validation points from the original
  # training set.

  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]

  # Our training set will be the first num_train points from the original
  # training set.
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]

  # We use the first num_test points of the original test set as our
  # test set.

  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess(X_train,X_val,X_test):

  # Preprocessing: reshape the image data into rows

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

  # As a sanity check, print out the shapes of the data
    print('Training data shape: %d,%d' %X_train.shape)
    print('Validation data shape: %d,%d' %X_val.shape)
    print('Test data shape: %d,%d' %X_test.shape)

  # Preprocessing: subtract the mean image
  # first: compute the image mean based on the training data

    mean_image = np.mean(X_train, axis=0)
#  plt.figure(figsize=(4,4))
#  plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image

  # second: subtract the mean image from train and test data

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

  # third: append the bias dimension of ones (i.e. bias trick) so that our softmax regressor
  # only has to worry about optimizing a single weight matrix theta.
  # Also, lets transform data matrices so that each image is a row.

    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    print('Training data shape with bias term: %d,%d' %X_train.shape)
    print('Validation data shape with bias term: %d,%d' %X_val.shape)
    print('Test data shape with bias term: %d,%d' %X_test.shape)

    return X_train, X_val, X_test
  

def sigmoid(X):
    return 1./(1. + np.exp(-X))
