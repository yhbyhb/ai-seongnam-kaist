import scipy.misc
import numpy as np

def onehot(X, n=None, negative_class=0.):
  X = np.asarray(X).flatten()
  if n is None:
      n = 10
  Xoh = np.ones((len(X), n)) * negative_class
  Xoh[np.arange(len(X)), X] = 1.
  return Xoh


def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
  h,w = X.shape[1], X.shape[2]
  img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

  for n,x in enumerate(X):
      j = n // nh_nw[1]
      i = n % nh_nw[1]
      img[j*h:j*h+h, i*w:i*w+w, :] = x

  scipy.misc.imsave(save_path, img)

