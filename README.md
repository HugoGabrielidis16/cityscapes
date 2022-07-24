## Cityscapes

# Issue encountered

To resolve : - When loading on LambdaLabs GPU got the following error :
-> 2022-07-21 13:03:26.910215: E tensorflow/stream_executor/cuda/cuda_driver.cc:1163] failed to enqueue async memcpy from device to host: CUDA_ERROR_INVALID_VALUE: invalid argument; host dst: 0x7fa1fe404700; GPU src: 0x7f97c144e040; size: 196608=0x30000

-> Google collab session stop :
Idea : set batch_size to 1

Resolved : - Error : keras_utils have no methods get_file :
-> from tensorflow.keras.utils import get_file + keras.utils.get_gile -> get_file


>>> Output has to be of shape (256,256,13) #13 the number of classes
-> Idea each color of the original image has to be shape to one of the 13 dimensions : examples blue = [1,0,0,0,0,...]
                    purple = [0,1,0,0...]
