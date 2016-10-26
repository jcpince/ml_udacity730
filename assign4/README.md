```text
The script notmnist_utils.py is a helper allowing to manipulate the datasets
found in ../data (nomnist training, valid and test).

The script 4_convolutions.py is the neural network script. It provides a few
functions like restart_training(), continue_training(),...

Typical usage:
1 - If not already done, download and build the pickle file (see
  helpers/build_nomnit_pickle.py)
2 - start a python3 interactive session:
  ipython3 -i 4_convolutions
3 - In the console, and if not already trained, run the command
  restart_training()
3.1 - If the training is done, you can refine it with:
  continue_training()
3.2 - If you're fine with your training, you can skip this step
4 - Evaluate the network on some test set:
  evaluate_cnn_on_ds(ds_name)
  # where ds_name can be 'train', 'valid' or 'test'
5 - Check what was badly predicted with:
  bad_pred = get_bad_predictions(ds_name):
  # where ds_name can be 'train', 'valid' or 'test'
5.1 - Visualize the errors with:
  show_image_index(ds_name, index)
  # where ds_name can be 'train', 'valid' or 'test'
  # and index is an element of bad_pred
```
