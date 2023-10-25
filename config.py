class DefaultConfigs(object):
    n_classes = 192  ## number of classes
    img_weight = 224  ## image width
    img_height = 224  ## image height
    batch_size = 16 ## batch size
    epochs = 50    ## epochs
    learning_rate=0.00005  ## learning rate
    model_name = "tf_efficientnet_b0"  ##Model_name

    folder_path = "/mnt/fast/nobackup/scratch4weeks/ds01502/MLDataset-Knife/" ## folder path
config = DefaultConfigs()
