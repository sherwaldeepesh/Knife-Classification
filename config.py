import args

class DefaultConfigs(object):
    opt, parser = args.argument_parser()
    n_classes = 192  ## number of classes
    img_weight = opt.width  ## image width
    img_height = opt.height  ## image height
    batch_size = 16 ## batch size
    epochs = 50    ## epochs
    learning_rate=0.00005  ## learning rate
    model_name = opt.modelname  ##Model_name
    inputs = opt   #Include all input
    folder_path = opt.root ## folder path
config = DefaultConfigs()
