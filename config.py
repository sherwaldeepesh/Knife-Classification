import args

class DefaultConfigs(object):
    opt, parser = args.argument_parser()
    n_classes = 192  ## number of classes
    img_weight = opt.width  ## image width
    img_height = opt.height  ## image height
    batch_size = opt.trainbatchsize ## batch size
    epochs = 50    ## epochs
    learning_rate = opt.lr  ## learning rate
    model_name = opt.modelname  ##Model_name
    inputs = opt   #Include all input
    folder_path = opt.root ## folder path
    name = opt.exname
    optimizer = opt.optim
    wtdecay = opt.wtdecay
    head = opt.classhead
    dropout = opt.dropout
config = DefaultConfigs()
