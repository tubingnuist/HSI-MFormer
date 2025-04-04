class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.01
    # training parameters
    train_epoch = 100
    test_epoch = 5
    BATCH_SIZE_TRAIN = 64
    norm_flag = True
    gpus = '0'
    # source data information
    data = 'PaviaU'  # PaviaU-9-103-0.95 / Indian-16-200-0.9  / Houston2018-21  / Houston2013-15-0.95
    num_classes = 9
    patch_size = 15
    pca_components = 30
    test_ratio = 0.95
    # model
    model_type = 'Parallel MT'   # 'Parallel MT'  'Interval MT'  'Series MT'  'Series TM'  'Parallel Transformer-Mamba'  'Series Transformer-Mamba'  'Series Mamba-Transformer'
    depth = 3
    embed_dim = 32
    d_state = 16
    ssm_ratio = 1
    pos = False
    cls = False
    # 3DConv parameters
    conv3D_channel = 32
    conv3D_kernel_1 = (5, 5, 5)  #(5, 5, 5)
    conv3D_kernel_2 = (7, 7, 7)  #(7, 7, 7)
    conv3D_kernel_3 = (9, 9, 9)  #(9, 9, 9)
    dim_patch = patch_size - conv3D_kernel_1[1] + 1  # 8
    dim_linear_1 = pca_components - conv3D_kernel_1[0] + 1  # 28
    dim_linear_2 = pca_components - conv3D_kernel_2[0] + 1  # 28
    dim_linear_3 = pca_components - conv3D_kernel_3[0] + 1  # 28
    # paths information
    checkpoint_path = ('./' + "checkpoint/" + data + '/' + model_type + '_TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN)\
                      + '/PatchSize' + str(patch_size) + '_TestRatio' + str(test_ratio) \
                      + '/'  + 'Depth' + str(depth) + '_embed' + str(embed_dim) + '_dstate' + str(d_state) + '_ratio' + str(ssm_ratio)
                      + '_3Dconv' + str(conv3D_channel) + '&' + str(conv3D_kernel_1) + '&' + str(conv3D_kernel_2) + '&' + str(conv3D_kernel_3) + '/')
    logs = checkpoint_path

config = DefaultConfigs()

