def default_model_options():
    options = {}
    options["hidden-dim"]=768
    options["num-classes"]=500
    options["relu-type"]='prelu'
    options["tcn_options"] = {
        'multiscale': True,
        'middle_pick': False,
        'aux_loss': True,
        'multibranch': True,
        'joint_dense': False,
        'num_layers': 4,
        'dropout': 0.2,
        'no_padding': False,
        'symmetric_chomp': True,
        'kernel_size': [3, 5, 7]}
    return options