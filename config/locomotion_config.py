import torch

from params_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc
    seed = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    bucket = '/home/kdyun/workspace/decidiff/code/weights/'
    dataset = 'hopper-medium-expert-v2'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianInvDynDiffusion'
    horizon = 56 # 100
    n_diffusion_steps = 100 # 200
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = False
    dim_mults = (1, 4, 8) # 1,4,8
    returns_condition = True
    calc_energy=False
    dim = 256 # 128
    condition_dropout = 0.25
    condition_guidance_w = 1.4
    test_ret = 0.9
    renderer = 'utils.RaisimRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 250 # 1000
    hidden_dim = 512 # 256 # inv network dimension
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 25000
    loss_type = 'l2'
    n_train_steps = 1e6
    batch_size = 32 # 32
    learning_rate = 2e-4 # 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 5000
    sample_freq = 1000
    eval_freq = 1000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False
