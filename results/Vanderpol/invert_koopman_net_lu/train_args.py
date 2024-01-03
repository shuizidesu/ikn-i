from tap import Tap


class Args(Tap):
    model: str = "invert_koopman_net_lu"
    """One of: 'invert_koopman_net', 'invert_koopman_net_lu'"""
    mode: str = "train"
    """One of: 'train', 'test'"""

    env: str = "Vanderpol"
    """One of: 'Vanderpol', 'DCMotor', 'DCMotor_delay', 'KdV', 'Franka'"""
    output_dir: str = "./results/" + env + "/" + model
    """The dictionary to save the results and the trained model to"""
    data_dir_save: str = "./data/" + env
    data_dir_load_train: str = "./data/" + env + "/train_data_50000_16.npy"
    data_dir_load_test: str = "./data/" + env + "/test_data_2000_900.npy"
    load_ckpt_dir = "./results/" + env + "/" + model + '/ckpt-299/model.pt'
    """The dictionary to save and load the data"""
    is_load: bool = True

    x_dim: int = 2
    u_dim: int = 1
    """Information of the system"""
    x_blocks: list = [2, 2]
    x_channels: list = [8, 16]
    x_hiddens: list = [64, 128]
    u_blocks: list = [2, 2]
    u_channels: list = [8, 16]
    u_hiddens: list = [64, 128]
    """Information for building the neural network"""
    act_fn: str = "gelu"
    """Activation function"""

    lr: float = 1e-3
    num_epochs: int = 300
    batch_size: int = 128
    eval_batch_size: int = 1024
    log_interval: int = 100
    eval_interval: int = 10
    loss_name: str = "mse"
    gamma: float = 0.8
    device: str = "cuda"
    """The training setting and loss function to use for training. "loss_name" One of: 'mse', 'nmse', 'mae', 'nmae'"""
