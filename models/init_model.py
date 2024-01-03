from models.invert_koopman_net import InvertKoopmanNet
from models.invert_koopman_net_lu import InvertKoopmanNetLu

from args import Args


def init_model(args: Args):
    print(f"Initiating {args.model}")
    if args.model == "invert_koopman_net":
        model = InvertKoopmanNet(
            args.x_dim,
            args.x_blocks,
            args.x_channels,
            args.x_hiddens,
            args.u_dim,
            args.u_blocks,
            args.u_channels,
            args.u_hiddens
        ).to(args.device)
        return model
    elif args.model == "invert_koopman_net_lu":
        model = InvertKoopmanNetLu(
            args.x_dim,
            args.x_blocks,
            args.x_channels,
            args.x_hiddens,
            args.u_dim,
            args.u_blocks,
            args.u_channels,
            args.u_hiddens
        ).to(args.device)
        return model
    else:
        raise ValueError(f"Model {args.model} not implemented!")