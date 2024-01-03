import time

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from tqdm import tqdm

from models.base_model import KoopmanNet
from models.init_model import init_model

from models.losses import k_linear_loss, pred_and_eval_loss

from args import Args
import os
import shutil

from utility import plot_predictions, plot_contour, dump_json


class Collater():
    def __init__(self, x_dim: int, u_dim: int, device: str = "cuda"):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.device = device

    def __call__(self, batch_list: list):
        batch_data = list(zip(*batch_list))[0]
        batch_data = torch.stack(batch_data, dim=0)
        return dict(
            x=batch_data[:, :, self.u_dim:].to(self.device),
            u=batch_data[:, :, :self.u_dim].to(self.device)
        )


def dataset_generate(
        args: Args,
):
    if not args.is_load:
        raise NotImplementedError("Not implemented for now!")
    else:
        print("Loading dataset...")
        if not (os.path.exists(args.data_dir_load_train) and os.path.exists(args.data_dir_load_test)):
            raise ValueError("Dataset not found!")
        train_data = np.load(args.data_dir_load_train)
        print(f"Train data has been loaded from {args.data_dir_load_train}",
              f"(shape: {train_data.shape}, "
              f"type,: {type(train_data)}, "
              f"dtype,: {train_data.dtype})")
        test_data = np.load(args.data_dir_load_test)
        print(f"Test data has been loaded from {args.data_dir_load_test}, "
              f"(shape: {test_data.shape}, "
              f"type,: {type(test_data)}, "
              f"dtype,: {test_data.dtype})")
        return train_data, test_data


def evaluate(
        model: KoopmanNet,
        test_data: np.ndarray,
        eval_batch_size: int,
        x_dim: int,
        u_dim: int,
        device: str
):
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_dataset = TensorDataset(test_data_tensor)
    collate_fn = Collater(x_dim, u_dim, device)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False)
    scores = []
    all_preds = []
    all_labels = dict(
        x=[],
        u=[]
    )

    print("=== Evaluating ===")
    print(f"# examples: {len(test_data)}")
    print(f"# batch size: {eval_batch_size}")
    print(f"# batches: {len(test_loader)}")
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):

            pred_and_error = pred_and_eval_loss(
                batch_data=batch,
                net=model,
            )
            pred = pred_and_error["pred"]
            error = pred_and_error["pred_loss"]

            scores.append(error.unsqueeze(0).cpu().detach())
            all_preds.append(pred.cpu().detach())
            for key in all_labels.keys():
                all_labels[key].append(batch[key].cpu().detach())

        scores = torch.cat(scores, dim=0).numpy()
        avg_error = np.mean(scores)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        for key in all_labels.keys():
            all_labels[key] = torch.cat(all_labels[key], dim=0).numpy()

    return dict(
        avg_error=avg_error,
        scores=scores,
        all_preds=all_preds,
        all_labels_x=all_labels["x"],
        all_labels_u=all_labels["u"]
    )


def train(
        model: KoopmanNet,
        train_data: np.ndarray,
        test_data: np.ndarray,
        output_dir: str,
        num_epochs: int,
        lr: float,
        batch_size: int,
        eval_batch_size: int,
        log_interval: int,
        eval_interval: int,
        x_dim: int,
        u_dim: int,
        loss_name: str,
        gamma: float,
        device: str
):
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_tensor)
    collate_fn = Collater(x_dim, u_dim, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    print("==== Training ====")
    print(f"Output dir: {output_dir}")
    print(f"# lr: {lr}")
    print(f"# batch: {batch_size}")
    print(f"# examples: {len(train_data)}")
    print(f"# step: {len(train_loader)}")
    print(f"# epoch: {num_epochs}")

    start_time = time.time()
    global_step = 0
    all_train_losses = dict(
        total_loss=[],
        koopman_loss=[],
        pred_loss=[],
        recon_loss=[]
    )
    model.train()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_train_losses = dict(
            total_loss=[],
            koopman_loss=[],
            pred_loss=[],
            recon_loss=[]
        )
        for step, batch in enumerate(train_loader):
            losses = k_linear_loss(
                batch_data=batch,
                net=model,
                loss_name=loss_name,
                gamma=gamma
            )
            loss = losses["total_loss"]

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            for key in losses.keys():
                epoch_train_losses[key] = epoch_train_losses[key] + [losses[key].item()]
            global_step += 1
            if global_step % log_interval == 0:
                avg_losses = dict(
                    total_loss=[],
                    koopman_loss=[],
                    pred_loss=[],
                    recon_loss=[]
                )
                for key in avg_losses.keys():
                    avg_losses[key] = sum(epoch_train_losses[key]) / (len(epoch_train_losses[key]) + 1e-5)
                log_info = {
                    "epoch": epoch,
                    "step": step,
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                    "total_loss": f"{avg_losses['total_loss']:.3e}",
                    "koopman_loss": f"{avg_losses['koopman_loss']:.3e}",
                    "pred_loss": f"{avg_losses['pred_loss']:.3e}",
                    "recon_loss": f"{avg_losses['recon_loss']:.3e}",
                    "time": round(time.time() - start_time)
                }
                print(log_info)

        # Evaluate and save
        if (epoch + 1) % eval_interval == 0:
            ckpt_dir = output_dir + f"/ckpt-{epoch}"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            eval_info = evaluate(
                model=model,
                test_data=test_data,
                eval_batch_size=eval_batch_size,
                x_dim=x_dim,
                u_dim=u_dim,
                device=device
            )
            print(f"Prediction error: {eval_info['avg_error']}")
            # Save checkpoint
            ckpt_path = ckpt_dir + "/model.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)
            if x_dim < 20:
                plot_predictions(
                    all_preds=eval_info['all_preds'],
                    all_labels_x=eval_info['all_labels_x'],
                    all_labels_u=eval_info['all_labels_u'],
                    plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
                    save_dir=ckpt_dir
                )
            else:
                plot_contour(
                    all_preds=eval_info['all_preds'],
                    all_labels_x=eval_info['all_labels_x'],
                    all_labels_u=eval_info['all_labels_u'],
                    plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
                    save_dir=ckpt_dir
                )

            # Save average scores
            epoch_scores = dict(
                epoch=epoch,
                train_total_loss=float(sum(epoch_train_losses['total_loss']) / (len(epoch_train_losses['total_loss']) + 1e-5)),
                koopman_loss=float(sum(epoch_train_losses['koopman_loss']) / (len(epoch_train_losses['total_loss']) + 1e-5)),
                pred_loss=float(sum(epoch_train_losses['pred_loss']) / (len(epoch_train_losses['pred_loss']) + 1e-5)),
                recon_loss=float(sum(epoch_train_losses['recon_loss']) / (len(epoch_train_losses['recon_loss']) + 1e-5)),
                evaluate_loss=float(eval_info['avg_error']),
                time=time.time() - epoch_start_time
            )
            dump_json(epoch_scores, ckpt_dir + "/scores.json")

        scheduler.step()
        for key in all_train_losses.keys():
            all_train_losses[key] = all_train_losses[key] + epoch_train_losses[key]

    dump_json(all_train_losses, output_dir + "/train_losses.json")


def main():
    args = Args().parse_args()
    print(args)
    # Loading dataset
    train_data, test_data = dataset_generate(args)

    # Model
    model = init_model(args)
    if args.mode == "train":
        print("==== Saving args ====")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        shutil.copy('./args.py', args.output_dir + '/train_args.py')

        train(
            model=model,
            train_data=train_data,
            test_data=test_data,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            loss_name=args.loss_name,
            gamma=args.gamma,
            device=args.device
        )
    elif args.mode == "test":
        print(f"Loading the checkpoint from {args.load_ckpt_dir}")
        if not os.path.exists(args.load_ckpt_dir):
            raise FileNotFoundError(f"No checkpoint!")
        model.load_state_dict(torch.load(args.load_ckpt_dir))
        eval_info = evaluate(
            model=model,
            test_data=test_data,
            eval_batch_size=args.eval_batch_size,
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            device=args.device
        )
        print(f"Prediction error: {eval_info['avg_error']}")
        # Save checkpoint
        save_path = args.output_dir + "/test"
        plot_predictions(
            all_preds=eval_info['all_preds'],
            all_labels_x=eval_info['all_labels_x'],
            all_labels_u=eval_info['all_labels_u'],
            plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
            save_dir=save_path
        )
        if args.x_dim > 10:
            plot_contour(
                all_preds=eval_info['all_preds'],
                all_labels_x=eval_info['all_labels_x'],
                all_labels_u=eval_info['all_labels_u'],
                plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
                save_dir=save_path
            )
        save_info = dict(
            avg_error=float(eval_info['avg_error'])
        )
        dump_json(save_info, save_path + "/scores.json")
        np.save(save_path + "/all_preds.npy", eval_info['all_preds'])
        np.save(save_path + "/all_labels_x.npy", eval_info['all_labels_x'])
        np.save(save_path + "/all_labels_u.npy", eval_info['all_labels_u'])

    else:
        raise ValueError("Mode not recognized!")


if __name__ == "__main__":
    main()
