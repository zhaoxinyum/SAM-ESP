import argparse
import torch
import torch.nn as nn
import logging
import wandb
import os
import random
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ["WANDB_MODE"] = "offline"
join = os.path.join
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from calculate_metric import calculate_dice_batch
from MyDataset import MyDataset
from SAM_ESP import SAMESP
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SAM_ESP model.")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument(
        "--checkpoint", type=str, default="parameterfault/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("--model_name", type=str, default="SAM_ESP", help="model (SAM or SAM_ESP)")
    parser.add_argument("--dataname", type=str, default="CASIAiris_data",
                        help="(CASIAiris_data) or (ACDC_data) or (DTUHerlev_data) or (Refuge_data) ")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
    )
    parser.add_argument("--num_iterate", type=int, default=100, help="number of ESP module")
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    return parser.parse_args()


# set random seed
def set_seed(seed):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    for step, (image, gt2D, boxes) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad(set_to_none=True)
        boxes_np = boxes.detach().cpu().numpy()
        gt2D = gt2D.to(device=device, dtype=torch.float32)
        if device == "cpu":
            output = model(image.numpy(), boxes_np)
            loss = criterion(output, gt2D)
            loss.backward()
            optimizer.step()
        else:
            # scaler = GradScaler()
            scaler = torch.amp.GradScaler()

            with torch.amp.autocast('cuda'):
                # with autocast():
                output = model(image.numpy(), boxes_np)
                loss = criterion(output, gt2D)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def val(model, val_loader, criterion, calculatedice, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    for image, gt2D, boxes in val_loader:
        boxes_np = boxes.detach().cpu().numpy()
        gt2D = gt2D.to(device=device, dtype=torch.float32)
        with torch.amp.autocast('cuda'), torch.no_grad():
            outputs = model(image.numpy(), boxes_np)
            loss = criterion(outputs, gt2D)
            dice = calculatedice(outputs, gt2D)
            total_loss += loss.item()
            total_dice += dice

    return total_loss / len(val_loader), total_dice / len(val_loader)


def setup_logging(log_file='training.log'):
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 创建一个logger
    logger = logging.getLogger(__name__)

    # 创建一个文件处理器，并将其级别设置为INFO
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建一个格式化器，并将其添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(file_handler)

    return logger


def main():
    # -------解析命令行参数--------
    args = parse_args()

    set_seed(args.seed)
    # 创建记录文件夹
    time_str = datetime.datetime.now().strftime(
        '%Y-%m-%d_%H.%M.%S')  # 执行脚本时的时间
    # 确定输出文件夹存在
    model_save_path = os.path.join("work_dir", args.model_name, args.dataname, f"{args.num_iterate}", time_str)
    os.makedirs(model_save_path, exist_ok=True)

    # -------logging------------
    logger = setup_logging(os.path.join(model_save_path, 'training.log'))
    logger.info("Arguments received: %s", args)
    logger.info(f"random seed: {args.seed}")
    # set wandb

    wandb.init(project=args.model_name + '_train', name='experiment_' + args.dataname, resume='allow', anonymous='must')
    wandb.config.update(dict(epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.lr,
                             ))

    # ---------model------------
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = SAMESP(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        device=args.device,
        num_iterate=args.num_iterate
    ).to(args.device)

    # ---------数据-----------
    train_image_path = join('../dataset', args.dataname, 'train', 'image')
    train_mask_path = join('../dataset', args.dataname, 'train', 'mask')

    val_image_path = join('../dataset', args.dataname, 'val', 'image')
    val_mask_path = join('../dataset', args.dataname, 'val', 'mask')

    train_dataset = MyDataset(train_image_path, train_mask_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_dataset = MyDataset(val_image_path, val_mask_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    # 优化器：只对encoder和maskdecoder的参数进行优化
    img_mask_encdec_params = list(model.image_encoder.parameters()) + list(
        model.mask_decoder.parameters())

    optimizer = torch.optim.Adam(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,
                                                           factor=0.5)

    BEST_SCORE = 0
    # 训练模型
    begin_time = datetime.datetime.now()
    logger.info(f'Training started at {begin_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(
        f"begin training {args.model_name} on {args.dataname} with learning_rate {args.lr} for {args.num_epochs} epochs, batch_size:{args.batch_size}")
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, args.device)

        val_loss, val_score = val(model, val_loader, criterion, calculate_dice_batch, args.device)
        scheduler.step(val_loss)
        logger.info(
            f"Epoch [{epoch + 1}/{args.num_epochs}]: Train Loss - {train_loss:.4f}, Validation Loss - {val_loss:.4f}, val_score-{val_score:.4f}")
        wandb.log({
            'train loss': train_loss,
            'epoch': epoch,
            'validation Loss': val_loss,
            'val score': val_score
        }
        )
        # 保存最新的模型
        # os.makedirs(join(model_save_path, args.dataname), exist_ok=True)
        # torch.save(model.state_dict(), join(model_save_path,"latest.pth"))
        if val_score > BEST_SCORE:
            BEST_SCORE = val_score
            torch.save(model.state_dict(), join(model_save_path, "model_best.pth"))
            logger.info(f'BEST {epoch + 1} saved!')

    end_time = datetime.datetime.now()
    elapsed_time = end_time - begin_time

    logger.info(f'Training finished at {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Total training time: {elapsed_time}')


if __name__ == "__main__":
    main()
