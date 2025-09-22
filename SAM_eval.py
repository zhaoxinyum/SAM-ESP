import argparse
from PIL import Image
import os

# from tqdm import tqdm
join = os.path.join
import torch
import logging
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from calculate_metric import *
from MyDataset import MyDataset
from SAM import SAM
import datetime


# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the SAM model.")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument(
        "--checkpoint_folder", type=str, default="SAM_fine/CASIAiris_data/2025-03-24_07.43.35"
    )
    parser.add_argument("--data_name", type=str, default="CASIAiris_data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


def setup_logging(log_file='evaluation.log'):
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
    import time
    args = parse_args()
    output_folder = os.path.join(args.checkpoint_folder, "test_result")
    masks_folder = os.path.join(output_folder, 'prediction')
    os.makedirs(masks_folder, exist_ok=True)
    logger = setup_logging(log_file=join(output_folder, 'evaluation.log'))

    checkpoint = os.path.join(args.checkpoint_folder, 'parameter/model_best.pth')
    logger.info(f'Using device {args.device}')
    logger.info(f"checkpoint:{checkpoint}")
    # ----model-------
    sam_model = sam_model_registry["vit_b"]()
    model = SAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        device=args.device
    ).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    np.random.seed(1)
    test_image_path = join('../dataset', args.data_name, 'test', 'image')
    test_mask_path = join('../dataset', args.data_name, 'test', 'mask')

    test_dataset = MyDataset(test_image_path, test_mask_path, eval=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # --------------evaluation------------------
    logger.info(f"begin evaluation of {args.data_name} using {args.checkpoint_folder}")
    begin_time = datetime.datetime.now()
    logger.info(f'Test started at {begin_time.strftime("%Y-%m-%d %H:%M:%S")}')
    model.eval()

    iou_scores, bd_scores, bdsd_scores, dice_scores = 0, 0, 0, 0
    with torch.no_grad():
        for i, (image, gt2D, boxes, imagename) in enumerate((test_loader)):
            boxes_np = boxes.detach().cpu().numpy()
            gt2D = gt2D.to(device=args.device, dtype=torch.float32)

            # 记录推理开始时间
            start_time = time.perf_counter()  # 高精度计时
            output = model(image.numpy(), boxes_np)

            # 计算推理耗时（毫秒）
            inference_time = (time.perf_counter() - start_time) * 1000  # 转为毫秒
            # 将输出结果转换为二进制
            binary_predictions = (output > 0).float()
            binary_predictions = binary_predictions.squeeze(0)
            gt2D = gt2D.squeeze(0)

            # 计算IoU,Dice,BD,BDSD
            iou = calculate_iou(binary_predictions, gt2D)
            dice = get_dice(binary_predictions, gt2D)
            bd, bdsd = calculate_bd(binary_predictions, gt2D)

            logger.info(
                f"Picture [{imagename[0]}]: iou-{iou:.4f},dice-{dice:.4f},bd-{bd:.4f},bdsd-{bdsd:.4f},time-{inference_time:.2f}ms")

            iou_scores = iou_scores + iou
            bd_scores = bd_scores + bd
            bdsd_scores = bdsd_scores + bdsd
            dice_scores = dice_scores + dice

            # 保存输出结果
            # 将二进制张量数据转换为 uint8 类型的 numpy 数组
            result = binary_predictions.mul(255).byte().cpu().numpy()

            # 创建 PIL 图像对象
            image = Image.fromarray(result, mode='L')
            file_name = f"{imagename[0]}_prediction.png"
            image.save(join(masks_folder, file_name))
    mean_iou = iou_scores / len(test_loader)
    mean_bd = bd_scores / len(test_loader)
    mean_bdsd = bdsd_scores / len(test_loader)
    mean_dice = dice_scores / len(test_loader)
    logger.info("Evaluation finish")
    logger.info(f"mean_iou - {mean_iou}, mean_dice - {mean_dice}, mean_bd - {mean_bd}, mean_bdsd - {mean_bdsd}")
    end_time = datetime.datetime.now()
    elapsed_time = end_time - begin_time

    logger.info(f'Test finished at {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Total eval time: {elapsed_time},test_image_num:{len(test_loader)}')


if __name__ == "__main__":
    main()
