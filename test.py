import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
from tqdm import tqdm
from PIL import Image
import cv2
import torch
import numpy as np
# from models.locate_func_handpre import Net as model
from models.GAAF_Dex import Net as model
from models.locate import Net as model_locate
from utils.viz import viz_pred_test, viz_pred_test_locate
from utils.util import set_seed, process_gt, normalize_map
from sklearn.metrics import accuracy_score
from utils.evaluation import cal_kl, cal_sim, cal_nss, AverageMeter, compute_cls_acc

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/')
parser.add_argument('--model_file', type=str, default='/')
parser.add_argument('--save_path', type=str, default='./save_preds')
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=448)
parser.add_argument('--resize_size', type=int, default=512)
#### test
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=4)
parser.add_argument('--gpu', type=str, default='5')
parser.add_argument('--viz', action='store_true', default=True)

args = parser.parse_args()

if args.divide == "Seen":
    aff_list = ['hold', "press", "click", "clamp", "grip", "open"]
else:
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]

if args.divide == "Seen":
    args.num_classes = 6
else:
    args.num_classes = 25

args.test_root = os.path.join(args.data_root, "Seen", "testset", "egocentric")
args.mask_root = args.test_root

if args.viz:
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)


if __name__ == '__main__':
    set_seed(seed=0)
    from data.datatest_func import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(aff_classes=args.num_classes).cuda()
    KLs = []
    SIM = []
    NSS = []
    all_preds = []
    all_hand_labels = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file))

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(args.divide + "_gt.t7")

    all_preds_per_task = {i: [] for i in range(6)}
    all_hand_labels_per_task = {i: [] for i in range(6)}
    # 用于存储不同任务和工具组合的预测和真实标签
    all_preds_per_task_tool = {}
    all_hand_labels_per_task_tool = {}

    for step, (image, label, mask_path, hand_label) in enumerate(tqdm(TestLoader)):
        ego_pred, func_ego_cam, hand_pred = model.func_test_forward(image.cuda(), label.long().cuda())
        cluster_sim_maps = []
        ego_pred0 = np.array(ego_pred.squeeze().data.cpu())
        ego_pred1 = normalize_map(ego_pred0, args.crop_size)

        func_ego_cam0 = np.array(func_ego_cam.squeeze().data.cpu())
        func_ego_cam1 = normalize_map(func_ego_cam0, args.crop_size)

        # ----------yf---------------------
        names = mask_path[0].split("/")
        task = names[-3]
        tool = names[-2]
        bianhao = names[-1].split('.')
        key = names[-3] + "_" + names[-2] + "_" + bianhao[-2] + "_heatmap." +bianhao[-1]

        GT_mask = GT_masks[key]
        GT_mask = GT_mask / 255.0
        GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))
        kld, sim, nss = cal_kl(func_ego_cam1, GT_mask), cal_sim(func_ego_cam1, GT_mask), cal_nss(func_ego_cam1, GT_mask)
        KLs.append(kld)
        SIM.append(sim)
        NSS.append(nss)

        if args.viz:
            img_name = key.split(".")[0]
            viz_pred_test(args, image, ego_pred1, GT_mask, aff_list, label, img_name, func_ego_cam1)
        # -----------------metric of grasp type---------------#
        hand_pred_probs = hand_pred.squeeze().data.cpu().numpy()
        hand_pred_labels = np.argmax(hand_pred_probs, axis=0)

        hand_label_index = hand_label.argmax(dim=1).item()  # Convert hand_label to the gesture index

        # 根据任务标签来分配预测和真实手势标签
        task_id = label.item()
        all_preds_per_task[task_id].append(hand_pred_labels)
        all_hand_labels_per_task[task_id].append(hand_label_index)

        # 提取任务和工具名称
        names = mask_path[0].split("/")
        tool = names[-2]
        task_tool_key = f"{task_id}_{tool}"

        # 初始化任务-工具组合的存储
        if task_tool_key not in all_preds_per_task_tool:
            all_preds_per_task_tool[task_tool_key] = []
            all_hand_labels_per_task_tool[task_tool_key] = []

        # 根据任务-工具组合存储预测和真实标签
        all_preds_per_task_tool[task_tool_key].append(hand_pred_labels)
        all_hand_labels_per_task_tool[task_tool_key].append(hand_label_index)

    # 计算6种任务一起的平均操作成功率
    all_preds = [pred for preds in all_preds_per_task.values() for pred in preds]
    all_labels = [label for labels in all_hand_labels_per_task.values() for label in labels]
    average_success_rate = accuracy_score(all_labels, all_preds)
    mKLD = sum(KLs) / len(KLs)
    mSIM = sum(SIM) / len(SIM)
    mNSS = sum(NSS) / len(NSS)

    print(f"KLD = {round(mKLD, 3)}\nSIM = {round(mSIM, 3)}\nNSS = {round(mNSS, 3)}")
    # 计算每个任务-工具组合的成功率
    precision_per_task_tool = {}
    recall_per_task_tool = {}

    for task_tool_key in all_preds_per_task_tool:
        preds = np.array(all_preds_per_task_tool[task_tool_key])
        labels = np.array(all_hand_labels_per_task_tool[task_tool_key])
        precision_per_task_tool[task_tool_key] = accuracy_score(labels, preds)

    # 计算每种任务的平均操作成功率
    precision_per_task = {}
    recall_per_task = {}

    for task_id in range(6):
        if all_hand_labels_per_task[task_id]:  # 确保不为空
            preds = np.array(all_preds_per_task[task_id])
            labels = np.array(all_hand_labels_per_task[task_id])
            precision_per_task[task_id] = accuracy_score(labels, preds)
        else:
            precision_per_task[task_id] = None
            recall_per_task[task_id] = None

    print("Average success rate across all tasks:", average_success_rate)
    print("Success rate per task-tool combination:", precision_per_task_tool)
    print("Average success rate per task:", precision_per_task)
    print('mKLD:',mKLD, 'mSIM', mSIM, 'mNSS', mNSS)
        # -----------------metric of grasp type---------------#
