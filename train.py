import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from config import config
from util import get_data_package, converter, tensor2str, saver, get_alphabet, must_in_screen, convert_char, get_radical_alphabet, get_alphabet_ddcm, get_radical_alphabet_ddcm
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, SequentialLR, MultiStepLR

import datetime
import time
from model.ocr_decoder import Transformer#, LabelSmoothing
import os
import torch.distributed as dist

from PIL import Image
import cv2
import pickle
import numpy as np

saver()

alphabet_ddcm = get_alphabet_ddcm()
radical_alphabet_ddcm = get_radical_alphabet_ddcm()
# print(f'alphabet:{alphabet}')
# print(f'alphabet_ddcm:{alphabet_ddcm}')
if 'PAD' in alphabet_ddcm:
    alphabet_ddcm.remove('PAD')

def setup_distributed():
    dist.init_process_group(backend="nccl")  
    local_rank = int(os.environ["LOCAL_RANK"]) 
    torch.cuda.set_device(local_rank) 
    return local_rank


local_rank = setup_distributed()

model = Transformer().cuda()
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))
    print('loading！！！')

optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9, weight_decay=1e-4)
# print(f"初始学习率: {optimizer.param_groups[0]['lr']}")
# 在第15个epoch时，学习率降为0.1倍
scheduler = MultiStepLR(
    optimizer,
    milestones=[10,15],  # 指定调整节点
    gamma=0.1  # 调整倍率
)

train_loader, test_loader, train_sampler = get_data_package()

#特征匹配
classify_data = torch.load(config['data_classify'])
text_features = classify_data["features"].to('cuda')   # [N, 2048]

log_file_path = './history/{}/train_log.txt'.format(config['exp_name'])
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
char_topk_path = './history/{}/topk_log.txt'.format(config['exp_name'])
os.makedirs(os.path.dirname(char_topk_path), exist_ok=True)

char_features_path = config['radical_encode']
char_features = torch.load(char_features_path)  # 字典格式：{char: [f_code, f_parent, f_child, f_depth]}
alphabet_size = len(alphabet)  # 3757 (包括 start 和 end)
print(alphabet_size)

feature_dim = len(char_features[alphabet_ddcm[0]][0]) 

f_code = torch.zeros((alphabet_size, feature_dim), dtype=torch.float32).cuda()
f_parent = torch.zeros((alphabet_size, feature_dim), dtype=torch.float32).cuda()
f_child = torch.zeros((alphabet_size, feature_dim), dtype=torch.float32).cuda()
f_depth = torch.zeros((alphabet_size, feature_dim), dtype=torch.float32).cuda()

for i, char in enumerate(alphabet_ddcm):
    f_code[i + 1] = torch.tensor(char_features[char][0], dtype=torch.float32)
    f_parent[i + 1] = torch.tensor(char_features[char][1], dtype=torch.float32)
    f_child[i + 1] = torch.tensor(char_features[char][2], dtype=torch.float32)
    f_depth[i + 1] = torch.tensor(char_features[char][3], dtype=torch.float32)


pos_entropy_path = config['pos_entropy_path']
pos_entropy_data = torch.load(pos_entropy_path)  # {char: 512维向量}
pos_dim = 512  # 确定嵌入维度
pos_entropy = torch.zeros((alphabet_size, pos_dim), dtype=torch.float32).cuda()  # 初始化 (3757, 512)
for i, char in enumerate(alphabet_ddcm):
    pos_entropy[i + 1] = pos_entropy_data.get(char, torch.zeros(pos_dim))  # 若不存在则填充 0 向量

def train(epoch, iteration, image, length, text_input, text_gt, ids_sequences):
    global times
    model.train()
    optimizer.zero_grad()


    reg_list = []
    for item in text_gt:
        reg_list.append(text_features[item].unsqueeze(0))
    reg = torch.cat(reg_list, dim=0)
    
    start_time = time.time()
    result = model(epoch, image, length, text_input, 
    f_code, f_depth, f_parent, f_child, pos_entropy=pos_entropy)
    
    text_pred = result['pred']
    conv_features = result['conv']
    topk_id = result['topk']

    text_pred = text_pred / text_pred.norm(dim=1, keepdim=True)
    final_res =  text_pred @ text_features.t()

    pred_indices = torch.argmax(final_res, dim=1) 
    loss_rec = criterion(final_res, text_gt) 
    loss_dis = - criterion_dis(text_pred, reg)  
    loss = loss_rec +  0.001 * loss_dis 

    end_tiime = time.time()
    if iteration % 100 == 0:
        log_message = (
            'epoch: {} | iter: {}/{} | loss_rec: {:.4f}  | loss_dis: {:.4f} |  learning_rate: {:.6f}'
            .format(epoch, iteration, len(train_loader), loss_rec.item(), loss_dis.item(),  optimizer.param_groups[0]['lr'])
        )

        print(log_message)
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_message)

        pred_results = []
        for i in range(config['batch']): 
            main_idx = pred_indices[i*2].item()
            main_char = alphabet[main_idx] if (main_idx != 3756 and main_idx < len(alphabet)) else "<END>"

            gt_idx = text_gt[i*2].item()
            gt_char = alphabet[gt_idx] if (gt_idx != 3756 and gt_idx < len(alphabet)) else "<END>"
            topk_chars = []
            for k in range(topk_id.size(1)):  # 遍历每个topk位置
                idx = topk_id[i, k].item()
                if idx == 3756:
                    topk_chars.append("<END>")
                elif idx < len(alphabet_ddcm):
                    topk_chars.append(alphabet[idx])
                else:
                    topk_chars.append("<UNK>")
            topk_str = ", ".join(topk_chars)
          
            pred_results.append(f"{i}-true: {gt_char}, char: {main_char}, topk: {topk_str}")

        with open(char_topk_path, 'a') as char_file:
            char_file.write("\n".join(pred_results) + "\n\n") 

    loss.backward()
    optimizer.step()

    times += 1

def register_hooks(model, layer):
    """
    Registers hooks to capture the activations and gradients.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        print("Forward hook triggered!") 
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        print("Backward hook triggered!")  
        gradients.append(grad_output[0])

    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)    #register_full_backward_hook
    
    return activations, gradients

def generate_grad_cam(activations, gradients, target_class):
    if len(activations) == 0 or len(gradients) == 0:
        raise ValueError("Activations or gradients are empty!")

    print(f"Activations shape: {activations[0].shape}")
    print(f"Gradients shape: {gradients[0].shape}")
    # Get the gradients and activations
    gradient = gradients[0][0]  # Get the gradient of the first image
    activation = activations[0][0]  # Get the activation of the first image

    weights = torch.mean(gradient, dim=[2, 3], keepdim=True)

    grad_cam = torch.sum(weights * activation, dim=1, keepdim=True)

    grad_cam = torch.nn.functional.relu(grad_cam)

    grad_cam = grad_cam.squeeze().cpu().detach().numpy()
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))

    return grad_cam

def apply_heatmap_on_image(image, grad_cam):
    grad_cam = cv2.resize(grad_cam, (image.shape[2], image.shape[3]))

    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    
    grad_cam = np.uint8(255 * grad_cam)
    heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return superimposed_img



@torch.no_grad()
def test(epoch):

    torch.cuda.empty_cache()
    global test_time
    test_time += 1
    result_file = open('./history/{}/result_file_test_{}.txt'.format(config['exp_name'], test_time), 'w+', encoding='utf-8')

    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('test:', test_loader_len)

    correct = 0
    total = 0
    inference_times = []
    
    # target_layer = model.module.encoder.layer3_conv  # 这里需要选择适合的目标层
    # activations, gradients = register_hooks(model, target_layer)

    for iteration in range(test_loader_len):
        data = dataloader.next()
        image, label, _ = data
        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

        length, text_input, text_gt, string_label, ids_sequences = converter(label)
        max_length = max(length)
        batch = image.shape[0]
        pred = torch.zeros(batch,1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()

        batch_inference_time = 0.0

        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1

            torch.cuda.synchronize()
            start_time = time.time()

            result = model(epoch, image, length_tmp, pred, conv_feature=image_features, f_code=f_code, f_depth=f_depth, f_parent=f_parent, f_child=f_child, pos_entropy=pos_entropy, test=True)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_duration = end_time - start_time
            batch_inference_time += inference_duration

            prediction = result['pred'][:, -1:, :].squeeze()
            prediction = prediction / prediction.norm(dim=1, keepdim=True)
            prediction = prediction @ text_features.t() # prediction : torch.Size([128, 3757])
            now_pred = torch.max(torch.softmax(prediction,1), 1)[1]

            # if torch.any(now_pred == 0) or torch.any(now_pred == 3756):
            #     prediction_for_target = prediction[0, now_pred[0]] 
            #     prediction_for_target = prediction_for_target.requires_grad_() 
            #     model.zero_grad()  
                
            #     prediction_for_target.backward(retain_graph=True)
            #     # prediction_for_target.backward(retain_graph=True)  


            #     grad_cam = generate_grad_cam(activations, gradients, now_pred[0])

            #     superimposed_img = apply_heatmap_on_image(image, grad_cam)

            #     plt.imshow(superimposed_img)
            #     plt.axis('off')
            #     plt.savefig(f'./history/{config["exp_name"]}/heatmap_{test_time}_{iteration}.png', bbox_inches='tight')
            #     plt.close()


            prob[:,i] = torch.max(torch.softmax(prediction,1), 1)[0]
            pred = torch.cat((pred, now_pred.view(-1,1)), 1)
            image_features = result['conv']
            attention_map = result['map']
 
        inference_times.append(batch_inference_time)

        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)
        
        start = 0
        for i in range(batch):
            state = False
            pred = tensor2str(text_pred_list[i])
            gt = tensor2str(text_gt_list[i])

            if pred == gt:
                correct += 1
                state = True

            start += i
            total += 1
            if iteration % 100 == 0:
                print('{} | {} | {} | {} | {} | {}'.format(total, pred, gt, state, text_prob_list[i],
                                                            correct / total))
            result_file.write(
                '{} | {} | {} | {} | {} \n'.format(total, pred, gt, state, text_prob_list[i]))

    total_inference_time = sum(inference_times)
    avg_inference_time_per_batch = total_inference_time / len(inference_times)
    avg_inference_time_per_image = (avg_inference_time_per_batch / batch) * 1000  # 单位转换成毫秒
    if dist.get_rank() == 0:
        print("平均每个批次推理时间: {:.4f}秒".format(avg_inference_time_per_batch))
        print("平均每张图片推理时间: {:.4f}ms".format(avg_inference_time_per_image))

    global_correct = torch.tensor(correct).cuda()
    global_total = torch.tensor(total).cuda()

    dist.all_reduce(global_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_total, op=dist.ReduceOp.SUM)
    
    global_acc = global_correct.item() / global_total.item()
    if dist.get_rank() == 0:
        print("ACC : {}".format(global_acc))
        global best_acc

        if global_acc > best_acc:
            best_acc = global_acc
            torch.save(model.state_dict(), './history/{}/best_model.pth'.format(config['exp_name']))

        f = open('./history/{}/record.txt'.format(config['exp_name']), 'a+', encoding='utf-8')
        f.write("Epoch : {} | ACC : {}\n".format(epoch, global_acc))
        f.close()

if __name__ == '__main__':
    print("-------------start-------------")
    for epoch in range(config['epoch']):
        train_sampler.set_epoch(epoch)
        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)
        for iteration in range(train_loader_len):
            data = dataloader.next()
            image, label, _ = data
            image = torch.nn.functional.interpolate(image, size=(32, 32))
            if iteration % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"learning rate: {current_lr}")
            length, text_input, text_gt, string_label, ids_sequences = converter(label)
            train(epoch, iteration, image, length, text_input, text_gt, ids_sequences)
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), f"./history/{config['exp_name']}/model_{epoch}.pth")
        test(epoch)
        scheduler.step()