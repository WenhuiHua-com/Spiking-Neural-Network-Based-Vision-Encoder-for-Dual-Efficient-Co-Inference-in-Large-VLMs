import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_retrieval import blip_retrieval
import json
from collections import Counter
from timm.models import create_model
from model import *
from spikingjelly.clock_driven import functional

# 加载Flickr30k的数据
with open('/media/data/huawenhui/BLIP-main/BLIP-main/flick/annotation/flickr30k_train.json', 'r') as file:
    data = json.load(file)

# 加载图像
def load_demo_image(image_size, device, picture_name, image_root):
    raw_image = Image.open(os.path.join(image_root, picture_name)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

# 获取图片文件名
image_name_list = [item['image'] for item in data]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_image_root = '/media/data/huawenhui/BLIP-main/BLIP-main/flick/flickr30k-images'

image_size = 224

# 加载模型
model_retrieval = blip_retrieval(pretrained='/media/data/huawenhui/BLIP-main/BLIP-main/model_base_retrieval_flickr.pth',
                                 image_size=224, vit='base',
                                 vit_grad_ckpt=True,
                                 queue_size=57600, negative_all_rank=False)
model_retrieval = model_retrieval.to(device)
model_retrieval.eval()

# 测试码本模型
def test_codebook_model(ori_image, model, codebook):
    sparse_image_embeds = model.visual_encoder(ori_image)
    token_indices = codebook.search_visual(sparse_image_embeds.squeeze())
    return token_indices

# 定义码本类
class CodeBook:
    def __init__(self, size, dim, device='cpu'):
        self.size = size
        self.dim = dim
        self.device = device
        self.book = torch.empty(0, dim, device=self.device)  # 初始化空的codebook
        self.weights = torch.empty(0, device=self.device)  # 存储对应的权重
        self.counter = 0  # 当前码本条目数量

    def add(self, tokens, importance_scores):
        tokens = tokens.to(self.device)
        importance_scores = importance_scores.to(self.device)
        token_num, token_dim = tokens.shape

        if (self.counter + token_num) <= self.size:
            self.book = torch.cat([self.book, tokens], dim=0)
            self.weights = torch.cat([self.weights, importance_scores], dim=0)
            self.counter += token_num
        else:
            sim_matrix = torch.mm(tokens, self.book.T)
            for i, token in enumerate(tokens):
                sim_scores, indices = torch.topk(sim_matrix[i], k=1, largest=True)
                index = indices[0]

                # 权重加权聚合
                wa = self.weights[index]
                wb = importance_scores[i]
                self.book[index] = (wa * self.book[index] + wb * token) / (wa + wb)
                self.weights[index] = wa + wb  # 更新合并后的权重

    def search(self, indices):
        indices = indices.to(self.device)
        ind = indices.argmax(dim=2)
        result_vectors = self.book[ind.squeeze()]  # 从codebook中取出替换向量
        return result_vectors  # 返回移除全零向量后的替换向量

    def search_visual(self, tokens):
        tokens = tokens.to(self.device)
        sim_matrix = torch.mm(tokens, self.book.T)  # 计算输入token与码本的相似度
        sim_scores, indices = torch.topk(sim_matrix, k=1, largest=True)  # 获取相似度分数和索引
        return indices.squeeze()  # 返回最相似的token索引

    def save(self, path):
        torch.save({'book': self.book, 'weights': self.weights}, path)  # 保存码本和权重

    def load(self, path):
        data = torch.load(path)
        self.book = data['book'].to(self.device)  # 加载码本
        self.weights = data['weights'].to(self.device)  # 加载权重
        self.counter = self.book.size(0)

    @staticmethod
    def test_codebook_model(ori_image, model, codebook):  # Make this static
        sparse_image_embeds = model.visual_encoder(ori_image)
        token_indices = codebook.search_visual(sparse_image_embeds.squeeze())  # Use codebook instance
        return token_indices

    def prune_codebook(self, image_name_list, model_retrieval):  # Add model_retrieval as argument
        # 统计每个token在所有样本中出现的频率
        token_counter = Counter()

        for picture_name in image_name_list:
            ori_image = load_demo_image(image_size=image_size, device=device, picture_name=picture_name, image_root=val_image_root)
            retrieval_indices = CodeBook.test_codebook_model(ori_image, model_retrieval, self)  # Call as static method
            token_indices = retrieval_indices.cpu().numpy()
            token_counter.update(token_indices)

        # 获取频率最高的前5w个token
        most_common_tokens = token_counter.most_common(5000)
        top_token_indices = [token for token, _ in most_common_tokens]

        # 创建一个新的codebook，包含前5w个最常用的token
        new_codebook = CodeBook(size=5000, dim=self.dim, device=self.device)

        # 将top_token_indices对应的tokens从原码本中取出并加入新码本
        for idx in top_token_indices:
            new_codebook.add(self.book[idx:idx+1], self.weights[idx:idx+1])  # 注意：这里假设book是按照token顺序存储的

        return new_codebook

# 执行修剪操作
def test_prune_codebook(image_name_list):
    codebook = CodeBook(size=524288, dim=768, device=device)
    codebook.load('/media/data/huawenhui/BLIP-main/BLIP-main/codebook_retrieval_New_10000_524288_384_0.5.pth')

    # 修剪码本
    new_codebook = codebook.prune_codebook(image_name_list, model_retrieval)

    # 保存新的码本
    new_codebook.save('flicker_purning_05w_codebook.pth')

# 执行修剪任务
test_prune_codebook(image_name_list)
