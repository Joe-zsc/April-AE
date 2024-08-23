from pytorch_transformers import BertModel, BertConfig, BertTokenizer

from torch import nn
import torch
import os
import numpy as np
import sys, os
from pathlib import Path
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from util import Configure


bert_pretrained_path=Path(curr_path)/"Embedding_models"/Configure.get("Embedding", "bert_model")
# bert_pretrained_path = os.path.join(
#     parent_path, Configure.get("NLP", "bert_pretrained_path"))
sentence_vector_dim = int(Configure.get('Embedding', 'sentence_vector_dim'))
word_vector_dim = int(Configure.get('Embedding', 'word_vector_dim'))
action_vector_dim = int(Configure.get('Embedding', 'action_vector_dim'))
# ——————构造模型——————
class TextNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained(
            os.path.join(bert_pretrained_path, 'config.json'))
        self.textExtractor = BertModel.from_pretrained(os.path.join(
            bert_pretrained_path, 'pytorch_model.bin'),
                                                       config=modelConfig)
        # modelConfig = BertConfig.from_pretrained(
        #     'bert_to_sentence_vector/pretrained_model/bert-base-uncased-config.json'
        # )
        # self.textExtractor = BertModel.from_pretrained(
        #     'bert_to_sentence_vector/pretrained_model/bert-base-uncased-pytorch_model.bin',
        #     config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens,
                                    token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features


textNet_sentence = TextNet(code_length=sentence_vector_dim)
textNet_word = TextNet(code_length=word_vector_dim)
textNet = TextNet(code_length=action_vector_dim)
tokenizer = BertTokenizer.from_pretrained(
    os.path.join(bert_pretrained_path, 'vocab.txt'))


def get_vector(text: str,dim=100):
    
    if text.find("[CLS]") == -1:
        text = "[CLS] " + text
    if text.find("[SEP]") == -1:
        text = text + " [SEP]"

    # ——————输入处理——————
    tokens, segments, input_masks = [], [], []

    tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
    if len(tokenized_text)>=512:
        tokenized_text=tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
    tokens.append(indexed_tokens)
    segments.append([0] * len(indexed_tokens))
    input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens])  # 最大的句子长度

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding


# segments列表全0，因为只有一个句子1，没有句子2
# input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
# 相当于告诉BertModel不要利用后面0的部分

# 转换成PyTorch tensors
    tokens_tensor = torch.tensor(tokens)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(input_masks)

    # ——————提取文本特征——————
    if dim==word_vector_dim:
        text_hashCodes = textNet_word(
            tokens_tensor, segments_tensors,
            input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
    elif dim==sentence_vector_dim:
        text_hashCodes = textNet_sentence(
            tokens_tensor, segments_tensors,
            input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
    else:
        
        text_hashCodes = textNet(
            tokens_tensor, segments_tensors,
            input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
    return text_hashCodes
def CosineDistance(x, y):
    
    x = np.array(x)
    y = np.array(y)
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
def JensenShannonDivergence(p, q):
    p = np.array(p)
    q = np.array(q)
    M = (p + q)/2
    return 0.5 * np.sum(p*np.log(p/M)) + 0.5 * np.sum(q*np.log(q/M))
def StandardizedEuclideanDistance(x, y):

    x = np.array(x)
    y = np.array(y)
    
    X = np.vstack([x,y])
    sigma = np.var(X, axis=0, ddof=1)
    return np.sqrt(((x - y) ** 2 /sigma).sum())
def test_similarity(str1,str2):
    
    output1 = get_vector(str1).flatten().detach().numpy()
    output2 = get_vector(str2).flatten().detach().numpy()
    score1=CosineDistance(output1,output2)
    print(f"CosineDistance of {str1} and {str2}:{score1}")
    
    
if __name__ == '__main__':
    # input = "Zenphoto Zenphoto; PHP+MySQL; zenphoto_ssl; 相馆"
    vector = np.zeros(sentence_vector_dim, dtype=np.float32)
    # output = get_vector(input).flatten().detach().numpy()
    str_= "# Possible XSS Vulnerability in Rails::Html::SanitizerThere is a possible XSS vulnerability with certain configurations of Rails::Html::Sanitizer.This vulnerability has been assigned the CVE identifier CVE-2022-32209.Versions Affected: ALLNot affected: NONEFixed Versions: v1.4.3## ImpactA possible XSS vulnerability with certain configurations of Rails::Html::Sanitizer may allow an attacker to inject content if the application developer has overridden the sanitizer's allowed tags to allow both `select` and `style` elements.Code is only impacted if allowed tags are being overridden. This may be done via application configuration:```ruby# In config/application.rbconfig.action_view.sanitized_allowed_tags = [\"select\", \"style\"]```see https://guides.rubyonrails.org/configuring.html#configuring-action-viewOr it may be done with a `:tags` option to the Action View helper `sanitize`:```<%= sanitize @comment.body, tags: [\"select\", \"style\"] %>```see https://api.rubyonrails.org/classes/ActionView/Helpers/SanitizeHelper.html#method-i-sanitizeOr it may be done with Rails::Html::SafeListSanitizer directly:```ruby# class-level optionRails::Html::SafeListSanitizer.allowed_tags = [\"select\", \"style\"]```or```ruby# instance-level optionRails::Html::SafeListSanitizer.new.sanitize(@article.body, tags: [\"select\", \"style\"])```All users overriding the allowed tags by any of the above mechanisms to include both \"select\" and \"style\" should either upgrade or use one of the workarounds immediately.## ReleasesThe FIXED releases are available at the normal locations.## WorkaroundsRemove either `select` or `style` from the overridden allowed tags.## CreditsThis vulnerability was responsibly reported by [windshock](https://hackerone.com/windshock?type=user)."
    vector+= get_vector(str_).flatten().detach().numpy()
    str1="Windows 7 or Windows Server 2008 or Windows Server 2008 or Windows 8 or Windows 8"
    str2="Windows 7"
    str3="Windows Server 2008 R2"
    ss=str1.split(" or ")
    for s in ss:
        vector+= get_vector(s).flatten().detach().numpy()
    vector=vector/len(ss)
    output2 = get_vector(str2).flatten().detach().numpy()
    a=CosineDistance(vector,output2)
    print(a)
    # while True:
    test_similarity("Struts2","http Jetty 9.2.11.v20150529")
    test_similarity("WebLogic","http Jetty 9.2.11.v20150529")
    test_similarity("WebLogic","Struts2")
    # score=JensenShannonDivergence(output1,output2)

