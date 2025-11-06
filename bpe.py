# 导入 OrderedDict，这是一个会记住键插入顺序的字典，对BPE词表很重要
from collections import OrderedDict
# 导入 pickle 模块，用于将Python对象（如我们的词表）序列化保存到文件
import pickle
# 导入 re 模块，用于正则表达式，这里主要用来切分特殊token
import re
# 导入 tqdm 库，用于在训练时显示一个漂亮的进度条
from tqdm import tqdm


# Byte-Pair Encoding (BPE) 分词器类
class BPETokenizer:
    # 类的初始化方法（构造函数）
    def __init__(self):
        # 初始化 b2i (bytes to id) 映射表，使用OrderedDict保持顺序
        self.b2i = OrderedDict()  # 字节(bytes) -> 整数ID
        # 初始化 i2b (id to bytes) 映射表，用于解码
        self.i2b = OrderedDict()  # 整数ID -> 字节(bytes)
        # next_id 计数器，用于为新合并的token分配下一个可用的ID
        self.next_id = 0

        # --- 特殊token部分 ---
        # sp_s2i (special string to id) 映射表
        self.sp_s2i = {}  # 特殊token字符串(str) -> 整数ID
        # sp_i2s (special id to string) 映射表
        self.sp_i2s = {}  # 整数ID -> 特殊token字符串(str)

    # 私有辅助方法：统计一个token序列中所有相邻token对的出现频率
    def _pair_stats(self, tokens, stats):
        # 遍历序列，直到倒数第二个元素（因为我们要看 i 和 i+1）
        for i in range(len(tokens) - 1):
            # 将相邻的两个token（它们都是bytes类型）合并成一个新的bytes
            new_token = tokens[i] + tokens[i + 1]
            # 检查这个新合并的token对是否已经在stats字典中
            if new_token not in stats:
                # 如果不在，初始化其计数为0
                stats[new_token] = 0
            # 将这个token对的计数加1
            stats[new_token] += 1

    # 私有辅助方法：在一个token序列中，合并所有出现的指定token对
    def _merge_pair(self, tokens, new_token):
        # 初始化一个新列表，用于存放合并后的token序列
        merged_tokens = []

        # 使用while循环和索引i来遍历原token列表
        i = 0
        while i < len(tokens):
            # 检查：1. 是否有下一个token (i+1 < len)； 2. 当前token和下一个token的组合是否等于我们要合并的 new_token
            if i + 1 < len(tokens) and tokens[i] + tokens[i + 1] == new_token:
                # 如果是，就将合并后的 new_token 添加到新列表中
                merged_tokens.append(tokens[i] + tokens[i + 1])
                # 索引 i 跳过2个位置（因为我们处理了 i 和 i+1）
                i += 2
            else:
                # 如果不是，只将当前的 token[i] 添加到新列表中
                merged_tokens.append(tokens[i])
                # 索引 i 只跳过1个位置
                i += 1
        # 返回合并完成后的新token序列
        return merged_tokens

    # 训练方法：从原始文本列表学习BPE词表
    def train(self, text_list, vocab_size):
        # --- 第1步：初始化词表 ---
        # BPE的基础词表是所有单个字节（0-255）
        #bit=(0,1)   字节---8个bit
        for i in range(256):
            # 将每个字节（如 b'\x00', b'\x01', ..., b'a', b'b', ...）映射到其对应的整数ID
            self.b2i[bytes([i])] = i
        # 将 next_id 设置为256，因为0-255已经被占用了
        self.next_id = 256

        # --- 第2步：准备语料 ---
        # 初始化一个列表，用于存放所有文本的初始token序列
        tokens_list = []
        # 遍历输入的每一个文本字符串
        for text in text_list:
            # 1. 将字符串 text 编码为 utf-8 字节流
            # 2. 将字节流拆分为单个字节的列表（例如 'abc' -> [b'a', b'b', b'c']）
            tokens = [bytes([b]) for b in text.encode('utf-8')]
            # 将这个初始的token序列（字节列表）添加到总列表中
            tokens_list.append(tokens)

        # --- 第3步：迭代合并 ---
        # 初始化进度条，总步数 = 目标词表大小 - 初始256个字节
        progress = tqdm(total=vocab_size - 256)

        # 开始主训练循环
        while True:
            # 检查是否已达到目标词表大小
            if self.next_id >= vocab_size:
                # 如果达到了，停止训练
                break

            # 初始化一个空字典，用于存储本轮迭代中所有token对的频率
            stats = {}
            # 遍历当前所有的token序列（这些序列会随着合并变得越来越短）
            for tokens in tokens_list:
                # 调用 _pair_stats 统计该序列中所有相邻token对的频率，并累加到 stats 字典中
                self._pair_stats(tokens, stats)

            # 检查 stats 是否为空。如果为空，说明没有可合并的token对了（例如所有文本都只剩1个token）
            if not stats:
                # 如果没有可合并的，提前停止训练
                break

                # --- 关键步骤：找到最高频的token对 ---
            # 使用 max() 找到 stats 字典中值（频率）最大的那个键（token对）
            new_token = max(stats, key=stats.get)

            # --- 关键步骤：在所有序列中执行合并 ---
            # 初始化一个新列表，用于存放本轮合并后的序列
            new_tokens_list = []
            # 遍历当前的token序列
            for tokens in tokens_list:
                # 调用 _merge_pair 将这个最高频的 new_token 在该序列中合并
                # （例如 [b't', b'h', b'e'] -> [b'th', b'e']）
                new_tokens_list.append(self._merge_pair(tokens, new_token))
            # 用合并后的新序列列表替换旧的序列列表，用于下一轮迭代
            tokens_list = new_tokens_list

            # --- 第4步：将新token加入词表 ---
            # 将这个新合并的 token (bytes) 添加到 b2i 词表中，并分配当前的 next_id
            self.b2i[new_token] = self.next_id
            # next_id 自增1，为下一个新token准备
            self.next_id += 1

            # 进度条更新1步
            progress.update(1)

        # --- 第5步：训练完成，生成反向词表 ---
        # 训练循环结束后，使用字典推导式快速生成 i2b (ID -> bytes) 的反向映射
        self.i2b = {v: k for k, v in self.b2i.items()}

    # 返回词表大小（普通token + 特殊token）
    def vocab_size(self):
        # next_id 始终指向下一个可用的ID，因此它也代表了当前的总词表大小
        return self.next_id

    # 返回一个完整的（ID -> bytes）词表，包括特殊token
    def vocab(self):
        # 初始化一个空字典
        v = {}
        # 首先，将所有普通token (ID -> bytes) 添加进去
        v.update(self.i2b)
        # 然后，将所有特殊token (ID -> str) 转换为 (ID -> bytes) 并添加进去
        v.update({id: token.encode('utf-8') for id, token in self.sp_i2s.items()})
        # 返回这个合并后的完整词表
        return v

    # 添加特殊token的方法
    def add_special_tokens(self, special_tokens):
        # 遍历要添加的特殊token列表（或元组）
        for token in special_tokens:
            # 检查这个token是否已经添加过了
            if token not in self.sp_s2i:
                # 如果没有，将其添加到 sp_s2i (str -> ID) 映射中，使用当前的 next_id
                self.sp_s2i[token] = self.next_id
                # 同时添加到 sp_i2s (ID -> str) 的反向映射中
                self.sp_i2s[self.next_id] = token
                # next_id 自增1
                self.next_id += 1

    # 编码方法：将原始字符串(str)转换为ID列表
    def encode(self, text):
        # --- 第1步：处理特殊token ---
        # 构建一个正则表达式，用于匹配所有已知的特殊token
        # re.escape(tok) 是为了防止token中的特殊字符（如'|', '?', '*'）被误认为是正则表达式元字符
        # '(...|...|...)' 是一个捕获组，re.split会保留匹配到的分隔符
        pattern = '(' + '|'.join([re.escape(tok) for tok in self.sp_s2i]) + ')'
        # 使用 re.split 将文本切分为“普通文本”和“特殊token”的交替列表
        # 例如："<|start|>hello" -> ["", "<|start|>", "hello"]
        splits = re.split(pattern, text)

        # --- 第2步：逐段编码 ---
        # 初始化用于存放最终ID的列表
        enc_ids = []
        # 初始化用于存放最终token(bytes)的列表（主要用于调试）
        enc_tokens = []
        # 遍历切分后的所有子字符串
        for sub_text in splits:
            # 如果这个子字符串是一个已知的特殊token
            if sub_text in self.sp_s2i:
                # 直接从 sp_s2i 字典中获取其ID
                enc_ids.append(self.sp_s2i[sub_text])
                # 将其字符串(str)编码为bytes存入
                enc_tokens.append(sub_text.encode('utf-8'))
            else:
                # --- 如果是普通文本，执行BPE编码（推理）---
                # 1. 将普通文本编码为UTF-8字节，并拆分为单字节列表
                tokens = [bytes([b]) for b in sub_text.encode('utf-8')]
                # 循环合并，直到没有可合并的token为止
                while True:
                    # 统计当前 tokens 序列中所有相邻token对的频率
                    stats = {}
                    self._pair_stats(tokens, stats)

                    # --- 关键步骤：找到“最佳”合并对 ---
                    # 这里的“最佳”指的是在词表中ID最小（即最早被学会）的那个token对
                    new_token = None
                    # 遍历所有在当前序列中出现的token对
                    for merge_token in stats:
                        # 检查：1. 这个token对是否在我们的BPE词表(b2i)中
                        # 2. ( new_token是None(第一次找到) 或者 当前token的ID < 已找到token的ID )
                        if merge_token in self.b2i and (
                                new_token is None or self.b2i[merge_token] < self.b2i[new_token]):
                            # 如果满足，更新 new_token 为这个“最佳”的合并对
                            new_token = merge_token

                    # 如果循环一遍后，new_token 仍然是 None，说明没有可合并的了
                    if new_token is None:
                        # 退出合并循环
                        break

                    # 如果找到了，就执行合并
                    tokens = self._merge_pair(tokens, new_token)

                # 合并循环结束后，tokens 列表就是最终的BPE token序列
                # 将这个序列中的所有token（bytes）转换为ID，并 *追加* 到 enc_ids 列表中
                enc_ids.extend([self.b2i[tok] for tok in tokens])
                # 将这个token（bytes）序列 *追加* 到 enc_tokens 列表中
                enc_tokens.extend(tokens)
        # 返回编码后的ID列表和token(bytes)列表
        return enc_ids, enc_tokens

    # 解码方法：将ID列表转换回原始字符串(str)
    def decode(self, ids):
        # 初始化一个列表，用于存放解码出的字节(bytes)片段
        bytes_list = []
        # 遍历输入的每一个ID
        for id in ids:
            # 检查这个ID是否在特殊token词表 (sp_i2s) 中
            if id in self.sp_i2s:
                # 如果是，获取其字符串(str)，编码为utf-8字节，并添加到列表中
                bytes_list.append(self.sp_i2s[id].encode('utf-8'))
            else:
                # 如果是普通token ID，从 i2b 词表中获取其对应的字节(bytes)
                bytes_list.append(self.i2b[id])
        # 使用 b''.join() 将列表中的所有字节片段拼接成一个单一的、完整的字节流
        # 然后使用 .decode('utf-8', errors='replace') 将字节流解码回字符串
        # errors='replace' 意味着如果遇到无效的utf-8序列，会用''字符替换，而不是抛出异常
        return b''.join(bytes_list).decode('utf-8', errors='replace')

    # 保存分词器状态到文件
    def save(self, file):
        # 以“写入字节”('wb')模式打开文件
        with open(file, 'wb') as fp:
            # 1. 将需要保存的核心状态 (b2i, sp_s2i, next_id) 打包成一个元组
            # 2. 使用 pickle.dumps 将这个元组序列化为字节流
            # 3. 将字节流写入文件
            fp.write(pickle.dumps((self.b2i, self.sp_s2i, self.next_id)))

    # 从文件加载分词器状态
    def load(self, file):
        # 以“读取字节”('rb')模式打开文件
        with open(file, 'rb') as fp:
            # 1. 读取文件中的全部字节内容
            # 2. 使用 pickle.loads 将字节流反序列化回元组
            # 3. 解包元组，恢复 self.b2i, self.sp_s2i, self.next_id 的状态
            self.b2i, self.sp_s2i, self.next_id = pickle.loads(fp.read())
        # --- 重建反向映射 ---
        # 根据加载的 b2i 重建 i2b
        self.i2b = {v: k for k, v in self.b2i.items()}
        # 根据加载的 sp_s2i 重建 sp_i2s
        self.sp_i2s = {v: k for k, v in self.sp_s2i.items()}


# 这是一个Python的标准写法，表示只有当这个 .py 文件被直接运行时，才执行以下代码
# 如果它被其他文件 import，则不执行
if __name__ == '__main__':
    # --- 演示如何使用 ---

    # 加载语料
    # 打开中文训练文件，模式为'r'(读)，并读取所有内容
    cn = open('dataset/train-cn.txt', 'r',encoding="UTF-8").read()
    # 打开英文训练文件，模式为'r'(读)，并读取所有内容
    en = open('dataset/train-en.txt', 'r',encoding="UTF-8").read()

    # 训练
    # 创建 BPETokenizer 类的一个实例（对象）
    tokenizer = BPETokenizer()
    # 调用 train 方法，传入中文和英文语料，目标词表大小为5000
    tokenizer.train(text_list=[cn, en], vocab_size=300)

    # 添加特殊token（注意：这会使词表大小在5000的基础上增加）
    tokenizer.add_special_tokens((['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|padding|>']))

    # 保存训练好的分词器状态
    tokenizer.save('tokenizer.bin')

    # --- 演示加载 ---
    # （可选）创建一个新的空分词器实例来模拟一个新进程
    tokenizer = BPETokenizer()
    # 从文件加载
    tokenizer.load('tokenizer.bin')
    # 打印词表大小（应为 5000 + 4 = 5004）
    print('vocab size:', tokenizer.vocab_size())

    # 编码测试
    # 定义一个复杂的测试字符串，包含特殊token、英文、中文和换行符
    text_to_encode = '<|im_start|>system\nyou are a helper assistant\n<|im_end|>\n<|im_start|>user\n今天的天气\n<|im_end|><|im_start|>assistant\n'
    # 调用 encode 方法
    ids, tokens = tokenizer.encode(text_to_encode)
    # 打印编码结果
    print('encode:', ids, tokens)

    # 解码测试
    # 调用 decode 方法，将ID列表解码回字符串
    s = tokenizer.decode(ids)
    # 打印解码结果（应与 text_to_encode 完全一致）
    print('decode:', s)

    # 打印词典（用于检查）
    print('vocab:',tokenizer.vocab())
