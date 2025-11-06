# 导入 OrderedDict，它是一种会记住键插入顺序的字典类型
from collections import OrderedDict
# 导入 pickle 模块，用于将 Python 对象（如词表）序列化保存到文件
import pickle
# 导入 re 模块，用于正则表达式操作，这里主要用于匹配特殊 token
import re
# 导入 tqdm 库，用于在词表训练（合并阶段）显示进度条
from tqdm import tqdm
# （保留备用）导入 math 模块（原用于处理无穷大或对数安全性，但当前未使用）
import math


class WordPieceTokenizer:
    """WordPiece 分词器的简化实现"""

    def __init__(self):
        # 初始化字节到ID的映射表（保持插入顺序）
        self.b2i = OrderedDict()  # bytes -> 整数ID
        # 初始化ID到字节的反向映射表
        self.i2b = OrderedDict()  # 整数ID -> bytes
        # 计数器，用于为新合并的 token 分配ID
        self.next_id = 0

        # --- 特殊 token 部分 ---
        # 特殊 token 字符串到ID的映射
        self.sp_s2i = {}  # str -> int
        # 特殊 token ID到字符串的映射
        self.sp_i2s = {}  # int -> str

    # --- WORDPIECE 改进 ---
    # 辅助方法：统计 token 序列中所有相邻 token 对的出现频率
    # 与 BPE 不同，这里使用 (token_a, token_b) 的元组作为键
    def _pair_stats(self, tokens, stats):
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            stats[pair] = stats.get(pair, 0) + 1

    # --- WORDPIECE 改进 ---
    # 辅助方法：在 token 序列中合并指定的 token 对
    def _merge_pair(self, tokens, pair_to_merge):
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair_to_merge:
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    # 训练方法：从原始文本列表中学习 WordPiece 词表
    def train(self, text_list, vocab_size):
        # --- 第1步：初始化基础词表 ---
        # 初始词表包含所有单个字节（0-255）
        for i in range(256):
            self.b2i[bytes([i])] = i
        self.next_id = 256

        # --- 第2步：将文本转为初始 token 序列 ---
        tokens_list = []
        for text in text_list:
            tokens = [bytes([b]) for b in text.encode('utf-8')]
            tokens_list.append(tokens)

        # --- 第3步：开始迭代合并 ---
        # 目标步数为 vocab_size - 256（减去基础字节数）
        progress = tqdm(total=vocab_size - 256)

        while True:
            if self.next_id >= vocab_size:
                break

            # 统计所有 token 对的频率
            stats = {}
            for tokens in tokens_list:
                self._pair_stats(tokens, stats)

            # 统计单个 token 的频率
            token_counts = {}
            for tokens in tokens_list:
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1

            if not stats:
                break

            # --- WordPiece 打分 ---
            # Score = count(A, B) / (count(A) * count(B))
            # （近似似然度，用于选择最优合并对）
            best_score = -1.0
            best_pair = None

            for pair, count in stats.items():
                tok_a, tok_b = pair
                count_a = token_counts.get(tok_a)
                count_b = token_counts.get(tok_b)

                if not count_a or not count_b:
                    score = -1.0
                else:
                    score = count / (count_a * count_b)

                if score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            # --- 执行合并 ---
            new_tokens_list = []
            for tokens in tokens_list:
                new_tokens_list.append(self._merge_pair(tokens, best_pair))
            tokens_list = new_tokens_list

            # --- 将新 token 加入词表 ---
            new_token_bytes = best_pair[0] + best_pair[1]
            self.b2i[new_token_bytes] = self.next_id
            self.next_id += 1
            progress.update(1)

        # --- 第4步：生成反向映射 ---
        self.i2b = {v: k for k, v in self.b2i.items()}

    # 返回当前词表大小（包括特殊 token）
    def vocab_size(self):
        return self.next_id

    # 返回完整词表（ID -> bytes），包括特殊 token
    def vocab(self):
        v = {}
        v.update(self.i2b)
        v.update({id: token.encode('utf-8') for id, token in self.sp_i2s.items()})
        return v

    # 添加特殊 token
    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            if token not in self.sp_s2i:
                self.sp_s2i[token] = self.next_id
                self.sp_i2s[self.next_id] = token
                self.next_id += 1

    # 编码：将字符串转换为 token ID 列表
    def encode(self, text):
        # --- 第1步：优先匹配特殊 token ---
        pattern = '(' + '|'.join([re.escape(tok) for tok in self.sp_s2i]) + ')'
        splits = re.split(pattern, text)

        # --- 第2步：逐段编码 ---
        enc_ids = []
        enc_tokens = []
        for sub_text in splits:
            if sub_text in self.sp_s2i:
                enc_ids.append(self.sp_s2i[sub_text])
                enc_tokens.append(sub_text.encode('utf-8'))
            else:
                # 使用贪心最长匹配进行 WordPiece 编码
                text_bytes = sub_text.encode('utf-8')
                current_pos = 0
                len_bytes = len(text_bytes)
                encoded_byte_tokens = []

                while current_pos < len_bytes:
                    best_tok = None
                    best_len = -1

                    # 从当前位置起，尝试找到最长匹配的子串
                    for end_pos in range(len_bytes, current_pos, -1):
                        substring_bytes = text_bytes[current_pos:end_pos]
                        if substring_bytes in self.b2i:
                            best_tok = substring_bytes
                            best_len = len(best_tok)
                            break

                    # 理论上不会出现 None（因为单字节一定存在）
                    encoded_byte_tokens.append(best_tok)
                    current_pos += best_len

                # 将编码结果添加到主列表
                enc_ids.extend([self.b2i[tok] for tok in encoded_byte_tokens])
                enc_tokens.extend(encoded_byte_tokens)

        return enc_ids, enc_tokens

    # 解码：将 ID 列表还原为原始字符串
    def decode(self, ids):
        bytes_list = []
        for id in ids:
            if id in self.sp_i2s:
                bytes_list.append(self.sp_i2s[id].encode('utf-8'))
            else:
                bytes_list.append(self.i2b[id])
        return b''.join(bytes_list).decode('utf-8', errors='replace')

    # 保存分词器状态（b2i、sp_s2i、next_id）到二进制文件
    def save(self, file):
        with open(file, 'wb') as fp:
            fp.write(pickle.dumps((self.b2i, self.sp_s2i, self.next_id)))

    # 从文件加载分词器状态
    def load(self, file):
        with open(file, 'rb') as fp:
            self.b2i, self.sp_s2i, self.next_id = pickle.loads(fp.read())
        # 重建反向映射
        self.i2b = {v: k for k, v in self.b2i.items()}
        self.sp_i2s = {v: k for k, v in self.sp_s2i.items()}


# 仅在脚本被直接运行时执行以下示例代码
if __name__ == '__main__':
    # 加载中英文语料
    cn = open('dataset/train-cn.txt', 'r', encoding="UTF-8").read()
    en = open('dataset/train-en.txt', 'r', encoding="UTF-8").read()

    # 创建并训练 WordPiece 分词器
    tokenizer = WordPieceTokenizer()
    tokenizer.train(text_list=[cn, en], vocab_size=300)

    # 添加特殊 token
    tokenizer.add_special_tokens(['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|padding|>'])

    # 保存训练好的分词器
    tokenizer.save('tokenizer.bin')

    # 加载并验证
    tokenizer = WordPieceTokenizer()
    tokenizer.load('tokenizer.bin')
    print('vocab size:', tokenizer.vocab_size())

    # 测试编码
    text_to_encode = '<|im_start|>system\nyou are a helper assistant\n<|im_end|>\n<|im_start|>user\n今天的天气\n<|im_end|><|im_start|>assistant\n'
    ids, tokens = tokenizer.encode(text_to_encode)
    print('encode:', ids, tokens)

    # 测试解码
    s = tokenizer.decode(ids)
    print('decode:', s)

    # 打印词表（ID→bytes，用于检查）
    print('vocab:', tokenizer.vocab())
