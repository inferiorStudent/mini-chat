<div align="center">
  中文 | <a href="./README_en.md">English</a>
</div>

<div align="center">
  <!-- <img src="./assets/logo.png" width="100" /> -->
  <h1 style="display: inline;">Milli-Chat</h1>
</div>


<details>
    <summary><strong>👉 博客讲解</strong></summary>

0. Transformer基础 [博客园：大模型基石——Transformer架构深度解构](https://www.cnblogs.com/skeinz/articles/19429739)
1. KV Cache推理加速 [博客园：推理加速KV Cache与显存优化](https://www.cnblogs.com/skeinz/articles/19436212)
2. 全量指令微调SFT [博客园：大模型全量指令微调——Full Parameter SFT](https://www.cnblogs.com/skeinz/articles/19450524)
3. 参数高效微调PEFT [CSDN：大模型参数高效微调PEFT](https://blog.csdn.net/m0_50203796/article/details/156774417?spm=1001.2014.3001.5501)
4. 人类偏好对齐
    - PPO在大语言模型中的应用 [CSDN：人类偏好对齐——大模型PPO算法原理](https://blog.csdn.net/m0_50203796/article/details/156953577?spm=1001.2014.3001.5501)
    - 直接偏好优化DPO

</details>

<div align="center">😭穷也想玩转大模型：尽可能只用最基本的库、用最朴素、不专业的方式动手写一遍，蹭免费GPU完成训练。</div>


## 项目结构介绍

```
├─📂config                    ---> 训练配置
├─📂dataset                   ---> 数据集存放目录
├─📂model                     ---> 模型源代码
│  ├─ configuration_slm.py
│  └─ model_slm.py
├─📂out                       ---> 模型参数存放 (已存放我自己训练好的分词器)
│  ├─ special_tokens_map.json
│  ├─ tokenizer_config.json
│  └─ tokenizer.json
├─📂scripts                   ---> 包含处理数据的脚本目录
│  ├─ chat_terminal.py
│  └─ ...
└─📂trainer                   ---> 训练脚本：tokenizer, pre, SFT, DPO...
   ├─ pretrain.py
   └─ ...
```

除了个别脚本可以单独运行（`prepare_shards.py`、`train_tokenizer.py`），上述所有文件均作为Python导入模块使用，即项目主目录下使用`main.py`来导入相应的函数方法。模型将基于Hugging Face所提供的transformers库来实现。

🚀 **环境准备**：`requirements.txt`文件是我的conda环境，有很多无用的包。本项目中所用到的包仅仅只有torch, transformers, tokenizers, numpy, zhconv，使用`pip install`即可。torch版本需要保证在2.0以上，因为在模型中使用了`torch.nn.functional.scaled_dot_product_attention`函数，如果你想自己实现的话，torch版本不需要这么高。

## 1 语料准备

首先需要准备大量的语料。我希望我的模型能够会中英双语，因此需要准备中文和英文的语料，最常见的就是[Wikipedia](https://dumps.wikimedia.org/)，下载下来的格式是`xxx.xml-xxxx.bz2`，然后使用[WikiExtractor](https://github.com/attardi/wikiextractor)将包中的数据提取到txt文件中。注意，Wikipedia的中文数据是繁体，可以使用`zhconv`库来将其转化为简体使用，可以参考`scripts/zh_simplify.py`脚本，当然你也可以采用自己的方式实现。如果是其它的语料类型，可以按照自己的方式处理，本项目的预训练以txt为语料数据的格式。

但是请记住这句话：Data is all you need。如果预训练的语料质量不高，那么后续训练的效果并不会好到哪里去。Wikipedia语料库处理之后会存在如`（；）`这样的标点，可能会对模型的训练产生影响，因此语料库的清洗一定要彻底，特别是对于小模型而言，其参数量还不能达到知识涌现的标准，很难自行过滤这些“乱码”。

## 2 训练分词器

在准备好txt语料之后，当你的内存足够的时候，可以一股脑将语料全都喂给训练器，直接运行脚本`trainer/train_tokenizer.py`即可。我希望模型能够生成语料库中没出现的低频token，比如表情，因此可以开启ByteLevel拆分方式。在只有16GB的内存中，基于BPE分词并开启了ByteLevel模式之后一定会导致内存爆炸。因为用uft-8编码，一个汉字通常占3个字节，有3500个常用汉字，意味着对于中文而言需要额外统计字节对。因此我决定只单独训练英文预料，而将一个汉字作为一个token，统计常用汉字并单独加入到分词器中。对于小模型而言这已经足够了。

训练好分词器之后，会在`/out`目录下输出相关的分词器文件，后续只需要通过如下方式调用即可：
```python
tokenizer = transformers.AutoTokenizer.from_pretrained('./out')
```

## 3 预训练

在进行预训练之前，需要处理一下语料。如果要训练一个双语或者多语言的模型，就必须将这些语言混合起来而不是分开训练，分开训练会导致遗忘灾难，即模型学了后面忘了前面，因此使用脚本`scripts/prepare_shards.py`将语料混合。除此之外，混合的同时也用token id来存储，即本项目中的词表大小设置为32000，每个token id可以用16位无符号整数来存储，这样数据文件的大小能够被进一步压缩。注意语料混合脚本有一个缺陷：该脚本将所有的语料都加载进来一并打散洗牌，这在内存不够的情况下是不可行的。

在本地没有庞大的GPU计算资源的情况下，本项目选择在线训练。由于担心网络波动导致训练中断，因此将数据分成多个文件，每训练完一个文件分片保存一次权重，方便下次继续训练。

此外，预训练需要尽可能地让模型多吃一些语料，即便是1G的语料就能使loss下降到2~3、后续的训练loss几乎不会下降，仍不建议停止训练，否则后续微调后的模型会胡言乱语，回答没有逻辑。预训练的目标是让模型能够流畅地完成句子续写，而不是当一个复读机、续写的过程中重复一个词或一句话。

## 4 全量有监督微调SFT

一般大模型的`tokenizer_config.json`文件中都有一个`chat_template`属性，使用Jinja2格式存放对话模板，在代码中使用`apply_chat_template`函数来封装提示词，这样就不需要手动构造了。但是我在训练完模型后保存tokenizer的时候，始终无法在代码中将该属性添加进去，而手动添加、即在文件中手动输入才能添加进去。因此对这个地方有疑惑。

## 5 人类偏好对齐

由于RLHF的工程量太大了，所以我采用DPO训练。不过PPO、GRPO相关的代码也会实现，这些代码能够在我的环境下测试跑通，但是代码实现可能有不正确的地方。

### PPO

> [!WARN]
>
> 直接使用原模型的参数权重会有如下警告信息：
> Some weights of SLMForSequenceClassification were not initialized from the model checkpoint at ./out/step_0 and are newly initialized: ['score.weight']
> You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

## 6 未完待续...

🛰 **说明**：按照`pretrain.py`中的模型默认配置，未开启MoE并测试，大约有124M个参数。本项目会持续不断地慢慢地更新完善，包括其它的微调方法实现、知识蒸馏、量化部署、模型评估等。

# 💡 TODO

在2025年12月31日草率地完成了模型的全量微调，经过测试发现效果并不理想：
- 由于预训练的语料不够多，只有1~2G，因此模型的知识库并不丰富，无法准确地回答用户问题；并且模型还没有完全收敛，句子续写仍会出现复读机现象或者没有逻辑性；中文部分主要采用的是新闻类语料，导致句子续写总是输出车轱辘话。
- 我希望该小模型具备两种语言能力，因此词汇表大小设置为了32000（准确来说是31900）。然而这是一个巨大的挑战。首先是Embedding层需要更新大量的参数，其次是训练tokenizer的时候内存不够用，最后是FFN层的脑容量不够大、难以同时掌握两种语言的对话能力。

基于以上教训，我将重新训练一个小模型：
1. 放弃双语，将词表大小降低为6400，仅训练一个中文聊天模型；
2. 修改checkpoint的保存方式：将step和loss保存到pt或pth文件中，并且降低每个分片文件的大小为50M左右；
3. 重新用更多、更高质量的语料库进行预训练。

初步目标：能顺利完成较长单步对话就行。