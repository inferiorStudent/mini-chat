<div align="center">
  中文 | <a href="./README_en.md">English</a>
</div>

<div align="center">
  <img src="./assets/logo.png" width="100" />
  <h1 style="display: inline;">Mini-Chat</h1>
</div>


<details>
    <summary><strong>👉 博客讲解</strong></summary>

- [cnblogs 还在施工中]()
- [CSDN 还在施工中]()

</details>



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

除了个别脚本可以单独运行（`prepare_shards.py`、`train_tokenizer.py`），上述所有文件均作为Python导入模块使用，即项目主目录下使用`main.py`来导入相应的函数方法。模型将基于Hugging Face所提供的transformers库来实现，以和开源大模型看齐。

🚀 **环境准备**：`requirements.txt`文件是我的conda环境，可能有无用的包。本项目中所用到的包仅仅只有torch, transformers, tokenizers, numpy, zhconv，使用`pip install`即可。torch版本需要保证在2.0以上，因为在模型中使用了`torch.nn.functional.scaled_dot_product_attention`函数，如果你想自己实现的话，torch版本不需要这么高。

## 1 语料准备

首先你需要准备大量的语料。我希望我的模型能够会中英双语，因此需要准备中文和英文的语料，最常见的就是[Wikipedia](https://dumps.wikimedia.org/)，你下载下来的格式是`xxx.xml-xxxx.bz2`，然后使用[WikiExtractor](https://github.com/attardi/wikiextractor)将包中的数据提取到txt文件中。注意，Wikipedia的中文数据是繁体，你可以使用`zhconv`库来将其转化为简体使用，可以参考`scripts/zh_simplify.py`脚本，当然你也可以采用自己的方式实现。

如果是其它的语料类型，你可以按照自己的方式处理，本项目的预训练以txt为语料数据的格式。

## 2 训练分词器

在准备好txt语料之后，当你的内存足够的时候，可以一股脑将语料全都喂给训练器，直接运行脚本`trainer/train_tokenizer.py`即可。我希望模型能够生成语料库中没出现的低频token，比如表情，因此可以开启ByteLevel拆分方式。在只有16GB的内存中，基于BPE分词并开启了ByteLevel模式之后一定会导致内存爆炸。因为用uft-8编码，一个汉字通常占3个字节，有3500个常用汉字，意味着对于中文而言需要额外统计字节对。因此我决定只单独训练英文预料，而将一个汉字作为一个token，统计常用汉字并单独加入到分词器中。对于小模型而言这已经足够了。

训练好分词器之后，会在`/out`目录下输出相关的分词器文件，后续只需要通过如下方式调用即可：
```python
tokenizer = transformers.AutoTokenizer.from_pretrained('./out')
```

## 3 预训练

在进行预训练之前，需要处理一下语料。如果要训练一个双语或者多语言的模型，就必须将这些语言混合起来而不是分开训练，分开训练会导致遗忘灾难，即模型学了后面忘了前面，因此使用脚本`scripts/prepare_shards.py`将语料混合。除此之外，混合的同时也用token id来存储，即本项目中的词表大小设置为32000，每个token id可以用16位无符号整数来存储，这样数据文件的大小能够被进一步压缩。

> [!WARNING]
>
> 注意语料混合脚本有一个缺陷：该脚本将所有的语料都加载进来一并打散洗牌，这在内存不够的情况下是不可行的。

在没有庞大的GPU计算资源的情况下，本项目选择在线训练。由于担心网络波动导致训练中断，因此将数据分成多个文件，每训练完一个文件分片保存一次权重，方便下次继续训练。

## 4 未完待续...

🛰 **说明**：按照`pretrain.py`中的模型默认配置，大约有124M个参数，尚且开启MoE并测试。本项目会持续不断地完善，包括其它的微调方法实现、知识蒸馏、量化部署、模型评估等，并随博客一起更新。