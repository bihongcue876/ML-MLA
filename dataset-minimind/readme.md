# 官方数据与权重

本目录存放外部获取的资产，全部来自 ModelScope 的 MiniMind 项目，**不入库**（见仓库根 `.gitignore`）。首次使用需自行下载并放到下表中的位置。

## 目录内容

| 路径 | 内容 | 体积 |
| --- | --- | --- |
| `corpus/pretrain_t2t_mini.jsonl` | 预训练语料，每行一个 JSON，取 `text` 字段 | 1.24 GB，1,270,238 行，约 2.1–2.4 亿 token |
| `tokenizer/tokenizer.json` | 词表 6400 的分词器主文件 | 451 KB |
| `tokenizer/tokenizer_config.json`、`special_tokens_map.json` | 分词器配置 | 共 8.6 KB |
| `model.safetensors` | MiniMind-3 官方权重，结构比对与外部参考之用 | 127.8 MB，90 个张量 |

## 来源

| 资产 | ModelScope 仓库 |
| --- | --- |
| 语料 | `gongjy/minimind_dataset` |
| tokenizer、官方权重 | `gongjy/minimind-3` |

HuggingFace 直连不可用，全部资产经 ModelScope 获取。

## 获取方式

```bash
pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
snapshot_download('gongjy/minimind_dataset', local_dir='dataset-minimind/corpus')
snapshot_download('gongjy/minimind-3', local_dir='dataset-minimind/minimind-3')
PY
```

下载后把 `dataset-minimind/minimind-3/` 下的 `tokenizer/` 与 `model.safetensors` 上移一层，去掉中间目录，最终结构为：

```text
dataset-minimind/
  corpus/pretrain_t2t_mini.jsonl
  tokenizer/{tokenizer.json, tokenizer_config.json, special_tokens_map.json}
  model.safetensors
```

## 用途边界

- 语料经 `MLATrain/prepare_data.py` 切分为 `data/tokens/{train,val,test}.bin`，切分后不再读原始 jsonl。
- tokenizer 只用于切分与评测解码；换 tokenizer 后旧权重不可复用。
- 官方权重不参与训练，也不与自训模型并列比较结构优劣，仅用于核对分词、词表与张量形状是否符合 MiniMind 的约定。

## 许可

资产版权与许可归 MiniMind 项目与原始语料来源所有，本仓库只记录获取路径与用途，不重新分发。
