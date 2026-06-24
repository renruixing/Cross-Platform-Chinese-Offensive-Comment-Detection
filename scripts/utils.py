"""公共工具：数据集封装、模型构建、训练与评测循环。被各步骤脚本复用。"""
import json, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import config as C


def set_seed(s):
    random.seed(s);
    np.random.seed(s);
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = list(texts);
        self.labels = list(labels)
        self.tok = tok;
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def build_model(path=None):
    path = path or C.PRETRAINED
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
    return tok, model.to(C.DEVICE)


def train_loop(model, train_ds, val_ds=None, hp=None, max_steps=None, eval_steps=None,
               log_history=None):
    """通用训练。返回 (loss_history, metric_history)。"""
    from torch.optim import AdamW
    hp = hp or {}
    bs = hp.get("batch_size", 16)
    lr = hp.get("lr", 2e-5);
    wd = hp.get("weight_decay", 0.01)
    dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_hist, metric_hist = [], []
    step = 0;
    model.train()
    epochs = hp.get("epochs")

    if epochs:  # epoch 模式（硬样本微调）
        total = epochs
        # ✅ 添加 epoch 级别的进度条
        pbar_epoch = tqdm(range(epochs), desc="Epochs", unit="epoch")
        for ep in pbar_epoch:
            # ✅ 添加 batch 级别的进度条
            pbar_batch = tqdm(dl, desc=f"Epoch {ep + 1}/{epochs}", unit="batch", leave=False)
            for batch in pbar_batch:
                batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
                out = model(**batch);
                out.loss.backward()
                opt.step();
                opt.zero_grad();
                step += 1
                loss_hist.append((step, out.loss.item()))
                # 更新进度条显示
                pbar_batch.set_postfix({'loss': f'{out.loss.item():.4f}'})
            if val_ds is not None:
                metrics = evaluate(model, val_ds)
                metric_hist.append((ep, metrics))
                pbar_epoch.set_postfix({
                    'f1': f'{metrics["f1"]:.4f}',
                    'acc': f'{metrics["accuracy"]:.4f}'
                })
        return loss_hist, metric_hist

    # step 模式（基准微调）
    max_steps = max_steps or hp.get("max_steps", 10000)
    eval_steps = eval_steps or hp.get("eval_steps", 500)

    print(f"\n🚀 开始训练 (总步数: {max_steps}, 评估间隔: {eval_steps})")
    print("-" * 60)

    # ✅ 创建主进度条
    pbar = tqdm(total=max_steps, desc="Training", unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    done = False
    while not done:
        for batch in dl:
            batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
            out = model(**batch);
            out.loss.backward()
            opt.step();
            opt.zero_grad();
            step += 1
            loss_hist.append((step, out.loss.item()))

            # ✅ 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{out.loss.item():.4f}',
                'step': f'{step}/{max_steps}'
            })

            if val_ds is not None and step % eval_steps == 0:
                metrics = evaluate(model, val_ds)
                metric_hist.append((step, metrics))
                model.train()
                # ✅ 在进度条上显示评估结果
                pbar.set_postfix({
                    'loss': f'{out.loss.item():.4f}',
                    'f1': f'{metrics["f1"]:.4f}',
                    'acc': f'{metrics["accuracy"]:.4f}'
                })
                # ✅ 打印评估结果
                print(
                    f"\n📊 Step {step}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, Loss={out.loss.item():.4f}")

            if step >= max_steps:
                done = True
                break

    pbar.close()
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    return loss_hist, metric_hist


@torch.no_grad()
def predict(model, ds, return_conf=False):
    model.eval()
    dl = DataLoader(ds, batch_size=64)
    preds, confs = [], []
    # ✅ 添加预测进度条
    pbar = tqdm(dl, desc="Predicting", unit="batch", leave=False)
    for batch in pbar:
        labels = batch.pop("labels", None)
        batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        prob = torch.softmax(logits, dim=-1)
        p = prob.argmax(-1).cpu().numpy()
        preds.extend(p.tolist())
        confs.extend(prob.max(-1).values.cpu().numpy().tolist())
    return (preds, confs) if return_conf else preds


@torch.no_grad()
def evaluate(model, ds):
    model.eval()
    dl = DataLoader(ds, batch_size=64)
    ys, ps, probs1 = [], [], []
    # ✅ 添加评估进度条
    pbar = tqdm(dl, desc="Evaluating", unit="batch", leave=False)
    for batch in pbar:
        y = batch.pop("labels").numpy()
        batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
        logit = model(**batch).logits
        prob = torch.softmax(logit, -1)
        ys.extend(y.tolist())
        ps.extend(prob.argmax(-1).cpu().numpy().tolist())
        probs1.extend(prob[:, 1].cpu().numpy().tolist())
    m = dict(
        accuracy=accuracy_score(ys, ps),
        precision=precision_score(ys, ps, zero_division=0),
        recall=recall_score(ys, ps, zero_division=0),
        f1=f1_score(ys, ps, zero_division=0),
    )
    try:
        m["auc"] = roc_auc_score(ys, probs1)
    except Exception:
        m["auc"] = float("nan")
    m["confusion"] = confusion_matrix(ys, ps, labels=[0, 1]).tolist()
    m["_y"], m["_prob1"] = ys, probs1
    return m


def save_json(obj, path):
    # 去掉不可序列化的中间数组
    def clean(d):
        if isinstance(d, dict):
            return {k: clean(v) for k, v in d.items() if not k.startswith("_")}
        if isinstance(d, list):
            return [clean(x) for x in d]
        return d

    json.dump(clean(obj), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", path)