import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.data import SequentialSampler
from dataset import Seq2SeqDataset, TestDataset
from model import TransformerModel
import argparse
import numpy as np
import os
from tqdm import tqdm
import logging
import transformers
import math
from torch.utils.data import Sampler
import random
import torch.multiprocessing
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=768, type=int)
    parser.add_argument("--hidden-size", default=3072, type=int)
    parser.add_argument("--num-layers", default=12, type=int)
    parser.add_argument("--batch-size", default=200, type=int)
    parser.add_argument("--test-batch-size", default=16, type=int)
    parser.add_argument("--lr", default=2.5e-4, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--num-epoch", default=10, type=int)
    parser.add_argument("--save-interval", default=7000000, type=int)
    parser.add_argument("--save-dir", default="model_1")
    parser.add_argument("--ckpt", default="ckpt_1.pt")
    parser.add_argument("--dataset", default="/root/metakg/data/")
    parser.add_argument("--label-smooth", default=0.5, type=float)
    parser.add_argument("--l-punish", default=False, action="store_true") # during generation, add punishment for length
    parser.add_argument("--beam-size", default=10, type=int) # during generation, beam size
    parser.add_argument("--encoder", default=True, action="store_true") # only use TransformerEncoder
    parser.add_argument("--trainset", default="part_0.txt")
    parser.add_argument("--max-len", default=10, type=int) # maximum number of hops considered
    parser.add_argument("--iter-batch-size", default=128, type=int)
    parser.add_argument("--warmup", default=50, type=float) # warmup steps ratio
    parser.add_argument("--output-path", default=False, action="store_true") # output top correct path in a file (for interpretability evaluation)
    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, args,):
    model.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = args.max_len
    # restricted_punish = -30
    mrr, hit, hit1, hit3, hit10, count = (0, 0, 0, 0, 0, 0)
    vocab_size = len(model.dictionary)
    eos = model.dictionary.eos()
    # bos = model.dictionary.bos()
    rev_dict = dict()
    lines = []
    for k in model.dictionary.indices.keys():
        v = model.dictionary.indices[k]
        rev_dict[v] = k
    with tqdm(dataloader, desc="testing") as pbar:
        for samples in pbar:
            pbar.set_description("MRR: %f, Hit@1: %f, Hit@3: %f, Hit@10: %f" % (mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit10/max(1, count)))
            batch_size = samples["source"].size(0)
            candidates = [dict() for i in range(batch_size)]
            candidates_path = [dict() for i in range(batch_size)]
            source = samples["source"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            source_typeid = samples["type_id"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            # prefix: save beam_size generated sequence
            prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            # print("prefix.shape=",prefix.shape, "source.shape=",source.shape)
            prefix[:, :, 0] = source.squeeze(-1)
            prefix_typeid = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix_typeid[:, :, 0] = source_typeid.squeeze(-1)
            lprob = torch.zeros([batch_size, beam_size]).to(device)
            clen = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token: choose beam_size from only vocab_size, initiate prefix
            tmp_source = samples["source"]
            tmp_type_id = samples["type_id"]
            logits = model.logits(tmp_source,tmp_type_id).squeeze()
            # if args.no_filter_gen:
            logits = F.log_softmax(logits, dim=-1)
            logits = logits.view(-1, vocab_size)
            argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]

            prefix[:, :, 1] = argsort[:, :]
            # decode and generate prefix_typeid
            for i in range(batch_size):
                for j in range(beam_size):
                    # Decode the id to symbols
                    decoded_symbols = model.dictionary.decode(prefix[i, j].tolist())
                    # Re-encode the symbols to get the corresponding type ids
                    _, reencoded_type_ids = model.dictionary.encode_line(
                        decoded_symbols, add_if_not_exist=False, append_eos=False
                    )
                    # Assign the re-encoded type ids to prefix_typeid
                    prefix_typeid[i, j, :len(reencoded_type_ids)] = reencoded_type_ids

            lprob += torch.gather(input=logits, dim=-1, index=argsort)
            clen += 1
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)    
                tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                # print("source.shape=", source.view(bb, -1).shape, "prefix.shape=", prefix.shape)
                # sour = torch.cat((source.view(bb, -1), prefix.view(bb, -1)), dim=-1)
                # typeid = torch.cat((source_typeid.view(bb, -1), prefix_typeid.view(bb, -1)), dim=-1)
                all_logits = model.logits(prefix.view(bb, -1), prefix_typeid.view(bb, -1))
                all_logits = all_logits.view(batch_size, beam_size, max_len, -1)
                # print("===all_logits.shape=",all_logits.shape)
                # all_logits = model.logits(sour, typeid).view(batch_size, beam_size, max_len, -1)
                logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                logits = F.log_softmax(logits, dim=-1)
                argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]

                tmp_clen = tmp_clen + 1
                tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)
                tmp_prefix, tmp_lprob, tmp_clen = tmp_prefix.view(batch_size, -1, max_len), tmp_lprob.view(batch_size, -1), tmp_clen.view(batch_size, -1)
                if l == max_len-1:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :(2*beam_size)]
                else:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]
                prefix = torch.gather(input=tmp_prefix, dim=1, index=argsort.unsqueeze(-1).repeat(1, 1, max_len))
                for i in range(batch_size):
                    for j in range(beam_size):
                        # Decode the id to symbols
                        decoded_symbols = model.dictionary.decode(prefix[i, j].tolist())
                        # Re-encode the symbols to get the corresponding type ids
                        # print("decoded_symbols======",decoded_symbols)
                        _, reencoded_type_ids = model.dictionary.encode_line(
                            decoded_symbols, add_if_not_exist=False, append_eos=False
                        )
                        # Assign the re-encoded type ids to prefix_typeid
                        prefix_typeid[i, j, :len(reencoded_type_ids)] = reencoded_type_ids
                lprob = torch.gather(input=tmp_lprob, dim=1, index=argsort)
                clen = torch.gather(input=tmp_clen, dim=1, index=argsort)
                # filter out next token after <end>, add to candidates
                for i in range(batch_size):
                    for j in range(beam_size):
                        if prefix[i][j][l].item() == eos:
                            candidate = prefix[i][j][l-1].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(l / 2)
                            else:
                                prob = lprob[i][j].item()
                            lprob[i][j] -= 10000
                            if candidate not in candidates[i]:
                                candidates[i][candidate] = math.exp(prob)

                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                                candidates[i][candidate] += math.exp(prob)
                # no <end> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate = prefix[i][j][l].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(max_len/2)
                            else:
                                prob = lprob[i][j].item()
                            if candidate not in candidates[i]:
                                candidates[i][candidate] = math.exp(prob)

                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                                candidates[i][candidate] += math.exp(prob)

            target = samples["target"].cpu()
            for i in range(batch_size):
                hid = samples["source"][i][0].item()
                count += 1
                candidate_ = sorted(zip(candidates[i].items(), candidates_path[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate = [pair[0][0] for pair in candidate_]
                candidate_path = [pair[1][1] for pair in candidate_]
                candidate = torch.from_numpy(np.array(candidate))
                ranking = (candidate[:] == target[i]).nonzero()
                path_token = rev_dict[hid] + " " + rev_dict[target[i].item()] + '\t'

                if ranking.nelement() != 0:
                    path = candidate_path[ranking]
                    for token in path[1:-1]:
                        path_token += (rev_dict[token]+' ')
                    path_token += (rev_dict[path[-1]]+'\t')
                    path_token += str(ranking.item())
                    ranking = 1 + ranking.item()
                    mrr += (1 / ranking)
                    hit += 1
                    if ranking <= 1:
                        hit1 += 1
                    if ranking <= 3:
                        hit3 += 1
                    if ranking <= 10:
                        hit10 += 1
                else:
                    path_token += "wrong"
                lines.append(path_token+'\n')
    
    if args.output_path:
        with open("test_output_Metaknowledge.txt", "w") as f:
            f.writelines(lines)
    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@10: %f]" % (mrr/count, hit1/count, hit3/count, hit10/count))
    return hit/count, hit1/count, hit3/count, hit10/count

def train(args):
    train_dataset = '/root/metakg/data/train/'
    save_path = os.path.join('/root/metakg/', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/train.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.info(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda:0"

    torch.multiprocessing.set_start_method('spawn', force=True)
    train_set = Seq2SeqDataset(data_path=train_dataset, vocab_file=args.dataset+"vocab.txt", device=device, args=args)
    # sampler = CustomRandomSampler(train_set, args.batch_size)
    sampler = SequentialSampler(train_set)
    train_loader = DataLoader(train_set, sampler=sampler, batch_size=args.batch_size, collate_fn=train_set.collate_fn)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)

    model = TransformerModel(args, train_set.dictionary).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    total_step_num = len(train_loader) * args.num_epoch
    print("steps is ==", total_step_num)
    warmup_steps = total_step_num / args.warmup
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_step_num)

    steps = 0
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            for samples in pbar:
                optimizer.zero_grad()
                loss = model(**samples)
                if loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                steps += 1
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))
        logging.info(
                "[Epoch %d/%d] [train loss: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses))
                )
        if (steps % args.save_interval == 0 ) or (steps ==total_step_num):
            torch.save(model.state_dict(), ckpt_path + "/ckpt_{}.pt".format(steps))
            print("it is evaluating =======")
            with torch.no_grad():
                valid_set = TestDataset(data_path=args.dataset, vocab_file=args.dataset + "vocab.txt", device=device,
                                        src_file="dev.txt")
                valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, collate_fn=valid_set.collate_fn,
                                          shuffle=True)
                evaluate(model.module, valid_loader, device, args)

def model_test(args):
    # args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('/root/metakg/', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(ckpt_path):
        print("Invalid path!")
        return
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/test.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/train.txt", device=device, args=args)
    test_set = TestDataset(data_path=args.dataset, vocab_file=args.dataset+"vocab.txt", device=device, src_file="test.txt")
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    model = TransformerModel(args, test_set.dictionary)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, args.ckpt)))
    model.args = args
    model = model.to(device)
    with torch.no_grad():
        evaluate(model.evaluate(), test_loader, device, args)

import torch


class CustomRandomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.indices = list(range(len(self.data_source)))

    def __iter__(self):
        # 在每个 epoch 开始时，重新随机化索引列表
        random.shuffle(self.indices)

        # 按 batch_size 切分索引列表并返回
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.data_source) // self.batch_size



if __name__ == "__main__":
    args = get_args()
    if args.test:
        model_test(args)
    else:
        train(args)
