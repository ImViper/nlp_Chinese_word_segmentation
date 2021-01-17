import timeit
import os
import argparse

import model.dict_generator as dg
from model.mm_seg import MMSeg
from model.unigram_seg import UniGramSeg
from model.hmm_seg import HmmSeg

def get_mmseg(dataset):
    seg = MMSeg()
    if dataset == 'pku':
        mdict = dg.json_read("model/dicts/pku_dict.json", encoding='utf-16')
    else:
        mdict = dg.json_read("model/dicts/shanxi_dict.json", encoding='utf-16')
    seg.set_dict(mdict)
    return seg

def get_uniseg(dataset):
    seg = UniGramSeg()
    if dataset == 'pku':
        mdict = dg.json_read("model/dicts/pku_dict.json", encoding='utf-16')
    else:
        mdict = dg.json_read("model/dicts/shanxi_dict.json", encoding='utf-16')
    seg.set_dict(mdict)
    return seg

def get_hmmseg(dataset):
    script_dir = os.path.dirname(__file__)
    seg = HmmSeg()
    if dataset == 'pku':
        model_path = os.path.join(script_dir, "model/hmm_para")
    else:
        model_path = os.path.join(script_dir, "model/sx_hmm_para")
    seg.load(model_path)
    return seg

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--segger', default='mm')
    parser.add_argument('--dataset', default='pku')
    args = parser.parse_args()

    if args.segger not in ['unigram', 'mm', 'hmm']:
        raise ValueError("[segger]参数不合法, 只有(unigram, mm, hmm).")

    if args.dataset not in ['pku', 'shanxi']:
        raise ValueError("[dataset]参数不合法, 只有(pku, shanxi).")

    seg = None
    if args.segger == 'unigram':
        seg = get_uniseg(args.dataset)
    elif args.segger == 'mm':
        seg = get_mmseg(args.dataset)
    else:
        seg = get_hmmseg(args.dataset)

    pred = []
    ch_count = 0
    time_cost = 0
    test_path = os.path.join('test_data', args.dataset, 'test.txt')
    
    encoding = 'utf-8' if args.dataset == 'pku' else 'utf-16'

    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        start = timeit.default_timer()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred.append(seg.cut(sent))
                ch_count += len(sent)
            except:
                pred.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed {0}/{1} ---- {2}".format(count, total_count, count/total_count))
        end = timeit.default_timer()
        time_cost = end - start

    
    print("Total number of characters: {0}.".format(ch_count))
    print("Time cost: {0}s.".format(time_cost))
    print("Processed characters per second: {0}.".format(int(ch_count / time_cost)) )

    # 保存分词结果
    save_path = test_path = os.path.join('test_data', args.dataset, args.segger+'_seg.txt')
    with open(save_path, "w", encoding='gb18030') as f:
        for words in pred:
            s = " ".join(words)
            s = s.strip() + '\n'
            s.encode('gb18030')
            f.write(s)

    print("Segmentation result is saved in {0}.".format(save_path))