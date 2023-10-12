from scapy.all import *
from tqdm import tqdm
import os
import json
import random

START_KEY = 0
END_KEY = 1


def get_len_seq(filename):
    reader = PcapReader(filename)
    len_seq = [START_KEY]

    for i_pak, packet in enumerate(reader):
        if i_pak >= 100:
            break
        val = len(packet) + 2
        len_seq.append(min(val, 4999))
    len_seq.append(END_KEY)

    reader.close()
    return len_seq


if __name__ == '__main__':
    data_path = 'D:\\workspace\\dl\\traffic-analysis\\2018YFF01012202-010-应用流量数据集\\原始流量数据'
    apps = os.listdir(data_path)
    train_seqs_of_app, test_seqs_of_app = [
        [[] for _ in range(len(apps))] for _ in range(2)]
    train_count, test_count, all_count = [[0] * len(apps) for _ in range(3)]
    n_train, n_test = 0, 0
    for i_app, app in enumerate(apps):
        print(f'App: {app}')
        app_path = os.path.join(data_path, app)
        for pcap in tqdm(os.listdir(app_path)):
            try:
                pcap_path = os.path.join(app_path, pcap)
                seq = get_len_seq(pcap_path)
                if random.random() < 0.8:
                    train_seqs_of_app[i_app].append(seq)
                    train_count[i_app] += 1
                    n_train += 1
                else:
                    test_seqs_of_app[i_app].append(seq)
                    test_count[i_app] += 1
                    n_test += 1
                all_count[i_app] += 1
            except Exception as e:
                print(f'Error on pcap_path={pcap_path}', e)
    with open('./dataset/train.json', 'w') as f:
        f.write(json.dumps(train_seqs_of_app))
    with open('./dataset/test.json', 'w') as f:
        f.write(json.dumps(test_seqs_of_app))
    with open('./dataset/meta.json', 'w') as f:
        f.write(json.dumps({
            'n_train': n_train,
            'n_test': n_test,
            'n_app': len(apps),
            'train_count': train_count,
            'test_count': test_count,
            'all_count': all_count
        }))
