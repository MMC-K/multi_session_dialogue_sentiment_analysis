import os
import gc
import pickle
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')


def _astype_f32(x):
    return x.astype(np.float32) if hasattr(x, "astype") else np.array(x, dtype=np.float32)

def _read_manifest_lines(path):
    """manifest(.txt)에는 샤드 pkl 파일 '파일명'들이 줄단위로 적혀있고,
    manifest와 같은 폴더에 pkl들이 존재한다고 가정한다.
    """
    with open(path, "r", encoding="utf-8") as f:
        files = [ln.strip() for ln in f if ln.strip()]
    base = os.path.dirname(os.path.abspath(path))
    return [os.path.join(base, p) for p in files]


class _ShardIterableDataset(IterableDataset):
    """MMDataset(샤드 모드)의 __iter__를 그대로 노출하여
    DataLoader가 Iterable 경로로 동작하게 한다.
    """
    def __init__(self, mm_ds: 'MMDataset'):
        super().__init__()
        self.mm_ds = mm_ds

    def __iter__(self):
        yield from self.mm_ds.__iter__()

    def __len__(self):
        try:
            return len(self.mm_ds)
        except Exception:
            return 0


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args

        self._shard_mode = False
        fp = self.args.get('featurePath', '')
        if isinstance(fp, str) and fp.lower().endswith('.txt') and os.path.isfile(fp):
            self._shard_mode = True
            self._init_sharded_from_featurePath_manifest()
            return

        # --- 기존 단일 PKL 모드 ---
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
        }
        DATASET_MAP[args['dataset_name']]()

    # 샤드(Manifest) 모드
    def _init_sharded_from_featurePath_manifest(self):
        manifest = self.args['featurePath']
        self._part_files = _read_manifest_lines(manifest)
        if not self._part_files:
            raise RuntimeError(f"manifest has no shard entries: {manifest}")


        self._shuffle_shards = (self.mode == 'train')
        self._shuffle_within = (self.mode == 'train')
        self._seed = int(self.args.get('seed', 42))

        # 첫 샤드로 메타 확인
        meta = self._scan_meta_from_first_shard(self._part_files[0])
        self._feature_dims = meta['feature_dims']   # (t_dim, a_dim, v_dim)
        self._seq_lens     = meta['seq_lens']       # (t_len, a_len, v_len)

        # 전체 길이
        self._global_len = self._fast_count_total_examples()

    def _safe_load_shard(self, p):
        with open(p, 'rb') as f:
            return pickle.load(f)

    def _scan_meta_from_first_shard(self, pkl_path):
        d = self._safe_load_shard(pkl_path)
        if self.mode not in d:
            raise KeyError(f"{pkl_path} has no key: {self.mode}")
        pack = d[self.mode]
        use_bert = bool(self.args.get('use_bert', False))
        txt_key = 'text_bert' if use_bert else 'text'

        text  = pack[txt_key]
        audio = pack['audio']
        vision= pack['vision']

        if use_bert:
            t_dim = 768
            t_len = text.shape[2]    # (N,3,L)
        else:
            t_dim = text.shape[2]    # (N,T,D)
            t_len = text.shape[1]
        a_dim, a_len = audio.shape[2], audio.shape[1]
        v_dim, v_len = vision.shape[2], vision.shape[1]
        del d; gc.collect()
        return {'feature_dims': (t_dim, a_dim, v_dim), 'seq_lens': (t_len, a_len, v_len)}

    def _fast_count_total_examples(self):
        total = 0
        for p in self._part_files:
            try:
                d = self._safe_load_shard(p)
                pack = d.get(self.mode, None)
                if pack is None:
                    continue
                labels = pack.get('regression_labels', None)
                if labels is None:
                    labels = pack.get('M', None)
                if labels is not None:
                    total += len(labels)
                del d; gc.collect()
            except Exception:
                continue
        return total

    def _iter_shard_order(self, rng):
        files = list(self._part_files)
        if self._shuffle_shards:
            rng.shuffle(files)
        return files

    def _normalize_time_major_mean(self, x):
        x = np.transpose(x, (1, 0, 2))   # (T,N,D)
        x = np.mean(x, axis=0, keepdims=True)  # (1,N,D)
        x[x != x] = 0
        x = np.transpose(x, (1, 0, 2))   # (N,1,D)
        return x

    def _make_text_missing(self, text):
        # text: (N,3,L)
        rate = self.args.missing_rate[0]
        seed = self.args.missing_seed[0]
        np.random.seed(seed)
        input_ids = text[:, 0, :]
        input_mask = text[:, 1, :]
        seg_ids = text[:, 2, :]
        input_len = np.argmin(input_mask, axis=1)
        missing_mask = (np.random.uniform(size=input_mask.shape) > rate) * input_mask
        for i, inst in enumerate(missing_mask):
            inst[0] = inst[input_len[i]-1] = 1
        unk = 100 * np.ones_like(input_ids)
        input_ids_m = missing_mask * input_ids + (input_mask - missing_mask) * unk
        text_m = np.concatenate([np.expand_dims(input_ids_m, 1),
                                 np.expand_dims(input_mask, 1),
                                 np.expand_dims(seg_ids, 1)], axis=1)
        return text_m, input_mask, missing_mask, input_len

    def _make_mod_missing(self, modality, lengths):
        # modality: (N,T,D), lengths: (N,)
        rate = self.args.missing_rate[1] if modality is not None else 0
        seed = self.args.missing_seed[1] if modality is not None else 0
        np.random.seed(seed)
        N, T, _ = modality.shape
        masks = []
        for L in lengths:
            L = int(L)
            masks.append(np.array([1]*L + [0]*(T-L)))
        mask = np.stack(masks, axis=0)               # (N,T)
        missing_mask = (np.random.uniform(size=mask.shape) > rate) * mask
        modality_m = missing_mask.reshape(N, T, 1) * modality
        return modality_m, mask, missing_mask

    def _yield_from_one_shard(self, shard_path, rng):
        """샤드 하나만 로드 → 샘플 yield → 즉시 언로드"""
        d = self._safe_load_shard(shard_path)
        if self.mode not in d:
            del d; gc.collect()
            return
        pack = d[self.mode]
        use_bert = bool(self.args.get('use_bert', False))
        txt_key = 'text_bert' if use_bert else 'text'

        text   = _astype_f32(pack[txt_key])
        audio  = _astype_f32(pack['audio'])
        vision = _astype_f32(pack['vision'])
        raw_text = pack.get('raw_text', [''] * len(text))
        ids      = pack.get('id', list(range(len(text))))

        labels_M = np.array(pack.get('regression_labels', pack.get('M', []))).astype(np.float32)
        labels = {'M': labels_M}
        if 'sims' in self.args.get('dataset_name', ''):
            for m in 'TAV':
                key = f'regression_labels_{m}'
                if key in pack:
                    labels[m] = np.array(pack[key]).astype(np.float32)

        N = len(labels_M)
        idxs = list(range(N))
        if self._shuffle_within:
            rng.shuffle(idxs)

        need_aligned = bool(self.args.get('need_data_aligned', True))
        if not need_aligned:
            audio_lengths = pack.get('audio_lengths', [audio.shape[1] * 1] * N)
            vision_lengths = pack.get('vision_lengths', [vision.shape[1] * 1] * N)
        else:
            audio_lengths = None
            vision_lengths = None

        audio[audio == -np.inf] = 0

        if bool(self.args.get('need_normalized', False)):
            vision = self._normalize_time_major_mean(vision)
            audio  = self._normalize_time_major_mean(audio)

        if bool(self.args.get('data_missing', False)):
            text_m, text_mask, text_missing_mask, t_lengths = self._make_text_missing(text)
            if need_aligned:
                audio_lengths = t_lengths
                vision_lengths = t_lengths
            audio_m, audio_mask, audio_missing_mask = self._make_mod_missing(audio, audio_lengths)
            vision_m, vision_mask, vision_missing_mask = self._make_mod_missing(vision, vision_lengths)
        else:
            text_m = text_mask = text_missing_mask = None
            audio_m = audio_mask = audio_missing_mask = None
            vision_m = vision_mask = vision_missing_mask = None

        for i in idxs:
            sample = {
                'raw_text': raw_text[i],
                'text': torch.tensor(text[i]),
                'audio': torch.tensor(audio[i]),
                'vision': torch.tensor(vision[i]),
                'index': i,
                'id': ids[i],
                'labels': {k: torch.tensor(v[i].reshape(-1)) for k, v in labels.items()},
            }
            if not need_aligned:
                sample['audio_lengths'] = int(audio_lengths[i])
                sample['vision_lengths'] = int(vision_lengths[i])
            if text_m is not None:
                sample['text_m'] = torch.tensor(text_m[i])
                sample['text_missing_mask'] = torch.tensor(text_missing_mask[i])
                sample['audio_m'] = torch.tensor(audio_m[i])
                sample['audio_lengths'] = int(audio_lengths[i])
                sample['audio_mask'] = audio_mask[i]
                sample['audio_missing_mask'] = torch.tensor(audio_missing_mask[i])
                sample['vision_m'] = torch.tensor(vision_m[i])
                sample['vision_lengths'] = int(vision_lengths[i])
                sample['vision_mask'] = vision_mask[i]
                sample['vision_missing_mask'] = torch.tensor(vision_missing_mask[i])

            yield sample

        del d, pack, text, audio, vision, raw_text, ids, labels_M, labels
        gc.collect()

    def __iter__(self):
        if not self._shard_mode:
            raise RuntimeError("Iterable iteration is only for manifest(shard) mode.")
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self._seed)
        files = self._iter_shard_order(rng)

        if worker is not None:
            files = files[worker.id::worker.num_workers]

        for p in files:
            for sample in self._yield_from_one_shard(p, rng):
                yield sample

    def get_seq_len(self):
        if self._shard_mode:
            return self._seq_lens
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        if self._shard_mode:
            return self._feature_dims
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __len__(self):
        if self._shard_mode:
            return self._global_len
        return len(self.labels['M'])

    # 기존(단일 PKL) 모드
    def __init_mosi(self):
        if self.args['custom_feature']:
            with open(self.args['custom_feature'], 'rb') as f:
                data = pickle.load(f)
        else:
            with open(self.args['featurePath'], 'rb') as f:
                data = pickle.load(f)
        
        if self.args.get('use_bert', None):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
            self.args['feature_dims'][0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            self.args['feature_dims'][0] = self.text.shape[2]
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # Overide with custom modality features
        if self.args['feature_T']:
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', None):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A']:
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V']:
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        if 'sims' in self.args['dataset_name']:
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            if self.args['feature_A']:
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V']:
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.args.get('data_missing'):
            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                        self.args.missing_rate[0], self.args.missing_seed[0], mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

            if self.args['need_data_aligned']:
                self.audio_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
                self.vision_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        self.args.missing_rate[1], self.args.missing_seed[1], mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                        self.args.missing_rate[2], self.args.missing_seed[2], mode='vision')

        if self.args.get('need_normalized'):
            self.__normalize()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    
    def __init_simsv2(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
        assert missing_mask.shape == input_mask.shape
        if mode == 'text':
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask)
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        return modality_m, input_len, input_mask, missing_mask

    def __truncate(self):
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for _ in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if (instance[index] == padding).all():
                        if (index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __getitem__(self, index):
        if self._shard_mode:
            raise RuntimeError("manifest(shard) mode uses iterable iteration; __getitem__ is disabled.")
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        if self.args.get('data_missing'):
            sample['text_m'] = torch.Tensor(self.text_m[index])
            sample['text_missing_mask'] = torch.Tensor(self.text_missing_mask[index])
            sample['audio_m'] = torch.Tensor(self.audio_m[index])
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['audio_mask'] = self.audio_mask[index]
            sample['audio_missing_mask'] = torch.Tensor(self.audio_missing_mask[index])
            sample['vision_m'] = torch.Tensor(self.vision_m[index])
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['vision_mask'] = self.vision_mask[index]
            sample['vision_missing_mask'] = torch.Tensor(self.vision_missing_mask[index])
        return sample


def MMDataLoader(args, num_workers):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test':  MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len()

    # manifest(.txt)면 IterableDataset 래퍼 사용
    fp = args.get('featurePath', '')
    is_shard = isinstance(fp, str) and fp.lower().endswith('.txt') and os.path.isfile(fp)
    if is_shard:
        for k in list(datasets.keys()):
            datasets[k] = _ShardIterableDataset(datasets[k])

    dataLoader = {
        ds: DataLoader(
            datasets[ds],
            batch_size=args['batch_size'],
            num_workers=num_workers,
            shuffle=(False if is_shard else True),  
            pin_memory=bool(args.get('pin_memory', False)),
            persistent_workers=bool(args.get('persistent_workers', False)) and num_workers > 0
        )
        for ds in datasets.keys()
    }
    return dataLoader
