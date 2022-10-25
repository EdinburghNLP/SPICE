import bisect
import gc
import glob
import random
import re
import torch

from others.logging import logger
from others.constants import *
from prepro.data_builder import SPICEDataset

class Batch(object):

    def _pad(self, data, pad_id, width=-1):

        if (width == -1):
            width = max(len(d) for d in data)

        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def _pad2D(self, data, pad_id, width1=-1, width2=-1):

        if (width1 == -1):
            width1 = max(len(d) for d in data)
        if (width2 == -1):
            width2 = max(len(d) for chunk in data for d in chunk)

        rtn_data = []
        for i in range(len(data)):
            chunks = [d + [pad_id] * (width2 - len(d)) for d in data[i]]
            chunk_size = len(chunks)
            chunks.extend([[pad_id] * width2 for k in range(width1- chunk_size )])
            rtn_data.append(chunks)

        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_input = [x[0] for x in data]
            pre_logical_form = [x[1] for x in data]
            str_logical_form = [x[2] for x in data]
            pre_segment = [x[3] for x in data]
            start = [x[4] for x in data]
            end = [x[5] for x in data]
            dynbin = [x[6] for x in data]
            entmap = [x[7] for x in data]

            # tensors with padding
            input = torch.tensor(self._pad2D(pre_input, 0))
            logical_form = torch.tensor(self._pad(pre_logical_form, 0))
            mask_input = ~(input == 0)
            mask_logical_form = ~(logical_form == 0)
            segment = torch.tensor(self._pad2D(pre_segment, 0))

            setattr(self, INPUT, input.to(device))
            setattr(self, LOGICAL_FORM, logical_form.to(device))
            setattr(self, SEGMENT, segment.to(device))
            setattr(self, 'mask_input', mask_input.to(device))
            setattr(self, 'mask_logical_form', mask_logical_form.to(device))

            # auxiliary instance data
            setattr(self, STR_LOGICAL_FORM, str_logical_form) ##could be moved to is_test
            setattr(self, START, start)
            setattr(self, END, end)
            setattr(self, DYNBIN, dynbin)
            setattr(self, ENTITY_MAP, entmap)

            qtype = [x[8] for x in data]
            setattr(self, QUESTION_TYPE, torch.tensor(qtype).to(device))
            desc = [x[9] for x in data]
            setattr(self, DESCRIPTION, desc)

            if is_test and len(data[0])>9:
                ques = [x[10] for x in data]
                setattr(self, QUESTION, ques)
                res = [x[11] for x in data]
                setattr(self, RESULTS, res)
                ans = [x[12] for x in data]
                setattr(self, ANSWER, ans)
                ans = [x[13] for x in data]
                setattr(self, TURN_ID, ans)

    def __len__(self):
        return self.batch_size



def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))

    if corpus_type in ["valid", "test"] and args.dosubset != '':
        patt = re.compile(r"" + args.bert_data_path + "." + corpus_type + "." + args.dosubset + ".bert.pt")
        pts = [fname for fname in pts if patt.match(fname)]
        logger.info(f'*** \t Loading subset: {args.dosubset}')

    logger.info(pts)

    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        self.batch_size_fn = abs_batch_size_fn
        self.spice = SPICEDataset(args)
        str_syn_voc, vocIds, start, end = self.spice.vocInputchunk()
        self.str_syn_voc = str_syn_voc
        self.vocIds = vocIds
        self.start = start
        self.end = end


    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):

        ## get additonal lines, temporarily done here to try, move if works to data_prepro

        input = ex[INPUT]
        logical_form = ex[LOGICAL_FORM]
        str_logical_form = ex[STR_LOGICAL_FORM]
        segment = ex[SEGMENT]
        starts = ex[START]
        ends = ex[END]
        dynbin = ex[DYNBIN]
        entmap = ex[ENTITY_MAP]
        if type(ex[QUESTION_TYPE]) == list and len(ex[QUESTION_TYPE]) == 2: # this is ['Clarification','Simple Question (Coreferenced)'] type
            qtype = ex[QUESTION_TYPE][0]
        else: qtype = ex[QUESTION_TYPE].strip()
        question_type = QTYPE_DICT[qtype]
        description = ex[DESCRIPTION]
        if is_test:
            results = ex[RESULTS]
            answer = ex[ANSWER]
            question = ex[QUESTION]
            turnID = ex[TURN_ID]

        # option 'args.input_syntax_voc' no tested enough, not used in the end
        if self.args.input_syntax_voc:
            #DONE TODO: ammend this here, ideally these repeated sequence should be added when data processing
            context_part = input[0][:starts[0][0]]
            new_input_line = context_part + self.str_syn_voc
            new_input_line = new_input_line[:511] + [context_part[-1]]
            input.insert(0,new_input_line)
            add_seg = [0]*len(context_part) + [1]*(len(self.str_syn_voc)+1)
            segment.insert(0, add_seg)
            starts.insert(0, self.start)
            ends.insert(0, self.end)
            dynbin.insert(0, self.vocIds)

            m = max([len(x) for x in segment])
            o = max([len(x) for x in input])
            assert  o == m, f'{segment} {input}'

            if len(input) > 10:
            # do this to reduce last line, in limit size cases TODO also handled with above in prepro. or solve with more memory
                input = input[:-1]
                segment = segment[:-1]
                starts = starts[:-1]
                ends = ends[:-1]
                dynbin = dynbin[:-1]

        ## could do some extra processing/formatting (e.g., masks) if needed here

        if(is_test): #TODO: will do anything special if test in the future???? otherwise remove
            return input, logical_form, str_logical_form, segment, starts, ends, dynbin, entmap, \
                   question_type, description, question, results, answer, turnID
        else:
            return input, logical_form, str_logical_form, segment, starts, ends, dynbin, entmap, question_type, description

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['input'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)

        gc.collect()
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
