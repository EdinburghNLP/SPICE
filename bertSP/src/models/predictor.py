#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import numpy as np
import json
import torch

from tensorboardX import SummaryWriter

from others.utils import  tile, ids_to_tokens_dynamic
from translate.beam import GNMTGlobalScorer
from others.constants import *

def build_predictor(args, vocab, symbols, tokenizer, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, vocab, symbols, tokenizer, global_scorer=scorer, logger=logger)
    return translator

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 tokenizer,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab # target fixed vocabulary
        self.tokenizer = tokenizer # input vocabulary
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        self.unk_token = symbols['UNK']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.str_logical_form, batch.input

        translations = []
        for b in range(batch_size):
            pred_sents = self.decode_lf_tokens(preds[b][0], batch.entity_map[b])
            gold_sent = ' '.join(tgt_str[b])
            src_chunks = []
            for chunk in range(len(src[b])):
                src_chunks.append(
                    ' '.join([self.tokenizer.ids_to_tokens[int(t)] for t in src[b][chunk]][:509]).replace(' ##',''))
            raw_src = '\n'.join(src_chunks)

            qtype = batch.question_type[b]
            if hasattr(batch, 'description'):
                translation = (pred_sents, gold_sent, raw_src, \
                               qtype, batch.description[b], batch.question[b], \
                               batch.answer[b], batch.results[b], batch.turnID[b])
            else:
                translation = (pred_sents, gold_sent, raw_src, qtype)

            translations.append(translation)

        return translations

    def decode_lf_tokens(self, predictions, entmap):
        return ' '.join([ids_to_tokens_dynamic(w, self.vocab, entmap) for w in predictions])

    def translate_single_batch(self, batch):
        """Called from training..."""
        batch_data = self.translate_batch(batch)
        translations = self.from_batch(batch_data)

        for trans in translations:
            pred, gold, src, qtype = trans
            pred_str = pred.replace('[PAD]', '').replace('[BOS]', '').replace('[EOS]', '').strip()
            gold_str = gold.strip()
            print('\nPred: ', pred_str)
            print('Gold: ', gold_str, '\n')


    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        qtype_path = self.args.result_path + '.%d.qtype' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.qtype_out_file = codecs.open(qtype_path, 'w', 'utf-8')
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        out_inferences = {}

        tkn_acc = []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                #if ct>10: break
                ## TODO: added the following control here until we debug the data creating
                src = batch.input
                segs = batch.segment
                tgt_str = batch.str_logical_form
                bsz, nbChunks, seqLen = src.size()
                bsz_s, nbChunks_s, seqLen_s = segs.size()
                if not (bsz==bsz_s and nbChunks==nbChunks_s and seqLen==seqLen_s):
                    print(f'src/segs diff size, {tgt_str} \n{src.shape} \n{segs.shape}')
                    continue

                if(self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src, qtype, desc, ques, ans, res, trunid = trans
                    pred_str = pred.replace('[PAD]', '').replace('[BOS]', '').replace('[EOS]', '').strip()
                    gold_str = gold.strip()
                    qtype_str = INV_QTYPE_DICT[qtype.item()]

                    # replace back the numerical argument for conditions
                    if 'num' in pred_str or 'num' in gold_str:
                        val = get_value(ques)
                        if val:
                            pred_str = pred_str.replace('num', val)
                            gold_str = gold_str.replace('value', val)

                    gold_tok = gold_str.split()
                    pred_tok = pred_str.split()[:len(gold_tok)]
                    tkn_acc.append(sum([1 for x, y in zip(gold_tok, pred_tok) if x==y])/len(gold_tok))

                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    self.qtype_out_file.write(qtype_str + '\n')
                    ct += 1

                    inference_actions = {
                                    QUESTION_TYPE: qtype_str,
                                    DESCRIPTION: desc,
                                    QUESTION: ques,
                                    ANSWER: ans,
                                    ACTIONS: pred_str,
                                    RESULTS: res,
                                    GOLD_ACTIONS: gold_str,
                                    TURN_ID: trunid
                                }

                    if qtype_str not in out_inferences.keys():
                        out_inferences[qtype_str] = []
                    out_inferences[qtype_str].append(inference_actions)


                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                self.qtype_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()
        self.qtype_out_file.close()

        out_qtype_jsons = {} # a dictionary, entries for each qtype, value is an open json file to write to
        for qtype in out_inferences.keys():
            if qtype not in out_qtype_jsons.keys():
                fileName = f'{self.args.result_path}.{step}.{self.args.test_split}_{qtype}.json'
                try:
                    out_qtype_jsons[qtype] = open(fileName, 'w', encoding='utf-8')
                except IOError:
                    print(f'Error trying to open json file for inference results. \n{fileName}')
                    return 0
            out_qtype_jsons[qtype].write(json.dumps(out_inferences[qtype], indent=4))
            out_qtype_jsons[qtype].close()


        atkn_acc = np.array(tkn_acc)
        print("Averaged Token Accuracy: \nMean {}\nMin {}\nMax {}\nStd {}".format(atkn_acc.mean(),
                                                                                  atkn_acc.min(),
                                                                                  atkn_acc.max(),
                                                                                  atkn_acc.std()))

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size

        src = batch.input
        tgt = batch.logical_form
        tgt_str = batch.str_logical_form
        segs = batch.segment
        starts = batch.start
        ends = batch.end
        dynbin = batch.dynbin
        entmap = batch.entity_map
        mask_src = batch.mask_input

        bsz, nbChunks, seqLen = src.size()

        src_features = self.model.bert(src.reshape(bsz * nbChunks, seqLen),
                                       segs.reshape(bsz * nbChunks, seqLen),
                                       mask_src.reshape(bsz * nbChunks, seqLen))
                                       # mask_src)
        ## steps before feeding to decoder
        src_features = src_features.reshape(bsz, nbChunks, seqLen, -1)
        # pooling eliminates the nbChunks dimension, will pull over the utterance dim
        # TODO: see other ways, max, etc.
        pooled_src_features = torch.mean(src_features, 1)  # [bsz * seqLen * dim]
        context_mask = self.model._get_utterance_mask_from_bert_input(starts, bsz, seqLen).to(src_features.device)
        # prepare dynamic output embeddings
        dyn_target_embedds, de_mask = self.model._compute_target_embedding(starts, ends, src_features, dynbin, entmap)

        #dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        dec_states = self.model.decoder.init_decoder_state(src[:,0,:], pooled_src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        pooled_src_features = tile(pooled_src_features, beam_size, dim=0)
        context_mask = tile(context_mask, beam_size, dim=0)
        dyn_target_embedds = tile(dyn_target_embedds, beam_size, dim=0)
        de_mask = tile(de_mask, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)

            dec_out, dec_states = self.model.decoder(decoder_input, pooled_src_features, dec_states,
                                                     self.args.input_syntax_voc,
                                                     dynEmbed=dyn_target_embedds,memory_masks=context_mask,
                                                     step=step)

            # Generator forward.
            log_probs = self.generator.forward({0: dec_out.transpose(0,1).squeeze(0), 1: (dyn_target_embedds, de_mask)})
            vocab_size = log_probs.size(-1)

            log_probs = log_probs.masked_fill(de_mask.squeeze(2).bool(), -1e20)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            if self.args.ban_unk_token:
                log_probs[:, self.unk_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)


            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids // vocab_size #topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)

            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    #finished_hyp = is_finished[i].nonzero().view(-1)
                    finished_hyp = torch.nonzero(is_finished[i], as_tuple=False).view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                #non_finished = end_condition.eq(0).nonzero().view(-1)
                non_finished = torch.nonzero(end_condition.eq(0), as_tuple=False).view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            #src_features = src_features.index_select(0, select_indices)
            pooled_src_features = pooled_src_features.index_select(0, select_indices)
            dyn_target_embedds = dyn_target_embedds.index_select(0, select_indices)
            de_mask = de_mask.index_select(0, select_indices)
            context_mask = context_mask.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))


        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
