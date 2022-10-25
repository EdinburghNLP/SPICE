import copy

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

from models.decoder import TransformerDecoder
from models.optimizers import Optimizer
from others.constants import *

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


class TransformerDecoderDynEmbed(TransformerDecoder):

    def forward(self, tgt, memory_bank, state, input_syntax_voc, memory_lengths=None,
                    step=None, cache=None, memory_masks=None, dynEmbed=None):

        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings({0: tgt, 1: dynEmbed, 2: input_syntax_voc})

        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input \
                = self.transformer_layers[i](
                output, src_memory_bank,
                src_pad_mask, tgt_pad_mask,
                previous_input=prev_layer_input,
                layer_cache=state.cache["layer_{}".format(i)]
                if state.cache is not None else None,
                step=step)
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return output, state

def get_classifier(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

def get_generator(vocab_size, dec_hidden_size, device, input_syntax_voc):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        DynamicLinear(dec_hidden_size, vocab_size, input_syntax_voc),
        gen_func
    )
    generator.to(device)

    return generator

class DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features, input_syntax_voc, bias=False, device=None):
        super(DynamicLinear, self).__init__()

        factory_kwargs = {'device': device}
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        self.input_syntax_voc = input_syntax_voc
        if input_syntax_voc:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))

    def forward(self, input):
        input, dynEmb = input[0], input[1]
        dynWeights, deMask = dynEmb

        scores = []
        for b in range(input.size(0)):
            combinedBias = None

            if self.input_syntax_voc:
                combinedEmbeddMatrix = dynWeights[b, :, :]
            else:
                combinedEmbeddMatrix = torch.cat((self.weight, dynWeights[b, :, :]), dim=0)

            sc = F.linear(input[b,:], combinedEmbeddMatrix, combinedBias)

            scores.append(sc)

        scores = torch.stack(scores)

        return scores

class DynamicEmbedding(nn.Embedding):
    """Just an embedding class that re-writes the forward method. On forward, it will append a dynamic set of weights
    to the embedding class weights."""

    def forward(self, input):
        """Dynamic embed each batch item with dynamic set of vectors"""
        # TODO: if this implementation is too slow, do implement lower level

        input, dynamic, input_syntax_voc = input[0], input[1], input[2]

        embeddings = []
        for b in range(dynamic.size(0)):
            if input_syntax_voc:
                combinedEmbeddMatrix = dynamic[b, :, :]
            else:
                combinedEmbeddMatrix = torch.cat((self.weight, dynamic[b, :, :]), dim=0)
            tmpEmbeds = F.embedding(
                input[b,:], combinedEmbeddMatrix, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
            embeddings.append(tmpEmbeds)

        embeddings = torch.stack(embeddings)

        return embeddings

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        """attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional) —
        Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are masked."""

        if(self.finetune):
            outputs = self.model(input_ids=x, token_type_ids = segs, attention_mask=mask)
            top_vec = outputs.last_hidden_state
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=x, token_type_ids = segs, attention_mask=mask)
                top_vec = outputs.last_hidden_state

        return top_vec

class BaseSparqlSemParser(nn.Module):
    def __init__(self, args, device, checkpoint=None, global_target_voc_size=None):
        super(BaseSparqlSemParser, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if self.args.predict_qtype:
            self.qtype_classifier = get_classifier(len(QTYPE_DICT.keys()), self.args.dec_hidden_size, device)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        self.global_target_voc_size = global_target_voc_size or self.bert.model.config.vocab_size

        tgt_embeddings = DynamicEmbedding(self.global_target_voc_size, self.bert.model.config.hidden_size, padding_idx=0)

        self.decoder = TransformerDecoderDynEmbed(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.global_target_voc_size, self.args.dec_hidden_size, device, self.args.input_syntax_voc)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

        self.to(device)

    def forward(self, src, tgt, segs, starts, ends, dynbin, entmap, mask_src, mask_tgt):
        """

        :param src: [batch * chunks * chunk len]
        :param tgt:
        :param segs: [batch * chunks * chunk len]
        :param starts:
        :param ends:
        :param dynbin:
        :param entmap:
        :param mask_src:
        :param mask_tgt:
        :return:
        """

        bsz, nbChunks, seqLen = src.size()

        # will create attention mask
        top_vec = self.bert(src.reshape(bsz * nbChunks, seqLen),
                            segs.reshape(bsz * nbChunks, seqLen),
                            mask_src.reshape(bsz * nbChunks, seqLen))

        top_vec = top_vec.reshape(bsz, nbChunks, seqLen, -1)

        # pooling eliminates the nbChunks dimension, will pull over the utterance dim
        # TODO: see other ways, max, etc.
        pooled_top_vec = torch.mean(top_vec, 1) # [bsz * seqLen * dim]
        context_mask = self._get_utterance_mask_from_bert_input(starts, bsz, seqLen).to(top_vec.device)

        # prepare dynamic output embeddings
        dyn_target_embedds, de_mask = self._compute_target_embedding(starts, ends, top_vec, dynbin, entmap)

        dec_state = self.decoder.init_decoder_state(src[:,0,:], pooled_top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], pooled_top_vec, dec_state, self.args.input_syntax_voc,
                                              memory_masks=context_mask,
                                              dynEmbed=dyn_target_embedds)


        return (decoder_outputs, dyn_target_embedds, de_mask), None


    # get the mask for utterance in the concatenated input to BERT
    def _get_utterance_mask_from_bert_input(self, schema_start, batch_size, max_len):
        assert batch_size == len(schema_start)
        mask = torch.zeros(batch_size, 1, max_len)
        for i, l in enumerate(schema_start):
            # some turns did not get any entity (specially counts and ellided follow-ups) so no KG neighbourhood
            # here we have only the sentences
            if len(l) > 0 and len(l[0])>0 and l[0][0] == -1:
                continue # TODO: leave all ones for this element? is ok if this mask is then combine w/ other src mask...

            s = l[0][0] if (len(l)>0 and l[0]) else 0
            # Care here about the meaning usage of True/False in mask!!!
            # Positions in the mask should be 1/True in the cases where we DO want mask. See definition as expects attn layer:
            # "mask: binary mask indicating which keys have non-zero attention `[batch, query_len, key_len]` "
            mask[i, 0, s:] = 1
        try:
            return mask.bool()
        except RuntimeError:
            mask = mask
            print(mask.shape)


    def _compute_target_embedding(self, schema_start, schema_end, bert_embeddings, dynbin, entmap):
        """
        entmap: a dictionary with the KG dynamic vocabulary, KGIDs->labels and labels->KGIDs, positions are ids for binarisation
                we expect one dictionary per batch entry.
        """

        batch_size = len(schema_start)
        maxDynVocLen = max([len(entmap[i].keys()) for i in range(len(entmap))])
        if self.args.input_syntax_voc:
            voc_dim = self.global_target_voc_size + maxDynVocLen
        else:
            voc_dim = maxDynVocLen
        target_embedding = bert_embeddings.new_zeros([batch_size, voc_dim, 768])
        target_embedding_cnt = bert_embeddings.new_zeros([batch_size, voc_dim, 1])

        for i in range(len(schema_start)):
            for k in range(len(schema_start[i])):
                assert len(schema_start[i][k]) == len(schema_end[i][k]), f'batch:{i} \n {schema_start} \n {schema_end}'
                for j in range(len(schema_start[i][k])):
                    if dynbin[i][k][j] == -1:
                        continue
                    start = schema_start[i][k][j]
                    end = schema_end[i][k][j]
                    avg_embedding = bert_embeddings[i][k][start: end + 1]
                    avg_embedding = torch.sum(avg_embedding, dim=0)
                    assert (end - start + 1) > 0
                    avg_embedding = avg_embedding / (end - start + 1)
                    if self.args.input_syntax_voc:
                        target_embedding[i][dynbin[i][k][j]  ] += avg_embedding
                        target_embedding_cnt[i][dynbin[i][k][j]  ] += 1
                    else:
                        target_embedding[i][dynbin[i][k][j] - self.global_target_voc_size ] += avg_embedding
                        target_embedding_cnt[i][dynbin[i][k][j] - self.global_target_voc_size ] += 1

        target_embedding_cnt[target_embedding_cnt.eq(0)] = 1 #those who were left without filling in, they where in the vocab but disappear from the input due to chunk to fit 512 size
        target_embedding = target_embedding / target_embedding_cnt.repeat(1, 1, 768)

        # prepare target embedding mask when using bsz>1 and some instances will have less elems in the vocabulary
        # recicle this tensor, ones for elements we need to exclude
        target_embedding_cnt.fill_(0)
        for b in range(batch_size):
            target_embedding_cnt[b, len(entmap[b].keys()):, :] = 1

        if not self.args.input_syntax_voc:
            outv_mask = torch.cat((bert_embeddings.new_zeros([batch_size, self.global_target_voc_size, 1]),
                                   target_embedding_cnt), dim=1)
        else:
            outv_mask = target_embedding_cnt

        # (batch_size, vocab_size, dim)
        return (target_embedding, outv_mask)
