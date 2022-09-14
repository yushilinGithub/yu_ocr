from ast import arg
from email.policy import default
import torch.nn as nn
from fairseq.models import FairseqEncoder, register_model, FairseqEncoderDecoderModel, register_model_architecture
from fairseq.models.transformer import TransformerDecoder, Embedding, TransformerModel
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq import utils
from fairseq.models.transformer import base_architecture as base_transformer
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .unilm_models import UniLMDecoder
from .cswin import CSWin
# from timm.models.vision_transformer import HybridEmbed, PatchEmbed, Block
from timm.models.layers import trunc_normal_
from timm.models import create_model
import torch
from torch.hub import load_state_dict_from_url
import os
from functools import partial
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024
@register_model('OKayOCR')
class OKayOCR(FairseqEncoderDecoderModel):
    
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        # parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
        #                     help='decoder embedding dimension')
        parser.add_argument(
            '--embed-dim', type=int, metavar='N',
            help='the patch size of h and w (h=w) of the ViT'
        )
        parser.add_argument(
            '--depths', type=list, metavar='N',
            help='the hidden size of the ViT'
        )
        parser.add_argument(
            '--num-heads', type=list, metavar='N',
            help='the layer num of the ViT'
        )
        parser.add_argument(
            "--encoder-fun",default="SwinViTEncoder",type=str,metavar="N",
            help="SwinEncoder | SwinViTEncoder | CswinEncoder"
            ) 
        parser.add_argument(
            '--reset-dictionary', action='store_true',
            help='if reset dictionary and related parameters'
        )
        parser.add_argument(
            '--adapt-dictionary', action='store_true',
            help='if adapt dictionary and related parameters'
        )
    @classmethod
    def build_model(cls, args, task):
        encoder = eval(args.encoder_fun)(
            args = args,
            dictionary = task.source_dictionary
        )

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        if getattr(args, "decoder_pretrained_url", None) and os.path.basename(args.decoder_pretrained_url).startswith("chinese_roberta"): 
           
            logger.info('Using the learned pos embedding version loading roberta.')
            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )
            """
            bert.embeddings.position_ids -> 
            bert.embeddings.position_embeddings.weight -> embed_positions.weight
            bert.embeddings.word_embeddings.weight -> embed_tokens.weight
            bert.embeddings.token_type_embeddings.weight -> 
            
            bert.embeddings.LayerNorm.weight -> layernorm_embedding.weight
            bert.embeddings.LayerNorm.bias -> layernorm_embedding.bias
            
            bert.encoder.layer.0.attention.self.query.weight  -> layers.0.self_attn.q_proj.weight
            bert.encoder.layer.0.attention.self.query.bias -> layers.0.self_attn.q_proj.bias
            bert.encoder.layer.0.attention.self.key.weight  -> layers.0.self_attn.k_proj.weight
            bert.encoder.layer.0.attention.self.key.bias -> layers.0.self_attn.k_proj.bias
            bert.encoder.layer.0.attention.self.value.weight -> layers.0.self_attn.v_proj.weight
            bert.encoder.layer.0.attention.self.value.bias -> layers.0.self_attn.v_proj.bias
            bert.encoder.layer.0.attention.output.dense.weight -> layers.0.self_attn.out_proj.weight
            bert.encoder.layer.0.attention.output.dense.bias -> layers.0.self_attn.out_proj.bias
            bert.encoder.layer.0.attention.output.LayerNorm.weight -> layers.0.self_attn_layer_norm.weight
            bert.encoder.layer.0.attention.output.LayerNorm.bias -> layers.0.self_attn_layer_norm.bias
            bert.encoder.layer.0.intermediate.dense.weight -> layers.0.fc1.weight
            bert.encoder.layer.0.intermediate.dense.bias -> layers.0.fc1.bias
            bert.encoder.layer.0.output.dense.weight -> layers.0.fc2.weight
            bert.encoder.layer.0.output.dense.bias -> layers.0.fc2.bias
            bert.encoder.layer.0.output.LayerNorm.weight -> layers.0.final_layer_norm.weight
            bert.encoder.layer.0.output.LayerNorm.bias -> layers.0.final_layer_norm.bias
            bert.encoder.layer.1.attention.self.query.weight -> layers.1.self_attn.q_proj.weight
            bert.encoder.layer.1.attention.self.query.bias -> layers.1.self_attn.q_proj.bias
            bert.encoder.layer.1.attention.self.key.weight -> layers.1.self_attn.k_proj.weight
            bert.encoder.layer.1.attention.self.key.bias -> layers.1.self_attn.k_proj.bias
            bert.encoder.layer.1.attention.self.value.weight -> layers.1.self_attn.v_proj.weight
            bert.encoder.layer.1.attention.self.value.bias -> layers.1.self_attn.v_proj.bias
            bert.encoder.layer.1.attention.output.dense.weight -> layers.1.self_attn.out_proj.weight
            bert.encoder.layer.1.attention.output.dense.bias -> layers.1.self_attn.out_proj.bias
            bert.encoder.layer.1.attention.output.LayerNorm.weight -> layers.1.self_attn_layer_norm.weight
            bert.encoder.layer.1.attention.output.LayerNorm.bias -> layers.1.self_attn_layer_norm.bias
            bert.encoder.layer.1.intermediate.dense.weight -> layers.1.fc1.weight
            bert.encoder.layer.1.intermediate.dense.bias -> layers.1.fc1.bias
            bert.encoder.layer.1.output.dense.weight -> layers.1.fc2.weight
            bert.encoder.layer.1.output.dense.bias -> layers.1.fc2.bias
            bert.encoder.layer.1.output.LayerNorm.weight -> layers.1.final_layer_norm.weight
            bert.encoder.layer.1.output.LayerNorm.bias -> layers.1.final_layer_norm.bias
            
            layers.0.encoder_attn.k_proj.weight
            layers.0.encoder_attn.k_proj.bias
            layers.0.encoder_attn.v_proj.weight
            layers.0.encoder_attn.v_proj.bias
            layers.0.encoder_attn.q_proj.weight
            layers.0.encoder_attn.q_proj.bias
            layers.0.encoder_attn.out_proj.weight
            layers.0.encoder_attn.out_proj.bias
            layers.0.encoder_attn_layer_norm.weight
            layers.0.encoder_attn_layer_norm.bias
            
            layers.1.encoder_attn.k_proj.weight
            layers.1.encoder_attn.k_proj.bias
            layers.1.encoder_attn.v_proj.weight
            layers.1.encoder_attn.v_proj.bias
            layers.1.encoder_attn.q_proj.weight
            layers.1.encoder_attn.q_proj.bias
            layers.1.encoder_attn.out_proj.weight
            layers.1.encoder_attn.out_proj.bias
            layers.1.encoder_attn_layer_norm.weight
            layers.1.encoder_attn_layer_norm.bias
            
            ...
            
            output_projection.weight
            
            cls.predictions.bias 
            cls.predictions.transform.dense.weight
            cls.predictions.transform.dense.bias
            cls.predictions.transform.LayerNorm.weight
            cls.predictions.transform.LayerNorm.bias
            cls.predictions.decoder.weight
            cls.predictions.decoder.bias
            """
            
            roberta_state_dict = torch.load(os.path.join(args.decoder_pretrained_url,"pytorch_model.bin"))


            decoder = TransformerDecoder(
                args,
                task.target_dictionary,
                decoder_embed_tokens,
                no_encoder_attn=False,
            )
            new_roberta_state_dict = OrderedDict()
            for key,value in roberta_state_dict.items():
                key = key.replace("bert.embeddings.word_embeddings","embed_tokens")
                key = key.replace("bert.embeddings.position_embeddings","embed_positions") 
                key = key.replace("bert.embeddings.LayerNorm","layernorm_embedding")
                key = key.replace("bert.encoder.layer","layers")
                key = key.replace("attention.self","self_attn")
                key = key.replace("query","q_proj")
                key = key.replace("key","k_proj")
                key = key.replace("value","v_proj")
                key = key.replace("attention.output.dense","self_attn.out_proj")
                key = key.replace("attention.output.LayerNorm","self_attn_layer_norm")
                key = key.replace("output.LayerNorm","final_layer_norm")
                key = key.replace("intermediate.dense","fc1")
                key = key.replace("output.dense","fc2")
                if key == "embed_positions.weight":
                    new_roberta_state_dict[key] = value[:args.max_target_positions+1]
                else:
                    new_roberta_state_dict[key] = value
            missing_keys, unexpected_keys = decoder.load_state_dict(
                new_roberta_state_dict, strict=False
            )

            
        if getattr(args, "decoder_pretrained", None) == 'minilm':
            logger.info('Decoder is pretrained using the minilm.')
            
            prefix_of_parameter = 'bert'

            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )

            decoder = UniLMDecoder(
                args,
                task.target_dictionary,
                decoder_embed_tokens,
                no_encoder_attn=False,
            )            
            # load pretrained model
            if hasattr(args, 'decoder_pretrained_url') and args.decoder_pretrained_url != None and args.decoder_pretrained_url != '':                
                unilm_url = args.decoder_pretrained_url
                logger.info('The unilm model url: {}.'.format(unilm_url[:unilm_url.find('?')]))
                if unilm_url.startswith("http"):
                    unilm_state_dict = torch.hub.load_state_dict_from_url(unilm_url)            
                else:
                    unilm_state_dict = torch.load(unilm_url)
                unilm_layers = OrderedDict([(k, unilm_state_dict[k]) for k in unilm_state_dict.keys() if k.startswith(prefix_of_parameter + '.encoder.layer.')])
                unilm_layers_num = []
                for k in unilm_layers.keys():
                    t = k.replace(prefix_of_parameter + '.encoder.layer.', '')
                    t = t[:t.find('.')]
                    unilm_layers_num.append(int(t))
                unilm_layers_num = max(unilm_layers_num) + 1

                offset = unilm_layers_num - len(decoder.layers)
                assert offset == 0

                decoder_dict = decoder.state_dict()
                # embedding
                new_pos_weight = torch.zeros_like(decoder_dict['embed_positions.weight'])
                # position padding will right offset padding idx + 1
                new_pos_weight[task.target_dictionary.pad() + 1:, :] = unilm_state_dict[prefix_of_parameter + '.embeddings.position_embeddings.weight']
                new_decoder_dict = {
                    'embed_tokens.weight': unilm_state_dict[prefix_of_parameter + '.embeddings.word_embeddings.weight'],
                    'embed_positions.weight': new_pos_weight,
                    'layernorm_embedding.weight': unilm_state_dict[prefix_of_parameter + '.embeddings.LayerNorm.weight'],
                    'layernorm_embedding.bias': unilm_state_dict[prefix_of_parameter + '.embeddings.LayerNorm.bias']
                }            

                # layers
                key_map = {
                    'self_attn.k_proj': 'attention.self.key',
                    'self_attn.v_proj': 'attention.self.value',                
                    'self_attn.q_proj': 'attention.self.query',
                    'self_attn.out_proj': 'attention.output.dense',
                    'self_attn_layer_norm': 'attention.output.LayerNorm',
                    'fc1': 'intermediate.dense',
                    'fc2': 'output.dense',
                    'final_layer_norm': 'output.LayerNorm'
                }
                for layer_id in range(unilm_layers_num):
                    unilm_prefix = prefix_of_parameter + '.encoder.layer.{}.'.format(layer_id)
                    decoder_prefix = 'layers.{}.'.format(layer_id)

                    for key in key_map:
                        for suffix in ['.weight', '.bias']:
                            decoder_key = decoder_prefix + key + suffix
                            unilm_key = unilm_prefix + key_map[key] + suffix
                            if decoder_key in decoder_dict and unilm_key in unilm_state_dict:
                                new_decoder_dict[decoder_key] = unilm_state_dict[unilm_key]
                            
                if hasattr(args, "reset_dictionary") and args.reset_dictionary:
                    logger.info('Reset token embedding weights during decoder initialization.')
                    del new_decoder_dict['embed_tokens.weight']
                elif hasattr(args, "adapt_dictionary") and args.adapt_dictionary:
                    unilm_embed_tokens_weight = new_decoder_dict['embed_tokens.weight']
                    logger.info('Adapt token embedding weights during decoder initialization from {} to {}'.format(unilm_embed_tokens_weight.shape[0], decoder_embed_tokens.weight.shape[0]))                
                    new_decoder_dict['embed_tokens.weight'] = torch.zeros_like(decoder_dict['embed_tokens.weight'])
                    new_decoder_dict['embed_tokens.weight'][:min(unilm_embed_tokens_weight.shape[0], decoder_dict['embed_tokens.weight'].shape[0]), :] = unilm_embed_tokens_weight[:min(unilm_embed_tokens_weight.shape[0], decoder_dict['embed_tokens.weight'].shape[0]), :]

                missing_keys, unexpected_keys = decoder.load_state_dict(
                    new_decoder_dict, strict=False
                )

            else:
                logger.warning('You must specify the unilm model url or the decoder is randomly initialized.')

            # freeze k_proj bias
            for layer in decoder.layers:
                layer.self_attn.k_proj.bias.requires_grad = False
        else:
            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )

            decoder = TransformerDecoder(
                args = args,
                dictionary=task.target_dictionary,
                embed_tokens=decoder_embed_tokens,
                no_encoder_attn=False
            )

        model = cls(encoder, decoder)
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad_token_id

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(self, imgs, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(imgs, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out



@register_model_architecture('OKayOCR', 'swin_tiny_patch4_window7')
def swin_tiny_patch4_window7(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 384)
    args.decoder_learned_pos = True
    args.layernorm_embedding = True
    args.swin_arch = getattr(args, "swin_arch", "swin_tiny_patch4_window7")
    
    import json
    with open(os.path.join(args.decoder_pretrained_url,"config.json"),"r") as f:
        decoder_config = json.load(f)
    
    args.decoder_layers = decoder_config["num_hidden_layers"]
    args.decoder_embed_dim = decoder_config["hidden_size"]
    args.decoder_ffn_embed_dim = decoder_config["intermediate_size"]
    args.decoder_attention_heads = decoder_config["num_attention_heads"]
    #args.max_target_positions = decoder_config["max_position_embeddings"]-1
    args.max_target_positions = 128
    args.activation_fn = decoder_config["hidden_act"]
    base_transformer(args)

    

@register_model_architecture('OKayOCR', 'swin_small_patch4_window7')
def swin_small_patch4_window7(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.decoder_learned_pos = True
    args.layernorm_embedding = True
    args.swin_arch = getattr(args, "swin_arch", "swin_small_patch4_window7")
    
    
    import json
    with open(os.path.join(args.decoder_pretrained_url,"config.json")) as f:
        decoder_config = json.load(f)
    
    args.decoder_layers = decoder_config["num_hidden_layers"]
    args.decoder_embed_dim = decoder_config["hidden_size"]
    args.decoder_ffn_embed_dim = decoder_config["intermediate_size"]
    args.decoder_attention_heads = decoder_config["num_attention_heads"]
    args.max_target_positions = 128
    args.activation_fn = decoder_config["hidden_act"]
    base_transformer(args)

@register_model_architecture("OKayOCR","CSWin_tiny")
def CSWin_tiny(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.depths = getattr(args,"depths",[1,2,21])
    args.num_heads = getattr(args,"num_heads",[2,4,8])
    args.split_size = getattr(args,"split_size",[1,2,7])
    args.cswin_arch = getattr(args,"cswin_arch","CSWin_64_12211_tiny_224")
    
    import json
    with open(os.path.join(args.decoder_pretrained_url,"config.json")) as f:
        decoder_config = json.load(f)
    
    args.decoder_layers = decoder_config["num_hidden_layers"]
    args.decoder_embed_dim = decoder_config["hidden_size"]
    args.decoder_ffn_embed_dim = decoder_config["intermediate_size"]
    args.decoder_attention_heads = decoder_config["num_attention_heads"]
    args.max_target_positions = 128
    args.activation_fn = decoder_config["hidden_act"]
    
    base_transformer(args)
    
@register_model_architecture("OKayOCR","CSWin_small")
def CSWin_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.depths = getattr(args,"depths",[2,4,32,2])
    args.num_heads = getattr(args,"num_heads",[2,4,8,16])
    args.split_size = getattr(args,"split_size",[1,2,7,7])
    args.cswin_arch = getattr(args,"cswin_arch","CSWin_64_24322_small_224")
    
    import json
    with open(os.path.join(args.decoder_pretrained_url,"config.json")) as f:
        decoder_config = json.load(f)
    
    args.decoder_layers = decoder_config["num_hidden_layers"]
    args.decoder_embed_dim = decoder_config["hidden_size"]
    args.decoder_ffn_embed_dim = decoder_config["intermediate_size"]
    args.decoder_attention_heads = decoder_config["num_attention_heads"]
    args.max_target_positions = 128
    args.activation_fn = decoder_config["hidden_act"]
        
    base_transformer(args)
    
@register_model_architecture("OKayOCR","swinv2_tiny_patch4_window8")
def CSWin_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 384)
    args.decoder_learned_pos = True
    args.layernorm_embedding = True
    
    
    import json
    with open(os.path.join(args.decoder_pretrained_url,"config.json")) as f:
        decoder_config = json.load(f)
    
    args.decoder_layers = decoder_config["num_hidden_layers"]
    args.decoder_embed_dim = decoder_config["hidden_size"]
    args.decoder_ffn_embed_dim = decoder_config["intermediate_size"]
    args.decoder_attention_heads = decoder_config["num_attention_heads"]
    #args.max_target_positions = decoder_config["max_position_embeddings"]-1
    args.max_target_positions = 128
    args.activation_fn = decoder_config["hidden_act"]
        
    base_transformer(args)
    
@register_model_architecture("OKayOCR","swinv2_small_patch4_window8")
def CSWin_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    import json
    with open(os.path.join(args.decoder_pretrained_url,"config.json")) as f:
        decoder_config = json.load(f)
    
    args.decoder_layers = decoder_config["num_hidden_layers"]
    args.decoder_embed_dim = decoder_config["hidden_size"]
    args.decoder_ffn_embed_dim = decoder_config["intermediate_size"]
    args.decoder_attention_heads = decoder_config["num_attention_heads"]
    #args.max_target_positions = decoder_config["max_position_embeddings"]-1
    args.max_target_positions = 128
    args.activation_fn = decoder_config["hidden_act"]
        
    base_transformer(args)
class SwinEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.swin_transformer = create_model(args.arch, pretrained_cfg=args, pretrained=True)
        #self.out_proj = nn.Linear(384,512)
        self.fp16 = args.fp16

    def forward(self, imgs):
        #if self.fp16:
        #    imgs = imgs.half()

        x, encoder_embedding = self.swin_transformer.forward_features(imgs)  # bs, n + 2, dim

        x = x.transpose(0, 1) # n + 2, bs, dim
        #x = self.out_proj(x)
        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
          """
          Reorder encoder output according to `new_order`.

          Args:
              encoder_out: output from the ``forward()`` method
              new_order (LongTensor): desired order

          Returns:
              `encoder_out` rearranged according to `new_order`
          """
          _encoder_out = encoder_out['encoder_out'][0]
          _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
          _encoder_embedding = encoder_out['encoder_embedding'][0]
          return {
              "encoder_out": [_encoder_out.index_select(1, new_order)],
                "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
                "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
                "encoder_states": [], 
                "src_tokens": [],
                "src_lengths": [],
        }

class SwinViTEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.swin_transformer = create_model(args.arch, pretrained_cfg=args, pretrained=True)
        
        self.vit_encoder = create_model("Vit_model",pretrained_cfg=args,pretrained=False)
        #self.out_proj = nn.Linear(384,512)
        self.fp16 = args.fp16

    def forward(self, imgs):
        #if self.fp16:
        #    imgs = imgs.half()

        x, encoder_embedding = self.swin_transformer.forward_features(imgs)  # bs, n + 2, dim
        x, encoder_embedding = self.vit_encoder.forward_features(x)
        x = x.transpose(0, 1) # n + 2, bs, dim
        #x = self.out_proj(x)
        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
          """
          Reorder encoder output according to `new_order`.

          Args:
              encoder_out: output from the ``forward()`` method
              new_order (LongTensor): desired order

          Returns:
              `encoder_out` rearranged according to `new_order`
          """
          _encoder_out = encoder_out['encoder_out'][0]
          _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
          _encoder_embedding = encoder_out['encoder_embedding'][0]
          return {
              "encoder_out": [_encoder_out.index_select(1, new_order)],
                "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
                "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
                "encoder_states": [], 
                "src_tokens": [],
                "src_lengths": [],
        }

class CswinEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.encoder = create_model(args.cswin_arch, pretrained_cfg=args, pretrained=True)
        #self.out_proj = nn.Linear(384,512)
        self.fp16 = args.fp16

    def forward(self, imgs):
        #if self.fp16:
        #    imgs = imgs.half()

        x = self.encoder.forward_features(imgs)  # bs, n + 2, dim

        x = x.transpose(0, 1) # n + 2, bs, dim
        #x = self.out_proj(x)
        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [None],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
          """
          Reorder encoder output according to `new_order`.

          Args:
              encoder_out: output from the ``forward()`` method
              new_order (LongTensor): desired order

          Returns:
              `encoder_out` rearranged according to `new_order`
          """
          _encoder_out = encoder_out['encoder_out'][0]
          _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
          _encoder_embedding = encoder_out['encoder_embedding'][0]
          return {
              "encoder_out": [_encoder_out.index_select(1, new_order)],
                "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
                "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
                "encoder_states": [], 
                "src_tokens": [],
                "src_lengths": [],
        }