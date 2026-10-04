# SPDX-License-Identifier: Apache-2.0
"""Attention layer MUSA GPUs."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import sys  # ADDED: REF-KV-DIAG/CACHE-LAYOUT-DIAG 用 sys.stderr 但漏 import
import torch

from vllm_musa import _musa_custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import PagedAttentionMetadata
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from vllm_musa.platforms.musa import on_ph1

logger = init_logger(__name__)

try:
    import mate
    logger.info("Use MUSA AI Tensor Engine attention backend.")
except (ImportError, ModuleNotFoundError):
    mate = None

class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "MUSA_FLASH"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]

        ops.copy_blocks(key_caches, value_caches, src_to_dists)

@dataclass
class FlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    _cached_prefill_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["FlashAttentionMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.block_tables is not None

        self._cached_prefill_metadata = FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=None if self.query_start_loc is None else
            self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=None if self.seq_start_loc is None else
            self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=None if self.context_lens_tensor is None else
            self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        # Batch may be composed of prefill|decodes, adjust query start indices
        # to refer to the start of decodes when the two are split apart.
        # E.g. in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """

        assert not turn_prefills_into_decodes, \
            ("Chunked prefill is not supported with rocm_flash_attn yet."
             "turn_prefills_into_decodes is a Multi-Step + Chunked-Prefill "
             "specific parameter.")

        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class FlashAttentionMetadataBuilder(
        CommonMetadataBuilder[FlashAttentionMetadata]):

    _metadata_cls = FlashAttentionMetadata


def _make_alibi_bias(alibi_slopes: torch.Tensor,
                     dtype: torch.dtype,
                     seq_lens: Optional[List[int]],
                     make_attn_mask: bool = True) -> List[torch.Tensor]:
    attn_biases = []
    if seq_lens:
        for seq_len in seq_lens:
            bias = torch.arange(seq_len, dtype=dtype)
            # NOTE(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(seq_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]

            num_heads = alibi_slopes.shape[0]
            bias = bias[None, :].repeat(
                (num_heads, 1, 1)).to(alibi_slopes.device)
            bias.mul_(alibi_slopes[:, None, None])
            if make_attn_mask:
                inf_mask = torch.empty(
                    (1, seq_len, seq_len),
                    dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1).to(
                        alibi_slopes.device)
                attn_biases.append((bias + inf_mask).to(dtype))
            else:
                attn_biases.append(bias.to(dtype))

    return attn_biases


def _get_seq_len_block_table_args(
    attn_metadata: FlashAttentionMetadata,
    attn_type: str,
) -> tuple:
    '''
    The particular choice of sequence-length
    attributes which should be extracted from attn_metadata is dependent
    on the type of attention operation.

    Decoder attn -> select entirely decoder self-attention-related fields
    Encoder/decoder cross-attn -> select encoder sequence lengths
    Encoder attn -> select encoder sequence lengths fields
    
    Arguments:

    * attn_metadata: Attention metadata structure associated with attention op
    * attn_type: encoder attention, decoder self-attention,
                encoder/decoder cross-attention

    Returns:

    * Appropriate sequence-lengths tensors for query and key
    * Appropriate max sequence-length scalar
    '''

    partial_prefix_sum = 0
    if attn_type == AttentionType.ENCODER:
        assert attn_metadata.encoder_seq_lens is not None
        assert attn_metadata.encoder_seq_lens_tensor is not None
        query_seq_start_loc = torch.tensor(
            [0] + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.encoder_seq_lens
            ],
            device=attn_metadata.encoder_seq_lens_tensor.device,
            dtype=attn_metadata.encoder_seq_lens_tensor.dtype)
        causal_mask = False

        # No block tables associated with encoder attention
        return (query_seq_start_loc, attn_metadata.max_encoder_seq_len,
                query_seq_start_loc, attn_metadata.max_encoder_seq_len,
                attn_metadata.encoder_seq_lens, causal_mask)
    elif attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None
        query_seq_start_loc = torch.tensor(
            [0] + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.seq_lens
            ],
            device=attn_metadata.seq_lens_tensor.device,
            dtype=attn_metadata.seq_lens_tensor.dtype)
        max_seq_len = attn_metadata.max_prefill_seq_len
        causal_mask = True

        return (query_seq_start_loc, max_seq_len, query_seq_start_loc,
                max_seq_len, attn_metadata.seq_lens, causal_mask)
    elif attn_type == AttentionType.ENCODER_DECODER:
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.encoder_seq_lens_tensor is not None
        query_start_loc = torch.tensor(
            [0] + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.seq_lens
            ],
            device=attn_metadata.encoder_seq_lens_tensor.device,
            dtype=attn_metadata.encoder_seq_lens_tensor.dtype)

        partial_prefix_sum = 0
        assert attn_metadata.encoder_seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None
        key_seq_start_loc = torch.tensor(
            [0] + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.encoder_seq_lens
            ],
            device=attn_metadata.seq_lens_tensor.device,
            dtype=attn_metadata.seq_lens_tensor.dtype)
        causal_mask = False

        # Enc/dec cross-attention KVs match encoder sequence length;
        # cross-attention utilizes special "cross" block tables
        return (query_start_loc, attn_metadata.max_prefill_seq_len,
                key_seq_start_loc, attn_metadata.max_encoder_seq_len,
                attn_metadata.seq_lens, causal_mask)
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


class FlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        **extra_impl_args,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")

        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            self.logits_soft_cap = 0.0
        else:
            self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.num_splits = 4
        if on_ph1():
            self.num_splits = 1

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return (x[:, :,
                  None, :].expand(tokens, n_kv_heads, n_rep,
                                  head_dim).reshape(tokens, n_kv_heads * n_rep,
                                                    head_dim))
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print("---------------FlashAttention---------------")
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        """
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0, (
            "key/v_scale is not supported in FlashAttention.")

        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")
        
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.reshape(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None
            
        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (
                    value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.attn_type != AttentionType.ENCODER:
            num_prefill_tokens = attn_metadata.num_prefill_tokens
        else:
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]

        if key is not None and value is not None \
            and self.attn_type != AttentionType.ENCODER_DECODER:
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

        # ==== CODEX MOD START: CACHE-LAYOUT-DIAG 2026-05-21 ====
        # Verify Path A's KV-cache read layout on fresh prefill. If this diff
        # is nonzero, key_cache[block_ids].reshape(-1, h, d) is not equivalent
        # to the original key/value layout written by reshape_and_cache_flash.
        if prefill_meta := attn_metadata.prefill_metadata:
            try:
                _cld_count = getattr(torch, '_codex_cache_layout_diag_count', 0)
                _cld_fresh = (
                    key is not None and value is not None
                    and kv_cache.numel() > 0
                    and prefill_meta.block_tables is not None
                    and prefill_meta.block_tables.numel() > 0
                    and attn_metadata.num_prefill_tokens == int(prefill_meta.seq_lens_tensor.sum().item())
                )
                if _cld_fresh and _cld_count < 2:
                    import os as _cld_os
                    torch._codex_cache_layout_diag_count = _cld_count + 1
                    _cld_key_cache = kv_cache[0]
                    _cld_value_cache = kv_cache[1]
                    _cld_block_size = _cld_key_cache.shape[1]
                    _cld_seq_lens = prefill_meta.seq_lens_tensor.detach().cpu().tolist()
                    _cld_h_kv = key.shape[1]
                    _cld_d_kv = key.shape[2]
                    _cld_seq0 = int(_cld_seq_lens[0])
                    _cld_num_blk0 = (_cld_seq0 + _cld_block_size - 1) // _cld_block_size
                    _cld_blk_ids0 = prefill_meta.block_tables[0][:_cld_num_blk0]
                    _cld_k_read = _cld_key_cache[_cld_blk_ids0].reshape(-1, _cld_h_kv, _cld_d_kv)[:_cld_seq0]
                    _cld_v_read = _cld_value_cache[_cld_blk_ids0].reshape(-1, _cld_h_kv, _cld_d_kv)[:_cld_seq0]
                    _cld_k_ref = key[:_cld_seq0]
                    _cld_v_ref = value[:_cld_seq0]
                    _cld_dk = (_cld_k_read - _cld_k_ref).abs().max().item()
                    _cld_dv = (_cld_v_read - _cld_v_ref).abs().max().item()
                    print(
                        f"[CODEX CACHE-LAYOUT-DIAG pid={_cld_os.getpid()} count={_cld_count}] "
                        f"key_cache_shape={tuple(_cld_key_cache.shape)} value_cache_shape={tuple(_cld_value_cache.shape)} "
                        f"seq0={_cld_seq0} block_size={_cld_block_size} blocks0={_cld_blk_ids0[:8].detach().cpu().tolist()} "
                        f"diff_k={_cld_dk} diff_v={_cld_dv}",
                        file=sys.stderr, flush=True)
            except Exception as _cld_e:
                print(f"[CODEX CACHE-LAYOUT-DIAG ERROR] {_cld_e}",
                      file=sys.stderr, flush=True)
        # ==== CODEX MOD END: CACHE-LAYOUT-DIAG 2026-05-21 ====

        # ==== CODEX MOD START: REF-KV-DIAG 2026-05-21 store fresh K/V by layer ====
        if prefill_meta := attn_metadata.prefill_metadata:
            try:
                _rkv_fresh = (
                    key is not None and value is not None
                    and attn_metadata.num_prefill_tokens == int(prefill_meta.seq_lens_tensor.sum().item())
                    and attn_metadata.num_prefill_tokens > 0
                )
                if _rkv_fresh:
                    _rkv_refs = getattr(torch, '_codex_ref_kv_by_layer', None)
                    if _rkv_refs is None:
                        _rkv_refs = {}
                        torch._codex_ref_kv_by_layer = _rkv_refs
                    _rkv_lid = id(layer)
                    if _rkv_lid not in _rkv_refs:
                        # Keep one full fresh prefill reference per layer/process.
                        _rkv_refs[_rkv_lid] = (
                            key.detach().clone().contiguous(),
                            value.detach().clone().contiguous(),
                        )
                        _rkv_count = getattr(torch, '_codex_ref_kv_store_prints', 0)
                        if _rkv_count < 4:
                            import os as _rkv_os
                            torch._codex_ref_kv_store_prints = _rkv_count + 1
                            print(
                                f"[CODEX REF-KV-DIAG store pid={_rkv_os.getpid()} count={_rkv_count}] "
                                f"layer_id={_rkv_lid} key_shape={tuple(key.shape)} value_shape={tuple(value.shape)}",
                                file=sys.stderr, flush=True)
            except Exception as _rkv_e:
                print(f"[CODEX REF-KV-DIAG store ERROR] {_rkv_e}",
                      file=sys.stderr, flush=True)
        # ==== CODEX MOD END: REF-KV-DIAG 2026-05-21 store fresh K/V by layer ====

        if prefill_meta := attn_metadata.prefill_metadata:
            # ===== PATH-A 2026-05-21: prefix-cache 命中,Python 层 concat + SDPA =====
            # 完全绕开 varlen_fa_seqlen_pad/unpad,在 Python 层手动从 KV cache 拉 cached K/V,
            # 拼到 new K/V 前面,然后调 SDPA(RunFlash 已知稳定)
            _path_a_hit = (
                self.attn_type == AttentionType.DECODER
                and kv_cache.numel() > 0
                and prefill_meta.block_tables is not None
                and prefill_meta.block_tables.numel() > 0
                # ==== CODEX MOD START: Path A real-hit gate 2026-05-21 ====
                # PATH-A-FIX gate+contiguous 2026-05-21:
                # Only real prefix-cache hits should use Path A. Fresh prefill
                # has num_prefill_tokens == sum(seq_lens), and the original
                # varlen_pad -> SDPA -> unpad path is already correct/stable.
                and attn_metadata.num_prefill_tokens < int(prefill_meta.seq_lens_tensor.sum().item())
                # ==== CODEX MOD END: Path A real-hit gate 2026-05-21 ====
            )
            if _path_a_hit:
                import sys as _pa_sys
                if not getattr(torch, '_path_a_announced', False):
                    print("[PATH-A] prefix-cache hit branch active (Python concat + SDPA)",
                          file=_pa_sys.stderr, flush=True)
                    torch._path_a_announced = True

                _pa_h_q = query.shape[1]
                _pa_h_kv = key.shape[1]
                _pa_d_q = query.shape[2]
                _pa_d_kv = key.shape[2]

                _pa_key_cache = kv_cache[0]      # shape (num_blocks, block_size, h_kv, d_kv)
                _pa_value_cache = kv_cache[1]
                _pa_block_size = _pa_key_cache.shape[1]

                _pa_block_tables = prefill_meta.block_tables   # (bs, max_blocks_per_seq)
                # ==== CODEX MOD START: use prefill metadata tensors 2026-05-21 ====
                _pa_seq_lens_tensor = prefill_meta.seq_lens_tensor   # (bs,) 完整序列长度
                # ==== CODEX MOD END: use prefill metadata tensors 2026-05-21 ====
                _pa_bs = _pa_block_tables.shape[0]

                # query_start_loc: cumsum of NEW token lens (优先用 attn_metadata)
                # ==== CODEX MOD START: use prefill query_start_loc 2026-05-21 ====
                _pa_query_start_loc = prefill_meta.query_start_loc
                # ==== CODEX MOD END: use prefill query_start_loc 2026-05-21 ====
                if _pa_query_start_loc is None:
                    # fallback: 单 batch 情况,所有 query 都是新的
                    _pa_query_start_loc = torch.tensor([0, query.shape[0]],
                                                       device=query.device, dtype=torch.int32)

                # 拿 host-side int 用于 loop
                _pa_seq_lens_cpu = _pa_seq_lens_tensor.cpu().tolist()
                _pa_query_start_cpu = _pa_query_start_loc.cpu().tolist()
                _pa_new_lens = [_pa_query_start_cpu[b+1] - _pa_query_start_cpu[b] for b in range(_pa_bs)]
                _pa_cached_lens = [_pa_seq_lens_cpu[b] - _pa_new_lens[b] for b in range(_pa_bs)]
                _pa_max_full_len = max(_pa_seq_lens_cpu)

                # ==== CODEX MOD START: PATH-A-DIAG 2026-05-21 metadata dump ====
                import os as _pa_os
                _pa_diag_count = getattr(torch, '_path_a_diag_count', 0)
                _pa_do_diag = _pa_diag_count < 4
                if _pa_do_diag:
                    torch._path_a_diag_count = _pa_diag_count + 1
                    _pa_bt0 = []
                    try:
                        _pa_num_blk0 = (_pa_cached_lens[0] + _pa_block_size - 1) // _pa_block_size
                        _pa_bt0 = _pa_block_tables[0][:_pa_num_blk0].detach().cpu().tolist()
                    except Exception as _pa_e:
                        _pa_bt0 = ["ERR", str(_pa_e)]
                    print(
                        f"[CODEX PATH-A-DIAG pid={_pa_os.getpid()} count={_pa_diag_count}] "
                        f"num_prefill_tokens={attn_metadata.num_prefill_tokens} "
                        f"seq_lens={_pa_seq_lens_cpu} query_start={_pa_query_start_cpu} "
                        f"new_lens={_pa_new_lens} cached_lens={_pa_cached_lens} "
                        f"block_size={_pa_block_size} block_ids0={_pa_bt0[:8]}",
                        file=_pa_sys.stderr, flush=True)
                # ==== CODEX MOD END: PATH-A-DIAG 2026-05-21 metadata dump ====

                # alloc padded buffers,zero-fill 让 attention 数学合法
                _pa_q_pad = torch.zeros((_pa_bs, _pa_h_q,  _pa_max_full_len, _pa_d_q),
                                        device=query.device, dtype=query.dtype)
                _pa_k_pad = torch.zeros((_pa_bs, _pa_h_kv, _pa_max_full_len, _pa_d_kv),
                                        device=query.device, dtype=query.dtype)
                _pa_v_pad = torch.zeros((_pa_bs, _pa_h_kv, _pa_max_full_len, _pa_d_kv),
                                        device=query.device, dtype=query.dtype)

                # 逐 batch 填数据
                for _pa_b in range(_pa_bs):
                    _pa_cl = _pa_cached_lens[_pa_b]
                    _pa_nl = _pa_new_lens[_pa_b]
                    _pa_fl = _pa_seq_lens_cpu[_pa_b]
                    _pa_ns = _pa_query_start_cpu[_pa_b]

                    # 1. 拉 cached K/V 到位置 [0:cached_len]
                    if _pa_cl > 0:
                        _pa_num_blk = (_pa_cl + _pa_block_size - 1) // _pa_block_size
                        _pa_blk_ids = _pa_block_tables[_pa_b][:_pa_num_blk]
                        # key_cache[blk_ids] shape: (num_blk, block_size, h_kv, d_kv)
                        _pa_cached_k = _pa_key_cache[_pa_blk_ids].reshape(-1, _pa_h_kv, _pa_d_kv)[:_pa_cl]
                        _pa_cached_v = _pa_value_cache[_pa_blk_ids].reshape(-1, _pa_h_kv, _pa_d_kv)[:_pa_cl]
                        # ==== CODEX MOD START: REF-KV-DIAG 2026-05-21 compare cached K/V to fresh reference ====
                        try:
                            _rkv_refs = getattr(torch, '_codex_ref_kv_by_layer', {})
                            _rkv_ref = _rkv_refs.get(id(layer))
                            _rkv_cmp_count = getattr(torch, '_codex_ref_kv_cmp_prints', 0)
                            if _rkv_ref is not None and _rkv_cmp_count < 8:
                                import os as _rkv_os
                                torch._codex_ref_kv_cmp_prints = _rkv_cmp_count + 1
                                _rkv_ref_k, _rkv_ref_v = _rkv_ref
                                _rkv_n = min(_pa_cl, _rkv_ref_k.shape[0])
                                _rkv_dk = (_pa_cached_k[:_rkv_n] - _rkv_ref_k[:_rkv_n]).abs().max().item()
                                _rkv_dv = (_pa_cached_v[:_rkv_n] - _rkv_ref_v[:_rkv_n]).abs().max().item()
                                print(
                                    f"[CODEX REF-KV-DIAG compare pid={_rkv_os.getpid()} count={_rkv_cmp_count}] "
                                    f"layer_id={id(layer)} cached_len={_pa_cl} n={_rkv_n} "
                                    f"diff_k={_rkv_dk} diff_v={_rkv_dv}",
                                    file=_pa_sys.stderr, flush=True)
                        except Exception as _rkv_e:
                            print(f"[CODEX REF-KV-DIAG compare ERROR] {_rkv_e}",
                                  file=_pa_sys.stderr, flush=True)
                        # ==== CODEX MOD END: REF-KV-DIAG 2026-05-21 compare cached K/V to fresh reference ====
                        # 转置后填入: (cl, h_kv, d) -> (h_kv, cl, d)
                        # ==== CODEX MOD START: contiguous cached K/V transpose 2026-05-21 ====
                        _pa_k_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_k.transpose(0, 1).contiguous()
                        _pa_v_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_v.transpose(0, 1).contiguous()
                        # ==== CODEX MOD END: contiguous cached K/V transpose 2026-05-21 ====

                    # 2. 填新 Q/K/V 到位置 [cached_len:full_len]
                    if _pa_nl > 0:
                        _pa_new_q = query[_pa_ns:_pa_ns+_pa_nl]   # (nl, h_q, d_q)
                        _pa_new_k = key  [_pa_ns:_pa_ns+_pa_nl]   # (nl, h_kv, d_kv)
                        _pa_new_v = value[_pa_ns:_pa_ns+_pa_nl]
                        # ==== CODEX MOD START: contiguous new Q/K/V transpose 2026-05-21 ====
                        _pa_q_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_q.transpose(0, 1).contiguous()
                        _pa_k_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_k.transpose(0, 1).contiguous()
                        _pa_v_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_v.transpose(0, 1).contiguous()
                        # ==== CODEX MOD END: contiguous new Q/K/V transpose 2026-05-21 ====

                # ==== CODEX MOD START: PATH-A-DIAG 2026-05-21 compare Python scatter vs varlen_pad on new-token region ====
                if _pa_do_diag:
                    try:
                        _pa_diag_q = torch.empty_like(_pa_q_pad)
                        _pa_diag_k = torch.empty_like(_pa_k_pad)
                        _pa_diag_v = torch.empty_like(_pa_v_pad)
                        ops.varlen_fa_seqlen_pad(
                            query, key, value,
                            _pa_diag_q, _pa_diag_k, _pa_diag_v,
                            _pa_query_start_loc, _pa_query_start_loc,
                            query.shape[0], max(_pa_new_lens), _pa_bs)
                        _pa_b0 = 0
                        _pa_cl0 = _pa_cached_lens[_pa_b0]
                        _pa_nl0 = _pa_new_lens[_pa_b0]
                        _pa_py_q = _pa_q_pad[_pa_b0, :, _pa_cl0:_pa_cl0 + _pa_nl0, :]
                        _pa_cpp_q = _pa_diag_q[_pa_b0, :, :_pa_nl0, :]
                        _pa_py_k = _pa_k_pad[_pa_b0, :, _pa_cl0:_pa_cl0 + _pa_nl0, :]
                        _pa_cpp_k = _pa_diag_k[_pa_b0, :, :_pa_nl0, :]
                        _pa_py_v = _pa_v_pad[_pa_b0, :, _pa_cl0:_pa_cl0 + _pa_nl0, :]
                        _pa_cpp_v = _pa_diag_v[_pa_b0, :, :_pa_nl0, :]
                        _pa_dq = (_pa_py_q - _pa_cpp_q).abs().max().item() if _pa_nl0 > 0 else -1
                        _pa_dk = (_pa_py_k - _pa_cpp_k).abs().max().item() if _pa_nl0 > 0 else -1
                        _pa_dv = (_pa_py_v - _pa_cpp_v).abs().max().item() if _pa_nl0 > 0 else -1
                        print(
                            f"[CODEX PATH-A-DIAG scatter-diff] q={_pa_dq} k={_pa_dk} v={_pa_dv}",
                            file=_pa_sys.stderr, flush=True)
                    except Exception as _pa_e:
                        print(f"[CODEX PATH-A-DIAG scatter-diff ERROR] {_pa_e}",
                              file=_pa_sys.stderr, flush=True)
                # ==== CODEX MOD END: PATH-A-DIAG 2026-05-21 compare Python scatter vs varlen_pad on new-token region ====

                # 调 SDPA(B2 已验证稳定)
                _pa_attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
                    _pa_q_pad, _pa_k_pad, _pa_v_pad, dropout_p=0.0, is_causal=True)
                # attn_out shape: (bs, h_q, max_full_len, d_q)

                # 从 attn_out 抽出每个 batch 的新 query 输出位置 [cached_len:full_len]
                _pa_sum_new = query.shape[0]
                _pa_output = torch.empty((_pa_sum_new, _pa_h_q * _pa_d_q),
                                         device=query.device, dtype=query.dtype)
                for _pa_b in range(_pa_bs):
                    _pa_cl = _pa_cached_lens[_pa_b]
                    _pa_nl = _pa_new_lens[_pa_b]
                    _pa_fl = _pa_seq_lens_cpu[_pa_b]
                    _pa_ns = _pa_query_start_cpu[_pa_b]
                    if _pa_nl > 0:
                        # attn_out[b, :, cl:fl, :].transpose(0,1) -> (nl, h_q, d_q) -> reshape (nl, h_q*d_q)
                        # ==== CODEX MOD START: contiguous output slice before reshape 2026-05-21 ====
                        _pa_seg = _pa_attn_out[_pa_b, :, _pa_cl:_pa_fl, :].transpose(0, 1).contiguous().reshape(_pa_nl, _pa_h_q * _pa_d_q)
                        # ==== CODEX MOD END: contiguous output slice before reshape 2026-05-21 ====
                        _pa_output[_pa_ns:_pa_ns+_pa_nl] = _pa_seg

                return _pa_output
            # ===== END PATH-A ===== 下面是原版 fresh prefill(无命中)路径
            # Prompt run.
            # normal attention and DECODER
            if self.attn_type == AttentionType.DECODER and (
                    kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                (query_seq_start_loc, query_max_seq_len, key_seq_start_loc,
                 key_max_seq_len, seq_lens,
                 causal_mask) = (prefill_meta.seq_start_loc,
                                 prefill_meta.max_prefill_seq_len,
                                 prefill_meta.seq_start_loc,
                                 prefill_meta.max_prefill_seq_len,
                                 attn_metadata.seq_lens, True)
            # prefix-enabled attention and ENCODER/ENCODER_DECODER
            else:
                (query_seq_start_loc, query_max_seq_len, key_seq_start_loc,
                 key_max_seq_len, seq_lens,
                 causal_mask) = _get_seq_len_block_table_args(
                     prefill_meta, self.attn_type)
            # Prompt run.
            if prefill_meta := attn_metadata.prefill_metadata:
                # if self.num_kv_heads != self.num_heads:
                #     # Interleave for MQA workaround.
                #     key = self.repeat_kv(key, self.num_queries_per_kv)
                #     value = self.repeat_kv(value, self.num_queries_per_kv)

                # output = sdpa_attention_with_torch_seqlen_pad(
                output = sdpa_attention_with_kernel_seqlen_pad(
                    query,
                    key,
                    value,
                    query_seq_start_loc,
                    query_max_seq_len,
                )

                return output

        if decode_meta := attn_metadata.decode_metadata:
            from vllm.attention.ops.triton_decode_attention import _decode_grouped_att_m_fwd, _decode_softmax_reducev_fwd

            decode_meta = attn_metadata.decode_metadata
            assert decode_meta is not None

            B = decode_query.shape[0]
            PAGE_SIZE = key_cache.size(1)

            if self.num_splits == 1:
                attn_logits = torch.zeros(
                    (
                        B,
                        self.num_heads,
                        self.num_splits,
                        self.head_size,
                    ),
                    dtype=query.dtype,
                    device=query.device,
                )
                output = attn_logits
            else:
                output = torch.empty(B,
                                self.num_heads,
                                self.head_size,
                                dtype=query.dtype,
                                device=query.device)
                attn_logits = torch.zeros(
                    (
                        B,
                        self.num_heads,
                        self.num_splits,
                        self.head_size + 1,
                    ),
                    dtype=torch.float32,
                    device=query.device,
                )

            if on_ph1() and mate != None:
                batch = decode_query.shape[0]
                head_size = decode_query.shape[2]
                num_head = decode_query.shape[1]
                decode_query = decode_query.view(batch, 1 ,num_head, head_size)
                attn_logits = attn_logits.permute(0, 2, 1, 3)
                mate.paged_fmha(decode_query, 
                                key_cache, 
                                value_cache,
                                decode_meta.seq_lens_tensor,
                                decode_meta.block_tables,
                                attn_logits,
                                self.scale, 
                                # self.num_splits, 
                                True
                                )
            elif on_ph1():
                ops.mp31_decode_mha(decode_query, 
                                    key_cache, 
                                    value_cache, 
                                    self.scale, 
                                    self.num_splits, 
                                    decode_meta.block_tables, 
                                    decode_meta.seq_lens_tensor, 
                                    attn_logits)
            else:
                # Run MQA
                _decode_grouped_att_m_fwd(
                    decode_query,
                    key_cache,
                    value_cache,
                    attn_logits,
                    decode_meta.block_tables,
                    decode_meta.seq_lens_tensor,
                    self.num_splits,
                    self.scale,
                    PAGE_SIZE,
                    0,
                )

            
            if self.num_splits > 1:
                _decode_softmax_reducev_fwd(attn_logits,
                                            decode_query,
                                            output, 
                                            value_cache, 
                                            decode_meta.seq_lens_tensor,
                                            self.num_splits)

            return output.view(-1, self.num_heads*self.head_size)

def sdpa_attention_with_kernel_seqlen_pad(
    query: torch.Tensor, #sum_seq, h_q, d_q 
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: List[int],
    max_prefill_seq_len: int,
    is_causal: bool = True,
) -> torch.Tensor:
    bs = seq_lens.shape[0] - 1
    sum_seq, h_q, d_q = query.shape
    _,h_kv, d_kv = key.shape
    device, dtype = query.device, query.dtype

    q_pad = torch.empty((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    k_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    v_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    output = torch.empty((sum_seq, h_q, d_q), device=device, dtype=dtype)

    ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
    attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
        q_pad,
        k_pad,
        v_pad,
        dropout_p=0.0,
        is_causal=is_causal)

    ops.varlen_fa_seqlen_unpad(attn_out, output, seq_lens, sum_seq, max_prefill_seq_len, d_q, h_q, bs)
    return output.view(-1, h_q * d_q)


def pad_seq(x: torch.Tensor, cu_seqlen:torch.Tensor, max_prefill_seq_len: int, num_heads: int) -> torch.Tensor:
    bs = cu_seqlen.shape[0] - 1
    padded = torch.zeros(bs, num_heads, max_prefill_seq_len, x.shape[-1], device=x.device, dtype=x.dtype)
    for i in range(bs):
        start = cu_seqlen[i].item()
        end = cu_seqlen[i + 1].item()
        seq_len = end - start 
        padded[i, :, :seq_len] = x[start:end, :, :].transpose(0, 1)
    return padded

def restore_tokens(attn_output: torch.Tensor, cu_seqlen: torch.Tensor, max_prefill_seq_len: int, num_heads: int) -> torch.Tensor:
    bs = cu_seqlen.shape[0] - 1
    attn_output_tensor = attn_output.transpose(1, 2).view(bs, max_prefill_seq_len, num_heads, -1)  
    restored_tokens = []
    for i in range(bs):
        start = cu_seqlen[i].item()
        end = cu_seqlen[i + 1].item()
        seq_len = end - start
        restored_tokens.append(attn_output_tensor[i, :seq_len])
    return torch.cat(restored_tokens, dim=0)


def sdpa_attention_with_torch_seqlen_pad(
    query: torch.Tensor, #sum_seq, h_q, d_q 
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: List[int],
    max_prefill_seq_len: int,
    is_causal: bool = True,
) -> torch.Tensor:
    sum_seq, h_q, d_q = query.shape
    _, h_kv, d_kv = key.shape
    
    q_pad = pad_seq(query, seq_lens, max_prefill_seq_len, h_q)
    k_pad = pad_seq(key, seq_lens, max_prefill_seq_len, h_kv)
    v_pad = pad_seq(value, seq_lens, max_prefill_seq_len, h_kv)
    
    output, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
        q_pad,
        k_pad,
        v_pad,
        dropout_p=0.0,
        is_causal=is_causal)
    
    output = restore_tokens(output, seq_lens, max_prefill_seq_len, h_q)
    return output.view(-1, h_q * d_q)
