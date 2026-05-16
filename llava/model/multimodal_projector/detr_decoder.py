import torch
import torch.nn as nn
from typing import Mapping, Any
import re

# DETR modules from transformers.
from transformers.models.detr.configuration_detr import DetrConfig
from transformers.models.detr.modeling_detr import (
    DetrPreTrainedModel,
    build_position_encoding,
    DetrDecoderLayer,
    _prepare_4d_attention_mask,
    DetrDecoderOutput

)

class DETRconnector(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: Custom configuration object
                (vision_width, bert_hidden_size, bert_num_hidden_layers,
                 bert_num_attention_heads, bert_intermediate_size, hidden_size_projector, etc.)
        """
        super().__init__()
        self.config = config

        self.detr_config = DetrConfig(
            d_model=config.vision_width,
            decoder_layers=config.detr_num_hidden_layers,
            decoder_ffn_dim=config.detr_intermediate_size,
            decoder_attention_heads=config.detr_num_attention_heads,
            num_queries=config.detr_num_queries,
            # num_hidden_layers=config.detr_num_hidden_layers,
        )

        self.query_position_embeddings = nn.Embedding(self.detr_config.num_queries, self.detr_config.d_model)

        self.position_embedding = build_position_encoding(self.detr_config)
        self.decoder = DetrDecoder(self.detr_config)
        
        
        self.input_projection = nn.Linear(config.vision_width, self.detr_config.d_model)
        self.llm_proj = nn.Linear(self.detr_config.d_model, config.hidden_size_projector)

        if self.config.detr_pre_caption_input == True:
            self.pre_caption_proj = nn.Sequential(
                nn.Linear(config.hidden_size_projector, self.detr_config.d_model),
                nn.GELU(),
                nn.Linear(self.detr_config.d_model, self.detr_config.d_model),
            )

    def forward(self, image_features: torch.Tensor, pre_captions_tuple=None) -> torch.Tensor:
        """
        Args:
            image_features: (bs, num_tokens, vision_width)
                - e.g., (bs, 576, 768)
                - 576 denotes a 24 x 24 patch grid
        Returns:
            (bs, num_tokens, hidden_size_projector)
        """
        output_attentions = self.detr_config.output_attentions
        output_hidden_states = (self.detr_config.output_hidden_states)
        return_dict = self.detr_config.use_return_dict
        
        bs, num_tokens = image_features.size()[:-1]
        projected_features = self.input_projection(image_features)

        w = h = int(num_tokens ** 0.5)
        assert h * w == num_tokens, "image_features from CLIP should be a perfect square"

        feat2d = projected_features.permute(0, 2, 1).reshape(bs, -1, h, w)
        mask_2d = torch.ones((bs, h, w), dtype=torch.bool, device=feat2d.device)
        flattened_mask = mask_2d.flatten(1)

        pos_embed_2d = self.position_embedding(feat2d, mask_2d).to(image_features.dtype)
        pos_embed_flat = pos_embed_2d.flatten(2).permute(0, 2, 1)  # (bs, num_patches, d_model)

        object_query_pos = self.query_position_embeddings.weight.unsqueeze(0).repeat(bs, 1, 1)
        object_query_init = torch.zeros_like(object_query_pos)

        if pre_captions_tuple is not None:
            pre_captions_embeds, pre_captions_attention_mask = pre_captions_tuple
            pre_captions_embeds = self.pre_caption_proj(pre_captions_embeds)  # (bs, seq_len, d_model)

            queries = torch.cat([object_query_init, pre_captions_embeds], dim=1)  # (bs, 100 + seq_len, d_model)
            
            query_position_embeddings = torch.cat([
                object_query_pos,
                torch.zeros_like(pre_captions_embeds) 
            ], dim=1)

            object_query_mask = torch.ones(bs, object_query_pos.size(1), dtype=torch.bool, device=feat2d.device)
            combined_query_mask = torch.cat([object_query_mask, pre_captions_attention_mask], dim=1)

            decoder_attention_mask = combined_query_mask
        else:
            queries = object_query_init
            query_position_embeddings = object_query_pos
            decoder_attention_mask = None

        decoder_outputs = self.decoder(
            inputs_embeds=queries, #  torch.Size([32, 122, 1024])
            attention_mask=decoder_attention_mask, #  torch.Size([32, 122])
            object_queries=pos_embed_flat, # torch.Size([32, 576, 1024])
            query_position_embeddings=query_position_embeddings, # torch.Size([32, 122])
            encoder_hidden_states=projected_features, # torch.Size([32, 576, 1024])
            encoder_attention_mask=flattened_mask, # torch.Size([32, 576])
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = decoder_outputs.last_hidden_state  # (bs, num_tokens, d_model) or # (bs, 100 + seq_len, d_model)
        object_query_output = hidden_states[:, : self.detr_config.num_queries, :]
        final_out = self.llm_proj(object_query_output)           # (bs, num_tokens, hidden_size_projector)
        return final_out

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if strict:
            result = super().load_state_dict(state_dict, strict=False)
            missing_keys = result.missing_keys
            if len(missing_keys) > 1:
                raise ValueError(f"Missing keys: {missing_keys}")
        else:
            result = super().load_state_dict(state_dict, strict=False)
        return result

def MLPConnector(detr_visual_connector, input_size, output_size):
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', detr_visual_connector)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(input_size, output_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_size, output_size))
        return nn.Sequential(*modules)

class FP_DETRconnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize projections for each level
        self.num_connector = len(config.mm_vision_select_layer.split(','))

        # Initialize DETR components
        self.detr_config = DetrConfig(
            d_model=config.vision_width,
            decoder_layers=config.detr_num_hidden_layers,
            decoder_ffn_dim=config.detr_intermediate_size,
            decoder_attention_heads=config.detr_num_attention_heads,
            num_queries=config.detr_num_queries,
        )
        self.query_position_embeddings = nn.Embedding(self.detr_config.num_queries, self.detr_config.d_model)
        self.position_embedding = build_position_encoding(self.detr_config)
        self.decoder = DetrDecoder(self.detr_config)

        self.input_projection = nn.ModuleList([
            MLPConnector(self.config.detr_visual_connector, config.vision_width, self.detr_config.d_model) for _ in range(self.num_connector)
        ])

        # Initialize layer embeddings for each intermediate level
        self.layer_embedding = nn.Parameter(torch.zeros(self.num_connector - 1, self.detr_config.d_model))
        
        self.llm_proj = nn.Linear(self.detr_config.d_model, config.hidden_size_projector)

        if self.config.detr_pre_caption_input == True:
            self.pre_caption_proj = nn.Sequential(
                nn.Linear(config.hidden_size_projector, self.detr_config.d_model),
                nn.GELU(),
                nn.Linear(self.detr_config.d_model, self.detr_config.d_model),
            )

    def forward(self, image_features, pre_captions_tuple=None):
        """
        Args:
            image_features: List of image feature tensors from multiple levels.
                            e.g. [level_1_features, level_2_features, level_3_features]
                            where each tensor is of shape (batch_size, num_patches, vision_width)
            pre_captions_tuple: Optional tuple containing pre-caption embeddings.
                If provided, it will be used to initialize part of the queries.
        Returns:
            Tensor of shape (batch_size, num_tokens, hidden_size_projector)
        """
        output_attentions = self.detr_config.output_attentions
        output_hidden_states = (self.detr_config.output_hidden_states)
        return_dict = self.detr_config.use_return_dict

        # Step 1: Project image features through input projection for each level
        projected_features = [proj(image_features[i]) for i, proj in enumerate(self.input_projection)]

        # Step 2: Define basic variables for the first level
        bs, num_tokens, _ = projected_features[0].shape
        h = w = int(num_tokens ** 0.5)  # Assuming square grid
        assert h * w == num_tokens, "image_features from CLIP should be a perfect square"

        # Step 3: Create mask for the first level
        feat2d = projected_features[0].permute(0, 2, 1).reshape(bs, -1, h, w)
        mask_2d = torch.ones((bs, h, w), dtype=torch.bool, device=feat2d.device)
        flattened_mask = mask_2d.flatten(1)  # Shape [batch_size, num_patches]

        # Step 4: Generate positional embeddings for the first level
        pos_embed_2d = self.position_embedding(feat2d, mask_2d).to(projected_features[0].dtype)
        pos_embed_flat_dump = pos_embed_2d.flatten(2).permute(0, 2, 1)  # (bs, num_patches, d_model)
        pos_embed_flat = pos_embed_flat_dump.repeat(1, self.num_connector, 1)

        # Step 5: Prepare encoder hidden states by concatenating features from different levels
        encoder_hidden_states = projected_features[0]  # Starting with the first level's feature

        for i in range(1, self.num_connector):
            projected_features[i] += self.layer_embedding[i - 1].unsqueeze(0).unsqueeze(1)
            encoder_hidden_states = torch.cat((encoder_hidden_states, projected_features[i]), dim=1)

        # Step 6: Generate object query embeddings
        object_query_pos = self.query_position_embeddings.weight.unsqueeze(0).repeat(bs, 1, 1)
        object_query_init = torch.zeros_like(object_query_pos)

        # Step 7: Handle pre-captions, if available
        if pre_captions_tuple is not None:
            pre_captions_embeds, pre_captions_attention_mask = pre_captions_tuple
            pre_captions_embeds = self.pre_caption_proj(pre_captions_embeds)  # (bs, seq_len, d_model)

            # Concatenate object queries and pre-captions embeddings
            queries = torch.cat([object_query_init, pre_captions_embeds], dim=1)

            # Generate position embeddings for the concatenated queries
            query_position_embeddings = torch.cat([
                object_query_pos,
                torch.zeros_like(pre_captions_embeds)
            ], dim=1)

            # Create a combined attention mask for object queries and pre-captions
            object_query_mask = torch.ones(bs, object_query_pos.size(1), dtype=torch.bool, device=feat2d.device)
            combined_query_mask = torch.cat([object_query_mask, pre_captions_attention_mask], dim=1)

            # Prepare the attention mask for the decoder
            decoder_attention_mask = combined_query_mask
        else:
            # If no pre-captions, use object queries only
            queries = object_query_init
            query_position_embeddings = object_query_pos
            decoder_attention_mask = None

        # Step 8: Ensure the encoder_attention_mask matches the shape of encoder_hidden_states
        # Create the encoder_attention_mask based on the number of levels (num_connector)
        expanded_mask = flattened_mask.repeat(1, self.num_connector)  # Repeat for each level

        # Step 9: Pass through the DETR decoder
        decoder_outputs = self.decoder(
            inputs_embeds=queries,  # torch.Size([32, 122, 1024])
            attention_mask=decoder_attention_mask, # torch.Size([32, 122])
            object_queries=pos_embed_flat, # torch.Size([32, 1152, 1024])
            query_position_embeddings=query_position_embeddings, # torch.Size([32, 122, 1024])
            encoder_hidden_states=encoder_hidden_states, # torch.Size([32, 1152, 1024])
            encoder_attention_mask=expanded_mask,  # torch.Size([32, 1152])
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Step 10: Extract the output and project to the final size
        hidden_states = decoder_outputs.last_hidden_state
        object_query_output = hidden_states[:, : self.detr_config.num_queries, :]
        final_out = self.llm_proj(object_query_output)
        return final_out



class DetrDecoder(DetrPreTrainedModel):
    """
        Most parts of this code are from transformers.models.detr.modeling_detr.DetrDecoder
        But, I fixed some parts to use gradient_checkpointing
        See 'FIXED' comments in the code
    """

    def __init__(self, config: DetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            object_queries = position_embeddings

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]

        combined_attention_mask = None
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            combined_attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    combined_attention_mask,
                    object_queries,             #### FIXED: original code does not this part, so error occurs ###
                    query_position_embeddings,  #### FIXED: original code does not this part, so error occurs
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    object_queries=object_queries,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # stack intermediate decoder activations
        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
                if v is not None
            )
        return DetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=intermediate,
        )
