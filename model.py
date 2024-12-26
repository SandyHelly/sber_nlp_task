import torch

from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel, XLMRobertaTokenizer
from transformers import PreTrainedModel, PretrainedConfig


class EmbedderConfig(PretrainedConfig):
  model_type = 'emb_test_basemodel'
  def __init__(self, version: int = 1, **kwargs):
    super().__init__(**kwargs)
    self.version = version
    self.hidden_size = 384
    self.base_model_name = 'microsoft/Multilingual-MiniLM-L12-H384'


class EmbedderModel(PreTrainedModel):
  config_class = EmbedderConfig

  def __init__(self, config: PretrainedConfig):
    super().__init__(config)
    self.config = config
    self.emb_base_model = BertModel.from_pretrained(config.base_model_name)

  def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, **kwargs) -> torch.tensor:
    '''for sentence encoding'''
    sentence_emb = self.emb_base_model(input_ids=input_ids,
                                       attention_mask=attention_mask)
    return sentence_emb

  def forward_(self, q_input: list, d_input: list, n_input: list) -> tuple:
    '''for training'''

    q_emb = self.emb_base_model(input_ids=q_input['input_ids'].squeeze(1),
                                attention_mask=q_input['attention_mask'].squeeze(1))
    q_emb = q_emb.last_hidden_state

    d_emb = self.emb_base_model(input_ids=d_input['input_ids'].squeeze(1),
                                attention_mask=d_input['attention_mask'].squeeze(1))
    d_emb = d_emb.last_hidden_state

    n_emb = self.emb_base_model(input_ids=n_input['input_ids'].squeeze(1),
                                attention_mask=n_input['attention_mask'].squeeze(1))
    n_emb = n_emb.last_hidden_state
    return q_emb, d_emb, n_emb
