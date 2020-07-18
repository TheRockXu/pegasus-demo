# Copyright 2020 The PEGASUS Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Parsing with public available ops.

This is a wrapper of sentencepiece ops for public release.
"""

from typing import List
import tensorflow as tf
import sentencepiece as sentencepiece_processor

_SHIFT_RESERVED_TOKENS = 103
_NEWLINE_SYMBOL = "<n>"


def create_text_encoder(encoder_type: str, vocab_filename: str):
  if encoder_type == "sentencepiece":
    return SentencePieceEncoder(vocab_filename)
  elif encoder_type == "sentencepiece_newline":
    return SentencePieceEncoder(vocab_filename, newline_symbol=_NEWLINE_SYMBOL)
  else:
    raise ValueError("Unsupported encoder type: %s" % encoder_type)


class SentencePieceEncoder(object):
  """SentencePieceEncoder.

  First two ids are pad=0, eos=1, rest ids are being shifted up by
  shift_reserved_tokens. If newline_symbol is provided, will replace newline in
  the text with that token.
  """

  def __init__(self,
               sentencepiece_model_file: str,
               shift_reserved_tokens: int = _SHIFT_RESERVED_TOKENS,
               newline_symbol: str = ""):
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._sp_model = tf.io.gfile.GFile(sentencepiece_model_file, "rb").read()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)
    self._shift_reserved_tokens = shift_reserved_tokens
    self._newline_symbol = newline_symbol

  @property
  def vocab_size(self) -> int:
    return self._tokenizer.GetPieceSize() + self._shift_reserved_tokens

  def encode(self, text: str) -> List[int]:
    if self._newline_symbol:
      text = text.replace("\n", self._newline_symbol)
    ids = self._tokenizer.EncodeAsIds(text)
    ids = [i + self._shift_reserved_tokens if i > 1 else i for i in ids]
    return ids

  def decode(self, ids: List[int]) -> str:
    ids = [
        i - self._shift_reserved_tokens
        if i > 1 + self._shift_reserved_tokens else i for i in ids
    ]
    text = self._tokenizer.DecodeIds(ids)
    if self._newline_symbol:
      text = text.replace(self._newline_symbol, "\n")
    return text
