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

"""Library for evaluating generative text models."""


import numpy as np



def ids2str(encoder, ids, num_reserved):
  """Decode ids."""
  if num_reserved:
    eos = np.where(ids == 1)[0]
    if np.any(eos):
      ids = ids[:eos[0]]
    reserved_tokens = np.where(ids < num_reserved)[0]
    if reserved_tokens.size:
      split_locations = np.union1d(reserved_tokens, reserved_tokens + 1)
      ids_list = np.split(ids, split_locations)
      text_list = [
          "<%d>" %
          i if len(i) == 1 and i < num_reserved else encoder.decode(i.tolist())
          for i in ids_list
      ]
      return " ".join(text_list)
  return encoder.decode(ids.flatten().tolist())

