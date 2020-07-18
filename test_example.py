import text_eval
import public_parsing_ops
import tensorflow as tf
import numpy as np

_SPM_VOCAB = 'ckpt/c4.unigram.newline.10pct.96000.model'
encoder = public_parsing_ops.create_text_encoder("sentencepiece",
                                                 _SPM_VOCAB)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--article", help="path of your example article", default="example_article")
    parser.add_argument("--model_dir", help="path of your model directory", default="model/")
    args = parser.parse_args()


    text = open(args.article, "r", encoding="utf-8").read()
    input_ids = encoder.encode(text)
    inputs = np.zeros(1024)
    idx = len(input_ids)
    if idx>1024: idx =1024
    inputs[:idx] = input_ids[:idx]
    imported = tf.saved_model.load('model/', tags='serve')
    example = tf.train.Example()
    example.features.feature["inputs"].int64_list.value.extend(inputs.astype(int))
    output = imported.signatures['serving_default'](examples=tf.constant([example.SerializeToString()]))

    print("\nPREDICTION >> ", text_eval.ids2str(encoder, output['outputs'].numpy(), None))