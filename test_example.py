import text_eval
import public_parsing_ops
import tensorflow as tf
import numpy as np

_SPM_VOCAB = 'ckpt/c4.unigram.newline.10pct.96000.model'
encoder = public_parsing_ops.create_text_encoder("sentencepiece",
                                                 _SPM_VOCAB)
shapes = {
    'cnn_dailymail': (1024, 128),
    'gigaword':(128, 32)
}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--article", help="path of your example article", default="example_article")
    parser.add_argument("--model_dir", help="path of your model directory", default="model/")
    parser.add_argument("--model_name", help="path of your model directory", default="cnn_dailymail")
    args = parser.parse_args()

    text = open(args.article, "r", encoding="utf-8").read()

    shape,_ = shapes[args.model_name]

    input_ids = encoder.encode(text)
    inputs = np.zeros(shape)
    idx = len(input_ids)
    if idx>shape: idx =shape

    inputs[:idx] = input_ids[:idx]
    imported = tf.saved_model.load(args.model_dir, tags='serve')
    example = tf.train.Example()
    example.features.feature["inputs"].int64_list.value.extend(inputs.astype(int))
    output = imported.signatures['serving_default'](examples=tf.constant([example.SerializeToString()]))

    print("\nPREDICTION >> ", text_eval.ids2str(encoder, output['outputs'].numpy(), None))