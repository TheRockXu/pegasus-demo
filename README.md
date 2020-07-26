
## *This is a trained model of [PEGASUS](https://github.com/google-research/pegasus) on cnn_dailymail* dataset. 



# Requirements - 
> numpy\
sentencepiece\
tensorflow==2.2.0

# PEGASUS library

Pre-training with Extracted Gap-sentences for Abstractive SUmmarization
Sequence-to-sequence models, or PEGASUS, uses self-supervised objective Gap
Sentences Generation (GSG) to train a transformer encoder-decoder model. The
paper can be found on [arXiv](https://arxiv.org/abs/1912.08777). ICML 2020 accepted.


!['screen'](https://1.bp.blogspot.com/-TSor4o51jGI/Xt50lkj6blI/AAAAAAAAGDs/TrDe9jv13WEwk9NQNebQL63jtY8n6JFGwCLcBGAsYHQ/s640/image1.gif)


If you use this code or these models, please cite the following paper:
```
@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# Usage

To run the demo, please download pre-trained model on cnn_dailymail from [here](https://drive.google.com/file/d/1FVzZto4bf5_TCmRy3tNeirhPDdLrvum5/view?usp=sharing) or gigaword from [here](https://drive.google.com/file/d/1ZF2qO6bAnsTF2LSndLMir3e7NrlFL288/view?usp=sharing). Unzip it and put it to `model/`, or anywhere
really if you just specify its location and where your article file is. Suppose your article is this [one](https://thehill.com/policy/national-security/507744-russian-hackers-return-to-spotlight-with-vaccine-research-attack)

`python test_example.py --article example_article --model_dir model/ --model_name cnn_dailymail`

You will see this output - `PREDICTION >>  The hacking group known as NC29 is largely believed to operate as part of Russia's security services .<n>The three countries allege that it is carrying out a persistent and ongoing cyber campaign to steal intellectual property about a possible coronavirus vaccine .
`

# Export Model

To export a model you have trained, please place the `ExportModel.ipynb` inside the [PEGASUS](https://github.com/google-research/pegasus) folder.
Just run the script inside by specifying which data model you want to export.
 
