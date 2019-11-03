#### Dynamic evaluation for out-of-domain language modeling with GPT-2

This code tests [dynamic evaluation](http://proceedings.mlr.press/v80/krause18a.html)'s ability to handle out-of-domain language modeling. Most language modeling benchmarks assume the training and test data come from the same distribution, where dynamic evaluation can already give large advantages on long sequences by exploiting locally re-occurring patterns in language. Dynamic evaluation can be especially useful for situations when the test data comes from a different distribution from the training data. The GPT-2 language model was trained on the WebText data set, and was evaluated on language modeling benchmarks from domains it had never seen before in [Open AI's work](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). This code can be used to apply dynamic evaluation to GPT-2 to obtain better results on out-of-domain language modeling tasks.

#### Requirements: [HuggingFace Transformers repository](https://github.com/huggingface/transformers) and all of its dependencies. 


#### Instructions for use:  

1. Download a data set to be used for evaluation (For instance [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)--use the raw character level version with this code). Set the `EVAL_FILE` in `run_dynamiceval.sh` to the path of the test set file.

2. Download a data set to be used for gradient statistics. Since I aimed to benchmark out-of-domain language modeling, I avoided using in-domain data for this. I used a subset of WebText that can be downloaded with from Open AI's [GPT-2 output data set repository](https://github.com/openai/gpt-2-output-dataset). Set the `TRAIN_FILE` in `run_dynamiceval.sh` to the path of the file to be used for gradient statistics.

3. Run dynamic evaluation with: `bash run_dynamiceval.sh`


#### Other Notes:

*  GPT-2 117M gets a perplexity of 24.5 on WikiText-2 with dynamic eval vs. 29.0 for static eval (set lr to 0) 

* Hyper parameters are tuned for GPT-2 117M, and may need to be re-tuned for other models. I used the WebText validation set (from the [same repository](https://github.com/openai/gpt-2-output-dataset) mentioned above) for hyper parameter tuning to avoid giving the model access to any in domain data.

* Decaying adapted parameters towards parameters learned during training was used in the original dynamic eval paper, but does not seem to help in an out-of-domain setting and is not included.
