export TRAIN_FILE=gpt-2-output-dataset/data/webtext.train.jsonl
export EVAL_FILE=wikitext-2-raw/wiki.test.raw

CUDA_VISIBLE_DEVICES=0 python dynamiceval_gpt2.py --do_dynamic_eval\
    --output_dir=output \
    --gradstat_data_file=$TRAIN_FILE \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --gs_steps=200 \
    --eval_data_file=$EVAL_FILE \
    --lr=2e-5 \
    --epsilon=1e-3 \
    --fix_emb_layer




