# PairwiseRL

To install the enrivonment, please run:

```pip install -r requirements.txt```


The link of the trained ECB+ model is [here](https://drive.google.com/file/d/18_MYR6UVCzO4YmAnm5WQ2djVsZt_-3ab/view?usp=sharing). Please put the model under the "model/" folder.


To evaluate the coreference model, please run:

``python -u train_event_coref.py --task_name rte --do_eval --do_lower_case --data_dir '' --output_dir '' --eval_batch_size 80 --learning_rate 1e-6 --max_seq_length 128 --kshot 3 --beta_sampling_times 1``

To train the coreference model, please run:

``python -u train_event_coref.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --data_dir '' --output_dir '' --train_batch_size 32 --eval_batch_size 80 --learning_rate 1e-6 --max_seq_length 128 --kshot 3 --beta_sampling_times 1``


After you get the pairwise coreference score between any two event mentions, to do clustering, please run:

``python clustering.py``
