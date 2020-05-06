[![Build Status](https://travis-ci.org/jilljenn/ktm.svg?branch=master)](https://travis-ci.org/jilljenn/ktm)
[![Codecov](https://img.shields.io/codecov/c/github/jilljenn/ktm.svg)](https://codecov.io/gh/jilljenn/ktm/)

# Knowledge Tracing Machines

- Presented at the AAAI 2019 conference in Honolulu, Hawaii on January 27, 2019.
- Applied in the [Best Paper Award](https://arxiv.org/abs/1905.06873) of the EDM 2019 conference in Montreal, Canada on July 2, 2019.

See our article: [Knowledge Tracing Machines: Factorization Machines for Knowledge Tracing [pdf]](https://arxiv.org/abs/1811.03388) [[slides]](http://jiji.cat/slides/aaai2019-ktm-slides.pdf).  
Comments are always welcome!

    @inproceedings{Vie2019,
      Author = {{Vie}, Jill-J{\^e}nn and {Kashima}, Hisashi},
      Booktitle = {Proceedings of the 33th {AAAI} Conference on Artificial Intelligence},
      Title = {{Knowledge Tracing Machines: Factorization Machines for Knowledge Tracing}},
      Pages = {750--757},
      Url = {https://arxiv.org/abs/1811.03388},
      Year = 2019}

Authors: [Jill-Jênn Vie](https://jilljenn.github.io), [Hisashi Kashima](https://hkashima.github.io/index_e.html)

## Follow our tutorial

Presented at the [Optimizing Human Learning](https://humanlearn.io) workshop in Kingston, Jamaica on June 4, 2019.

Slides from the tutorial are available [here](doc/tuto.pdf). A notebook on Colab will be available "soon", but the priority is to have tests in this repository.

The tutorial makes you play with the models to assess **weak generalization**. To assess **strong generalization** and reproduce the experiments of the paper, you want to look at how folds are created in [dataio.py](https://github.com/jilljenn/ktm/blob/master/dataio.py#L12).

## Install

    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt  # Will install numpy, scipy, pandas, scikit-learn, pywFM

If you also want to get the factorization machines running (KTM for *d* > 0), you should also do:

    make libfm

## Prepare data

Select a dataset and the features you want to include.

### Case 1: There is only one skill per item.

`data/<dataset>/data.csv` should contain the following columns:

    user, item, skill, correct, wins, fails

where wins and fails are the number of successful and unsuccessful
attempts at the corresponding skill.

### Case 2: There may be several skills associated to an item.

`data/<dataset>/needed.csv` needs to contain:

    user_id, item_id, correct

(Note the difference.)

And `data/<dataset>/q_mat.npz` should be a q-matrix under `scipy.sparse` format.

If you want to compute wins and fails like in PFA or DAS3H,
you should run `encode_tw.py` instead of this file, with the `--pfa` option for PFA or `--tw` for DAS3H.

## Run

### Encoding data into sparse features (quick start)

    python encode.py --users --items  # To get the encodings (npz)
    python lr.py data/dummy/X-ui.npz  # To get results (txt)

You can also download the [Assistments 2009 dataset](https://jiji.cat/weasel2018/data.csv) into `data/assistments09` and change the dataset:

    python encode.py --dataset assistments09 --skills --wins --fails  # Will encode PFA sparse features into X-swf.npz

If you are lazy, you can also just do `make` and try to understand what is going on in the [Makefile](Makefile).

### Encoding time windows

Choffin et al. proposed the DAS3H model, and we implemented it using queues. This code is faster than the original KTM encoding.

To prepare a dataset like Assistments, see examples in the `data` folder.  
Skill information should be available either as `skill_id`, or `skill_ids` separated with `~~`, or in a q-matrix `q_mat.npz`.

    python encode_tw.py --dataset dummy_tw --tw  # Will encode DAS3H sparse features into X.npz

Then you can run `lr.py` or `fm.py`, see below.

### Running a ML model

If you want to encode PFA features:

    python encode.py --skills --wins --fails  # Will create X-swf.npz

For logistic regression:

    python lr.py data/dummy/X-swf.npz
	# Will save weights in coef0.npy

For factorization machines of size *d* = 5:

    python fm.py --d 5 data/dummy/X-swf.npz
	# Will save weights in w.npy and V.npy

NEW! For an online MIRT model:

    python omirt.py --d 0 data/assist09/needed.csv  # Will load LR: coef0.npy
	python omirt.py --d 5 data/assist09/needed.csv  # Will load FM: w.npy and V.npy

	# Will train a IRT model on Fraction dataset with learning rate 0.01
	python omirt.py --d 0 data/fraction/needed.csv --lr 0.01 --lr2 0.

NEW! For an IRT or deeper model with Keras, for batching and early stopping:

    python dmirt.py data/assist09/needed.csv

It will also create a model.png file with the architecture (here just IRT with L2 regularization):

![](model.png)

## Results

### Weak generalization

On the Assistments 2009 dataset:

| AUC time    | users + items  | skills + wins + fails | items + skills + wins + fails |
|:------------|:---------------|:----------------------|:------------------------------|
| LR          | **0.734** (IRT) 2s | 0.651 (PFA) 9s        | 0.737 23s                     |
| FM *d* = 20 | 0.730 2min9s   | **0.652** 43s             | **0.739** 2min30s                 |

Computation times are given for a i7 with 2.6 GHz, with 200 epochs of FM training.

### Strong generalization

On the Assistments 2009 dataset:

| Model | Dimension | AUC | Improvement |
|:-----:|:---------:|:---:|:-----------:|
| KTM: items, skills, wins, fails, extra | 5 | **0.819** | |
| KTM: items, skills, wins, fails, extra | 5 | 0.815 | +0.05 |
| KTM: items, skills, wins, fails | 10 | 0.767 | |
| KTM: items, skills, wins, fails | 0 | 0.759 | +0.02 |
| (*DKT* (Wilson et al., 2016)) | 100 | 0.743 | +0.05 |
| IRT: users, items | 0 | 0.691 | |
| PFA: skills, wins, fails | 0 | 0.685 | +0.07 |
| AFM: skills, attempts | 0 | 0.616 | |

On the [Duolingo](http://sharedtask.duolingo.com) French dataset:

| Model | Dimension | AUC | Improvement |
|:-----:|:---------:|:---:|:-----------:|
| KTM   | 20        | **0.822** | +0.01 |
| DeepFM | 20       | 0.814 | +0.04 |
| Logistic regression + L2 reg | 0 | 0.771 |

We also showed that Knowledge Tracing Machines (Bayesian FMs) got better results than Deep Factorization Machines on the [Duolingo dataset](http://sharedtask.duolingo.com). See our article: [Deep Factorization Machines for Knowledge Tracing](https://arxiv.org/abs/1805.00356) and [poster](https://github.com/jilljenn/ktm/blob/master/poster/dfm-kt-poster.pdf) at the [BEA](https://www.cs.rochester.edu/~tetreaul/naacl-bea13.html) workshop at New Orleans, LA on June 5, 2018.

    @inproceedings{Vie2018,
      Author = {{Vie}, Jill-J{\^e}nn},
      Booktitle = {{Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications}},
      Pages = {370--373},
      Title = {{Deep Factorization Machines for Knowledge Tracing}},
      Url = {http://arxiv.org/abs/1805.00356},
      Year = 2018}
