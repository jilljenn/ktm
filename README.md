# Knowledge Tracing Machines

- To be presented at the [AAAI 2019](https://aaai.org/Conferences/AAAI-19/) conference in Honolulu, Hawaii on January 27, 2019.
- Early results were presented as a [poster](https://github.com/jilljenn/ktm/blob/master/poster/ktm-poster.pdf) at the [AIP-IIS-MLGT](https://sites.google.com/view/aip-fau-mlgt-2018/home) workshop at Georgia Tech, Atlanta, GA on March 8, 2018.

See our article: [Knowledge Tracing Machines: Factorization Machines for Knowledge Tracing](https://arxiv.org/abs/1811.03388).

Check our slides: [PDF](http://jiji.cat/slides/aaai2019-ktm-slides.pdf) (comments are welcome!).

    @inproceedings{Vie2019,
      Author = {{Vie}, Jill-J{\^e}nn and {Kashima}, Hisashi},
      Booktitle = {Proceedings of the 33th {AAAI} Conference on Artificial Intelligence},
      Title = {{Knowledge Tracing Machines: Factorization Machines for Knowledge Tracing}},
      Pages = {to appear},
      Url = {http://arxiv.org/abs/1811.03388},
      Year = 2019}

Authors: [Jill-Jênn Vie](https://jilljenn.github.io), [Hisashi Kashima](http://www.geocities.co.jp/kashi_pong/index_e.html)

## Follow our tutorial

Presented at the [Optimizing Human Learning](https://humanlearn.io) workshop in Montreal, CA on June 12, 2018.

Slides from the tutorial are available [here](doc/tuto.pdf). A notebook will be available soon (please complain in the issues to speed up the process).

The tutorial makes you play with the models to assess **weak generalization**. To assess **strong generalization** and reproduce the experiments of the paper, you may want to borrow code from another repository: [jilljenn/TF-recomm](https://github.com/jilljenn/TF-recomm/blob/master/fm.py#L106).

## Install

    python3 -m venv venv   # Python 2 should work as well, but we suggest you to use virtualenv
    . venv/bin/activate
    pip install -r requirements.txt  # Will install numpy, scipy, pandas, scikit-learn, pywFM

If you also want to get the factorization machines running (KTM for *d* > 0), you should also do:

    make libfm

## Run

    make  # To get the encodings (npz)
    make  # To get results (txt)

You can also download the [Assistments 2009 dataset](http://jiji.cat/weasel2018/data.csv) into `data/assistments09` and do:

    make big

To understand what is going on, look at the [Makefile](Makefile). Basically there are two steps:

### Encoding data into sparse features

    python encode.py --dataset dummy --skills --wins --fails  # Will encode PFA sparse features into X-swf.npz

### Running a ML model

For logistic regression:

    python lr.py data/dummy/X-swf.npz

For factorization machines of size *d* = 5:

    python fm.py --d 5 data/dummy/X-swf.npz

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
