# Knowledge Tracing Machines

- Presented at the [AIP-IIS-MLGT](https://sites.google.com/view/aip-fau-mlgt-2018/home) workshop at Georgia Tech, Atlanta, GA on March 8, 2018.
- Presented at the [BEA](https://www.cs.rochester.edu/~tetreaul/naacl-bea13.html) workshop at New Orleans, LA on June 5, 2018.
- To be presented at the [AAAI 2019](https://aaai.org/Conferences/AAAI-19/) conference in Honolulu, Hawaii on January 27, 2019.

See [article](https://arxiv.org/abs/1811.03388) on arXiv:

    @inproceedings{Vie2019,
      Author = {{Vie}, Jill-J{\^e}nn and {Kashima}, Hisashi},
      Booktitle = {To appear at the 33th AAAI Conference on Artificial Intelligence},
      Title = {{Knowledge Tracing Machines: Factorization Machines for Knowledge Tracing}},
      Url = {http://arxiv.org/abs/1811.03388},
      Year = 2019}

Authors: [Jill-Jênn Vie](https://jilljenn.github.io), [Hisashi Kashima](http://www.geocities.co.jp/kashi_pong/index_e.html)

## Install

    python3 -m venv venv   # Python 2 should work as well, but we suggest to you to use virtualenv
    . venv/bin/activate
    pip install -r requirements.txt

If you also want to get the factorization machines running, follow the [pywFM](https://github.com/jfloff/pywFM) (weird) suggestion:

    git clone https://github.com/srendle/libfm  # In the same ktm folder, it's better
    cd libfm
    git reset --hard 91f8504a15120ef6815d6e10cc7dee42eebaab0f
    make all

## Run

    make  # To get the encodings (npz)
    make  # To get results (txt)

You can also download the Assistments 2009 dataset in `data/assistments09` and get:

    make big

## Results

On the Assistments 2009 dataset:

| AUC time    | users + items  | skills + wins + fails | items + skills + wins + fails |
|:------------|:---------------|:----------------------|:------------------------------|
| LR          | 0.734 (IRT) 2s | 0.651 (PFA) 9s        | 0.737 23s                     |
| FM *d* = 20 | 0.730 2min9s   | 0.652 43s             | 0.739 2min30s                 |

Computation times are given for a i7 with 2.6 GHz, with 200 epochs of FM training.
