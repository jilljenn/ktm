% Knowledge Tracing Machines\newline Towards an unification of DKT, IRT & PFA
% Jill-Jênn Vie
% Optimizing Human Learning, June 12, 2018
---
theme: Frankfurt
section-titles: false
handout: true
biblio-style: authoryear
header-includes:
    - \usepackage{booktabs}
    - \usepackage{multicol}
    - \usepackage{bm}
    - \DeclareMathOperator\logit{logit}
    - \def\ReLU{\textnormal{ReLU}}
---

# Introduction

## Predicting student performance

### Data

A population of students answering questions

### Goal

- Measure their knowledge
- Potentially optimize their learning

### Assumption

Good model for prediction $\rightarrow$ Good adaptive policy for teaching

## Learning outcomes of this tutorial

> - Logistic regression is amazing \vspace{1cm}
> - Factorization machines are even more amazing \vspace{1cm}
> - Recurrent neural networks are boring \footnotesize (but useful)

## Families of models

- Factorization Machines [@rendle2012factorization]
    - Multidimensional Item Response Theory
    - Logistic Regression
        - Item Response Theory
        - Performance Factor Analysis
- Recurrent Neural Networks
    - Deep Knowledge Tracing [@piech2015deep]

\vspace{1cm}

\fullcite{rendle2012factorization}

\fullcite{piech2015deep}

## Problems

### Weak generalization

Filling the blanks: some students did not attempt all questions

### Strong generalization

Cold-start: some new students are not in the train set

## Dummy dataset

\begin{columns}
\begin{column}{0.6\linewidth}
\begin{itemize}
\item User 1 answered Item 1 correct
\item User 1 answered Item 2 incorrect
\item User 2 answered Item 1 incorrect
\item User 2 answered Item 1 correct
\item User 2 answered Item 2 ???
\end{itemize}
\end{column}
\begin{column}{0.4\linewidth}
\centering
\input{tables/dummy-ui}
\end{column}
\end{columns}

# Logistic Regression

## Task 1: Item Response Theory

Learn abilities $\theta_i$ for each user $i$  
Learn easiness $e_j$ for each item $j$ such that:
$$ \begin{aligned}
Pr(\textnormal{User $i$ Item $j$ OK}) & = \sigma(\theta_i + e_j)\\
\logit Pr(\textnormal{User $i$ Item $j$ OK}) & = \theta_i + e_j
\end{aligned}$$

### Logistic regression

Learn $\alert{\bm{w}}$ such that $\logit Pr(\bm{x}) = \langle \alert{\bm{w}}, \bm{x} \rangle$

## Graphically: IRT as logistic regression

Encoding of "User $i$ answered Item $j$":

\centering

![](figures/lr.pdf)

$$ \logit Pr(\textnormal{User $i$ Item $j$ OK}) = \langle \bm{w}, \bm{x} \rangle = \theta_i + e_j $$

## Time to experiment

Cf. `README.md` @ `https://github.com/jilljenn/ktm`

\vspace{1cm}

    git clone https://github.com/jilljenn/ktm
    cd ktm
    python3 -m venv venv   # Python 2 OK
    . venv/bin/activate
    pip install -r requirements.txt

    git clone https://github.com/srendle/libfm
    cd libfm
    git reset --hard 91f8504a15120ef6815d6e10cc7dee42eebaab0f
    make all

## Encoding

`python encode.py --users --items`  
`# Creates data/dummy/X-ui.npz`

\centering

\input{tables/show-ui}

\raggedright
Then logistic regression can be run on the sparse features:

`python lr.py data/dummy/X-ui.npz`

## Oh, there's a problem

`python encode.py --users --items`

`python lr.py data/dummy/X-ui.npz`

\input{tables/pred-ui}

We predict the same thing when there are several attempts.

## Count successes and failures

Keep track of what the student has done before:

\centering

\input{tables/dummy-uiswf}

## Task 2: Performance Factor Analysis

$W_{ik}$: how many successes of user $i$ over skill $k$ ($F_{ik}$: #failures)

Learn $\alert{\beta_k}$, $\alert{\gamma_k}$, $\alert{\delta_k}$ for each skill $k$ such that:
$$ \logit Pr(\textnormal{User $i$ Item $j$ OK}) = \sum_{\textnormal{Skill } k \textnormal{ of Item } j} \alert{\beta_k} + W_{ik} \alert{\gamma_k} + F_{ik} \alert{\delta_k} $$

`python encode.py --skills --wins --fails`

\centering
\input{tables/show-swf}

## Better!

`python encode.py --skills --wins --fails`

`python lr.py data/dummy/X-swf.npz`

\input{tables/pred-swf}

## Task 3: a new model (but still logistic regression)

`python encode.py --items --skills --wins --fails`

`python lr.py data/dummy/X-iswf.npz`

# Factorization Machines

## Here comes a new challenger

How to model \alert{side information} in recommender systems?

### Logistic Regression

Learn a \alert{bias} for each feature (each user, item, etc.)

### Factorization Machines

Learn a \alert{bias} and an \alert{embedding} for each feature

## The power of embeddings

\centering

![](figures/embedding1.png){width=60%}

## Interpreting the components

![](figures/embedding2.png)

## Interpreting the components

![](figures/embedding3.png)

## Graphically: logistic regression

\centering

![](figures/lr.pdf)

## Graphically: factorization machines

\centering

![](figures/fm.pdf)

![](figures/fm2.pdf)

## Formally: factorization machines

Learn bias \alert{$w_k$} and embedding \alert{$\bm{v_k}$} for each feature $k$ such that:
$$ \logit p(\bm{x}) = \mu + \underbrace{\sum_{k = 1}^N \alert{w_k} x_k}_{\textnormal{logistic regression}} + \underbrace{\sum_{1 \leq k < l \leq N} x_k x_l \langle \alert{\bm{v_k}}, \alert{\bm{v_l}} \rangle}_{\textnormal{pairwise interactions}} $$


\begin{block}{Particular cases}
\begin{itemize}
\item Multidimensional item response theory: $\logit p = \langle \bm{u_i}, \bm{v_j} \rangle + e_j$
\item SPARFA: $\bm{v_j} > \bm{0}$ and $\bm{v_j}$ sparse
\item GenMA: $\bm{v_j}$ is constrained by the zeroes of a q-matrix
\end{itemize}
\end{block}

\footnotesize
\fullcite{lan2014sparse}

\fullcite{Vie2016ECTEL}

## Assistments 2009 dataset

278608 attempts of 4163 students over 196457 items on 124 skills.

- Download `http://jiji.cat/weasel2018/data.csv`
- Put it in `data/assistments09`

`python fm.py data/assistments09/X-ui.npz`  
etc. or `make big`

\vspace{1cm}

\input{tables/results}

Results obtained with FM $d = 20$

# Deep Learning

## Deep Factorization Machines

Learn layers \alert{$W^{(\ell)}$} and \alert{$b^{(\ell)}$} such that:
$$ \begin{aligned}[c]
\bm{a}^{0}(\bm{x}) & = (\alert{\bm{v_{\texttt{user}}}}, \alert{\bm{v_{\texttt{item}}}}, \alert{\bm{v_{\texttt{skill}}}}, \ldots)\\
\bm{a}^{(\ell + 1)}(\bm{x}) & = \ReLU(\alert{W^{(\ell)}} \bm{a}^{(\ell)}(\bm{x}) + \alert{\bm{b}^{(\ell)}}) \quad \ell = 0, \ldots, L - 1\\
y_{DNN}(\bm{x}) & = \ReLU(\alert{W^{(L)}} \bm{a}^{(L)}(\bm{x}) + \alert{\bm{b}^{(L)}})
\end{aligned} $$

$$ \logit p(\bm{x}) = y_{FM}(\bm{x}) + y_{DNN}(\bm{x}) $$

When trained, performance was lower than Bayesian FMs.

\fullcite{Duolingo2018}

## What 'bout recurrent neural networks?

Deep Knowledge Tracing: model the problem as sequence prediction

- Each student on skill $q_t$ has performance $a_t$
- How to predict outcomes $\bm{y}$ on every skill $k$?
- Spoiler: by measuring the evolution of a latent state $\alert{\bm{h_t}}$

## Graphically: deep knowledge tracing

\centering

![](figures/dkt1.pdf)

## Graphically: there is a MIRT in my DKT

\centering

![](figures/dkt2.pdf)

## Drawback

DKT does not model individual differences.

By estimating on-the-fly the student's learning ability, we managed to get a better model.

\centering
\input{tables/results-dkt}

\raggedright
\fullcite{Minn2018}

## Take home message

\alert{Factorization machines} are a strong baseline that take many models as special cases

\alert{Recurrent neural networks} are powerful because they track the evolution of the latent state (try simpler dynamic models?)

\alert{Deep factorization machines} may require more data/tuning, but neural collaborative filtering offer promising directions

## Any suggestions are welcome!

`vie@jill-jenn.net`

`github.com/jilljenn/ktm`

Questions?
