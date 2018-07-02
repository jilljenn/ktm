% Knowledge Tracing Machines:\newline Families of models\newline for predicting student performance
% Jill-Jênn Vie\newline RIKEN Center for Advanced Intelligence Project, Tokyo\newline\newline\includegraphics[width=3cm]{figures/aip.png}
% Optimizing Human Learning, June 12, 2018\newline Polytechnique Montréal, June 18, 2018
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
biblatexoptions:
    - maxbibnames=99
    - maxcitenames=5
---

# Introduction

## Predicting student performance

### Data

A population of students answering questions

- Events: "Student $i$ answered question $j$ correctly/incorrectly"

### Goal

- Learn the difficulty of questions automatically from data
- Measure the knowledge of students
- Potentially optimize their learning

### Assumption

Good model for prediction $\rightarrow$ Good adaptive policy for teaching

## Learning outcomes of this tutorial

- \alert{Logistic regression} is amazing
    - Unidimensional
    - Takes IRT, PFA as special cases\vspace{1cm}
- \alert{Factorization machines} are even more amazing
    - Multidimensional
    - Take MIRT as special case\vspace{1cm}
- It makes sense to consider \alert{deep neural networks}
    - What does deep knowledge tracing model exactly?

## Families of models

- Factorization Machines [@rendle2012factorization]
    - Multidimensional Item Response Theory
    - Logistic Regression
        - Item Response Theory
        - Performance Factor Analysis
- Recurrent Neural Networks
    - Deep Knowledge Tracing [@piech2015deep]

\vspace{5mm}

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
\input{tables/dummy-ui}\vspace{5mm}

\texttt{dummy.csv}
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

Usually with L2 regularization: ${||\bm{w}||}_2^2$ penalty $\leftrightarrow$ Gaussian prior

## Graphically: IRT as logistic regression

Encoding of "User $i$ answered Item $j$":

\centering

![](figures/lr.pdf)

$$ \logit Pr(\textnormal{User $i$ Item $j$ OK}) = \langle \bm{w}, \bm{x} \rangle = \theta_i + e_j $$

## Encoding

`python encode.py --users --items`  

\centering

\input{tables/show-ui}

`data/dummy/X-ui.npz`

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

`data/dummy/data.csv`

## Task 2: Performance Factor Analysis

$W_{ik}$: how many successes of user $i$ over skill $k$ ($F_{ik}$: #failures)

Learn $\alert{\beta_k}$, $\alert{\gamma_k}$, $\alert{\delta_k}$ for each skill $k$ such that:
$$ \logit Pr(\textnormal{User $i$ Item $j$ OK}) = \sum_{\textnormal{Skill } k \textnormal{ of Item } j} \alert{\beta_k} + W_{ik} \alert{\gamma_k} + F_{ik} \alert{\delta_k} $$

`python encode.py --skills --wins --fails`

\centering
\input{tables/show-swf}

`data/dummy/X-swf.npz`

## Better!

`python encode.py --skills --wins --fails`

`python lr.py data/dummy/X-swf.npz`

\input{tables/pred-swf}

## Task 3: a new model (but still logistic regression)

`python encode.py --items --skills --wins --fails`

`python lr.py data/dummy/X-iswf.npz`

# Factorization Machines

## Here comes a new challenger

How to model \alert{side information} in, say, recommender systems?

### Logistic Regression

Learn a \alert{bias} for each feature (each user, item, etc.)

### Factorization Machines

Learn a \alert{bias} and an \alert{embedding} for each feature

## What can be done with embeddings?

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
\item GenMA: $\bm{v_j}$ is constrained by the zeroes of a q-matrix $(q_{ij})_{i, j}$
\end{itemize}
\end{block}

\footnotesize
\fullcite{lan2014sparse}

\fullcite{Vie2016ECTEL}

## Tradeoff expressiveness/interpretability

\centering

\begin{tabular}{@{}ccccc@{}} \toprule
NLL & $\logit p$ & 4 q & 7 q & 10 q\\ \midrule
Rasch & $\theta_i + e_j$ & 0.469 (79\%) & 0.457 (79\%) & 0.446 (79\%)\\
DINA & $1 - s_j$ or $g_j$ & 0.441 (80\%) & 0.410 (82\%) & 0.406 (82\%)\\
MIRT & $\langle \bm{u_i}, \bm{v_j} \rangle + e_j$ & 0.368 (83\%) & 0.325 (86\%) & 0.316 (86\%)\\
GenMA & $\langle \bm{u_i}, \bm{\tilde{q}_j} \rangle + e_j$ & 0.459 (79\%) & 0.355 (85\%) & 0.294 (88\%)\\ \bottomrule
\end{tabular}

<!-- ![](figures/fraction-mean.pdf){width=50%}
![](figures/timss-mean.pdf){width=50%} -->

\includegraphics[width=0.53\linewidth]{figures/fraction-mean.pdf}\includegraphics[width=0.53\linewidth]{figures/timss-mean.pdf}

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

<!-- When trained, performance was lower than Bayesian FMs. -->

\fullcite{Duolingo2018}

## Comparison

- FM: $y_{FM}$ factorization machine with $\lambda = 0.01$
- Deep: $y_{DNN}$: multilayer perceptron
- DeepFM: $y_{DNN} + y_{FM}$ with shared embedding
- Bayesian FM: $\alert{w_k}, \alert{v_{kf}} \sim \mathcal{N}(\alert{\mu_f}, 1/\alert{\lambda_f})$  
$\alert{\mu_f} \sim \mathcal{N}(0, 1)$, $\alert{\lambda_f} \sim \Gamma(1, 1)$ (trained using Gibbs sampling)

### Various types of side information

- first: `<discrete>` (`user`, `token`, `countries`, etc.)
- last: `<discrete>` + `<continuous>` (`time` + `days`)
- pfa: `<discrete>` + `wins` + `fails`

## Duolingo dataset

![](figures/duolingo.png)

![](figures/duolingo2.png)

## Results

\begin{tabular}{cccccccc}
\toprule
  Model &  $d$ &     epoch &  train &  first &   last &    pfa \\
\midrule
Bayesian FM &   20 &   500/500 &     -- &  0.822 &     -- &     -- \\
Bayesian FM &   20 &   500/500 &     -- &     -- &  0.817 &     -- \\
     DeepFM &   20 &   15/1000 &  0.872 &  0.814 &     -- &     -- \\
Bayesian FM &   20 &   100/100 &     -- &     -- &  0.813 &     -- \\
         FM &   20 &   20/1000 &  0.874 &  0.811 &     -- &     -- \\
Bayesian FM &   20 &   500/500 &     -- &     -- &     -- &  0.806 \\
         FM &   20 &   21/1000 &  0.884 &     -- &     -- &  0.805 \\
         FM &   20 &   37/1000 &  0.885 &     -- &    0.8 &     -- \\
     DeepFM &   20 &   77/1000 &   0.89 &     -- &  0.792 &     -- \\
       Deep &   20 &    7/1000 &  0.826 &  0.791 &     -- &     -- \\
       Deep &   20 &  321/1000 &  0.826 &     -- &   0.79 &     -- \\
         LR &    0 &     50/50 &     -- &     -- &     -- &  0.789 \\
         LR &    0 &     50/50 &     -- &  0.783 &     -- &     -- \\
         LR &    0 &     50/50 &     -- &     -- &  0.783 &     -- \\
\bottomrule
\end{tabular}

## Duolingo ranking

\centering

\begin{tabular}{cccc} \toprule
Rank & Team & Algo & AUC\\ \midrule
1 & SanaLabs & RNN + GBDT & .857\\
2 & singsound & RNN & .854\\
2 & NYU & GBDT & .854\\
4 & CECL & LR + L1 (13M feat.) & .843\\
5 & TMU & RNN & .839\\ \midrule
7 (off) & JJV & Bayesian FM & .822\\
8 (off) & JJV & DeepFM & .814\\
10 & JJV & DeepFM & .809\\ \midrule
15 & Duolingo & LR & .771\\ \bottomrule
\end{tabular}

\raggedright
\small
\fullcite{Settles2018}

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

## Drawback of Deep Knowledge Tracing

DKT does not model individual differences.

Actually, Wilson even managed to beat DKT with (1-dim!) IRT.

By estimating on-the-fly the student's learning ability, we managed to get a better model.

\centering
\input{tables/results-dkt}

\raggedright \small
\fullcite{Minn2018}

# Conclusion

## Take home message

\alert{Factorization machines} are a strong baseline that take many models as special cases

\alert{Recurrent neural networks} are powerful because they track the evolution of the latent state (try simpler dynamic models?)

\alert{Deep factorization machines} may require more data/tuning, but neural collaborative filtering offer promising directions

## Any suggestions are welcome!

Feel free to chat:

\centering
`vie@jill-jenn.net`

\raggedright
All code:

\centering
`github.com/jilljenn/ktm`

\raggedright
Do you have any questions?
