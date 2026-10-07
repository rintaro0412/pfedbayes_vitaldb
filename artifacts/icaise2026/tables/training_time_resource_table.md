# Training Time and Resource Table

| Method | Training time (min) | Type | Peak CPU RSS tree (MB) | Peak GPU memory (MB) |
|---|---:|---|---:|---:|
| Centralized | 65.2 | measured | 342110.5 | 518.0 |
| Local | 72.0 | measured | 362690.9 | 550.0 |
| FedAvg | 185.8 | measured | 60266.4 | 378.0 |
| FedProx | 228.9 | measured | 51058.6 | 376.0 |
| SCAFFOLD | 187.3 | measured | 52453.4 | 378.0 |
| FedNova | 191.3 | measured | 58454.3 | 376.0 |
| Per-FedAvg | 239.0 | measured | 56889.9 | 526.0 |
| pFedMe | 478.8 | measured | 54694.1 | 524.0 |
| pFedBayes | 2160.0 | projected | 97455.4 | 1918.0 |

Note: CPU memory is reported as the summed resident set size over the process tree, so it should be described as "peak process-tree RSS" rather than physical RAM usage. The pFedBayes runtime is a 200-round projection from the first 37 completed rounds.

LaTeX-ready compact version:

```latex
\begin{table}[t]
\centering
\caption{Training time and peak resource usage. CPU memory is summed process-tree RSS. The pFedBayes runtime is projected from the first 37 completed rounds.}
\label{tab:training_resource}
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{lrrr}
\toprule
Method & Time (min) & Peak CPU RSS (MB) & Peak GPU mem. (MB) \\
\midrule
Centralized & 65.2 & 342110.5 & 518.0 \\
Local & 72.0 & 362690.9 & 550.0 \\
FedAvg & 185.8 & 60266.4 & 378.0 \\
FedProx & 228.9 & 51058.6 & 376.0 \\
SCAFFOLD & 187.3 & 52453.4 & 378.0 \\
FedNova & 191.3 & 58454.3 & 376.0 \\
Per-FedAvg & 239.0 & 56889.9 & 526.0 \\
pFedMe & 478.8 & 54694.1 & 524.0 \\
pFedBayes$^\dagger$ & 2160.0 & 97455.4 & 1918.0 \\
\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{$^\dagger$Projected 200-round runtime from the first 37 completed rounds.}
\end{table}
```
