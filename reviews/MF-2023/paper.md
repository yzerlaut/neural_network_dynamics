# A general mean-field formalism for networks of spiking neurons, and its application to modeling brain dynamics at large scales

_Yann Zerlaut[^1] & Alain Destexhe[^2]_

[^1]: ICM
[^2]: Neuropsi

### Summary

Population/Rate models of neural network activity are the most common building block to investigate brain dynamics at large scales. However, including the biophysical complexity characterizing the cellular and synaptic scale in such models has proven difficult. We review here a general formalism to design mean-field models filling this gap. The formalism combines a Markovian description with a semi-analytical procedure where the biophysical complexity is captured by neuronal transfer functions fitted from single cell simulations. This framework allow to design models for arbitrarily complex neuron types and synaptic dynamics, leading to biologically plausible population models that include features such as membrane conductances, synaptic receptors, or spike-frequency adaptation.  We illustrate the application of this formalism to investigate larger scales, from the mesoscopic (millimeter) scale, with the example of propagrating waves in the monkey visual cortex, up to the whole brain, with the example of the spread of stimulus-evoked activity in the human brain.

### Introduction


- The brain can be seen as an interconnected ensemble of local neural networks (Figure [@fig:motiv] a). 
- Local neural network can thus be used as a fundamental unit to model brain dynamics. Detailed modeling is tough. So more abstract modeling, a model describing the average activity of the population. See Figure \ref{fig:motiv}b,c.
- Two approaches to design such models. Top-Down vs Bottom-Up. Here Bottom-Up
- Bottom-Up means that there is a cellular and synaptic dynamics underlying the model (Figure \ref{fig:motiv}b).


![**A mean field description accounts for the average firing rate dynamics in an underlying spiking network model.** **(a)** Scale applicable to mean-field models. ~5k neurons. **(b)** Numerical simulations of the spiking network model. Synaptic and Cellular Dynamics together with the Spiking Activity in the whole network. **(c)** Population rate summary of the network dynamics (num. sim.) together with the mean-field prediction.](figures/motiv.png){#fig:motiv}

### Framework presentation

\begin{itemize}
    \item Markovian hypothesis for the dynamics. Need to set a time scale T. Should be a compromise between: small-enough so that spiking remains a binary process (no multiple spikes within T), large-enough to allow a continuous description of rates (for very small T, even only 1 spike translate into a large change of firing rate activity, $\Delta \nu = \frac{1}{N \cdot T} $.
    \item Continuous description. Originally  to be consistent with the Fokker-Planck approach at the neuronal level (EB \& D, 2009). Allows the derivation of a Master Equation.
    \item Ergodic hypothesis. The ensemble average (i.e. over neurons, what we want) is approximated by the temporal average (what we have from the numerical sim.). The formalism indeed relies on stationary transfer functions . 
    \item Includes an estimate of second order dynamics at the time scale T. Fluctuations due to finite-size effects within the window T are estimated (via Gaussian approx. of binomial population spiking within T).
\end{itemize}

\begin{table}[htb!]
    \centering
    \begin{tabular}{c|c}
         step  &  \\
         \hline \\
         1 & Analyze the spiking network model. \\ &  Find the system of ODE with population rate variables corresponding to the network. \\
         \hline \\
         2 & Perform single cell simulations to generate the numerical transfer function data for each population. \\
         \hline \\
         3 & Perform transfer function fitting. \\    
         \hline \\
         4 & Numerically simulate the system of ODEs to obtain the population rates. \\
         \hline \\
    \end{tabular}
    \caption{ \textbf{Steps to obtain the mean-field description from a given spiking network model}. \newline 
    }
    \label{tab:steps}
\end{table}

### Properties of the obtained mean-field models

$\rightarrow$ predict some of the emergent properties of the spiking network system.

\begin{itemize}
    \item Predictions of stationary dynamics. (via fixed-point analysis)
    \item Predictions of stimulus-evoked dynamics (gain modulation, etc...)
    \item Predictions of oscillatory behavior
    \item ...
\end{itemize}

\section{Limitations}

\begin{itemize}
    \item Misses fast synchronization events below this time scale.
    \item Not particularly restrictive for multi-dimensional systems. Allow oscillations if slow enough, etc...
    \item 
\end{itemize}

### Large scale modeling

\begin{itemize}
    \item Stim.-evoked activity as traveling waves in V1. Lateral interactions between stim.-evoked activity in Monkey V1
    \item 
\end{itemize}


\begin{figure}[!ht]
\begin{center}
% \includegraphics[trim=0 0 0 0, clip, width=0.8\columnwidth]{new-figs/tf-determinatino.png}
\caption{ 
\textbf{Determining the transfer function for arbitrarily complex neuronal and synaptic models}. \newline
\textbf{(a)} Singe cell simulations at various levels of population activity of presynaptic afferents of the associated network.  \newline
\textbf{(b)} Neuronal transfer function for the different neuronal types: Average firing rate response \newline
}
\label{fig:tf-determination}
\end{center}
\end{figure}

\begin{figure}[!ht]
\begin{center}
% \includegraphics[trim=0 0 0 0, clip, width=0.8\columnwidth]{new-figs/power-limit.png}
\caption{ 
\textbf{Explanatory power and limitations of the mean-field description}. \newline
\textbf{(a)} Fixed point analysis predicting the stationary dynamics of the networks in a few network settings.  \newline
\textbf{(b)} Frequency response curve in the mean-field and numerical simulations. \newline
}
\label{fig:tf-determination}
\end{center}
\end{figure}


### Discussion

[...]
