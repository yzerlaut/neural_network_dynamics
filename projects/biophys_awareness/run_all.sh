<<<<<<< HEAD
Nsim=2
TSTOP=5000
=======
Nsim=3
TSTOP=500000
>>>>>>> 473f5ea89438be766bd01fc1c8faf9080b7fe422
# RS-FS configuration
# python contrib_of_single_spike.py -f data/active.npz --tstop $TSTOP --nsim $Nsim
# SR configuration
# python contrib_of_single_spike.py -f data/RS.npz --Qi 0.5 --tstop $TSTOP --nsim $Nsim
# Vogels-Abbott configuration
python contrib_of_single_spike.py -f data/VA.npz --NRNe LIF_Vreset_-60 --NRNi LIF_Vreset_-60 --NTWK Vogels-Abbott --fext_stat 0. --tstop $TSTOP --nsim $Nsim --fext_kick 4. --kick_length 50
