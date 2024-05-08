#!/bin/bash
for (( i=0; i < 10; i++))
do
scp -P 2222 kiminnibaev_1@cluster.hpc.hse.ru:/home/kiminnibaev_1/centers_lorenz-for10k$i.npy /home/karim/PycharmProjects/Time-Series/lorentz_research/centers_by_eps/
sshpass -p 00^2mCJa
done

