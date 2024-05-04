#!/bin/bash
conda create -n span_bert python=3.9 cuda-version=11.6
sleep 10
conda init
source ~/.bashrc
conda activate span_bert