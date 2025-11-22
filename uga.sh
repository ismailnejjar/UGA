#!/usr/bin/env bash
python uga.py --gpu_id 0 --src s --tgt n --cmixup 0 --uncertainty_alignment feature | tee ./feature_cmix_s2n.log
python uga.py --gpu_id 0 --src s --tgt c --cmixup 0 --uncertainty_alignment feature| tee ./feature_cmix_s2c.log
python uga.py --gpu_id 0 --src n --tgt c --cmixup 0 --uncertainty_alignment feature| tee ./feature_cmix_n2c.log
python uga.py --gpu_id 0 --src n --tgt s --cmixup 0 --uncertainty_alignment feature| tee ./feature_cmix_n2s.log
python uga.py --gpu_id 0 --src c --tgt s --cmixup 0 --uncertainty_alignment feature| tee ./feature_cmix_c2s.log
python uga.py --gpu_id 0 --src c --tgt n --cmixup 0 --uncertainty_alignment feature| tee ./feature_cmix_c2n.log
