#!/bin/bash

EXPNR=000
SAVEPATH="/dccstor/saentis/log/03_rpm/"
DATAPATH="/dccstor/saentis/data/RAVEN/RAVEN-10000"
DATAPATHIRAVEN="/dccstor/saentis/data/I-RAVEN/"
EXPNAME="nvsa_test2"
EXPPATH="${SAVEPATH}/${EXPNAME}/"
SEED=1234
RUN=0

# NVSA specific backend
source /opt/share/anaconda3-2019.03/x86_64/bin/activate py37_torch111

# Center
jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_6h \
python raven/main_nvsa_marg.py --mode train --config center_single --epochs 50 --s 7 --trainable-s \
 --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN

# 2x2
jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_24h \
python raven/main_nvsa_marg.py --mode train --config distribute_four --epochs 150 --s 6 --trainable-s \
--exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN

# 3x3
jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_24h \
python raven/main_nvsa_marg.py --mode train --config distribute_nine --epochs 150 --s 2 --trainable-s --batch-size 8 \
--exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN

# Left-right
jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
python raven/main_nvsa_marg.py --mode train --config left_center_single_right_center_single --epochs 100 --s 5 --trainable-s \
--exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN

# Up-down
jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
python raven/main_nvsa_marg.py --mode train --config up_center_single_down_center_single --epochs 100 --s 5 --trainable-s \
--exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN

# Out-in center
jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
python raven/main_nvsa_marg.py --mode train --config in_center_single_out_center_single --epochs 100 --s 5 --trainable-s \
--exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN

# Out-in grid
jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
python raven/main_nvsa_marg.py --mode train --config in_center_single_out_center_single --epochs 100 --s 5 --trainable-s \
--exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN