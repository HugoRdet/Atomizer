#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda



## === Then load the module and activate your env ===
conda activate venv


#sh TrainEval.sh test_Atos_lancement_scale config_test-ScaleMAE.yaml regular



MODEL_NAME=config_test-Atomiser_Atos_One.yaml

#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_1
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_2
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_3
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_4
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_5
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_6
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" test_1
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" test_2
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" test_3
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" train_1
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r1
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r2
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r3
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r4
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r5
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r6