import argparse
from ml_functions import get_argparser, savedmodel_default
import os
import pdb
from train_stormrpts_dnn import make_fhr_str

cmds = open("cmds.txt","r").readlines()

fhr = list(range(1,49))
nfits = 5
for i in range(nfits):
    c = f"cmds_{i}.txt"
    if os.path.exists(c):
        os.remove(c)

for icmd, cmd in enumerate(cmds):
    words = cmd.split()
    parser = get_argparser()
    args = parser.parse_args()
    args.glm = True
    args.neurons = [int(words[2])]
    args.layers = int(words[4])
    args.optimizer = words[6]
    args.learning_rate = float(words[8])
    args.dropout = float(words[10])
    args.reg_penalty = float(words[12])
    args.epochs = int(words[14])
    if words[-1] == "--batchnorm": args.batchnorm=True
    fhr = list(range(1,49))
    kfold = args.kfold
    savedmodel = savedmodel_default(args, make_fhr_str(fhr) )
    scores = f"nn/nn_{savedmodel}.{kfold}fold.scores.txt"
    if not os.path.exists(scores): print(scores)
    for i in range(nfits):
        missing = False
        for ifold in range(kfold):
            model_i = f"nn/nn_{savedmodel}_{i}/{kfold}fold{ifold}"
            if not os.path.exists(model_i) or not os.path.exists(f"{model_i}/config.yaml"): missing = True
        if missing:
            f = open(f"cmds_{i}.txt", "a")
            f.write(cmd)
            f.close()

