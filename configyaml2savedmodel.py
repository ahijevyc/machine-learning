import ml_functions
import sys
import yaml
ifile = sys.argv[1]

with open(ifile, "rb") as stream:
    yl = yaml.load(stream, Loader=yaml.Loader)
    args = yl["args"]
    savedmodel = ml_functions.get_savedmodel_path(args)
    print(savedmodel)
