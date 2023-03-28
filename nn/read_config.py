import pprint
import sys
import yaml

ifile = sys.argv[1]
with open(ifile) as file:
    yl = yaml.load(file, Loader=yaml.Loader)
    pprint.pprint(yl)
