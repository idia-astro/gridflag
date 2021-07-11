#Copyright (C) 2020 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

import os
import sys

import config_parser
from config_parser import validate_args as va
import bookkeeping

from recipes.setOrder import setToCasaOrder

import logging
from time import gmtime

logging.Formatter.converter = gmtime
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)

def do_reorder(visname, fields, dirs='*MHz'):

    if visname[-1]=='/':
        visname = visname[:-1]

    basename, ext = os.path.splitext(visname)

    newvis = basename + '_reorder' + ext

    setToCasaOrder(visname, newvis)

    return newvis

def main(args,taskvals):

    visname = va(taskvals, 'data', 'vis', str)

    newvis = do_reorder(visname)

    config_parser.overwrite_config(args['config'], conf_dict={'vis' : "'{0}'".format(newvis)}, conf_sec='data')

if __name__ == '__main__':

    bookkeeping.run_script(main)