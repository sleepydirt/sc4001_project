"""Supported entry points; no work runs without an explicit command."""
import argparse
import importlib
import sys


def main():
    modules={'workflow':'workflow','models':'prepare_models','report':'reporting','notebook':'notebook','audit':'audit','select':'selection',
             'train':'training','acquire':'acquire','evaluate':'inference'}
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=modules)
    args,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
    importlib.import_module('nano_learning.'+modules[args.command]).main()

if __name__=='__main__':main()
