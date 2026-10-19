"""Reproduce the audited workbook, all analyses, and the standalone report."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))

import audit,clean,kalshi,analyze,kelly,robustness,strategy_tradeoffs,report,verify

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    source=parser.parse_args().source
    audit.audit(source)
    clean.clean(source)
    for stage in (kalshi,analyze,kelly,robustness,strategy_tradeoffs,report):
        print('Running',stage.__name__,flush=True)
        stage.run()
    verify.run(source)
