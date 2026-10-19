"""Reproduce the audited workbook, all analyses, and the standalone report."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))

import audit,clean,kalshi,analyze,kelly,robustness,strategy_tradeoffs,timing_audit,interval_trajectories,interval_report,report,verify,verify_intervals

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    source=parser.parse_args().source
    audit.audit(source)
    clean.clean(source)
    for stage in (kalshi,analyze,kelly,robustness,strategy_tradeoffs,timing_audit,interval_trajectories,interval_report,report):
        print('Running',stage.__name__,flush=True)
        stage.run()
    verify.run(source)
    verify_intervals.run()
