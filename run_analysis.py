"""Reproduce the audited workbook, all analyses, and the standalone report."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))

import audit,clean,kalshi,analyze,kelly,robustness,strategy_tradeoffs,timing_audit,interval_trajectories,interval_report,report,verify,verify_intervals
import prospective,calibration_uncertainty,sequential_evidence,prospective_report,verify_prospective
import lower_confidence,lower_confidence_report,verify_lower_confidence
import aggressive_growth,aggressive_growth_report,verify_aggressive_growth
import sequential_kelly,sequential_kelly_report,verify_sequential_kelly
import trace_sequential_drawdown
import capped_kelly,capped_kelly_report,verify_capped_kelly
import confidence23,confidence23_report,verify_confidence23
import confidence_signal
import shrinkage_backtest,shrinkage_report,verify_shrinkage

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    source=parser.parse_args().source
    audit.audit(source)
    clean.clean(source)
    for stage in (kalshi,analyze,kelly,robustness,strategy_tradeoffs,timing_audit,interval_trajectories,
                  prospective,calibration_uncertainty,sequential_evidence,lower_confidence,lower_confidence_report,
                  aggressive_growth,aggressive_growth_report,sequential_kelly,trace_sequential_drawdown,sequential_kelly_report,
                  capped_kelly,capped_kelly_report,confidence23,confidence23_report,confidence_signal,
                  shrinkage_backtest,shrinkage_report,
                  prospective_report,interval_report,report):
        print('Running',stage.__name__,flush=True)
        stage.run()
    verify.run(source)
    verify_intervals.run()
    verify_prospective.run()
    verify_lower_confidence.run()
    verify_aggressive_growth.run()
    verify_sequential_kelly.run()
    verify_capped_kelly.run()
    verify_confidence23.run()
    verify_shrinkage.run()
