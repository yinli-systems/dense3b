"""Use the documented EvalPlus overlay; sandbox only untrusted test children."""
import argparse,json,os,sys,tempfile
from pathlib import Path
from isolation import restrict

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['humaneval','mbpp']);p.add_argument('--samples');p.add_argument('--workers',type=int,default=4);p.add_argument('--probe',action='store_true');a=p.parse_args()
 if a.probe:
  info=restrict();print(json.dumps(info));return
 import evalplus.eval as backend
 import evalplus.eval.utils as utils
 original=backend.reliability_guard
 def guarded(*args,**kwargs):
  restrict()
  return original(*args,**kwargs)
 backend.reliability_guard=guarded;utils.reliability_guard=guarded
 from evalplus.evaluate import evaluate
 evaluate(dataset=a.dataset,samples=a.samples,parallel=a.workers,test_details=True)
if __name__=='__main__':main()
