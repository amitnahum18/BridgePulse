import argparse
import json
from .pipeline import BridgePredictor, DEFAULT_MODEL, write_result

def main():
    parser=argparse.ArgumentParser(description='BridgePulse recording-level damage classification')
    parser.add_argument('--h5',required=True,help='Raw H5 path, opened read-only')
    parser.add_argument('--recording-path',required=True,help='Group containing sensor groups, e.g. baseline_1/acceleration/test_1')
    parser.add_argument('--model',default=str(DEFAULT_MODEL))
    parser.add_argument('--output',help='Optional JSON result path')
    args=parser.parse_args()
    try:result=BridgePredictor(args.model).predict_h5(args.h5,args.recording_path)
    except (ValueError,OSError,KeyError) as error:parser.exit(2,f'Prediction failed: {error}\n')
    if args.output:write_result(result,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

