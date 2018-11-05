# -*- coding: utf-8 -*-

import re
import argparse
import json

def report2json(dir_report, model_id, model_info, training_info, tag, target_dir):

	assert tag in ["gold", "pred","both"]

	report = {"model_id": model_id, 
			  "model_info": model_info, 
			  "training_info":training_info,
			  "UAS":{}, "LAS":{}}
	tag_seen = []
	with open(dir_report,"r") as f:
		for row in f:
			if row.startswith("SCORE"):
				print("DEBUG --> ", row)
				match = re.search("SCORE ON ([^\s]*) ([^\s]*)", row)
				report["test_data"] = match.group(1)
				report[match.group(1)] = {"UAS":{},"LAS":{}}
				tag = match.group(2)
				tag_seen.append(tag)
			elif row.startswith("  Unlabeled attachment score:"):
				
				report[match.group(1)]["UAS"][tag] = re.search(".*  Unlabeled attachment score:.*= (.*) %$", row).group(1)
				# to be remove (redundant) : 
				report["UAS"][tag] = re.search(".*  Unlabeled attachment score:.*= (.*) %$", row).group(1)
			elif row.startswith("  Labeled   attachment score:"):
				report[match.group(1)]["LAS"][tag] = re.search(".*  Labeled   attachment score:.*= (.*) %$", row).group(1)
				# to be remove (redundant) : 
				report["LAS"][tag] = re.search(".*  Labeled   attachment score:.*= (.*) %$", row).group(1)

	json.dump(report,open(target_dir, "w"))
	print("json {} dumped {}".format(report, target_dir))
	print("REPORTING {}".format(str(tag_seen)))

if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--model_id',required=True)
	parser.add_argument('--model_info',required=True)
	parser.add_argument('--training_info',required=True)
	parser.add_argument('--tag',required=True)
	parser.add_argument('--dir_report',required=True)
	parser.add_argument('--target_dir',required=True)

	args = parser.parse_args()


	report2json(args.dir_report, 
		model_info=args.model_info, model_id=args.model_id, 
		training_info=args.training_info, tag=args.tag, target_dir = args.target_dir)

	  #Labeled   attachment score: 16350 / 20575 * 100 = 79.47 %