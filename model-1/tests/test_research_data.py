import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from aim.research_data import world_id, world_partitions, make_record, load_training_data
from aim.research_format import parse_hypothesis
from aim.memory import Memory


class ResearchDataTests(unittest.TestCase):
    def test_partition_counts_disjointness_and_determinism(self):
        splits=world_partitions()
        self.assertEqual(splits,world_partitions())
        self.assertEqual({s:len(v) for s,v in splits.items()},dict(train=512,validation=64,test=64,ood=64))
        groups=[{row[0] for row in rows} for rows in splits.values()]
        self.assertEqual(sum(map(len,groups)),len(set.union(*groups)))
        for split,rows in splits.items():
            if split=="ood":
                self.assertTrue(all(len(row[1])==4 and row[1][-1]!=0 for row in rows))
            else:
                self.assertEqual(sum(row[2]=="linear" for row in rows),128 if split=="train" else 16)

    def test_labels_match_exact_world_and_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            memory=Memory(Path(directory))
            try:
                coefs=(-3,2,1)
                row=make_record(world_id(coefs),coefs,"quadratic","train",memory)
                candidate=parse_hypothesis(row["response"],row["aliases"])[0]
                x=row["case"]["target_x"]
                actual=sum(Fraction(str(c))*x**i for i,c in enumerate(candidate.coefficients))
                self.assertEqual(actual,Fraction(row["case"]["experiment"]["observations"][0][1]))
                self.assertEqual(candidate.evidence_ids,tuple(row["aliases"].values()))
                self.assertNotIn("world_coefficients",row["prompt"])
                self.assertNotIn("experiment",row["prompt"])
            finally: memory.close()

    def test_holdout_file_is_not_training_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"holdout.json"
            path.write_text(json.dumps({"schema":"aim-research-holdout-v1"}))
            with self.assertRaises(ValueError): load_training_data(path)


if __name__=='__main__': unittest.main()
