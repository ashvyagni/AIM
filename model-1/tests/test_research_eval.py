import unittest

from aim.research_eval import paired_comparison, summarize, wilson, SingleReferenceResearcher
from aim.contracts import Evidence, ResearchState


class ResearchEvaluationTests(unittest.TestCase):
    def row(self,id,verified):
        return {"id":id,"verified":verified,"contract_valid":True,"json_valid":True,"contradicted":not verified,
                "error_category":None,"unbacked_verified_claims":0,"tool_actions":2,"seconds":0.1,"family":"quadratic"}

    def test_wilson_boundaries(self):
        self.assertIsNone(wilson(0,0))
        self.assertAlmostEqual(wilson(0,64)[0],0)
        self.assertAlmostEqual(wilson(64,64)[1],1)
        self.assertLess(wilson(32,64)[0],0.5)
        self.assertGreater(wilson(32,64)[1],0.5)

    def test_paired_gain_and_loss(self):
        initial=[self.row("a",False),self.row("b",True),self.row("c",False)]
        trained=[self.row("a",True),self.row("b",False),self.row("c",True)]
        metrics=paired_comparison(initial,trained)
        self.assertEqual(metrics["gained_worlds"],2)
        self.assertEqual(metrics["lost_worlds"],1)
        self.assertAlmostEqual(metrics["verified_rate_change"],1/3)

    def test_invalid_worlds_stay_in_denominator(self):
        invalid={**self.row("b",False),"contract_valid":False,"json_valid":False,"contradicted":False}
        result=summarize([self.row("a",True),invalid])
        self.assertEqual(result["verified_rate"],0.5)
        self.assertEqual(result["unresolved_worlds"],1)

    def test_reference_has_same_one_candidate_budget(self):
        state=ResearchState("1","test","topic",4)
        state.observations=[[0,1],[1,4],[2,9]]
        state.evidence=[Evidence("e","s","h",0,1,"x","h","chars:0:1")]
        hypotheses=SingleReferenceResearcher().hypothesize(state)
        self.assertEqual(len(hypotheses),1)
        self.assertEqual(hypotheses[0].coefficients,(1.,2.,1.))


if __name__=='__main__': unittest.main()
