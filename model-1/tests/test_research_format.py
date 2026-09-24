import json
import tempfile
import unittest
from pathlib import Path

from aim.contracts import ContractError, ResearchState, Status, ToolResult, Outcome
from aim.controller import Controller
from aim.datasets import polynomial_case
from aim.memory import Memory
from aim.research_format import VERSION, ResearchOutputError, parse_hypothesis, prompt_for, target_response
from aim.researcher import TransformerResearcher
from aim.tracking import Run


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.memory = Memory(self.path/"memory")
        self.addCleanup(self.memory.close)
        self.case = polynomial_case("format",[1,2,1])
        self.state = ResearchState("1",self.case["question"],self.case["topic"],4)
        for source in self.case["sources"]:
            sid=self.memory.ingest(**source)
            self.state.evidence.append(self.memory.span(sid,0,len(source["text"])))

    def test_prompt_contains_observations_not_measurement(self):
        prompt, aliases=prompt_for(self.state)
        record=json.loads(prompt)
        self.assertEqual(record["evidence"],{"E0":[[0,1],[1,4],[2,9]]})
        self.assertNotIn("experiment",record)
        self.assertNotIn("25",prompt)
        self.assertEqual(aliases["E0"],self.state.evidence[0].id)

    def test_valid_output_maps_to_real_evidence(self):
        _, aliases=prompt_for(self.state)
        hypotheses=parse_hypothesis(target_response([1,2,1]),aliases)
        self.assertEqual(hypotheses[0].evidence_ids,(self.state.evidence[0].id,))

    def test_invalid_json_and_duplicate_fields_fail(self):
        for text in ('', '{} trailing','{"coefficients":[],"coefficients":[],"evidence":[]}'):
            with self.subTest(text=text),self.assertRaises(ResearchOutputError):
                parse_hypothesis(text,{"E0":"source"})

    def test_fabricated_and_missing_aliases_fail(self):
        for references in (["invented"],[],["E0","E0"],[True]):
            with self.subTest(references=references),self.assertRaises(ResearchOutputError) as error:
                parse_hypothesis(json.dumps({"coefficients":[[1,2,1]],"evidence":references}),{"E0":"source"})
            self.assertEqual(error.exception.category,"evidence")

    def test_nonfinite_boolean_and_extra_candidates_fail(self):
        for candidates in ([[1,True,3]],[[1,float('nan'),3]],[[1,2]],[[1,2,3],[1,2,3]]):
            with self.subTest(candidates=candidates),self.assertRaises(ResearchOutputError):
                parse_hypothesis(json.dumps({"coefficients":candidates,"evidence":["E0"]}),{"E0":"source"})

    def test_post_action_state_is_rejected(self):
        self.state.results=[ToolResult("a",Outcome.PASS,25,"measurement",0,"1")]
        with self.assertRaises(ContractError):
            prompt_for(self.state)

    def stub_researcher(self,text):
        # Test double only: production adapter always calls its own checkpoint.
        class StubModel:
            def generate_text(self,*args): return text
        researcher=object.__new__(TransformerResearcher)
        researcher.record={"research_contract":VERSION}
        researcher.model=StubModel();researcher.tokenizer=None
        researcher.max_new_tokens=80;researcher.model_id="test-double";researcher.last_trace=None
        return researcher

    def test_adapter_loop_binds_and_verifies_claim(self):
        researcher=self.stub_researcher(target_response([1,2,1]))
        with Run(self.path,"adapter-test",{}) as run:
            state=Controller(researcher=researcher).run(self.case,run)
        self.assertEqual(state.claims[0].status,Status.VERIFIED)
        self.assertIn('RESEARCHER_OUTPUT',(run.path/'events.jsonl').read_text())
        self.assertTrue(researcher.last_trace['valid'])

    def test_invalid_generation_has_no_reference_fallback(self):
        researcher=self.stub_researcher('')
        with Run(self.path,"adapter-test",{}) as run:
            state=Controller(researcher=researcher).run(self.case,run)
        self.assertEqual(state.claims,[])
        self.assertEqual(researcher.last_trace['error_category'],'syntax')
        self.assertTrue(state.unknowns)

    def test_reused_adapter_does_not_retain_trace_after_prompt_failure(self):
        researcher=self.stub_researcher(target_response([1,2,1]))
        researcher.hypothesize(self.state)
        self.state.evidence=[]
        with self.assertRaises(ContractError): researcher.hypothesize(self.state)
        self.assertIsNone(researcher.last_trace['raw_output'])
        self.assertFalse(researcher.last_trace['valid'])
        self.assertEqual(researcher.last_trace['error_category'],'context_or_contract')


if __name__=='__main__': unittest.main()
