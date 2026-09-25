import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aim.contracts import ContractError, Hypothesis, Outcome, Status
from aim.controller import Controller
from aim.datasets import polynomial_case
from aim.memory import Memory
from aim.observation_verifier import ObservationConsistencyVerifier
from aim.research_eval import SingleReferenceResearcher
from aim.tracking import Run, file_hash


class ObservationVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.memory=Memory(self.path/'memory');self.addCleanup(self.memory.close)
        self.verifier=ObservationConsistencyVerifier()

    def evidence(self,points,topic='instrument'):
        text=json.dumps({'topic':topic,'observations':points})
        source=self.memory.ingest(text,uri='fixture://observation',version='1',rights='fixture',title='fixture')
        return self.memory.span(source,0,len(text))

    def check(self,ev,coefficients=(1,2,1),topic='instrument'):
        return self.verifier.verify(Hypothesis('h',coefficients,(ev.id,),'candidate'),[ev],self.memory,topic)

    def test_exact_observation_fit_is_scoped_and_bound_to_hypothesis(self):
        ev=self.evidence([[0,1],[1,4],[2,9]])
        result=self.check(ev)
        self.assertEqual(result.outcome,Outcome.PASS);self.assertEqual(result.points_checked,3)
        self.assertIn('no future prediction',result.scope)
        self.assertNotEqual(result.hypothesis_hash,self.check(ev,(4,1,1)).hypothesis_hash)

    def test_forged_or_missing_span_fails(self):
        ev=self.evidence([[0,1],[1,4],[2,9]])
        self.assertEqual(self.check(replace(ev,quote=ev.quote+' ')).outcome,Outcome.FAIL)
        h=Hypothesis('h',(1,2,1),(ev.id,),'candidate')
        self.assertEqual(self.verifier.verify(h,[],self.memory,'instrument').outcome,Outcome.FAIL)

    def test_missing_unsupported_or_wrong_topic_is_unknown(self):
        for rows in ([],[[True,1]],[[0,float('nan')]],[[0]],'not rows'):
            with self.subTest(rows=rows): self.assertEqual(self.check(self.evidence(rows)).outcome,Outcome.UNKNOWN)
        self.assertEqual(self.check(self.evidence([[0,1]],topic='other')).outcome,Outcome.UNKNOWN)

    def test_conflicting_observations_fail(self):
        self.assertEqual(self.check(self.evidence([[0,1],[0,2]])).outcome,Outcome.FAIL)

    def test_read_only_memory_cannot_write_or_create_store(self):
        ev=self.evidence([[0,1]])
        before=file_hash(self.path/'memory/memory.sqlite')
        memory=Memory(self.path/'memory',read_only=True)
        try:
            self.assertTrue(memory.validate(ev))
            with self.assertRaises(ContractError): memory.append('INVALID_WRITE',{})
            objects=set(memory.objects.iterdir())
            with self.assertRaises(ContractError): memory.ingest('new',uri='new',version='1',rights='test',title='test')
            self.assertEqual(objects,set(memory.objects.iterdir()))
            with self.assertRaises(sqlite3.OperationalError): memory.db.execute("DELETE FROM edges")
        finally: memory.close()
        self.assertEqual(before,file_hash(self.path/'memory/memory.sqlite'))
        with self.assertRaises(ContractError): Memory(self.path/'absent',read_only=True)
        self.assertFalse((self.path/'absent').exists())

    def test_target_match_does_not_imply_observation_fit(self):
        class Counterexample(SingleReferenceResearcher):
            def hypothesize(self,state):
                return [Hypothesis('counterexample',(4,1,1),tuple(e.id for e in state.evidence),'unit-test counterexample')]
        with Run(self.path,'counterexample',{}) as run:
            state=Controller(researcher=Counterexample()).run(polynomial_case('counterexample',[1,2,1],target=3),run)
        self.assertEqual(state.claims[0].status,Status.VERIFIED)
        memory=Memory(run.path/'memory',read_only=True)
        try: check=self.verifier.verify(state.hypotheses[0],state.evidence,memory,state.topic)
        finally: memory.close()
        self.assertEqual(check.outcome,Outcome.FAIL)
        self.assertEqual(state.claims[0].status,Status.VERIFIED)

    def test_observation_fit_does_not_imply_future_match(self):
        with Run(self.path,'cubic-counterexample',{}) as run:
            state=Controller(researcher=SingleReferenceResearcher()).run(polynomial_case('cubic',[1,2,1,1],target=3),run)
        self.assertEqual(state.claims[0].status,Status.CONTRADICTED)
        memory=Memory(run.path/'memory',read_only=True)
        try: check=self.verifier.verify(state.hypotheses[0],state.evidence,memory,state.topic)
        finally: memory.close()
        self.assertEqual(check.outcome,Outcome.PASS)


if __name__=='__main__': unittest.main()
