"""Opt-in shared decisions and paid observations; verifiers still own claim status."""
import copy
from dataclasses import asdict
from fractions import Fraction

from .contracts import ContractError, Decision, Outcome, Status, finite_number
from .controller import Controller
from .judge import TARGET
from .judge_shift_features import extract, payload
from .researcher import PolynomialResearcher, predict
from .tools import ToolRunner
from .tracking import canonical, digest, write_json

VERSION = 'aim-paid-evidence-v1'
KINDS = {'verify_all','abstain_all','fit_direct','acquire_fit','judge_direct','acquire_judge','selective_judge'}


def aggregate(predictions, probabilities):
    if len(predictions)!=len(probabilities) or not predictions:
        raise ContractError('Candidate forecast shape mismatch')
    groups={}
    for value,p in zip(predictions,probabilities):
        finite_number(value)
        if not 0<=finite_number(p)<=1: raise ContractError('Invalid candidate probability')
        key=str(Fraction(str(value)))
        groups[key]=max(groups.get(key,0.),p)
    return min(1.,sum(groups.values())),groups


class SharedPolicy:
    def __init__(self, kind, config, forecaster=None):
        if kind not in KINDS: raise ContractError('Unknown paid-evidence policy')
        if 'judge' in kind and forecaster is None: raise ContractError('Frozen forecaster required')
        self.kind=kind;self.forecaster=forecaster
        self.target_cost=finite_number(config['target_cost']);self.acquisition_cost=finite_number(config['acquisition_cost'])
        if not 0<=self.target_cost<=1 or not 0<=self.acquisition_cost<=1: raise ContractError('Invalid utility cost')
        self.band=[finite_number(v) for v in config['acquisition_band']]
        if len(self.band)!=2 or not 0<=self.band[0]<self.band[1]<=1: raise ContractError('Invalid acquisition band')

    def assess(self,state,hypotheses):
        if not hypotheses: raise ContractError('No supported candidates')
        inputs=[copy.deepcopy(payload(state,h)) for h in hypotheses]
        fits=[bool(extract(p,'rich')[5]) for p in inputs]
        predictions=[predict(h.coefficients,state.target_x) for h in hypotheses]
        probabilities=[self.forecaster.forecast(p) for p in inputs] if self.forecaster is not None else None
        joint,groups=aggregate(predictions,probabilities) if probabilities is not None else (None,{})
        if self.kind=='verify_all': measure=True
        elif self.kind=='abstain_all': measure=False
        elif self.kind in {'fit_direct','acquire_fit'}: measure=any(fits)
        else: measure=joint>=self.target_cost
        return {'policy':self.kind,'hypothesis_ids':[h.id for h in hypotheses],'inputs':inputs,
            'predictions':predictions,'candidate_probabilities':probabilities,'observed_fits':fits,
            'aggregate':joint,'distinct_prediction_groups':groups,'measure':measure,
            'scope':'Heuristic aggregate, not calibrated question probability or proof'}

    def acquire(self, assessment):
        return self.kind in {'acquire_fit','acquire_judge'} or (
            self.kind=='selective_judge' and self.band[0]<assessment['aggregate']<self.band[1])


class SharedJudge:
    def __init__(self,policy):
        self.policy=policy;self.record=None;self.cached=None;self.recorder=None
        self.model_id='shared-'+VERSION+'-'+policy.kind
        if policy.forecaster is not None: self.model_id+='-'+policy.forecaster.model_id

    def decide(self,state,hypothesis,claim):
        if (claim.hypothesis_id!=hypothesis.id or claim.target_x!=state.target_x
            or abs(claim.predicted-predict(hypothesis.coefficients,state.target_x))>1e-8):
            raise ContractError('Forecast claim mismatch')
        key=digest([payload(state,h) for h in state.hypotheses])
        if self.cached!=key:
            self.record=self.policy.assess(state,state.hypotheses);self.cached=key
            if self.recorder: self.recorder('TARGET_DECISION',self.record)
        index=self.record['hypothesis_ids'].index(hypothesis.id)
        values=self.record['candidate_probabilities']
        return Decision(claim.id,values[index] if values is not None else None,TARGET,
            'VERIFY' if self.record['measure'] else 'ABSTAIN',self.model_id,
            'Individual forecast; shared action chosen by separately recorded question policy')


class MeteredTools:
    """Count dispatches, including failures, without changing the underlying tool."""
    def __init__(self,runner): self.runner=runner;self.dispatches=[]

    def execute(self,action):
        row={'action':asdict(action),'result':None}
        self.dispatches.append(row)
        result=self.runner.execute(action)
        row['result']=asdict(result)
        return result


class PaidEvidenceController(Controller):
    def __init__(self,policy,*,tools=None,max_actions=8):
        self.policy=policy;self.meter=MeteredTools(tools or ToolRunner())
        super().__init__(researcher=PolynomialResearcher(),judge=SharedJudge(policy),tools=self.meter,max_actions=max_actions)
        self.decisions=[]

    def prepare_evidence(self,state,memory,case,run,invoke):
        def record(kind,value):
            item={'kind':kind,**copy.deepcopy(value)}
            self.decisions.append(item);memory.append(kind,item);run.event(kind,item)
        self.judge.recorder=record
        # Initial assessment has no acquisition or target outcome fields.
        initial=self.policy.assess(state,self.researcher.hypothesize(copy.deepcopy(state)))
        purchase=self.policy.acquire(initial)
        record('ACQUISITION_DECISION',{**initial,'purchase':purchase,'coordinate':3})
        state.assumptions.append('Paid evidence policy: '+self.policy.kind+'; question reward capped at one; utility costs are illustrative')
        if not purchase: return
        if sorted(x for x,_ in state.observations)!=[0.,1.,2.] or state.target_x<=3:
            raise ContractError('Acquisition requires initial x=0,1,2 and a later target')
        request=case.get('acquisition')
        if not isinstance(request,dict) or set(request)!={'x','available','observations'} or request['x']!=3:
            raise ContractError('Missing or incompatible acquisition environment')
        measured=invoke('MEASURE',copy.deepcopy(request))
        if measured.outcome!=Outcome.PASS:
            raise ContractError('Additional observation unresolved: '+measured.outcome.value)
        text=canonical({'topic':state.topic,'observations':[[3,measured.value]]})
        sid=memory.ingest(text,uri=f"aim://acquisition/{case['id']}/x3",version=VERSION,
            rights='project-generated-fixture',title=state.topic+' purchased observation')
        ev=memory.span(sid,0,len(text));state.evidence.append(ev);state.observations.append([3.,measured.value])
        memory.append('ACQUISITION_RESULT',{'result':asdict(measured),'evidence_id':ev.id})
        memory.append('STATE',state.to_dict())
        run.event('ACQUISITION_RESULT',{'result':asdict(measured),'evidence_id':ev.id})

    def run(self,case,run):
        if self.meter.dispatches or self.decisions: raise ContractError('Use a fresh paid Controller per investigation')
        state=super().run(case,run)
        attempts={'acquisition':0,'target':0,'calculation':0}
        for row in self.meter.dispatches:
            action=row['action']
            if action['name']=='CALCULATE': attempts['calculation']+=1
            elif action['arguments']['x']==state.target_x: attempts['target']+=1
            else: attempts['acquisition']+=1
        if attempts['target']>1 or attempts['acquisition']>1: raise ContractError('Shared measurement budget violated')
        success=any(c.status==Status.VERIFIED for c in state.claims)
        from .verifiers import claim_identity
        checks={v.id:v for v in state.verifications}
        backed=sum(c.status==Status.VERIFIED and len(c.verification_ids)==2 and all(
            checks[i].outcome==Outcome.PASS and checks[i].claim_hash==digest(claim_identity(c))
            for i in c.verification_ids) for c in state.claims)
        utility=int(success)-self.policy.target_cost*attempts['target']-self.policy.acquisition_cost*attempts['acquisition']
        summary={'version':VERSION,'policy':self.policy.kind,'success':success,'utility':utility,
            'target_cost':self.policy.target_cost,'acquisition_cost':self.policy.acquisition_cost,
            'attempts':attempts,'tool_dispatches':len(self.meter.dispatches),
            'tool_elapsed_seconds':sum(d['result']['elapsed_seconds'] for d in self.meter.dispatches),
            'unknown_claims':sum(c.status==Status.UNKNOWN for c in state.claims),'unknown_reasons':state.unknowns,
            'verified_claims':sum(c.status==Status.VERIFIED for c in state.claims),
            'verified_with_passing_checks':backed,'decisions':self.decisions,
            'dispatches':self.meter.dispatches,'state_sha256':None}
        from .tracking import file_hash
        summary['state_sha256']=file_hash(run.path/'state.json')
        write_json(run.path/'policy-summary.json',summary)
        return state,summary
