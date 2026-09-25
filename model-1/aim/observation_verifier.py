"""Scoped hypothesis checking. No claim mutation, repair or inference oracle."""
import json
from dataclasses import asdict, dataclass
from fractions import Fraction

from .contracts import ContractError, Outcome, finite_number
from .research_format import _unique_object, _invalid_constant
from .tracking import digest


@dataclass(frozen=True)
class ObservationCheck:
    id: str
    hypothesis_hash: str
    evidence_hash: str
    verifier: str
    version: str
    outcome: Outcome
    scope: str
    evidence_ids: tuple[str,...]
    points_checked: int
    residuals: tuple[str,...]
    detail: str


class ObservationConsistencyVerifier:
    name="hypothesis-observation-consistency"
    version="1"

    def verify(self,hypothesis,evidence,memory,topic):
        identity=digest(asdict(hypothesis))
        evidence_hash=digest([asdict(e) for e in evidence])
        references=tuple(hypothesis.evidence_ids)

        def result(outcome,detail,residuals=()):
            return ObservationCheck("vo-"+digest([identity,evidence_hash,topic,outcome])[:24],identity,evidence_hash,
                self.name,self.version,outcome,"Polynomial agreement with all cited observations at tolerance 1e-8; no future prediction or global-law proof",
                references,len(residuals),tuple(str(r) for r in residuals),detail)

        if not references: return result(Outcome.UNKNOWN,"No cited observations")
        if len(references)!=len(set(references)): return result(Outcome.FAIL,"Repeated evidence references")
        index={e.id:e for e in evidence}
        if len(index)!=len(evidence): return result(Outcome.FAIL,"Duplicate evidence identities")
        if any(ref not in index for ref in references): return result(Outcome.FAIL,"Missing cited source span")
        rows=[]
        for ref in references:
            ev=index[ref]
            try:
                if not memory.validate(ev): return result(Outcome.FAIL,"Source span integrity failed")
            except (ContractError,OSError): return result(Outcome.FAIL,"Source unavailable or altered")
            try:
                document=json.loads(ev.quote,object_pairs_hook=_unique_object,parse_constant=_invalid_constant)
                if not isinstance(document,dict) or document.get("topic")!=topic:
                    return result(Outcome.UNKNOWN,"Cited observations have an incompatible topic/schema")
                points=document.get("observations")
                if not isinstance(points,list) or not 1<=len(points)<=100: raise ContractError("Invalid observation count")
                for pair in points:
                    if not isinstance(pair,list) or len(pair)!=2: raise ContractError("Invalid observation pair")
                    for value in pair: finite_number(value)
                    rows.append(tuple(Fraction(str(v)) for v in pair))
                if len(rows)>1000: raise ContractError("Observation budget exceeded")
            except (ValueError,TypeError,KeyError):
                return result(Outcome.UNKNOWN,"Unsupported or nonfinite observation data")
        seen={}
        for x,y in rows:
            if x in seen and seen[x]!=y: return result(Outcome.FAIL,"Conflicting observations at the same coordinate")
            seen[x]=y
        coefficients=[Fraction(str(c)) for c in hypothesis.coefficients]
        residuals=[abs(sum(c*x**i for i,c in enumerate(coefficients))-y) for x,y in rows]
        outcome=Outcome.PASS if all(r<=Fraction(1,100_000_000) for r in residuals) else Outcome.FAIL
        return result(outcome,"Every cited point evaluated independently with rational arithmetic",residuals)
