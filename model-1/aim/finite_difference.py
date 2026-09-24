"""Opt-in worked-output contract and independent observational diagnostics."""
import json
from fractions import Fraction

from .contracts import ContractError, finite_number
from .research_format import VERSION, ResearchOutputError, _unique_object, _invalid_constant, parse_hypothesis

EXPERIMENT = "aim-finite-difference-comparison-v1"
CONTRACT = "aim-finite-difference-output-v1"


def worked_response(coefficients):
    c0,c1,c2=coefficients
    return json.dumps({"d1":c1+c2,"d2":2*c2,"coefficients":[[c0,c1,c2]],"evidence":["E0"]},separators=(",",":"))


def parse_output(text,aliases,contract):
    if contract==VERSION: return parse_hypothesis(text,aliases)
    if contract!=CONTRACT: raise ContractError("Unknown research output contract")
    try:
        record=json.loads(text,object_pairs_hook=_unique_object,parse_constant=_invalid_constant)
    except (ValueError,TypeError) as exc:
        if isinstance(exc,ResearchOutputError): raise
        raise ResearchOutputError("syntax","Worked output must be JSON") from exc
    if not isinstance(record,dict) or set(record)!={"d1","d2","coefficients","evidence"}:
        raise ResearchOutputError("schema","Worked output requires d1,d2,coefficients,evidence")
    try:
        finite_number(record["d1"],1e6);finite_number(record["d2"],1e6)
    except ContractError as exc:
        raise ResearchOutputError("schema",str(exc)) from exc
    return parse_hypothesis(json.dumps({k:record[k] for k in ("coefficients","evidence")}),aliases)


def diagnostics(text,prompt,aliases,contract):
    """Check generated numbers; never supply an answer or alter Controller status."""
    try:
        hypothesis=parse_output(text,aliases,contract)[0]
    except ContractError:
        return {"valid":False,"coefficient_observation_agreement":[False]*3,
                "d1_correct":False if contract==CONTRACT else None,
                "d2_correct":False if contract==CONTRACT else None,
                "all_steps_correct":False if contract==CONTRACT else None}
    observations=json.loads(prompt)["evidence"]["E0"]
    if [p[0] for p in observations]!=[0,1,2]: raise ContractError("Process diagnostic expects x=0,1,2")
    y0,y1,y2=[Fraction(str(p[1])) for p in observations]
    d1=y1-y0;d2=y2-2*y1+y0
    expected=[y0,d1-d2/2,d2/2]
    agrees=[Fraction(str(got))==want for got,want in zip(hypothesis.coefficients,expected)]
    d1_ok=d2_ok=None
    if contract==CONTRACT:
        record=json.loads(text)
        d1_ok=Fraction(str(record["d1"]))==d1;d2_ok=Fraction(str(record["d2"]))==d2
    return {"valid":True,"coefficient_observation_agreement":agrees,"d1_correct":d1_ok,"d2_correct":d2_ok,
            "all_steps_correct":bool(d1_ok and d2_ok and all(agrees)) if contract==CONTRACT else None}
