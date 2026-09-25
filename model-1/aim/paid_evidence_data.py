"""Fresh groups; known-family and hidden-fifth-degree factorial stress fixture."""
import itertools
from pathlib import Path

from .datasets import polynomial_case, world_value
from .paid_evidence import VERSION
from .tracking import Run, digest, write_json


def bases():
    result=[]
    for linear in (True,False):
        pool=sorted((digest([VERSION,q]),q) for q in itertools.product(range(120,201),range(-20,21),range(-3,4))
                    if (q[2]==0)==linear)
        result.extend(pool[:16])
    return sorted(result)


def build_dataset(runs):
    with Run(Path(runs),'paid-evidence-data',{'version':VERSION}) as run:
        rows=[]
        for group,q in bases():
            a,b,c=q;k=(-1,1)[int(group[:8],16)%2];target=(4,5,6)[int(group[8:16],16)%3]
            variants={'base':q,'cubic':(a,b+2*k,c-3*k,k),'quintic':(a,b-6*k,c+5*k,5*k,-5*k,k)}
            for family,coefs in variants.items():
                wid=digest([VERSION,coefs]);case=polynomial_case(wid,coefs,target=target)
                for source in case['sources']: source['version']=VERSION
                case['acquisition']={'x':3,'available':True,'observations':[[3,world_value(coefs,3)]]}
                rows.append({'id':wid,'group':group,'family':family,'case':case})
        write_json(run.path/'worlds.json',{'version':VERSION,'rows':rows})
    return run.path
