"""New cross-domain stress inputs; reuse immutable exact closed-world graders."""
import hashlib
import json
from pathlib import Path
import random

import frontier_tasks as base

ORIGIN = 'prospective_replanning_stress_v1'
LEVELS = {'logic':(2,4),'sql':(5,7),'events':(4,6),'evidence':(4,6)}
GOALS = {
 'logic':('Find the satisfying assignments using every supplied clause; a single satisfying assignment does not establish entailment.',
          'Classify every variable across all satisfying assignments, retaining undetermined values when both possibilities occur.'),
 'sql':('Determine account eligibility and the latest eligible event per account, respecting the cutoff, ban reason, exclusive endpoints and ordering ties.',
        'Evaluate the joins and aggregates with bag multiplicities and SQL NULL semantics; combine every eligible account into the result.'),
 'events':('Replay each job epoch and terminal-state transition in the given global event order; preserve version resets and ignore stale events.',
           'Track current ownership through transfers, cancellation and finish races; apply only events authorized by the exact state and owner.'),
 'evidence':('Evaluate each query independently using matching primary scope/topic, issuance, validity, expiry and revocation at its exact time.',
             'Apply permanent authorized supersession before resolving values and all surviving IDs; retain unknowns and conflicts without using recency as authority.'),
}


def evidence_case(partition,level):
    seed=int(hashlib.sha256((ORIGIN+partition+str(level)).encode()).hexdigest()[:16],16)
    rng=random.Random(seed)
    sources=[]
    for group in range(3):
        scope,topic=f'project_{group%2}',f'field_{group}'
        prefix=f's{group}'
        for suffix,issued,kind,targets in (
                ('a',1,'primary',[]),('b',3,'primary',[prefix+'a']),
                ('c',4,'primary',[]),('d',5,'derived',[prefix+'b',prefix+'c'])):
            sources.append({'id':prefix+suffix,'scope':scope,'topic':topic,'kind':kind,
                'value':rng.choice(['amber','birch','coral']),'issued_at':issued,
                'valid_from':issued if suffix!='b' else 2,'valid_to':None,
                'expires_at':(5 if suffix=='b' else rng.choice([None,6]) if suffix=='c' else None),
                'revoked_at':(6 if suffix=='c' else None),'supersedes':targets})
        if group==0:
            sources[-2]['value']=next(v for v in ('amber','birch','coral') if v!=sources[-3]['value'])
        elif group==1:
            sources[-2]['value']=sources[-3]['value']
        # A foreign primary attempts to supersede current matching sources;
        # authority remains scoped even though the IDs name real records.
        sources.append({'id':prefix+'x','scope':'foreign','topic':topic,'kind':'primary',
            'value':'foreign','issued_at':2,'valid_from':1,'valid_to':None,
            'expires_at':None,'revoked_at':None,'supersedes':[prefix+'b',prefix+'c']})
        if level>=6:
            # Supersession remains permanent after the replacement expires.
            sources.append({'id':prefix+'e','scope':scope,'topic':topic,'kind':'primary',
                'value':rng.choice(['amber','birch','coral']),'issued_at':5,'valid_from':5,
                'valid_to':None,'expires_at':6,'revoked_at':None,
                'supersedes':[prefix+'b',prefix+'c']})
    queries=[{'id':f'q{i:02d}','scope':f'project_{g%2}','topic':f'field_{g}','time':t}
        for i,(g,t) in enumerate((g,t) for g,times in enumerate(((2,3,4,5,6),(3,4,6),(3,4,5,6))) for t in times)]
    rng.shuffle(sources);rng.shuffle(queries)
    data={'sources':sources,'queries':queries}
    return data,('This is a closed-world source-selection exercise under explicit symbolic rules, '
        'not a request to retrieve or certify real-world facts.\nRules: '+base.EVIDENCE_RULES
        +'\nInput: '+json.dumps(data,separators=(',',':')))


def make_case(phase,tier,domain):
    level=LEVELS[domain][tier]
    partition=ORIGIN+':'+phase
    case=base.make_case(partition,level,domain)
    case.update(id=f'{phase}:{tier}:{domain}',partition=phase,tier=tier)
    text=case['question'].split('\nRequirements:\n',1)[0]
    if domain=='evidence':
        case['data'],text=evidence_case(partition,level)
        case['expected']=base.evidence_answer(case['data'])
    goals=GOALS[domain]+('Using step 1 and step 2, return only the complete JSON answer, without Markdown, explanations, '
        'omitted items or extra fields. Use JSON integers and nulls where required. Shape: '+base.SHAPES[domain],)
    case['question']=text+'\nRequirements:\n'+'\n'.join(f'{i}. {g}' for i,g in enumerate(goals,1))
    if len(case['question'])>10000 or any(len(g)>400 for g in goals):
        raise ValueError('Task exceeds bounded request/goal limits')
    return case


def build_fixture():
    return {'schema_version':1,'origin':ORIGIN,'clock':1794182400,
        'cases':[make_case(phase,tier,domain) for phase in ('main','confirmation')
            for tier in ((0,1) if phase=='main' else (1,)) for domain in base.DOMAINS]}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit('Refuse to replace a frozen fixture')
    args.output.write_text(json.dumps(build_fixture(),indent=2,ensure_ascii=False)+'\n')
