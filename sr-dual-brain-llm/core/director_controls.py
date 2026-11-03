"""Bounded Director controls derived only from already-authoritative facts.

This is local steering, not model output or a successful latency observation.
Question text, memory presence and persistence intent are deliberately absent.
"""
from copy import deepcopy
from dataclasses import dataclass


LOCAL_DIRECTOR_VERSION = 'trusted_facts_v1'
LOCAL_DIRECTOR_REASONS = frozenset({'caller_system2_on', 'existing_long_term_restriction'})


@dataclass(frozen=True)
class LocalDirectorPlan:
    _force_consult: bool = False
    _restrict_long_term: bool = False

    @property
    def source(self):
        return 'local_facts'

    @property
    def version(self):
        return LOCAL_DIRECTOR_VERSION

    @property
    def controls(self):
        """Fresh sparse controls: absent facts never become invented defaults."""
        result = {}
        if self._force_consult is True:
            result['consult'] = 'force'
        if self._restrict_long_term is True:
            result['memory'] = {'long_term': 'drop'}
        return result

    @property
    def reasons(self):
        return tuple(reason for active, reason in (
            (self._force_consult is True, 'caller_system2_on'),
            (self._restrict_long_term is True, 'existing_long_term_restriction'),
        ) if active)

    @property
    def nonneutral(self):
        return self._force_consult is True or self._restrict_long_term is True

    def merge(self, remote=None):
        """Keep remote capabilities while authoritative local fields prevail.

        This merges controls; it does not approve remote claims, infer missing
        controls, or perform a memory operation. Neither input is mutated.
        """
        merged = deepcopy(remote) if isinstance(remote, dict) else {}
        controls = self.controls
        if 'consult' in controls:
            merged['consult'] = controls['consult']
        if 'memory' in controls:
            memory = merged.get('memory')
            merged['memory'] = memory if isinstance(memory, dict) else {}
            merged['memory'].update(controls['memory'])
        return merged

    def to_payload(self):
        return {'source': self.source, 'version': self.version,
                'reasons': list(self.reasons), 'controls': self.controls}


def build_local_director_plan(*, system2_mode='auto', system2_capable=False,
                              system2_active=False, long_term_restricted=False):
    """Project explicit normalized facts, with neutral output for unknowns.

    System2 OFF does not mean ordinary consultation is forbidden. Only explicit
    ON with a capable, active subsystem establishes a consultation requirement.
    A long-term restriction must be supplied by an existing memory policy; it
    cannot be inferred from absent memory or a caller's source-write setting.
    """
    return LocalDirectorPlan(
        _force_consult=(type(system2_mode) is str and system2_mode == 'on'
                        and system2_capable is True and system2_active is True),
        _restrict_long_term=long_term_restricted is True,
    )
