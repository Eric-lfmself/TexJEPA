"""Shared explicit synthetic markers; absence is not proof of training history."""
SYNTHETIC_EVIDENCE=('synthetic_smoke','synthetic','fixture','mock','dummy')
SYNTHETIC_SOURCES=(*SYNTHETIC_EVIDENCE,'explicit_random_initialization','random_init')


def has_synthetic_ancestry(value):
    if isinstance(value,dict):
        if (value.get('mock') is True or value.get('epoch_is_lineage_label') is True
                or value.get('class_mapping_is_synthetic') is True
                or value.get('evidence') in SYNTHETIC_EVIDENCE
                or value.get('source') in SYNTHETIC_SOURCES):
            return True
        return any(has_synthetic_ancestry(item) for item in value.values())
    if isinstance(value,(list,tuple)):
        return any(has_synthetic_ancestry(item) for item in value)
    return False
