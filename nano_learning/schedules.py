"""Recorded schedules expressed in local stage optimizer updates."""
SPECS = {
    'warmup': dict(updates=640, offset=0, base_updates=0, full=(256,512,640), mini_at_full=False),
    'pilot64': dict(updates=64, offset=640, base_updates=0, full=(64,), mini_at_full=False),
    'continuation1024': dict(updates=1024, offset=704, base_updates=0, full=(1024,), mini_at_full=True),
    'sft4096': dict(updates=3072, offset=1728, base_updates=1024, full=tuple(range(512,3073,512)), mini_at_full=True),
}


def validation_kinds(stage, step):
    spec=SPECS[stage]
    if not 0 <= step <= spec['updates']: raise ValueError('Step outside stage')
    full=step in spec['full']
    mini=(step % 64 == 0 and (step > 0 or spec['mini_at_full']) and (not full or spec['mini_at_full']))
    return (['small'] if mini else []) + (['full'] if full else [])
