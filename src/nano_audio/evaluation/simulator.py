"""Deterministic local state machine; traces preserve every side effect."""
from copy import deepcopy
from nano_audio.data.task import SchemaFamily

def simulate(program, initial_state=None, family=None):
    family = family or SchemaFamily(renamed=False)
    family.encode(program)  # validates everything before any state mutation
    state = deepcopy(initial_state or {'devices': {}, 'timers': {}})
    events = []
    for call in program:
        op, args = call['operation'], call['arguments']
        if op == 'start_timer':
            state['timers'][args['label']] = args['seconds']
        elif op == 'cancel_timer':
            state['timers'].pop(args['label'], None)
        else:
            device = state['devices'].setdefault(args['device'], {})
            key = {'set_power': 'power', 'set_brightness': 'brightness', 'set_volume': 'volume', 'set_temperature': 'celsius', 'set_playback': 'playback'}.get(op)
            if key:
                device[key] = next(v for k, v in args.items() if k != 'device')
        event = deepcopy(call)
        if op == 'query_device_state':
            event['result'] = deepcopy(state['devices'][args['device']])
        events.append(event)
    return {'state': state, 'events': events}
