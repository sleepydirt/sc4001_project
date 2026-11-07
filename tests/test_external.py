import pytest
from nano_audio.evaluation.external import parse_call,audit_record

def test_external_rejects_actual_release_style_disagreement():
    row={'id':1,'query_idx':2100,'tier':'tier1_direct','domain':'smart_home','duration':2.697,'path':'some.wav','tool_name':'setLighting','expected_tool_call':'setLighting()','extracted_params':'{}','functions':[{'tool_name':'setLightState'}]}
    assert audit_record(row,{'smart_home':[{'name':'setLightState'}]})==(None,'tool_registry_label_disagreement')

def test_external_safe_literals_and_evaluation_only():
    assert parse_call('setVolume(percent=25)')['arguments']=={'percent':25}
    with pytest.raises((ValueError,TypeError)):parse_call('f(value=__import__("os"))')
    with pytest.raises(ValueError):parse_call('f(value=1,value=2)')
    with pytest.raises(ValueError):audit_record({'split':'train'}, {})
