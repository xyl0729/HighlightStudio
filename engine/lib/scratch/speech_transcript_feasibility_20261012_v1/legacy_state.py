import torch


def text_state(full_state,target_keys,current_positions):
    subset={k:v for k,v in full_state.items() if k.startswith('text_model.') or k=='text_projection.weight'}
    key='text_model.embeddings.position_ids'
    proof={'legacy_position_ids_present':key in subset,'legacy_position_ids_equal_current_buffer':False}
    if key in subset and key not in target_keys:
        if not torch.equal(subset[key],current_positions):raise ValueError('legacy positions differ from current model buffer')
        del subset[key];proof['legacy_position_ids_equal_current_buffer']=True
    if set(subset)!=set(target_keys):raise ValueError(f'unexpected/missing text state: extra={set(subset)-set(target_keys)}, missing={set(target_keys)-set(subset)}')
    return subset,proof
