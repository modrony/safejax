import jax
import jax.numpy
import numpy as np
from safetensors.numpy import save_file, load_file
from jax.tree_util import tree_flatten, tree_unflatten

def save_safetensors(fpath, state):
    arrays, treedef = jax.tree.flatten_with_path(state)
    flat_state = {}

    # Convert all JAX arrays to NumPy for saving
    for kp, leaf in arrays:
        key = jax.tree_util.keystr(kp)
        if isinstance(leaf, jax.Array):
            flat_state[key] = np.array(leaf)
        elif isinstance(leaf, np.ndarray):
            flat_state[key] = leaf
        else:
            raise TypeError(f"Unsupported leaf type: {key}: {type(leaf)}")

    save_file(flat_state, fpath)

def load_safetensors(fname, target):
    arrays, treedef = jax.tree.flatten_with_path(target)
    arrkeys = [jax.tree_util.keystr(x[0]) for x in arrays]
    arrvals = [x[1] for x in arrays]
    key_index={k:i for i,k in enumerate(arrkeys)}
    # Load tensor data
    print(f"load {fname}")
    flat_state = load_file(fname)
    missing = set(arrkeys)-set(flat_state.keys())
    if len(missing)>0:
        print("missing keys: {}".format(', '.join(sorted(list(missing)))))
    for k,v in flat_state.items():
        index = key_index.get(k,-1)
        if index<0:
            print("unknown array key: {}".format(k))
            continue
        template = arrvals[index]
        if isinstance(template, jax.Array):
            if template.shape==v.shape:
                vv=v
            elif v.shape[1:] == template.shape[1:]:
                vv=v.mean(0)[None] #reduce over device dim
            else:
                raise ValueError(f"Shape mismatch for key '{k}': {v.shape} vs {template.shape}")
                
            arrvals[index] = jax.device_put(
                jax.numpy.broadcast_to(
                    jax.numpy.asarray(vv, dtype=template.dtype),
                    template.shape), 
                template.sharding)
        elif isinstance(template, np.ndarray):
            arrvals[index] = v
        else:
            raise TypeError(f"Unsupported template type at key {k}: {type(template)}")
    return tree_unflatten(treedef, arrvals)
