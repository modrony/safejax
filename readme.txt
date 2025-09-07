save:

save_safetensors(filename, state)

load:

state = load_safetensors(filename, state)

doesn't save treedef, so load needs a new state to get it from.
