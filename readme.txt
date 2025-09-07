save:

save_safetensors(filename, state)

load:

state = load_safetensors(filename, state)

doesn't save treedef or sharding, so load needs a new state to get those from.
does a reduce mean over axis-0 if state was saved with different sharding.
