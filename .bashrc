# Set the encoding for SSH since ssh can't inherit the ENV
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Set HOME
export HOME="/teamspace/studios/this_studio"

# >>> lightning managed. do not modify >>>
[ -f /settings/.lightningrc ] && source /settings/.lightningrc bash
# <<< lightning managed. do not modify <<<

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
