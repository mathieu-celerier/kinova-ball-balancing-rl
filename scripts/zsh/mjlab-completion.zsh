# Zsh completions for project-local `uv run train` / `uv run play` task IDs.
#
# Load this after uv's own completion so we can delegate back to `_uv` for
# every other `uv` command.

autoload -Uz compinit
if [[ -z ${functions[compdef]+x} ]]; then
  compinit -i >/dev/null 2>&1
fi

typeset -g _MJLAB_COMPLETION_SCRIPT_DIR=${${(%):-%N}:A:h}
typeset -ga _MJLAB_TASK_IDS_CACHE

if [[ -n ${functions[_uv]+x} && -z ${functions[_mjlab_uv_original]+x} ]]; then
  functions[_mjlab_uv_original]=$functions[_uv]
fi

_mjlab_repo_root() {
  builtin cd "${_MJLAB_COMPLETION_SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd
}

_mjlab_task_ids() {
  local repo_root registry_file

  if (( ${#_MJLAB_TASK_IDS_CACHE[@]} > 0 )); then
    printf '%s\n' "${_MJLAB_TASK_IDS_CACHE[@]}"
    return 0
  fi

  repo_root=$(_mjlab_repo_root)
  registry_file="${repo_root}/src/mjlab_kinova/tasks/__init__.py"

  if [[ ! -r "${registry_file}" ]]; then
    return 1
  fi

  _MJLAB_TASK_IDS_CACHE=(${(f)"$(
    rg -o '"Mjlab-[^"]+"' "${registry_file}" \
      | tr -d '"'
  )"})

  (( ${#_MJLAB_TASK_IDS_CACHE[@]} > 0 )) || return 1
  printf '%s\n' "${_MJLAB_TASK_IDS_CACHE[@]}"
}

_mjlab_complete_uv_task() {
  local -a task_ids
  task_ids=(${(f)"$(_mjlab_task_ids)"})
  (( ${#task_ids[@]} > 0 )) || return 1
  compadd -- "${task_ids[@]}"
}

_mjlab_uv_completion() {
  if (( CURRENT == 4 )) && [[ ${words[2]} == run ]] && [[ ${words[3]} == (train|play) ]]; then
    _mjlab_complete_uv_task && return 0
  fi

  if [[ -n ${functions[_mjlab_uv_original]+x} ]]; then
    _mjlab_uv_original "$@"
    return $?
  fi

  return 1
}

if [[ -n ${functions[compdef]+x} ]]; then
  compdef _mjlab_uv_completion uv
fi
