#!/bin/sh

set -eu

CONSOLE_WIDTH=68
MAX_VISIBLE_OPTIONS=5
if [ -t 0 ]; then
	TTY_DEVICE="/dev/stdin"
else
	TTY_DEVICE="/dev/tty"
fi
terminal_state=""
setup_tmp_dir=""
cursor_hidden="false"
spinner_pid=""
spinner_output_file=""
spinner_error_file=""
curl_auth_config=""
selection_header_open=""
selection_default_index=0
selection_filter_enabled="false"
selection_query=""
selection_initial_query=""
selection_loading_input_enabled="true"
selection_hint=""
key_read_mode="blocking"

if [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
	BLUE='\033[1;38;2;31;143;255m'
	GREEN='\033[1;32m'
	YELLOW='\033[1;33m'
	RED='\033[1;31m'
	BOLD='\033[1m'
	DIM='\033[38;2;140;150;165m'
	LINE='\033[38;2;95;105;120m'
	RESET='\033[0m'
else
	BLUE=''
	GREEN=''
	YELLOW=''
	RED=''
	BOLD=''
	DIM=''
	LINE=''
	RESET=''
fi

restore_terminal() {
	if [ -n "$terminal_state" ]; then
		stty "$terminal_state" <"$TTY_DEVICE" 2>/dev/null || true
		terminal_state=""
	fi
	if [ "$cursor_hidden" = "true" ]; then
		printf '\033[?25h' >&2
		cursor_hidden="false"
	fi
}

cleanup() {
	restore_terminal
	if [ -n "$spinner_pid" ]; then
		kill "$spinner_pid" 2>/dev/null || true
		spinner_pid=""
	fi
	if [ -n "$spinner_output_file" ]; then
		rm -f "$spinner_output_file"
		spinner_output_file=""
	fi
	if [ -n "$spinner_error_file" ]; then
		rm -f "$spinner_error_file"
		spinner_error_file=""
	fi
	if [ -n "$curl_auth_config" ]; then
		rm -f "$curl_auth_config"
		curl_auth_config=""
	fi
	if [ -n "$setup_tmp_dir" ] && [ -d "$setup_tmp_dir" ]; then
		rm -rf "$setup_tmp_dir"
	fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

die() {
	printf '\n%b×  %s%b\n' "$RED" "$*" "$RESET" >&2
	exit 1
}

header() {
	printf '%b╭  MLflow Tracing Setup%b\n' "$BLUE" "$RESET" >&2
	printf '%b│%b\n' "$LINE" "$RESET" >&2
}

footer() {
	printf '%b╰  %s%b\n' "$BLUE" "$1" "$RESET" >&2
}

detail() {
	text=$1
	if command -v fold >/dev/null 2>&1; then
		printf '%s\n' "$text" | fold -s -w $((CONSOLE_WIDTH - 3)) | while IFS= read -r detail_line; do
			printf '%b│%b  %b%s%b\n' "$LINE" "$RESET" "$DIM" "$detail_line" "$RESET" >&2
		done
	else
		printf '%b│%b  %b%s%b\n' "$LINE" "$RESET" "$DIM" "$text" "$RESET" >&2
	fi
}

primary_detail() {
	text=$1
	if command -v fold >/dev/null 2>&1; then
		printf '%s\n' "$text" | fold -s -w $((CONSOLE_WIDTH - 3)) | while IFS= read -r detail_line; do
			printf '%b│%b  %s\n' "$LINE" "$RESET" "$detail_line" >&2
		done
	else
		printf '%b│%b  %s\n' "$LINE" "$RESET" "$text" >&2
	fi
}

success() {
	label=$1
	shift
	printf '%b●%b  %b%s%b\n' "$GREEN" "$RESET" "$BOLD" "$label" "$RESET" >&2
	for value in "$@"; do
		detail "$value"
	done
	printf '%b│%b\n' "$LINE" "$RESET" >&2
}

progress() {
	label=$1
	shift
	printf '%b◐%b  %b%s%b\n' "$BLUE" "$RESET" "$BOLD" "$label" "$RESET" >&2
	for value in "$@"; do
		detail "$value"
	done
	printf '%b│%b\n' "$LINE" "$RESET" >&2
}

warning() {
	label=$1
	shift
	printf '%b!%b  %b%s%b\n' "$YELLOW" "$RESET" "$YELLOW" "$label" "$RESET" >&2
	for value in "$@"; do
		detail "$value"
	done
	printf '%b│%b\n' "$LINE" "$RESET" >&2
}

run_with_spinner() {
	spinner_label=$1
	shift
	spinner_output_file=$(mktemp "${TMPDIR:-/tmp}/mlflow-spinner-output.XXXXXX")
	spinner_error_file=$(mktemp "${TMPDIR:-/tmp}/mlflow-spinner-error.XXXXXX")
	"$@" >"$spinner_output_file" 2>"$spinner_error_file" &
	spinner_pid=$!
	spinner_frame=0
	while kill -0 "$spinner_pid" 2>/dev/null; do
		case "$spinner_frame" in
		0) spinner_icon="◐" ;;
		1) spinner_icon="◓" ;;
		2) spinner_icon="◑" ;;
		*) spinner_icon="◒" ;;
		esac
		printf '\r%b%s%b  %b%s%b' "$BLUE" "$spinner_icon" "$RESET" "$BOLD" "$spinner_label" "$RESET" >&2
		spinner_frame=$(((spinner_frame + 1) % 4))
		sleep 0.12
	done
	if wait "$spinner_pid"; then
		spinner_status=0
	else
		spinner_status=$?
	fi
	spinner_pid=""
	printf '\r\033[2K' >&2
	spinner_output=$(cat "$spinner_output_file")
	if [ "$spinner_status" -ne 0 ]; then
		cat "$spinner_error_file" >&2
	fi
	rm -f "$spinner_output_file" "$spinner_error_file"
	spinner_output_file=""
	spinner_error_file=""
	return "$spinner_status"
}

render_loading_option() {
	loading_option_text=$loading_manual_label
	if [ -n "$loading_query" ]; then
		loading_option_text="$loading_manual_label: $loading_query"
	fi
	printf '%b│%b  %b❯ %s%b\r\n' "$LINE" "$RESET" "$BLUE" "$loading_option_text" "$RESET" >&2
	render_loading_status
}

render_loading_status() {
	case "$spinner_frame" in
	0) spinner_icon="◐" ;;
	1) spinner_icon="◓" ;;
	2) spinner_icon="◑" ;;
	*) spinner_icon="◒" ;;
	esac
	printf '%b│%b    %b%s %s%b\r\n' "$LINE" "$RESET" "$DIM" "$spinner_icon" "$loading_label" "$RESET" >&2
}

load_with_manual_option() {
	loading_select_label=$1
	loading_manual_label=$2
	loading_label=$3
	shift 3
	loading_manual_selected="false"
	loading_query=""
	loading_input_enabled=$selection_loading_input_enabled
	selection_loading_input_enabled="true"
	spinner_output_file=$(mktemp "${TMPDIR:-/tmp}/mlflow-spinner-output.XXXXXX")
	spinner_error_file=$(mktemp "${TMPDIR:-/tmp}/mlflow-spinner-error.XXXXXX")
	"$@" >"$spinner_output_file" 2>"$spinner_error_file" &
	spinner_pid=$!
	spinner_frame=0
	require_tty
	terminal_state=$(stty -g <"$TTY_DEVICE")
	stty -icanon -echo min 0 time 1 <"$TTY_DEVICE"
	key_read_mode="timed"
	printf '\033[?25l' >&2
	cursor_hidden="true"
	render_loading_option
	while kill -0 "$spinner_pid" 2>/dev/null; do
		read_key
		loading_input_changed="false"
		case "$key_code" in
		3)
			restore_terminal
			exit 130
			;;
		10 | 13)
			kill "$spinner_pid" 2>/dev/null || true
			wait "$spinner_pid" 2>/dev/null || true
			spinner_pid=""
			selection_query=$loading_query
			loading_manual_selected="true"
			printf '\033[4A\033[J' >&2
			restore_terminal
			key_read_mode="blocking"
			printf '%b●%b  %b%s%b\n' "$GREEN" "$RESET" "$BOLD" "$loading_select_label" "$RESET" >&2
			detail "$loading_manual_label"
			printf '%b│%b\n' "$LINE" "$RESET" >&2
			selection_header_open=""
			rm -f "$spinner_output_file" "$spinner_error_file"
			spinner_output_file=""
			spinner_error_file=""
			return 0
			;;
		8 | 127)
			if [ "$loading_input_enabled" = "true" ]; then
				loading_query=${loading_query%?}
				loading_input_changed="true"
			fi
			;;
		*)
			case "$key_code" in
			3[2-9] | [4-9][0-9] | 1[01][0-9] | 12[0-6])
				if [ "$loading_input_enabled" = "true" ]; then
					typed_character=$(printf "\\$(printf '%03o' "$key_code")")
					loading_query="$loading_query$typed_character"
					loading_input_changed="true"
				fi
				;;
			esac
			;;
		esac
		spinner_frame=$(((spinner_frame + 1) % 4))
		if [ "$loading_input_changed" = "true" ]; then
			printf '\033[2A\033[J' >&2
			render_loading_option
		else
			printf '\033[1A\r\033[2K' >&2
			render_loading_status
		fi
	done
	if wait "$spinner_pid"; then
		spinner_status=0
	else
		spinner_status=$?
	fi
	spinner_pid=""
	printf '\033[2A\033[J' >&2
	restore_terminal
	key_read_mode="blocking"
	spinner_output=$(cat "$spinner_output_file")
	if [ "$spinner_status" -ne 0 ]; then
		cat "$spinner_error_file" >&2
	fi
	rm -f "$spinner_output_file" "$spinner_error_file"
	spinner_output_file=""
	spinner_error_file=""
	selection_initial_query=$loading_query
	return "$spinner_status"
}

require_tty() {
	[ -r "$TTY_DEVICE" ] || die "An interactive terminal is required."
}

prompt_text() {
	prompt_label=$1
	prompt_default=${2:-}
	prompt_description=${3:-}
	require_tty
	printf '%b○%b  %b%s%b\n' "$BLUE" "$RESET" "$BOLD" "$prompt_label" "$RESET" >&2
	if [ -n "$prompt_description" ]; then
		detail "$prompt_description"
	fi
	if [ -n "$prompt_default" ]; then
		if [ -n "$prompt_description" ]; then
			detail "Default: $prompt_default"
		else
			detail "Press Enter to use $prompt_default"
		fi
	fi
	printf '%b│%b  %b❯ %b' "$LINE" "$RESET" "$BLUE" "$RESET" >&2
	IFS= read -r prompt_value <"$TTY_DEVICE" || die "Could not read from the terminal."
	if [ -z "$prompt_value" ]; then
		prompt_value=$prompt_default
	fi
	printf '%b│%b\n' "$LINE" "$RESET" >&2
}

prompt_secret() {
	prompt_label=$1
	require_tty
	terminal_state=$(stty -g <"$TTY_DEVICE")
	stty -echo <"$TTY_DEVICE"
	printf '%b○%b  %b%s%b\n' "$BLUE" "$RESET" "$BOLD" "$prompt_label" "$RESET" >&2
	printf '%b│%b  %b❯ %b' "$LINE" "$RESET" "$BLUE" "$RESET" >&2
	IFS= read -r prompt_value <"$TTY_DEVICE" || true
	restore_terminal
	printf '\n%b│%b\n' "$LINE" "$RESET" >&2
}

option_at() {
	wanted=$1
	current=0
	while IFS= read -r option_value; do
		if [ "$current" -eq "$wanted" ]; then
			printf '%s' "$option_value"
			return
		fi
		current=$((current + 1))
	done <<EOF
$filtered_options
EOF
}

render_options() {
	active=$1
	window_start=$2
	current=0
	while IFS= read -r option_value; do
		if [ "$current" -ge "$window_start" ] && [ "$current" -lt $((window_start + MAX_VISIBLE_OPTIONS)) ]; then
			display_option_value=$option_value
			if [ "$current" -eq 0 ] && [ "${menu_filter_enabled:-false}" = "true" ] && [ -n "${filter_query:-}" ]; then
				case "$option_value" in
				*:) display_option_value="$option_value $filter_query" ;;
				*) display_option_value="$option_value: $filter_query" ;;
				esac
			fi
			if [ "$current" -eq "$active" ]; then
				printf '%b│%b  %b❯ %s%b\r\n' "$LINE" "$RESET" "$BLUE" "$display_option_value" "$RESET" >&2
			else
				printf '%b│%b    %s\r\n' "$LINE" "$RESET" "$display_option_value" >&2
			fi
		fi
		current=$((current + 1))
	done <<EOF
$filtered_options
EOF
}

apply_option_filter() {
	filter_source=$1
	filter_prefix=$2
	filter_enabled=$3
	filtered_options=$(printf '%s\n' "$filter_source" | awk -v prefix="$filter_prefix" -v enabled="$filter_enabled" '
		BEGIN { lower_prefix = tolower(prefix) }
		NR == 1 || enabled != "true" || prefix == "" || index(tolower($0), lower_prefix) == 1 { print }
	')
	filtered_option_count=0
	while IFS= read -r filter_option; do
		[ -n "$filter_option" ] && filtered_option_count=$((filtered_option_count + 1))
	done <<EOF
$filtered_options
EOF
}

render_filter() {
	printf '%b│%b  %bFilter: %b%s\r\n' "$LINE" "$RESET" "$DIM" "$RESET" "$filter_query" >&2
}

begin_selection() {
	selection_header_open=$1
	printf '%b○%b  %b%s%b\n' "$BLUE" "$RESET" "$BOLD" "$selection_header_open" "$RESET" >&2
	if [ -n "${2:-}" ]; then
		detail "$2"
	fi
}

read_key() {
	key_code=$(dd if="$TTY_DEVICE" bs=1 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')
	if [ "$key_code" = "27" ]; then
		stty min 0 time 1 <"$TTY_DEVICE"
		key_two=$(dd if="$TTY_DEVICE" bs=1 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')
		key_three=$(dd if="$TTY_DEVICE" bs=1 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')
		if [ "$key_read_mode" = "blocking" ]; then
			stty min 1 time 0 <"$TTY_DEVICE"
		fi
		key_code="$key_code,$key_two,$key_three"
	fi
}

select_option() {
	select_label=$1
	shift
	[ "$#" -gt 0 ] || die "No options were provided for $select_label."
	require_tty
	if [ "$selection_header_open" = "$select_label" ]; then
		selection_header_open=""
		selection_header_line_count=2
	else
		printf '%b○%b  %b%s%b\n' "$BLUE" "$RESET" "$BOLD" "$select_label" "$RESET" >&2
		selection_header_line_count=1
	fi
	menu_filter_enabled=$selection_filter_enabled
	selection_filter_enabled="false"
	filter_query=$selection_initial_query
	selection_initial_query=""
	selection_query=""
	menu_hint=$selection_hint
	selection_hint=""
	if [ -n "$menu_hint" ]; then
		detail "$menu_hint"
	fi
	if [ "$menu_filter_enabled" = "true" ]; then
		detail "Type to filter · ↑/↓ navigate · Enter to select"
		render_filter
	else
		detail "↑/↓ navigate · Enter to select"
	fi
	printf '%b│%b\n' "$LINE" "$RESET" >&2
	all_options=$(printf '%s\n' "$@")
	apply_option_filter "$all_options" "$filter_query" "$menu_filter_enabled"
	option_count=$filtered_option_count
	selected_index=$selection_default_index
	selection_default_index=0
	if [ -n "$filter_query" ]; then
		selected_index=0
	fi
	if [ "$selected_index" -ge "$option_count" ]; then
		selected_index=0
	fi
	window_start=0
	if [ "$option_count" -lt "$MAX_VISIBLE_OPTIONS" ]; then
		visible_option_count=$option_count
	else
		visible_option_count=$MAX_VISIBLE_OPTIONS
	fi
	rendered_option_count=$visible_option_count
	render_options "$selected_index" "$window_start"

	terminal_state=$(stty -g <"$TTY_DEVICE")
	stty -icanon -echo min 1 time 0 <"$TTY_DEVICE"
	key_read_mode="blocking"
	printf '\033[?25l' >&2
	cursor_hidden="true"
	while :; do
		read_key
		case "$key_code" in
		3)
			restore_terminal
			die "Setup cancelled."
			;;
		10 | 13)
			selected_value=$(option_at "$selected_index")
			selection_query=$filter_query
			selection_static_line_count=$((selection_header_line_count + 2))
			if [ -n "$menu_hint" ]; then
				selection_static_line_count=$((selection_static_line_count + 1))
			fi
			if [ "$menu_filter_enabled" = "true" ]; then
				selection_static_line_count=$((selection_static_line_count + 1))
			fi
			printf '\033[%sA\033[J' $((visible_option_count + selection_static_line_count)) >&2
			restore_terminal
			printf '%b●%b  %b%s%b\n' "$GREEN" "$RESET" "$BOLD" "$select_label" "$RESET" >&2
			detail "$selected_value"
			printf '%b│%b\n' "$LINE" "$RESET" >&2
			return
			;;
		27,91,65 | 27,79,65)
			selected_index=$(((selected_index - 1 + option_count) % option_count))
			;;
		27,91,66 | 27,79,66)
			selected_index=$(((selected_index + 1) % option_count))
			;;
		8 | 127)
			[ "$menu_filter_enabled" = "true" ] || continue
			if [ "$selected_index" -eq 0 ]; then custom_entry_selected="true"; else custom_entry_selected="false"; fi
			filter_query=${filter_query%?}
			apply_option_filter "$all_options" "$filter_query" "$menu_filter_enabled"
			option_count=$filtered_option_count
			if [ "$custom_entry_selected" = "true" ]; then selected_index=0; elif [ "$option_count" -gt 1 ]; then selected_index=1; else selected_index=0; fi
			window_start=0
			if [ "$option_count" -lt "$MAX_VISIBLE_OPTIONS" ]; then visible_option_count=$option_count; else visible_option_count=$MAX_VISIBLE_OPTIONS; fi
			printf '\033[%sA\033[J' $((rendered_option_count + 2)) >&2
			render_filter
			printf '%b│%b\n' "$LINE" "$RESET" >&2
			render_options "$selected_index" "$window_start"
			rendered_option_count=$visible_option_count
			continue
			;;
		*)
			case "$key_code" in
			3[2-9] | [4-9][0-9] | 1[01][0-9] | 12[0-6])
				[ "$menu_filter_enabled" = "true" ] || continue
				if [ "$selected_index" -eq 0 ]; then custom_entry_selected="true"; else custom_entry_selected="false"; fi
				typed_character=$(printf "\\$(printf '%03o' "$key_code")")
				filter_query="$filter_query$typed_character"
				apply_option_filter "$all_options" "$filter_query" "$menu_filter_enabled"
				option_count=$filtered_option_count
				if [ "$custom_entry_selected" = "true" ]; then selected_index=0; elif [ "$option_count" -gt 1 ]; then selected_index=1; else selected_index=0; fi
				window_start=0
				if [ "$option_count" -lt "$MAX_VISIBLE_OPTIONS" ]; then visible_option_count=$option_count; else visible_option_count=$MAX_VISIBLE_OPTIONS; fi
				printf '\033[%sA\033[J' $((rendered_option_count + 2)) >&2
				render_filter
				printf '%b│%b\n' "$LINE" "$RESET" >&2
				render_options "$selected_index" "$window_start"
				rendered_option_count=$visible_option_count
				continue
				;;
			*) continue ;;
			esac
			;;
		esac
		if [ "$selected_index" -lt "$window_start" ]; then
			window_start=$selected_index
		elif [ "$selected_index" -ge $((window_start + visible_option_count)) ]; then
			window_start=$((selected_index - visible_option_count + 1))
		fi
		printf '\033[%sA\033[J' "$visible_option_count" >&2
		render_options "$selected_index" "$window_start"
	done
}

json_first_string() {
	json_key=$1
	if command -v jq >/dev/null 2>&1; then
		jq -r --arg key "$json_key" '[.. | objects | .[$key]? | select(type == "string")][0] // empty'
	else
		sed -n 's/.*"'"$json_key"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1
	fi
}

json_all_strings() {
	json_key=$1
	if command -v jq >/dev/null 2>&1; then
		jq -r --arg key "$json_key" '.. | objects | .[$key]? | select(type == "string")'
	else
		sed -n 's/.*"'"$json_key"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
	fi
}

json_tag_value() {
	tag_key=$1
	if command -v jq >/dev/null 2>&1; then
		jq -r --arg wanted "$tag_key" '[.. | objects | select(.key? == $wanted) | .value? | select(type == "string")][0] // empty'
	else
		awk -v wanted="$tag_key" '
		/"key"[[:space:]]*:/ {
			line=$0
			sub(/^.*"key"[[:space:]]*:[[:space:]]*"/, "", line)
			sub(/".*$/, "", line)
			current_key=line
		}
		/"value"[[:space:]]*:/ {
			line=$0
			sub(/^.*"value"[[:space:]]*:[[:space:]]*"/, "", line)
			sub(/".*$/, "", line)
			if (current_key == wanted) { print line; exit }
			current_key=""
		}
	'
	fi
}

json_experiment_strings() {
	json_key=$1
	if command -v jq >/dev/null 2>&1; then
		jq -r --arg key "$json_key" '.experiments[]? | .[$key] | select(type == "string")'
	else
		sed '
s/"experiment_id"/\
"experiment_id"/g
s/"name"/\
"name"/g
' | json_all_strings "$json_key"
	fi
}

json_escape() {
	printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

trim_whitespace() {
	printf '%s' "$1" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

normalize_workspace_url() {
	workspace_value=$1
	case "$workspace_value" in
	http://* | https://*) ;;
	*) workspace_value="https://$workspace_value" ;;
	esac
	printf '%s' "$workspace_value" | sed 's#\(https\{0,1\}://[^/]*\).*#\1#; s#/$##'
}

normalize_tracking_uri() {
	tracking_value=$1
	case "$tracking_value" in
	http://* | https://*) ;;
	*) tracking_value="https://$tracking_value" ;;
	esac
	printf '%s' "$tracking_value" | sed 's/[?#].*$//; s#/$##'
}

WORKSPACE_URL=""
PROFILE=""
TRACKING_URI=""
EXPERIMENT_ID=""
EXPERIMENT_NAME=""
UC_SCHEMA=""
WAREHOUSE_ID=""
AGENT_NAME=""
WORKSPACE_URL_EXPLICIT="false"
PROFILE_EXPLICIT="false"

usage() {
	printf '%s\n' \
		"Usage: setup.sh [options]" \
		"" \
		"Options:" \
		"  --workspace-url <url>         Databricks workspace URL" \
		"  --profile <name>              Databricks CLI profile" \
		"  --tracking-uri <url>          Existing OSS MLflow server" \
		"  --experiment-id <id>          Existing experiment" \
		"  --experiment-name <name>      Experiment name or workspace path" \
		"  --uc-schema <catalog.schema>  Unity Catalog trace storage" \
		"  --warehouse-id <id>           Databricks SQL warehouse" \
		"  --agent <name>                claude, codex, or opencode" \
		"  -h, --help                    Show this help"
}

parse_args() {
	while [ "$#" -gt 0 ]; do
		case "$1" in
		--workspace-url)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			WORKSPACE_URL=$2
			WORKSPACE_URL_EXPLICIT="true"
			shift 2
			;;
		--profile)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			PROFILE=$2
			PROFILE_EXPLICIT="true"
			shift 2
			;;
		--tracking-uri)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			TRACKING_URI=$2
			shift 2
			;;
		--experiment-id)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			EXPERIMENT_ID=$2
			shift 2
			;;
		--experiment-name)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			EXPERIMENT_NAME=$2
			shift 2
			;;
		--uc-schema)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			UC_SCHEMA=$2
			shift 2
			;;
		--warehouse-id)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			WAREHOUSE_ID=$2
			shift 2
			;;
		--agent)
			[ "$#" -ge 2 ] || die "$1 requires a value."
			AGENT_NAME=$2
			shift 2
			;;
		-h | --help)
			usage
			exit 0
			;;
		*) die "Unknown option: $1" ;;
		esac
	done
}

inspect_repository() {
	if repo_root=$(git rev-parse --show-toplevel 2>/dev/null); then
		repo_name=$(basename "$repo_root")
		git_changes=$(git -C "$repo_root" status --short --untracked-files=no 2>/dev/null || true)
		if [ -n "$git_changes" ]; then
			warning "Git changes detected" "This repository has local changes:"
			change_number=1
			printf '%s\n' "$git_changes" | while IFS= read -r change; do
				change_path=$(printf '%s' "$change" | sed 's/^...//')
				detail "$change_number. $change_path"
				change_number=$((change_number + 1))
			done
			printf '%b│%b\n' "$LINE" "$RESET" >&2
			select_option "Continue with setup?" \
				"Exit and protect current changes" \
				"Continue anyway"
			if [ "$selected_index" -eq 0 ]; then
				die "Setup cancelled. Commit or stash the changes, then rerun setup."
			fi
		fi
		success "Repository detected" "$repo_root"
	else
		repo_root=$(pwd)
		repo_name=$(basename "$repo_root")
		warning "Git repository not found" "The agent's edits cannot be reviewed or reverted with Git."
		select_option "Continue with setup?" "Exit and initialize Git first" "Continue without Git"
		[ "$selected_index" -eq 1 ] || die "Setup cancelled."
	fi
}

show_manual_setup() {
	printf '%b●%b  %bContinue manually%b\n' "$GREEN" "$RESET" "$BOLD" "$RESET" >&2
	printf '%b│%b\n' "$LINE" "$RESET" >&2
	primary_detail "Use the resources configured above to add MLflow Tracing to this project:"
	printf '%b│%b\n' "$LINE" "$RESET" >&2
	case "$backend" in
	databricks)
		primary_detail "1. Install mlflow-tracing in this project."
		primary_detail "2. Set MLFLOW_TRACKING_URI=$TRACKING_URI and MLFLOW_EXPERIMENT_ID=$EXPERIMENT_ID."
		if [ -n "$WAREHOUSE_ID" ]; then
			primary_detail "3. Set MLFLOW_TRACING_SQL_WAREHOUSE_ID=$WAREHOUSE_ID."
		fi
		if [ -n "$PROFILE" ]; then
			primary_detail "Authenticate with: $DATABRICKS_BIN auth login --host $WORKSPACE_URL --profile $PROFILE"
		else
			primary_detail "Set DATABRICKS_HOST=$WORKSPACE_URL and authenticate with: $DATABRICKS_BIN auth login --host $WORKSPACE_URL"
		fi
		primary_detail "Enable tracing for your framework, run one request, and confirm the trace appears in MLflow."
		;;
	remote)
		primary_detail "1. Install mlflow-tracing in this project."
		primary_detail "2. Set MLFLOW_TRACKING_URI=$TRACKING_URI and MLFLOW_EXPERIMENT_ID=$EXPERIMENT_ID."
		if [ -n "${MLFLOW_TRACKING_USERNAME:-}" ] && [ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]; then
			primary_detail "3. Export MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD in your application environment. Do not commit their values."
		elif [ -n "${MLFLOW_TRACKING_TOKEN:-}" ]; then
			primary_detail "3. Export MLFLOW_TRACKING_TOKEN in your application environment. Do not commit its value."
		fi
		if [ -n "${MLFLOW_WORKSPACE:-}" ]; then
			primary_detail "Set MLFLOW_WORKSPACE=$MLFLOW_WORKSPACE."
		fi
		primary_detail "Enable tracing for your framework, run one request, and confirm the trace appears in MLflow."
		;;
	local)
		primary_detail "1. Install mlflow-tracing in this project."
		primary_detail "2. Configure it to use $TRACKING_URI and experiment $EXPERIMENT_NAME."
		primary_detail "3. Run one request and confirm the trace appears in MLflow."
		;;
	esac
	printf '%b│%b\n' "$LINE" "$RESET" >&2
	primary_detail "MLflow Tracing quickstart:"
	printf '%b│%b  %b%s%b\n' "$LINE" "$RESET" "$YELLOW" "$manual_setup_docs" "$RESET" >&2
	printf '%b│%b\n' "$LINE" "$RESET" >&2
	footer "Setup complete"
}

validate_agent_name() {
	case "$AGENT_NAME" in
	"" | claude | codex | opencode) ;;
	*) die "Unsupported coding agent: $AGENT_NAME" ;;
	esac
}

choose_agent() {
	manual_setup_docs="https://mlflow.org/docs/latest/genai/tracing/quickstart/"
	agent_options=""
	for candidate in claude codex opencode; do
		if command -v "$candidate" >/dev/null 2>&1; then
			agent_options="${agent_options}${candidate}\n"
		fi
	done
	if [ -n "$AGENT_NAME" ]; then
		command -v "$AGENT_NAME" >/dev/null 2>&1 || die "Coding agent '$AGENT_NAME' is not installed."
		agent_choice=$AGENT_NAME
	else
		set --
		while IFS= read -r candidate; do
			[ -n "$candidate" ] && set -- "$@" "$candidate"
		done <<EOF
$(printf '%b' "$agent_options")
EOF
		set -- "$@" "Configure manually"
		begin_selection "Choose a coding agent" "It can instrument this project and verify a trace in MLflow."
		select_option "Choose a coding agent" "$@"
		agent_choice=$selected_value
		if [ "$agent_choice" = "Configure manually" ]; then
			show_manual_setup
			exit 0
		fi
	fi
	case "$agent_choice" in
	claude) agent_display="Claude Code" ;;
	codex) agent_display="OpenAI Codex" ;;
	opencode) agent_display="OpenCode" ;;
	*) die "Unsupported coding agent: $agent_choice" ;;
	esac
	success "Coding agent" "$agent_display"
}

choose_backend() {
	configured_tracking_uri=${MLFLOW_TRACKING_URI:-}
	if [ -n "$WORKSPACE_URL" ] || [ -n "$PROFILE" ]; then
		backend="databricks"
	elif [ -n "$TRACKING_URI" ]; then
		backend="remote"
	elif [ "$configured_tracking_uri" = "databricks" ] || printf '%s' "$configured_tracking_uri" | grep -q '^databricks://'; then
		backend="databricks"
		if [ -z "$PROFILE" ]; then
			PROFILE=$(printf '%s' "$configured_tracking_uri" | sed 's#^databricks://##')
			[ "$PROFILE" = "databricks" ] && PROFILE=""
		fi
	elif [ -n "$configured_tracking_uri" ]; then
		backend="remote"
		TRACKING_URI=$configured_tracking_uri
	else
		select_option "Where should MLflow store traces?" \
			"Databricks" \
			"Existing OSS MLflow server" \
			"New local MLflow server"
		case "$selected_index" in
		0) backend="databricks" ;;
		1) backend="remote" ;;
		2) backend="local" ;;
		esac
	fi
}

find_compatible_databricks_cli() {
	for candidate_dir in "${XDG_BIN_HOME:-}" "$HOME/.local/bin" "${CARGO_HOME:-$HOME/.cargo}/bin"; do
		[ -n "$candidate_dir" ] || continue
		candidate="$candidate_dir/databricks"
		if [ -x "$candidate" ] && "$candidate" auth profiles --help >/dev/null 2>&1; then
			printf '%s' "$candidate"
			return
		fi
	done
	if command -v databricks >/dev/null 2>&1; then
		candidate=$(command -v databricks)
		if "$candidate" auth profiles --help >/dev/null 2>&1; then
			printf '%s' "$candidate"
		fi
	fi
}

databricks_cli_target() {
	case "$(uname -s)" in
	Darwin) cli_os="darwin" ;;
	Linux) cli_os="linux" ;;
	*) die "Automatic Databricks CLI setup supports macOS and Linux." ;;
	esac
	case "$(uname -m)" in
	x86_64 | amd64) cli_arch="amd64" ;;
	arm64 | aarch64) cli_arch="arm64" ;;
	*) die "Unsupported architecture: $(uname -m)" ;;
	esac
	printf '%s|%s' "$cli_os" "$cli_arch"
}

ensure_databricks_cli() {
	DATABRICKS_BIN=$(find_compatible_databricks_cli)
	[ -z "$DATABRICKS_BIN" ] || return 0
	command -v curl >/dev/null 2>&1 || die "curl is required to download the Databricks CLI."
	command -v unzip >/dev/null 2>&1 || die "unzip is required to install the Databricks CLI."
	if command -v databricks >/dev/null 2>&1; then
		progress "Databricks CLI update required" "Installing the latest release…"
	else
		progress "Databricks CLI not found" "Installing the latest release…"
	fi
	release_json=$(curl -fsSL -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/databricks/cli/releases/latest) || die "Could not determine the latest Databricks CLI release."
	cli_tag=$(printf '%s' "$release_json" | json_first_string tag_name)
	[ -n "$cli_tag" ] || die "Could not determine the latest Databricks CLI release."
	cli_version=${cli_tag#v}
	cli_target=$(databricks_cli_target)
	cli_os=${cli_target%%|*}
	cli_arch=${cli_target#*|}
	cli_asset="databricks_cli_${cli_version}_${cli_os}_${cli_arch}.zip"
	cli_url="https://github.com/databricks/cli/releases/download/${cli_tag}/${cli_asset}"
	setup_tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlflow-databricks-cli.XXXXXX")
	curl -fsSL "$cli_url" -o "$setup_tmp_dir/$cli_asset" || die "Could not download Databricks CLI $cli_tag for ${cli_os}/${cli_arch}."
	unzip -q "$setup_tmp_dir/$cli_asset" -d "$setup_tmp_dir" || die "Could not extract the Databricks CLI."
	[ -f "$setup_tmp_dir/databricks" ] || die "Databricks CLI installation completed, but the CLI could not be found."
	cli_install_dir=${XDG_BIN_HOME:-$HOME/.local/bin}
	mkdir -p "$cli_install_dir" || die "Could not create $cli_install_dir."
	chmod +x "$setup_tmp_dir/databricks"
	mv "$setup_tmp_dir/databricks" "$cli_install_dir/databricks" || die "Could not install the Databricks CLI in $cli_install_dir."
	DATABRICKS_BIN="$cli_install_dir/databricks"
	rm -rf "$setup_tmp_dir"
	setup_tmp_dir=""
	[ -x "$DATABRICKS_BIN" ] || die "Databricks CLI installation completed, but the CLI could not be found."
	success "Databricks CLI ready" "$DATABRICKS_BIN"
}

dbx_json() {
	if [ -n "$PROFILE" ]; then
		DATABRICKS_HOST= "$DATABRICKS_BIN" "$@" --output json --profile "$PROFILE"
	elif [ -n "$WORKSPACE_URL" ]; then
		DATABRICKS_CONFIG_PROFILE= DATABRICKS_HOST="$WORKSPACE_URL" "$DATABRICKS_BIN" "$@" --output json
	else
		"$DATABRICKS_BIN" "$@" --output json
	fi
}

databricks_token_user() {
	if [ -n "$PROFILE" ]; then
		token_json=$("$DATABRICKS_BIN" auth token "$PROFILE" --output json 2>/dev/null) || return
	elif [ -n "$WORKSPACE_URL" ]; then
		token_json=$(DATABRICKS_CONFIG_PROFILE= DATABRICKS_HOST="$WORKSPACE_URL" "$DATABRICKS_BIN" auth token --output json 2>/dev/null) || return
	else
		token_json=$("$DATABRICKS_BIN" auth token --output json 2>/dev/null) || return
	fi
	access_token=$(printf '%s\n' "$token_json" | json_first_string access_token)
	case "$access_token" in
	*.*.*) ;;
	*) return ;;
	esac
	token_payload=$(printf '%s' "$access_token" | cut -d. -f2 | tr '_-' '/+')
	case $((${#token_payload} % 4)) in
	2) token_payload="${token_payload}==" ;;
	3) token_payload="${token_payload}=" ;;
	esac
	if decoded_token=$(printf '%s' "$token_payload" | base64 -d 2>/dev/null); then
		:
	elif decoded_token=$(printf '%s' "$token_payload" | base64 -D 2>/dev/null); then
		:
	else
		return
	fi
	token_subject=$(printf '%s\n' "$decoded_token" | json_first_string sub)
	case "$token_subject" in
	*@*) printf '%s' "$token_subject" ;;
	esac
}

list_databricks_profiles() {
	"$DATABRICKS_BIN" auth profiles --skip-validate 2>/dev/null | awk 'NR > 1 && $1 != "" && $2 ~ /^https?:\/\// { print $1 "|" $2 }'
}

databricks_auth_valid() {
	if [ -n "$PROFILE" ]; then
		"$DATABRICKS_BIN" auth token "$PROFILE" --output json >/dev/null 2>&1
	elif [ -n "$WORKSPACE_URL" ]; then
		DATABRICKS_CONFIG_PROFILE= DATABRICKS_HOST="$WORKSPACE_URL" "$DATABRICKS_BIN" auth token --output json >/dev/null 2>&1
	else
		"$DATABRICKS_BIN" auth token --output json >/dev/null 2>&1
	fi
}

resolve_databricks_profile() {
	profile_selection_needed="false"
	if [ "$WORKSPACE_URL_EXPLICIT" = "false" ] && [ -z "$PROFILE" ] && [ -n "${DATABRICKS_CONFIG_PROFILE:-}" ]; then
		PROFILE=$DATABRICKS_CONFIG_PROFILE
	fi
	if [ "$PROFILE_EXPLICIT" = "false" ] && [ -z "$WORKSPACE_URL" ] && [ -n "${DATABRICKS_HOST:-}" ]; then
		WORKSPACE_URL=$DATABRICKS_HOST
	fi
	if [ -n "$WORKSPACE_URL" ]; then
		WORKSPACE_URL=$(normalize_workspace_url "$WORKSPACE_URL")
	fi
	if [ -z "$PROFILE" ] && [ -z "$WORKSPACE_URL" ]; then
		profile_selection_needed="true"
		begin_selection "Choose a Databricks workspace" "Select a saved profile, or type a profile name or URL."
	fi
	if [ "$profile_selection_needed" = "true" ]; then
		load_with_manual_option "Choose a Databricks workspace" "Enter a profile name or workspace URL" "Loading available Databricks profiles…" list_databricks_profiles || true
		if [ "$loading_manual_selected" = "true" ]; then
			profiles=""
		else
			profiles=$spinner_output
		fi
	elif run_with_spinner "Loading Databricks profiles…" list_databricks_profiles; then
		profiles=$spinner_output
	else
		profiles=""
	fi
	if [ -n "$PROFILE" ]; then
		profile_host=$(printf '%s\n' "$profiles" | awk -F '|' -v profile="$PROFILE" '$1 == profile { print $2; exit }')
		if [ -n "$profile_host" ]; then
			profile_host=$(normalize_workspace_url "$profile_host")
		fi
		if [ -n "$WORKSPACE_URL" ] && [ -n "$profile_host" ] && [ "$WORKSPACE_URL" != "$profile_host" ]; then
			die "Databricks profile '$PROFILE' points to $profile_host, not $WORKSPACE_URL."
		fi
		if [ -z "$WORKSPACE_URL" ]; then
			WORKSPACE_URL=$profile_host
		fi
		[ -n "$WORKSPACE_URL" ] || die "Databricks profile '$PROFILE' was not found."
		return
	fi
	if [ -n "$WORKSPACE_URL" ]; then
		PROFILE=$(printf '%s\n' "$profiles" | awk -F '|' -v host="$WORKSPACE_URL" '$2 == host { print $1; exit }')
		return
	fi
	if [ "$loading_manual_selected" = "true" ]; then
		selected_value="Enter a profile name or workspace URL"
	else
		set -- "Enter a profile name or workspace URL"
		while IFS='|' read -r profile_name profile_url; do
			[ -n "$profile_name" ] && set -- "$@" "$profile_name    $(printf '%s' "$profile_url" | sed 's#https\{0,1\}://##')"
		done <<EOF
$profiles
EOF
		if [ -n "$profiles" ]; then
			selection_default_index=1
		fi
		selection_filter_enabled="true"
		select_option "Choose a Databricks workspace" "$@"
	fi
	if [ "$selected_value" = "Enter a profile name or workspace URL" ]; then
		entered_profile=$selection_query
		if [ -z "$entered_profile" ]; then
			prompt_text "Databricks profile name or workspace URL" ""
			entered_profile=$prompt_value
		fi
		profile_host=$(printf '%s\n' "$profiles" | awk -F '|' -v profile="$entered_profile" '$1 == profile { print $2; exit }')
		if [ -n "$profile_host" ]; then
			PROFILE=$entered_profile
			WORKSPACE_URL=$profile_host
		else
			case "$entered_profile" in
			http://* | https://* | *.*)
				WORKSPACE_URL=$(normalize_workspace_url "$entered_profile")
				PROFILE=""
				;;
			*) die "Databricks profile '$entered_profile' was not found." ;;
			esac
		fi
	else
		selected_profile=$(printf '%s' "$selected_value" | awk '{print $1}')
		PROFILE=$selected_profile
		WORKSPACE_URL=$(printf '%s\n' "$profiles" | awk -F '|' -v profile="$PROFILE" '$1 == profile { print $2; exit }')
	fi
}

authenticate_databricks() {
	current_user=""
	if ! databricks_auth_valid; then
		progress "Sign in to Databricks" "Opening $WORKSPACE_URL in your browser…"
		if [ -n "$PROFILE" ]; then
			"$DATABRICKS_BIN" auth login --host "$WORKSPACE_URL" --profile "$PROFILE" <"$TTY_DEVICE" || die "Databricks authentication failed."
		else
			DATABRICKS_CONFIG_PROFILE= "$DATABRICKS_BIN" auth login --host "$WORKSPACE_URL" <"$TTY_DEVICE" || die "Databricks authentication failed."
		fi
		databricks_auth_valid || die "Databricks authentication could not be verified."
	fi
	current_user=$(databricks_token_user || true)
	if [ -n "$PROFILE" ]; then
		success "Workspace" "$PROFILE · $WORKSPACE_URL"
	else
		success "Workspace" "$WORKSPACE_URL"
	fi
	if [ -n "$current_user" ]; then
		success "Authenticated" "$current_user"
	else
		if [ -n "$PROFILE" ]; then
			success "Authenticated" "Profile $PROFILE"
		else
			success "Authenticated" "Databricks unified authentication"
		fi
	fi
}

resolve_databricks_experiment() {
	experiment_created="false"
	existing_experiment_selected="false"
	if [ -n "$EXPERIMENT_ID" ]; then
		experiment_json=$(dbx_json experiments get-experiment "$EXPERIMENT_ID") || die "Experiment '$EXPERIMENT_ID' was not found."
		EXPERIMENT_NAME=$(printf '%s\n' "$experiment_json" | json_first_string name)
	else
		if [ -z "$EXPERIMENT_NAME" ]; then
			if [ -n "$current_user" ]; then
				default_experiment="/Users/$current_user/$repo_name"
				begin_selection "Choose an MLflow experiment" "Experiments group traces. Create one or connect an existing experiment."
				selection_hint="Default path: $default_experiment"
				select_option "Choose an MLflow experiment" \
					"Create a new experiment" \
					"Use existing experiment path or ID"
				if [ "$selected_index" -eq 0 ]; then
					prompt_text "New experiment path" "$default_experiment" "Type an experiment path, or press Enter to use the default."
					EXPERIMENT_NAME=$(trim_whitespace "$prompt_value")
					[ -n "$EXPERIMENT_NAME" ] || die "An absolute Databricks experiment path is required."
				else
					prompt_text "Existing experiment path or ID" ""
					existing_experiment=$(trim_whitespace "$prompt_value")
					[ -n "$existing_experiment" ] || die "An experiment path or ID is required."
					case "$existing_experiment" in
					/*)
						EXPERIMENT_NAME=$existing_experiment
						existing_experiment_selected="true"
						;;
					*)
						EXPERIMENT_ID=$existing_experiment
						;;
					esac
				fi
			else
				prompt_text "Experiment path" ""
				EXPERIMENT_NAME=$(trim_whitespace "$prompt_value")
				[ -n "$EXPERIMENT_NAME" ] || die "An absolute Databricks experiment path is required."
			fi
		fi
		if [ -n "$EXPERIMENT_ID" ]; then
			experiment_json=$(dbx_json experiments get-experiment "$EXPERIMENT_ID") || die "Experiment '$EXPERIMENT_ID' was not found."
			EXPERIMENT_NAME=$(printf '%s\n' "$experiment_json" | json_first_string name)
		elif experiment_json=$(dbx_json experiments get-by-name "$EXPERIMENT_NAME" 2>/dev/null); then
			EXPERIMENT_ID=$(printf '%s\n' "$experiment_json" | json_first_string experiment_id)
		else
			[ "$existing_experiment_selected" = "false" ] || die "Experiment '$EXPERIMENT_NAME' was not found."
			experiment_json=$(dbx_json experiments create-experiment "$EXPERIMENT_NAME") || die "Could not create experiment '$EXPERIMENT_NAME'."
			EXPERIMENT_ID=$(printf '%s\n' "$experiment_json" | json_first_string experiment_id)
			[ -n "$EXPERIMENT_ID" ] || die "Databricks did not return an experiment ID."
			dbx_json experiments set-experiment-tag "$EXPERIMENT_ID" mlflow.experimentKind genai_development >/dev/null 2>&1 || true
			experiment_created="true"
		fi
	fi
	[ -n "$EXPERIMENT_ID" ] || die "Databricks did not return an experiment ID."
	if [ "$experiment_created" = "true" ]; then
		experiment_label="Experiment created"
	else
		experiment_label="Experiment"
	fi
	success "$experiment_label" "$EXPERIMENT_NAME · $EXPERIMENT_ID"
	trace_destination=$(printf '%s\n' "$experiment_json" | json_tag_value mlflow.experiment.databricksTraceDestinationPath)
}

validate_uc_schema() {
	printf '%s\n' "$1" | awk -F '.' 'NF == 2 && $1 != "" && $2 != "" { valid=1 } END { exit !valid }'
}

select_uc_schema() {
	if [ -n "$UC_SCHEMA" ]; then
		validate_uc_schema "$UC_SCHEMA" || die "--uc-schema must use catalog.schema format."
		return
	fi
	begin_selection "Choose a Unity Catalog catalog" "Choose where MLflow stores its trace tables."
	load_with_manual_option "Choose a Unity Catalog catalog" "Enter catalog.schema" "Loading available Unity Catalog catalogs…" dbx_json catalogs list --max-results 100 || die "Could not list Unity Catalog catalogs."
	if [ "$loading_manual_selected" = "true" ]; then
		selected_value="Enter catalog.schema"
	else
		catalog_json=$spinner_output
		set -- "Enter catalog.schema"
		while IFS= read -r catalog_name; do
			[ -n "$catalog_name" ] && set -- "$@" "$catalog_name"
		done <<EOF
$(printf '%s\n' "$catalog_json" | json_all_strings name | sort -u)
EOF
		selection_default_index=0
		selection_filter_enabled="true"
		select_option "Choose a Unity Catalog catalog" "$@"
	fi
	if [ "$selected_value" = "Enter catalog.schema" ]; then
		UC_SCHEMA=$selection_query
		if [ -z "$UC_SCHEMA" ]; then
			prompt_text "Unity Catalog destination (catalog.schema)" ""
			UC_SCHEMA=$prompt_value
		fi
		validate_uc_schema "$UC_SCHEMA" || die "Unity Catalog destination must use catalog.schema format."
		success "Trace storage" "$UC_SCHEMA"
		return
	fi
	catalog_name=$selected_value
	begin_selection "Choose a Unity Catalog schema" "Choose a schema in $catalog_name."
	load_with_manual_option "Choose a Unity Catalog schema" "Enter a schema name" "Loading available schemas in ${catalog_name}…" dbx_json schemas list "$catalog_name" --max-results 100 || die "Could not list schemas in '$catalog_name'."
	if [ "$loading_manual_selected" = "true" ]; then
		selected_value="Enter a schema name"
	else
		schema_json=$spinner_output
		set -- "Enter a schema name"
		while IFS= read -r schema_name; do
			[ -n "$schema_name" ] && set -- "$@" "$schema_name"
		done <<EOF
$(printf '%s\n' "$schema_json" | json_all_strings name | sort -u)
EOF
		selection_default_index=0
		selection_filter_enabled="true"
		select_option "Choose a Unity Catalog schema" "$@"
	fi
	if [ "$selected_value" = "Enter a schema name" ]; then
		schema_name=$selection_query
		if [ -z "$schema_name" ]; then
			prompt_text "Schema name" ""
			schema_name=$prompt_value
		fi
		[ -n "$schema_name" ] || die "A schema name is required."
	else
		schema_name=$selected_value
	fi
	UC_SCHEMA="$catalog_name.$schema_name"
	success "Trace storage" "$UC_SCHEMA"
}

select_warehouse() {
	if [ -n "$WAREHOUSE_ID" ]; then
		success "SQL warehouse" "$WAREHOUSE_ID"
		return
	fi
	begin_selection "Choose a SQL warehouse" "Required compute for creating and querying trace tables."
	load_with_manual_option "Choose a SQL warehouse" "Enter a warehouse ID" "Loading available SQL warehouses…" dbx_json warehouses list || die "Could not list SQL warehouses."
	if [ "$loading_manual_selected" = "true" ]; then
		selected_value="Enter a warehouse ID"
	else
		warehouse_json=$spinner_output
		warehouse_rows=$(printf '%s\n' "$warehouse_json" | awk '
		/"id"[[:space:]]*:/ { line=$0; sub(/^.*"id"[[:space:]]*:[[:space:]]*"/, "", line); sub(/".*$/, "", line); id=line }
		/"name"[[:space:]]*:/ { line=$0; sub(/^.*"name"[[:space:]]*:[[:space:]]*"/, "", line); sub(/".*$/, "", line); name=line }
		/"state"[[:space:]]*:/ { line=$0; sub(/^.*"state"[[:space:]]*:[[:space:]]*"/, "", line); sub(/".*$/, "", line); state=line; if (id != "" && name != "") { print id "|" name "|" state; id=""; name=""; state="" } }
	')
		set -- "Enter a warehouse ID"
		while IFS='|' read -r warehouse_id warehouse_name warehouse_state; do
			[ -n "$warehouse_id" ] && set -- "$@" "$warehouse_name    $warehouse_state · $warehouse_id"
		done <<EOF
$warehouse_rows
EOF
		if [ -n "$warehouse_rows" ]; then
			selection_default_index=1
		else
			selection_default_index=0
		fi
		selection_filter_enabled="true"
		select_option "Choose a SQL warehouse" "$@"
	fi
	if [ "$selected_value" = "Enter a warehouse ID" ]; then
		WAREHOUSE_ID=$selection_query
		if [ -z "$WAREHOUSE_ID" ]; then
			prompt_text "SQL warehouse ID" ""
			WAREHOUSE_ID=$prompt_value
		fi
		[ -n "$WAREHOUSE_ID" ] || die "A SQL warehouse ID is required."
		success "SQL warehouse" "$WAREHOUSE_ID"
	else
		WAREHOUSE_ID=$(printf '%s' "$selected_value" | awk '{print $NF}')
		[ -n "$WAREHOUSE_ID" ] || die "Could not resolve the selected warehouse."
		success "SQL warehouse" "$selected_value"
	fi
}

link_uc_trace_storage() {
	[ -z "$trace_destination" ] || {
		UC_SCHEMA=$(printf '%s' "$trace_destination" | awk -F '.' '{print $1 "." $2}')
		success "Trace storage" "$trace_destination"
		return
	}
	select_uc_schema
	catalog_name=$(printf '%s' "$UC_SCHEMA" | cut -d . -f 1)
	schema_name=$(printf '%s' "$UC_SCHEMA" | cut -d . -f 2)
	escaped_catalog_name=$(json_escape "$catalog_name")
	escaped_schema_name=$(json_escape "$schema_name")
	escaped_experiment_id=$(json_escape "$EXPERIMENT_ID")
	escaped_warehouse_id=$(json_escape "$WAREHOUSE_ID")
	create_body=$(printf '{"uc_table_prefix":{"catalog_name":"%s","schema_name":"%s","table_prefix":"%s"},"sql_warehouse_id":"%s"}' "$escaped_catalog_name" "$escaped_schema_name" "$escaped_experiment_id" "$escaped_warehouse_id")
	progress "Configuring Unity Catalog trace storage" "$UC_SCHEMA"
	location_json=$(dbx_json api post /api/5.0/mlflow/tracing/locations --json "$create_body") || die "Could not create Unity Catalog trace storage."
	resolved_catalog=$(printf '%s\n' "$location_json" | json_first_string catalog_name)
	resolved_schema=$(printf '%s\n' "$location_json" | json_first_string schema_name)
	resolved_prefix=$(printf '%s\n' "$location_json" | json_first_string table_prefix)
	[ -n "$resolved_catalog" ] || resolved_catalog=$catalog_name
	[ -n "$resolved_schema" ] || resolved_schema=$schema_name
	[ -n "$resolved_prefix" ] || resolved_prefix=$EXPERIMENT_ID
	escaped_resolved_catalog=$(json_escape "$resolved_catalog")
	escaped_resolved_schema=$(json_escape "$resolved_schema")
	escaped_resolved_prefix=$(json_escape "$resolved_prefix")
	resolved_location=$(printf '{"catalog_name":"%s","schema_name":"%s","table_prefix":"%s"' "$escaped_resolved_catalog" "$escaped_resolved_schema" "$escaped_resolved_prefix")
	for table_field in spans_table_name logs_table_name metrics_table_name annotations_table_name location_id; do
		table_value=$(printf '%s\n' "$location_json" | json_first_string "$table_field")
		if [ -n "$table_value" ]; then
			escaped_table_value=$(json_escape "$table_value")
			resolved_location="$resolved_location,$(printf '"%s":"%s"' "$table_field" "$escaped_table_value")"
		fi
	done
	resolved_location="$resolved_location}"
	link_body=$(printf '{"experiment_id":"%s","uc_table_prefix":%s}' "$escaped_experiment_id" "$resolved_location")
	dbx_json api post "/api/5.0/mlflow/experiments/$EXPERIMENT_ID/trace-location:link" --json "$link_body" >/dev/null || die "Could not link the experiment to Unity Catalog trace storage."
	trace_destination="$UC_SCHEMA.$EXPERIMENT_ID"
	success "Trace storage" "$trace_destination"
}

configure_databricks() {
	ensure_databricks_cli
	resolve_databricks_profile
	authenticate_databricks
	if [ -n "$PROFILE" ]; then
		TRACKING_URI="databricks://$PROFILE"
	else
		TRACKING_URI="databricks"
		DATABRICKS_HOST=$WORKSPACE_URL
		export DATABRICKS_HOST
	fi
	resolve_databricks_experiment
	if [ "$experiment_created" = "true" ]; then
		select_warehouse
		link_uc_trace_storage
	elif [ -n "$trace_destination" ]; then
		UC_SCHEMA=$(printf '%s' "$trace_destination" | awk -F '.' '{print $1 "." $2}')
		success "Trace storage" "$trace_destination"
		select_warehouse
	else
		success "Trace storage" "Existing experiment uses workspace storage"
	fi
}

tracking_uri_authority() {
	tracking_authority=${TRACKING_URI#*://}
	printf '%s' "${tracking_authority%%/*}"
}

validate_tracking_uri() {
	case "$(tracking_uri_authority)" in
	*@*) die "Do not include credentials in the MLflow tracking server URL. Enter them when prompted instead." ;;
	esac
}

require_secure_auth_transport() {
	case "$TRACKING_URI" in
	https://*) return ;;
	esac
	case "$(tracking_uri_authority)" in
	localhost | localhost:* | 127.0.0.1 | 127.0.0.1:* | \[::1\] | \[::1\]:*) return ;;
	esac
	die "Authentication requires HTTPS for non-local MLflow tracking servers."
}

curl_config_escape() {
	if printf '%s' "$1" | LC_ALL=C grep '[[:cntrl:]]' >/dev/null 2>&1; then
		die "MLflow credentials cannot contain control characters."
	fi
	printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

oss_curl() {
	set -- --connect-timeout 10 --max-time 60 "$@"
	if [ -n "${MLFLOW_WORKSPACE:-}" ]; then
		set -- -H "X-MLFLOW-WORKSPACE: $MLFLOW_WORKSPACE" "$@"
	fi
	case "${MLFLOW_TRACKING_INSECURE_TLS:-}" in
	1 | true | TRUE | True)
		[ -z "${MLFLOW_TRACKING_SERVER_CERT_PATH:-}" ] || die "MLFLOW_TRACKING_INSECURE_TLS and MLFLOW_TRACKING_SERVER_CERT_PATH cannot both be set."
		set -- --insecure "$@"
		;;
	esac
	if [ -n "${MLFLOW_TRACKING_SERVER_CERT_PATH:-}" ]; then
		set -- --cacert "$MLFLOW_TRACKING_SERVER_CERT_PATH" "$@"
	fi
	if [ -n "${MLFLOW_TRACKING_CLIENT_CERT_PATH:-}" ]; then
		set -- --cert "$MLFLOW_TRACKING_CLIENT_CERT_PATH" "$@"
	fi
	curl_auth_config=""
	if [ -n "${MLFLOW_TRACKING_USERNAME:-}" ] && [ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]; then
		require_secure_auth_transport
		curl_auth_config=$(mktemp "${TMPDIR:-/tmp}/mlflow-curl-auth.XXXXXX")
		chmod 600 "$curl_auth_config"
		curl_user=$(curl_config_escape "${MLFLOW_TRACKING_USERNAME:-}:${MLFLOW_TRACKING_PASSWORD:-}")
		printf 'user = "%s"\n' "$curl_user" >"$curl_auth_config"
	elif [ -n "${MLFLOW_TRACKING_TOKEN:-}" ]; then
		require_secure_auth_transport
		curl_auth_config=$(mktemp "${TMPDIR:-/tmp}/mlflow-curl-auth.XXXXXX")
		chmod 600 "$curl_auth_config"
		curl_token=$(curl_config_escape "$MLFLOW_TRACKING_TOKEN")
		printf 'header = "Authorization: Bearer %s"\n' "$curl_token" >"$curl_auth_config"
	fi
	if [ -n "$curl_auth_config" ]; then
		set -- --config "$curl_auth_config" "$@"
	fi
	if curl "$@"; then
		curl_status=0
	else
		curl_status=$?
	fi
	if [ -n "$curl_auth_config" ]; then
		rm -f "$curl_auth_config"
		curl_auth_config=""
	fi
	return "$curl_status"
}

check_oss_server() {
	if [ -z "$setup_tmp_dir" ]; then
		setup_tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlflow-setup.XXXXXX")
	fi
	auth_attempts=0
	while :; do
		status=$(oss_curl -sS -o "$setup_tmp_dir/response.json" -w '%{http_code}' \
			-X POST -H 'Content-Type: application/json' \
			-d '{"max_results":1}' \
			"$TRACKING_URI/api/2.0/mlflow/experiments/search" || true)
		case "$status" in
		200) return ;;
		401 | 403)
			auth_attempts=$((auth_attempts + 1))
			[ "$auth_attempts" -le 3 ] || die "MLflow authentication failed after 3 attempts."
			require_secure_auth_transport
			select_option "Authentication required" "Use an access token" "Use a username and password"
			if [ "$selected_index" -eq 0 ]; then
				prompt_secret "MLflow access token"
				MLFLOW_TRACKING_TOKEN=$prompt_value
				unset MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
				export MLFLOW_TRACKING_TOKEN
			else
				prompt_text "MLflow username" ""
				MLFLOW_TRACKING_USERNAME=$prompt_value
				prompt_secret "MLflow password"
				MLFLOW_TRACKING_PASSWORD=$prompt_value
				unset MLFLOW_TRACKING_TOKEN
				export MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
			fi
			;;
		*) die "MLflow server returned HTTP $status." ;;
		esac
	done
}

search_oss_experiments() {
	experiments_file="$setup_tmp_dir/experiments.json"
	page_file="$setup_tmp_dir/experiments-page.json"
	: >"$experiments_file"
	page_token=""
	while :; do
		if [ -n "$page_token" ]; then
			escaped_page_token=$(json_escape "$page_token")
			search_body=$(printf '{"max_results":100,"page_token":"%s"}' "$escaped_page_token")
		else
			search_body='{"max_results":100}'
		fi
		oss_curl -fsS -X POST -H 'Content-Type: application/json' -d "$search_body" \
			"$TRACKING_URI/api/2.0/mlflow/experiments/search" >"$page_file"
		cat "$page_file" >>"$experiments_file"
		next_page_token=$(json_first_string next_page_token <"$page_file")
		[ -n "$next_page_token" ] || break
		[ "$next_page_token" != "$page_token" ] || die "MLflow returned the same experiment page token twice."
		page_token=$next_page_token
	done
}

resolve_oss_experiment() {
	experiment_label="Experiment"
	if [ -n "$EXPERIMENT_ID" ]; then
		status=$(oss_curl -sS -o "$setup_tmp_dir/experiment.json" -w '%{http_code}' \
			"$TRACKING_URI/api/2.0/mlflow/experiments/get?experiment_id=$EXPERIMENT_ID")
		[ "$status" = "200" ] || die "Experiment '$EXPERIMENT_ID' was not found."
		EXPERIMENT_NAME=$(json_first_string name <"$setup_tmp_dir/experiment.json")
		return
	fi
	search_oss_experiments
	experiment_ids_file="$setup_tmp_dir/experiment-ids"
	experiment_names_file="$setup_tmp_dir/experiment-names"
	json_experiment_strings experiment_id <"$setup_tmp_dir/experiments.json" >"$experiment_ids_file"
	json_experiment_strings name <"$setup_tmp_dir/experiments.json" >"$experiment_names_file"
	if [ -n "$EXPERIMENT_NAME" ]; then
		experiment_line=$(awk -v wanted="$EXPERIMENT_NAME" '$0 == wanted { print NR; exit }' "$experiment_names_file")
		if [ -n "$experiment_line" ]; then
			EXPERIMENT_ID=$(sed -n "${experiment_line}p" "$experiment_ids_file")
		fi
	else
		set -- "Create a new experiment"
		while IFS= read -r experiment_name; do
			[ -n "$experiment_name" ] && set -- "$@" "$experiment_name"
		done <<EOF
$(cat "$experiment_names_file")
EOF
		select_option "Choose an experiment" "$@"
		if [ "$selected_index" -gt 0 ]; then
			EXPERIMENT_NAME=$selected_value
			EXPERIMENT_ID=$(sed -n "${selected_index}p" "$experiment_ids_file")
		else
			prompt_text "Experiment name" "$repo_name"
			EXPERIMENT_NAME=$prompt_value
		fi
	fi
	if [ -z "$EXPERIMENT_ID" ]; then
		escaped_experiment_name=$(json_escape "$EXPERIMENT_NAME")
		create_body=$(printf '{"name":"%s","tags":[{"key":"mlflow.experimentKind","value":"genai_development"}]}' "$escaped_experiment_name")
		oss_curl -fsS -X POST -H 'Content-Type: application/json' -d "$create_body" \
			"$TRACKING_URI/api/2.0/mlflow/experiments/create" >"$setup_tmp_dir/created.json"
		EXPERIMENT_ID=$(json_first_string experiment_id <"$setup_tmp_dir/created.json")
		[ -n "$EXPERIMENT_ID" ] || die "MLflow did not return an experiment ID."
		experiment_label="Experiment created"
	else
		experiment_label="Experiment"
	fi
}

configure_remote() {
	if [ -z "$TRACKING_URI" ]; then
		prompt_text "MLflow tracking server URL" ""
		TRACKING_URI=$prompt_value
	fi
	TRACKING_URI=$(normalize_tracking_uri "$TRACKING_URI")
	validate_tracking_uri
	check_oss_server
	success "MLflow server" "$TRACKING_URI"
	success "Authentication verified"
	resolve_oss_experiment
	success "$experiment_label" "$EXPERIMENT_NAME · $EXPERIMENT_ID"
	rm -rf "$setup_tmp_dir"
	setup_tmp_dir=""
}

wait_for_local_server() {
	while ! curl -fsS --max-time 1 "$TRACKING_URI/health" >/dev/null 2>&1; do
		sleep 0.5
	done
}

configure_local() {
	local_port=5000
	if command -v lsof >/dev/null 2>&1; then
		while lsof -nP -iTCP:"$local_port" -sTCP:LISTEN >/dev/null 2>&1; do
		local_port=$((local_port + 1))
		done
	fi
	TRACKING_URI="http://127.0.0.1:$local_port"
	command -v curl >/dev/null 2>&1 || die "curl is required to wait for the local MLflow server."
	printf '%b○%b  %bStart a local MLflow server%b\n' "$BLUE" "$RESET" "$BOLD" "$RESET" >&2
	primary_detail "Run this command in another terminal:"
	printf '%b│%b  %bmlflow server --port %s%b\n' "$LINE" "$RESET" "$YELLOW" "$local_port" "$RESET" >&2
	primary_detail "This setup will continue when the server is ready. Press Ctrl+C to stop waiting."
	printf '%b│%b\n' "$LINE" "$RESET" >&2
	run_with_spinner "Waiting for the local MLflow server…" wait_for_local_server
	success "Local MLflow server connected" "$TRACKING_URI"
	if [ -z "$EXPERIMENT_NAME" ]; then
		prompt_text "Experiment name" "$repo_name"
		EXPERIMENT_NAME=$prompt_value
	fi
	success "Experiment" "$EXPERIMENT_NAME"
}

build_agent_prompt() {
	printf '%s\n' \
		"# MLflow Tracing Setup" \
		"" \
		"Instrument the application in this repository with MLflow Tracing." \
		"Inspect the project first. If there is more than one application entry point, ask which one to instrument; otherwise proceed without setup questions." \
		""
	case "$backend" in
	databricks)
		printf '%s\n' \
			"The setup wizard already provisioned these resources. Do not recreate them:" \
			"- Tracking URI: $TRACKING_URI" \
			"- Experiment ID: $EXPERIMENT_ID" \
			"- Experiment name: $EXPERIMENT_NAME" \
			""
		if [ -n "$trace_destination" ]; then
			printf '%s\n' "- Unity Catalog trace destination: $trace_destination"
		fi
		if [ -n "$WAREHOUSE_ID" ]; then
			printf '%s\n' "- SQL warehouse ID: $WAREHOUSE_ID"
		fi
		printf '%s\n' \
			"" \
			"Add the latest mlflow-tracing package and configure these non-secret values using the project's conventions:" \
			"MLFLOW_TRACKING_URI=$TRACKING_URI" \
			"MLFLOW_EXPERIMENT_ID=$EXPERIMENT_ID"
		if [ -z "$PROFILE" ]; then
			printf '%s\n' "DATABRICKS_HOST=$WORKSPACE_URL"
		fi
		if [ -n "$WAREHOUSE_ID" ]; then
			printf '%s\n' "MLFLOW_TRACING_SQL_WAREHOUSE_ID=$WAREHOUSE_ID"
		fi
		if [ -n "$trace_destination" ]; then
			uc_catalog_name=${trace_destination%%.*}
			uc_location_remainder=${trace_destination#*.}
			uc_schema_name=${uc_location_remainder%%.*}
			uc_table_prefix=${uc_location_remainder#*.}
			printf '%s\n' \
				"" \
				"For a Python application, activate the UC-backed experiment before enabling autologging:" \
				"from mlflow.entities.trace_location import UnityCatalog" \
				"mlflow.set_experiment(" \
				"    experiment_id=\"$EXPERIMENT_ID\"," \
				"    trace_location=UnityCatalog(" \
				"        catalog_name=\"$uc_catalog_name\"," \
				"        schema_name=\"$uc_schema_name\"," \
				"        table_prefix=\"$uc_table_prefix\"," \
				"    )," \
				")" \
				"Use this exact Unity Catalog destination. Do not replace it with MlflowExperimentLocation, which targets MLflow experiment storage rather than the configured UC tables."
		fi
		;;
	remote)
		printf '%s\n' \
			"The setup wizard already resolved these resources. Do not recreate them:" \
			"- Tracking URI: $TRACKING_URI" \
			"- Experiment ID: $EXPERIMENT_ID" \
			"- Experiment name: $EXPERIMENT_NAME" \
			"" \
			"Add the latest mlflow-tracing package and configure MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID using the project's conventions."
		if [ -n "${MLFLOW_WORKSPACE:-}" ]; then
			printf '%s\n' "MLFLOW_WORKSPACE=$MLFLOW_WORKSPACE"
		fi
		;;
	local)
		printf '%s\n' \
			"The user already started a local MLflow server at $TRACKING_URI. Do not start another server." \
			"Add the latest mlflow-tracing package and use experiment $EXPERIMENT_NAME."
		;;
	esac
	printf '%s\n' \
		"" \
		"Enable the framework-specific tracing integration before the LLM client is created, such as mlflow.openai.autolog() or mlflow.langchain.autolog()." \
		"Do not add evaluation code, write credentials into the repository, or create setup-only files." \
		"Run the application, exercise one traced operation, and confirm a trace reaches the experiment." \
		"Finally report the MLflow version, modified files, and trace URL."
}

launch_agent() {
	agent_prompt=$(build_agent_prompt)
	footer "Launching ${agent_display}…"
	if [ "${MLFLOW_SETUP_DRY_RUN:-}" = "1" ]; then
		printf '%s\n' "$agent_prompt"
		return
	fi
	cd "$repo_root"
	case "$agent_choice" in
	claude | codex) exec "$agent_choice" "$agent_prompt" ;;
	opencode) exec opencode --prompt "$agent_prompt" ;;
	esac
}

main() {
	parse_args "$@"
	validate_agent_name
	header
	inspect_repository
	choose_backend
	case "$backend" in
	databricks) configure_databricks ;;
	remote) configure_remote ;;
	local) configure_local ;;
	esac
	choose_agent
	launch_agent
}

if [ "${MLFLOW_SETUP_SKIP_MAIN:-0}" != "1" ]; then
	main "$@"
fi
