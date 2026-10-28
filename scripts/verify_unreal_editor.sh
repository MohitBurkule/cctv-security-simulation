#!/usr/bin/env bash
# Check an installed Unreal Editor actually has what we need it for.
#
# "It installed" is not the question. The reason for installing 5.8 at all is
# the MetaHuman parametric body and wardrobe — the one capability the packaged
# UnrealZoo runtime does not expose — so this checks for that specifically
# rather than reporting success because a binary exists.
#
# This is the same lesson as the twice-repeated capability bug in docs/11: a
# feature was reported working because a command appeared in a list. Presence of
# a plugin directory is still only [A] "it is there", not [V] "it works" — but
# it is a far better signal than the installer's exit code.

set -uo pipefail

UE_ROOT=${UE_ROOT:-/opt/unreal-engine}
ENGINE="$UE_ROOT/Engine"
EDITOR="$ENGINE/Binaries/Linux/UnrealEditor"

ok=0
say() { printf '  %-42s %s\n' "$1" "$2"; }

echo "Unreal Editor at $UE_ROOT"

if [ ! -d "$ENGINE" ]; then
  echo "  NOT INSTALLED — no $ENGINE" >&2
  exit 1
fi

# --- version ----------------------------------------------------------------
version_file="$ENGINE/Build/Build.version"
if [ -f "$version_file" ]; then
  ver=$(grep -oE '"MajorVersion": *[0-9]+|"MinorVersion": *[0-9]+|"PatchVersion": *[0-9]+' \
        "$version_file" | grep -oE '[0-9]+$' | paste -sd.)
  say "engine version" "$ver"
  major=${ver%%.*}; rest=${ver#*.}; minor=${rest%%.*}
  # Parametric MetaHuman bodies arrived in 5.6. Below that, the whole reason
  # for this install is absent and nothing downstream will work.
  if [ "${major:-0}" -gt 5 ] || { [ "${major:-0}" -eq 5 ] && [ "${minor:-0}" -ge 6 ]; }; then
    say "  supports parametric MetaHuman" "yes (>= 5.6)"
  else
    say "  supports parametric MetaHuman" "NO — needs >= 5.6"; ok=1
  fi
else
  say "engine version" "unknown (no Build.version)"
fi

# --- editor binary ----------------------------------------------------------
if [ -x "$EDITOR" ]; then
  say "editor binary" "present"
else
  say "editor binary" "MISSING at $EDITOR"; ok=1
fi

# --- the plugins we came for ------------------------------------------------
echo "  plugins:"
for name in MetaHuman MetaHumanCharacter MetaHumanSDK RigLogic ControlRig \
            LiveLink Animation; do
  hit=$(find "$ENGINE/Plugins" -maxdepth 4 -type d -iname "*${name}*" 2>/dev/null | head -1)
  if [ -n "$hit" ]; then
    say "    $name" "${hit#$ENGINE/Plugins/}"
  else
    say "    $name" "not found"
  fi
done

# --- what is NOT here -------------------------------------------------------
# Worth stating plainly so nobody assumes the install finished the job.
cat <<'NOTE'

  Still required before wardrobe/gait work:
    - a UE project to work in (the UnrealZoo maps are cooked .pak content and
      cannot be un-cooked into an editable project)
    - the UnrealCV+ plugin compiled against this engine, so Python can drive it
NOTE

exit $ok
