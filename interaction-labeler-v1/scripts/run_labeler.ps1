param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Arguments
)

$Project = Resolve-Path (Join-Path $PSScriptRoot "..")
uv run --project $Project interaction-labeler run @Arguments
