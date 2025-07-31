<#
    create_submission_archive.ps1
    Usage:
        # With a custom submission name
        ./create_submission_archive.ps1 MyProject

        # Without a custom submission name
        ./create_submission_archive.ps1
#>

param(
    [string]$SubmissionName = ""
)

# ----------------------------------------------------------------------
# 1. Build archive file-name:  submission-[name-]YYYYMMDD-HHMMSS.zip
# ----------------------------------------------------------------------
$timestamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
$archiveName = if ($SubmissionName) {
                   "submission-$SubmissionName-$timestamp.zip"
               } else {
                   "submission-$timestamp.zip"
               }

# ----------------------------------------------------------------------
# 2. Define items to exclude (top-level folders and individual files)
# ----------------------------------------------------------------------
$excludeDirs  = @('scripts', 'examples', 'notebooks')          # folders
$excludeFiles = @('README.md', 'TODO.md', '.gitignore')        # files

# ----------------------------------------------------------------------
# 3. Collect files we *do* want
# ----------------------------------------------------------------------
$itemsToZip = Get-ChildItem -Recurse -File | Where-Object {
    $relative = $_.FullName.Substring($PWD.Path.Length + 1)

    # Does the item live under an excluded top-level directory?
    $topDir   = $relative.Split([IO.Path]::DirectorySeparatorChar)[0]
    if ($excludeDirs -contains $topDir)   { return $false }

    # Is the item one of the excluded root-level files?
    if ($excludeFiles -contains $relative) { return $false }

    return $true
}

# ----------------------------------------------------------------------
# 4. Create the archive
#    (remove any prior archive with the same name so Compress-Archive
#     doesn’t append to it)
# ----------------------------------------------------------------------
if (Test-Path $archiveName) { Remove-Item $archiveName }

Compress-Archive -Path $itemsToZip.FullName `
                 -DestinationPath $archiveName `
                 -Force

Write-Host "Created archive: $archiveName"
