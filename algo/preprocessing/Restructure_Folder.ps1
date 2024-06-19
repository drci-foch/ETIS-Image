# Define the root directory
$rootDir = "D:\THROMBMICS-ALARMS_20240531"

# Define the mapping of source directories to target directories
$dirMapping = @{
    "CLOT_SEG" = "MASK"
    "T2star_" = "SWI"
    "TOF3D" = "TOF3D"
}

# Create target directories if they do not exist
foreach ($targetDir in $dirMapping.Values) {
    $fullTargetDir = Join-Path -Path $rootDir -ChildPath $targetDir
    if (-not (Test-Path -Path $fullTargetDir)) {
        New-Item -ItemType Directory -Path $fullTargetDir
    }
}

# Get the list of unique item directories
$itemDirs = Get-ChildItem -Path $rootDir -Directory | Where-Object { $_.Name -ne "CLOT_SEG" -and $_.Name -ne "T2star_" -and $_.Name -ne "TOF3D" }

foreach ($itemDir in $itemDirs) {
    foreach ($sourceDirName in $dirMapping.Keys) {
        $targetDirName = $dirMapping[$sourceDirName]
        $sourceDir = Join-Path -Path $itemDir.FullName -ChildPath $sourceDirName
        $destinationDir = Join-Path -Path $rootDir -ChildPath $targetDirName

        if (Test-Path -Path $sourceDir) {
            # Move all files from the source to the destination directory
            Get-ChildItem -Path $sourceDir -File | ForEach-Object {
                $sourceFile = $_.FullName
                $destinationFile = Join-Path -Path $destinationDir -ChildPath $_.Name
                Move-Item -Path $sourceFile -Destination $destinationFile
            }
        }
    }

    # Optionally, remove the empty item directory after moving files
    #Remove-Item -Path $itemDir.FullName -Recurse -Force
}

Write-Host "Directory structure has been updated."