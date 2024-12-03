# Define source and destination directories
$sourceDirectory = "C:\Users\nq9093\CodeSpace\AI\presentation\repo"  # Change this to your source directory
$destinationDirectory = "C:\Users\nq9093\CodeSpace\AI\presentation\slides"  # Change this to your destination directory

# Create the destination directory if it doesn't exist
if (-Not (Test-Path -Path $destinationDirectory)) {
    New-Item -ItemType Directory -Path $destinationDirectory
}

# Get all PDF files recursively from the source directory and copy them to the destination directory
Get-ChildItem -Path $sourceDirectory -Recurse -Filter *.pdf | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $destinationDirectory
}

Write-Host "All PDF files have been copied to $destinationDirectory"