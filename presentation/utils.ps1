# Define the original filename
# List all possible files that start with Course-3
$files = Get-ChildItem -Path "C:/Users/nq9093/CodeSpace/AI/presentation/slides" -Filter "Course-2*"

# Filter files with length equal to "Course-2-Week-4.pdf"
$files = $files | Where-Object { $_.Name.Length -eq "Course-2-Week-4.pdf".Length }

# Define the original filename
foreach ($file in $files) {
    $originalFilename = $file.Name

    # Use regex to replace the first occurrence of "3" with "2"
    $updatedFilename = $originalFilename -replace "Course-2-", "Course-3-"

    # Rename the file
    Rename-Item -Path $file.FullName -NewName $updatedFilename

    # Output the updated filename
    Write-Output $updatedFilename
}

