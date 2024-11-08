function Process-CutFile {
    param (
        [string]$cutFilePath,
        [string]$generation,
        [bool]$parallel,
        [string]$outputFolder
    )
    try {
        .\CutFileParserCLI.exe $cutFilePath $generation $parallel $outputFolder
    } catch {
        $parallelOption = if ($parallel) { "with" } else { "without" }
        Write-Host "Failed to process $cutFilePath $parallelOption parallel option."
        Write-Host $_.Exception.Message
    }
}

function Compare-Files {
    param (
        [string]$parallelFolder,
        [string]$nonParallelFolder
    )
    $parallelFiles = Get-ChildItem -Path $parallelFolder -Filter *.csv
    $nonParallelFiles = Get-ChildItem -Path $nonParallelFolder -Filter *.csv

    foreach ($parallelFile in $parallelFiles) {
        $nonParallelFile = $nonParallelFiles | Where-Object { $_.Name -eq $parallelFile.Name }
        if ($nonParallelFile) {
            try {
                .\csv_comparator.exe $parallelFile.FullName $nonParallelFile.FullName 10
                Write-Host "Success: $($parallelFile.Name) matches."
            } catch {
                Write-Host "Error: $($parallelFile.Name) does not match."
                Write-Host $_.Exception.Message
            }
        } else {
            Write-Host "Error: Corresponding file for $($parallelFile.Name) not found in non_parallel folder."
        }
    }
}

# Process MissyDataSet
Process-CutFile "C:\Data\MissyDataSet\Missy_Disc1\Cutfiles\CoroPlus_240918-135243.cut" "Gen2" $true "C:\Users\nq9093\Downloads\missy_parallel"
Process-CutFile "C:\Data\MissyDataSet\Missy_Disc1\Cutfiles\CoroPlus_240918-135243.cut" "Gen2" $false "C:\Users\nq9093\Downloads\missy_not_parallel"

# Process JorgensData
Process-CutFile "C:\Data\JorgensData\batch\Heat Treated HRC48_SS2541_TR-DC1304-F 4415.cut" "Gen1" $true "C:\Users\nq9093\Downloads\J_parallel"
Process-CutFile "C:\Data\JorgensData\batch\Heat Treated HRC48_SS2541_TR-DC1304-F 4415.cut" "Gen1" $false "C:\Users\nq9093\Downloads\J_not_parallel"

# Compare files
Compare-Files "C:\Users\nq9093\Downloads\missy_parallel" "C:\Users\nq9093\Downloads\missy_not_parallel"
Compare-Files "C:\Users\nq9093\Downloads\J_parallel" "C:\Users\nq9093\Downloads\J_not_parallel"
