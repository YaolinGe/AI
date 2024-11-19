git add .

$commit_message = Read-Host "Enter commit message"

git commit -m "$commit_message"
git push --all