#!/bin/bash

# Prompt for a commit message if not provided as an argument
if [ -z "$1" ]; then
    read -p "Enter commit message: " commit_message
else
    commit_message="$1"
fi

# Add all changes to staging
git add .

# Commit with the provided message
git commit -m "$commit_message"

# Push changes to the remote repository
git push

echo "Changes have been pushed successfully."