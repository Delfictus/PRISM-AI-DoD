#!/bin/bash

for i in 2 3 4 5 6 7 8; do
    WORKER_DIR="/home/diddy/Desktop/PRISM-Worker-$i"
    
    echo "Committing Worker $i vault..."
    cd "$WORKER_DIR"
    
    git add .worker-vault/
    git commit -m "Add Worker $i specialized vault with constitution and tasks"
    git push origin worker-$i-*
    
    echo "✅ Worker $i vault pushed"
done

echo ""
echo "✅ All 8 workers have specialized vaults committed and pushed"
