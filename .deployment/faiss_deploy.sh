#!/bin/bash
set -e

ssh -o StrictHostKeyChecking=no riya_verma@34.93.240.227 << 'EOF'
  set -e
  echo "Connected to server"

  cd /home/sarthak_singh/verismart-frt
  echo "Changed to project directory"

  git stash || echo "No local changes to save"
  echo "Stashed changes (if any)"

  git checkout faiss
  git fetch origin faiss
  git reset --hard origin/faiss
  git pull origin faiss
  echo "Git updated to latest faiss branch"

  echo "Restarting verismart-frt.service..."
  sudo systemctl restart verismart-frt.service
  echo "Restart complete"
EOF

exit 0
