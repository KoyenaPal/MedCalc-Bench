#!/bin/bash
mkdir -p logs
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/$ensemble_$timestamp.log"
echo "Running: python run_ensemble.py --prompt zero_shot"
echo "Logging to: $logfile"
    
# Run the command
python run_ensemble.py --prompt zero_shot > "$logfile" 2>&1

# Check exit code
if [ $? -ne 0 ]; then
  echo "❌ Error running: model=$model target_model=$targetmodel  (check $logfile)"
  # Uncomment next line to exit on error:
  # exit 1
fi


echo "✅ All runs completed."
