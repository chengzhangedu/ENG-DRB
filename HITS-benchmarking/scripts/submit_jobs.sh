#!/bin/bash

# Script to submit DISRPT training jobs to SLURM

echo "=== DISRPT Training Job Submission ==="
echo "Timestamp: $(date)"

# Make sure scripts are executable
chmod +x scripts/task12_tutorial.slurm
chmod +x scripts/task3_tutorial.slurm

# Submit Task 1&2 job
echo "Submitting Task 1&2 (Segmentation + Connective Detection) job..."
task12_job=$(sbatch scripts/task12_tutorial.slurm | awk '{print $4}')
echo "Task 1&2 job submitted with ID: $task12_job"

# Submit Task 3 job (can run in parallel or sequential)
echo "Submitting Task 3 (Relation Classification) job..."
task3_job=$(sbatch scripts/task3_tutorial.slurm | awk '{print $4}')
echo "Task 3 job submitted with ID: $task3_job"

echo ""
echo "Jobs submitted successfully!"
echo "Task 1&2 Job ID: $task12_job"
echo "Task 3 Job ID: $task3_job"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $task12_job,$task3_job"
echo ""
echo "Check outputs:"
echo "  tail -f disrpt_task12_${task12_job}.out"
echo "  tail -f disrpt_task3_${task3_job}.out"
echo ""
echo "Cancel jobs if needed:"
echo "  scancel $task12_job $task3_job"