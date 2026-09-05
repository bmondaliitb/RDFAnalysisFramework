echo "Activate venv ...."
source /mnt/nvme0n1p4/HEP/Work/Project-W-Z/MotorHead/venv/bin/activate

echo "Settting up root ...."
sroot

echo "Adding analysis files to PATH .."
analyses_dir=/mnt/nvme0n1p4/HEP/Work/Project-W-Z/MotorHead/rdf-analysis-framework/RDFAnalysisFramework/src/rdf_analysis/analyses/
export PATH=$PATH:$analyses_dir
