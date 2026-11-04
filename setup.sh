
#echo "Settting up root ...."
#sroot
setupATLAS; asetup Athena,25.0.69

echo "Activate venv ...."
SCRIPT_DIR=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
source $SCRIPT_DIR/../venv/bin/activate

echo "Adding package files to PYTHONPATH .."
src_dir=/mnt/nvme0n1p4/HEP/Work/Project-W-Z/MotorHead/rdf-analysis-framework/RDFAnalysisFramework/src
export PYTHONPATH=$src_dir:${PYTHONPATH}

echo "Adding analysis files to PATH .."
analyses_dir=$src_dir/rdf_analysis/analyses
export PATH=$PATH:$analyses_dir
