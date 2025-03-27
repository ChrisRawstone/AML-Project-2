
# First, log into the A100 node and then execute the following commands
a100sh << 'EOF'
module load python3/3.10.14
python3 -m venv venv
source venv/bin/activate
module load cuda/11.8
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
EOF
