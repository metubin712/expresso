#ssh gcn1
module load 2019
module load eb
module unload Python
module load Python/3.6.6-fosscuda-2018b
module load  TensorFlow/2.2.0-fosscuda-2018b-Python-3.6.6 
pip install --user pyyaml
python -c "import tensorflow as tf; tf.test.is_gpu_available();"
