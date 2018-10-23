#Compiling MXNet

In order to compile MXNet for the Lambda function, you would need an environment running the AMI that the Lambda service is using. Currently the version is:

AMI – amzn-ami-hvm-2017.03.1.20170812-x86_64-gp2

https://docs.aws.amazon.com/lambda/latest/dg/current-supported-versions.html


```
sudo yum groupinstall -y "Development Tools" && sudo yum install -y git

sudo yum install atlas-devel

sudo yum install openblas openblas-devel.x86_64 lapack-devel.x86_64

git clone --recursive https://github.com/apache/incubator-mxnet mxnet
cd mxnet

make -j $(nproc) USE_OPENCV=0 USE_CUDNN=0 USE_CUDA=0 USE_BLAS=openblas USE_LAPACK=1

cd python

sudo python setup.py install

cd

mkdir mxnetpackage

cp -r /usr/local/lib/python2.7/site-packages/mxnet-1.3.1-py2.7.egg/mxnet mxnetpackage/

cd mxnetpackage

sudo pip install numpy

cp -r /usr/local/lib/python2.7/site-packages/numpy-1.15.2-py2.7-linux-x86_64.egg/numpy/ .

mkdir lib

cp /usr/lib64/atlas/libatlas.so.3 lib/

cp /usr/lib64/atlas/libcblas.so.3 lib/

cp /usr/lib64/atlas/libclapack.so.3 lib/

cp /usr/lib64/atlas/libf77blas.so.3 lib/

cp /usr/lib64/libgfortran.so.3 lib/

cp /usr/lib64/libgfortran.so.3.0.0 lib/

cp /usr/lib64/libgomp.so.1 lib/

cp /usr/lib64/libgomp.so.1.0.0 lib/

cp /usr/lib64/atlas/liblapack.so.3 lib/

cp /usr/lib64/libopenblas.so.0 lib/

cp /usr/lib64/atlas/libptcblas.so.3 lib/

cp /usr/lib64/atlas/libptf77blas.so.3 lib/

cp /usr/lib64/libquadmath.so.0 lib/

tar cfz mxnet.tar.gz *

aws s3 cp mxnet.tar.gz s3://{your_bucket}/

```