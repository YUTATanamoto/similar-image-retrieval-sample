# Similar image retrieval on Docker  
## Usage  
```
$ sudo docker run -it -v $PWD:/simila-image-retrieval --gpus all --name simila-image-retrieval -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3  
$ cd simila-image-retrieval
$ pip install -r requirements.txt
$ jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root
```
