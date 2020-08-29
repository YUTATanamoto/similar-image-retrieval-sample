# Runway Retriever  
Sample code for similar runway image retrieval  

## Requirement  
- docker  
- docker-compose  

## Usage  
### Clone this repository  
```
$ git clone https://github.com/YUTATanamoto/similar-image-retrieval-sample.git  
```
### Start container
```
$ cd similar-image-retrieval-sample  
$ docker-compose up
```
- above command will launch jupyter notebook  
- access url start with 'http://127.0.0.1:8888' displayed on the terminal  
- open notebooks/deeplab-efficientnet-annoy.ipynb  

### Stop and delete container
```
$ docker-compose down --rmi all --volumes
```
