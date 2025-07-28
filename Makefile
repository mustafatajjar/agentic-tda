ENV_NAME=test

env:
	git clone https://github.com/ad-freiburg/grasp
	conda create -n $(ENV_NAME) python=3.12 -y:
	cd grasp
	conda config --add channels conda-forge
	conda config --add channels nvidia
	conda config --add channels pytorch
	conda config --set channel_priority strict
	conda install -n $(ENV_NAME) -c pytorch -c nvidia faiss-gpu=1.11.0
	conda run -n $(ENV_NAME) pip install -e . -r requirements.txt
	cd ..

data:
	mkdir -p grasp/data/kg-index
	wget -P grasp/data/kg-index https://ad-publications.cs.uni-freiburg.de/grasp/kg-index/wikidata.tar.gz
	tar -xzf grasp/data/kg-index/wikidata.tar.gz -C grasp/data/kg-index
