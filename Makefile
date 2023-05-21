all:
	./ca_env/bin/python main.py 
	
build:
	python -m venv ca_env
	./ca_env/bin/pip install -r ./requirements.txt

