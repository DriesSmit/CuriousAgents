all:
	tensorboard --bind_all --logdir='./logs'  &
	./ca_env/bin/python main.py hydra.job.chdir=False hydra.output_subdir=null hydra.run.dir=.
	
build:
	python -m venv ca_env
	./ca_env/bin/pip install -r ./requirements.txt

sudo_clean:
	sudo rm -r ./logs
