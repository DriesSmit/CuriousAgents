run:
	tensorboard --bind_all --logdir='./logs'  &
	./ca_env/bin/python main.py hydra.job.chdir=False hydra.output_subdir=null hydra.run.dir=.
	
build:
	sudo apt-get install xvfb
	python -m venv ca_env
	./ca_env/bin/pip install --upgrade pip
	./ca_env/bin/pip install -r ./requirements.txt
	./ca_env/bin/pip install jax==0.4.9
	./ca_env/bin/pip install jaxlib==0.4.7+cuda11.cudnn82  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
	
sudo_clean:
	sudo rm -r ./logs
