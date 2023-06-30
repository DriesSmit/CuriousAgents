run:
	tensorboard --bind_all --logdir='./logs'  &
	./ca_env/bin/python main.py hydra.job.chdir=False hydra.output_subdir=null hydra.run.dir=.
	
build:
	sudo apt-get install xvfb
	python -m venv ca_env
	./ca_env/bin/pip install --upgrade "jax[cuda]"
	./ca_env/bin/pip install -r ./requirements.txt
	
	# ./ca_env/bin/pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

sudo_clean:
	sudo rm -r ./logs
