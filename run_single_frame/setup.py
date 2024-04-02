import subprocess

subprocess.call('pip install --upgrade pip', shell=True)
subprocess.call('pip install -r requirements.txt', shell=True)
subprocess.call('pip uninstall openexr', shell=True)
subprocess.call('uninstall openexr', shell=True)
subprocess.call('sudo apt-get remove -y openexr', shell=True)
subprocess.call('sudo apt-get install -y openexr', shell=True)
subprocess.call('pip install git+https://github.com/jamesbowman/openexrpython.git', shell=True)