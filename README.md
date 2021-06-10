# faceMaskDetector

for working on raspberry pi 4:

opencv:

sudo apt-get install libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
pip3 install opencv-contrib-python

tf: 

sudo apt install -y libatlas-base-dev liblapacke-dev gfortran
sudo apt install -y libhdf5-dev libhdf5-103

keras==2.3.1
numpy==1.18.2
streamlit==0.65.2

wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.4.0-cp37-none-linux_armv7l_download.sh"
sudo bash ./tensorflow-2.4.0-cp37-none-linux_armv7l_download.sh
pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl

MLX sensor:

install through pip install -e PyMLX90614

audio: 

sudo apt-get install ffmpeg libavcodec-extra
pip3 install pydub
