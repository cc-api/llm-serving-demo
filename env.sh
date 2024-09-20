# 依赖包裹
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
sudo tar xzf Python-3.8.12.tgz

cd Python-3.8.12
sudo ./configure --enable-optimizations
sudo make altinstall


python3.8 --version


sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 1
sudo update-alternatives --config python3

sudo apt install python3-pip

pip3 install fastapi uvicorn


apt-get install python3-venv

# 创建虚拟环境
python3 -m venv rag_env

# 激活虚拟环境
source rag_env/bin/activate

# 安装依赖
pip install -r requirements.txt

