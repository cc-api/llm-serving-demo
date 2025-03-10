# 环境
sudo apt-get install software-properties-common
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone git@hf.co:mistralai/Mistral-7B-v0.1

# 依赖包裹
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
sudo tar xzf Python-3.8.12.tgz

cd Python-3.8.12
sudo ./configure --enable-optimizations --with-system-ffi
sudo make altinstall


python3.8 --version


curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.8 get-pip.py
pip3 --version



apt-get install python3-venv

# 创建虚拟环境
python3 -m venv rag_env

# 激活虚拟环境
source rag_env/bin/activate

# 安装依赖
pip install --upgrade pip

pip install --upgrade accelerate autoawq transformers

pip install -r requirements.txt

